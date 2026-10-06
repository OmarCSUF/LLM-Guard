"""
Dataset Builder
===============
Builds the Layer-1 training pool and the held-out evaluation benchmarks from
the source files fetched by scripts/fetch_datasets.py.

Replaces scripts/generate_prompt_pool.py (a template pool in which every test
prompt also appears in training) and scripts/prepare_data.py (unlabelled or
single-class benchmarks, written where run_evaluation.py never looked).

Evaluation benchmarks, held out and never trained on
(<benchmarks_dir>/<name>/test.jsonl, the layout run_evaluation.py reads):
  jailbreakbench : JBB-Behaviors, harmful + benign
  xstest         : XSTest, safe + unsafe
  harmbench      : HarmBench standard subset, harmful only
  toxicchat      : ToxicChat 0124 test split
  orbench        : OR-Bench hard-1k (safe prompts that sound unsafe) + OR-Bench
                   toxic (harmful); an over-refusal test from a different
                   generator than the WildJailbreak training prompts

Training pool for Layer 1 (<distillation_pool_dir>/{train,val,test}.jsonl):
  benign  : Dolly-15k instructions without a context passage;
            ToxicChat 0124 train prompts that are neither toxic nor jailbreaking;
            WildJailbreak vanilla_benign (written to resemble harmful prompts)
  harmful : BeaverTails prompts whose responses are mostly unsafe;
            ToxicChat 0124 train prompts that are toxic or jailbreaking;
            WildJailbreak vanilla_harmful
  BeaverTails prompts whose responses are mostly safe are left out rather than
  used as benign, because a safe response can be a refusal of a harmful request.

Cleaning, in order:
  1. normalise text (NFKC, lowercase, collapsed whitespace); drop empty prompts
  2. group near-duplicates (character 5-gram Jaccard >= threshold; candidates
     from MinHash LSH, every pair confirmed exactly) and keep one prompt per
     group; groups whose prompts disagree on the label are dropped
  3. drop every pool prompt that is a near-duplicate of an evaluation prompt
  4. balance the classes by downsampling, capped per class. WildJailbreak
     supplies the same number of prompts to both classes, so its writing
     style carries no label signal; the rest of each class comes from the
     other sources. (WildJailbreak is subsampled per class before step 1
     to keep the near-duplicate pass fast.)
  5. split each class into train/val/test; with groups collapsed, no prompt
     has a near-duplicate in another split

The overlap checks at the end use independent hash permutations, so they can
catch pairs the clustering pass missed. metadata.json records counts at each
step, the checks (all expected 0), source revisions, parameters and run_info.

Nothing reaches the destinations until those checks pass: the benchmark files,
the pool splits and metadata.json are staged under <destination>.staging beside
each output directory and moved into place one rename at a time only when every
check is 0. A run whose checks fail leaves the previous dataset exactly as it
was and the rejected build in the staging directories, its failing counts under
overlap_checks_failed in the staged metadata.json, and exits 1.

Labels follow config.yaml → data.labels: 1 = allow, 0 = mitigate.

Usage:
    python scripts/build_datasets.py
    python scripts/build_datasets.py --max-per-class 8000 --near-dup-threshold 0.8
    python scripts/build_datasets.py --paired-per-class 4000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import gzip
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("build_datasets")

ALLOW, MITIGATE = 1, 0

# Source files, relative to the fetch output directory (see fetch_datasets.py).
FILES = {
    "jbb_harmful":     "JailbreakBench/JBB-Behaviors/data/harmful-behaviors.csv",
    "jbb_benign":      "JailbreakBench/JBB-Behaviors/data/benign-behaviors.csv",
    "xstest":          "walledai/XSTest/data/train-00000-of-00001.parquet",
    "harmbench":       "walledai/HarmBench/standard/train-00000-of-00001.parquet",
    "toxicchat_train": "lmsys/toxic-chat/data/0124/toxic-chat_annotation_train.csv",
    "toxicchat_test":  "lmsys/toxic-chat/data/0124/toxic-chat_annotation_test.csv",
    "dolly":           "databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl",
    "beavertails":     "PKU-Alignment/BeaverTails/round0/30k/train.jsonl.gz",
    "wildjailbreak":   "allenai/wildjailbreak/train/train.tsv",
    "orbench_hard":    "bench-llm/or-bench/or-bench-hard-1k.csv",
    "orbench_toxic":   "bench-llm/or-bench/or-bench-toxic.csv",
}

BENCHMARKS = ("jailbreakbench", "xstest", "harmbench", "toxicchat", "orbench")
SHINGLE = 5

# Sources that write both classes with one generator; balance() takes the
# same number of prompts from each of their classes.
PAIRED_SOURCES = ("wildjailbreak",)

# Sources that can only ever supply one class let a classifier learn the source
# instead of the harm; --source-balance caps or drops them, and source_shortcut()
# measures what is left.
WILDJAILBREAK_TYPES = {
    "vanilla_harmful": MITIGATE, "vanilla_benign": ALLOW,
    "adversarial_harmful": None, "adversarial_benign": None,   # jailbreak rewrites: not used
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Layer-1 pool and evaluation benchmarks")
    p.add_argument("--raw-dir", default=None,
                   help="Fetched sources (default: <data.raw_dir>/hf)")
    p.add_argument("--pool-dir", default=None,
                   help="Pool output (default: data.distillation_pool_dir)")
    p.add_argument("--benchmarks-dir", default=None,
                   help="Benchmark output (default: data.benchmarks_dir)")
    p.add_argument("--near-dup-threshold", type=float, default=0.8,
                   help="Character 5-gram Jaccard at or above which prompts count as near-duplicates")
    p.add_argument("--max-per-class", type=int, default=10000)
    p.add_argument("--source-balance", choices=["off", "cap", "strict"], default="off",
                   help="off: the shipped behaviour. cap: a source that only appears in one "
                        "class fills at most --single-class-cap of it. strict: only sources "
                        "that supply both classes, in equal numbers, so the source of a "
                        "prompt says nothing about its label")
    p.add_argument("--single-class-cap", type=float, default=0.25,
                   help="With --source-balance cap, the largest share of a class one "
                        "single-class source may fill")
    p.add_argument("--paired-per-class", type=int, default=5000,
                   help="WildJailbreak prompts per class in the pool (the same number in both classes)")
    p.add_argument("--wildjailbreak-candidates", type=int, default=15000,
                   help="WildJailbreak prompts per class sampled before cleaning")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--num-perm", type=int, default=64, help="MinHash permutations")
    p.add_argument("--bands", type=int, default=16, help="LSH bands (must divide --num-perm)")
    return p.parse_args()


# ── Text ──────────────────────────────────────────────────────────────────────

def normalise(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _record(rid: str, text, label: int, category, source: str) -> dict:
    return {
        "id":       rid,
        "text":     text.strip() if isinstance(text, str) else "",
        "label":    int(label),
        "category": category if isinstance(category, str) and category else "none",
        "source":   source,
    }


def _label_counts(records) -> dict:
    c = Counter(r["label"] for r in records)
    return {"allow": c.get(ALLOW, 0), "mitigate": c.get(MITIGATE, 0)}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _csv(raw: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(raw / FILES[key], keep_default_na=False)


def load_jailbreakbench(raw: Path) -> list:
    out = []
    for key, label in (("jbb_harmful", MITIGATE), ("jbb_benign", ALLOW)):
        kind = key.split("_", 1)[1]
        for r in _csv(raw, key).to_dict("records"):
            out.append(_record(f"jbb_{kind}_{r['Index']}", r["Goal"], label, r["Category"], "jailbreakbench"))
    return out


def load_xstest(raw: Path) -> list:
    df = pd.read_parquet(raw / FILES["xstest"])
    mapping = {"safe": ALLOW, "unsafe": MITIGATE}
    unknown = sorted(set(df["label"]) - set(mapping))
    if unknown:
        raise ValueError(f"XSTest has unexpected label values {unknown}; expected safe/unsafe")
    return [_record(f"xstest_{i}", r["prompt"], mapping[r["label"]], r["type"], "xstest")
            for i, r in enumerate(df.to_dict("records"))]


def load_harmbench(raw: Path) -> list:
    df = pd.read_parquet(raw / FILES["harmbench"])
    return [_record(f"harmbench_{i}", r["prompt"], MITIGATE, r["category"], "harmbench")
            for i, r in enumerate(df.to_dict("records"))]


def load_orbench(raw: Path) -> list:
    """OR-Bench hard-1k prompts are safe (allow); OR-Bench toxic prompts are harmful."""
    out = []
    for key, label in (("orbench_hard", ALLOW), ("orbench_toxic", MITIGATE)):
        kind = key.split("_", 1)[1]
        for i, r in enumerate(_csv(raw, key).to_dict("records")):
            out.append(_record(f"orbench_{kind}_{i}", r["prompt"], label, r["category"], "orbench"))
    return out


def load_toxicchat(raw: Path, split: str) -> tuple:
    """Harmful when toxic or jailbreaking. Returns (records, n_human_annotated)."""
    df = _csv(raw, f"toxicchat_{split}")
    out = []
    for r in df.to_dict("records"):
        toxic, jailbreak = int(r["toxicity"]) == 1, int(r["jailbreaking"]) == 1
        category = "jailbreak" if jailbreak else ("toxicity" if toxic else "none")
        out.append(_record(f"toxicchat_{split}_{r['conv_id']}", r["user_input"],
                           MITIGATE if (toxic or jailbreak) else ALLOW, category, f"toxicchat_{split}"))
    human = int(df["human_annotation"].astype(str).str.lower().isin(["true", "1"]).sum())
    return out, human


def load_dolly(raw: Path) -> tuple:
    """Instructions without a context passage. Returns (records, n_skipped_with_context)."""
    out, skipped = [], 0
    with open(raw / FILES["dolly"], encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if str(r.get("context") or "").strip():
                skipped += 1
                continue
            out.append(_record(f"dolly_{i}", r["instruction"], ALLOW, r.get("category"), "dolly"))
    return out, skipped


def load_beavertails(raw: Path) -> tuple:
    """
    One record per prompt, harmful when most of its responses are unsafe.

    is_safe labels a question-answer pair, so prompts whose responses are mostly
    safe are excluded (a safe response may be a refusal) and ties are dropped.
    """
    votes = defaultdict(lambda: [0, 0])     # prompt -> [unsafe, safe]
    cats = defaultdict(Counter)
    pairs = 0
    with gzip.open(raw / FILES["beavertails"], "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pairs += 1
            if r["is_safe"]:
                votes[r["prompt"]][1] += 1
            else:
                votes[r["prompt"]][0] += 1
                cats[r["prompt"]].update(k for k, v in (r.get("category") or {}).items() if v)

    out, mostly_safe, ties = [], 0, 0
    for i, (prompt, (unsafe, safe)) in enumerate(votes.items()):
        if unsafe == safe:
            ties += 1
        elif safe > unsafe:
            mostly_safe += 1
        else:
            top = cats[prompt].most_common(1)
            out.append(_record(f"beavertails_{i}", prompt, MITIGATE,
                               top[0][0] if top else "unsafe", "beavertails"))
    stats = {"qa_pairs": pairs, "unique_prompts": len(votes), "harmful_kept": len(out),
             "mostly_safe_excluded": mostly_safe, "ties_dropped": ties}
    return out, stats


def load_wildjailbreak(raw: Path) -> tuple:
    """
    Vanilla harmful and benign prompts from WildJailbreak train.

    One generator wrote both classes and the benign prompts were made to
    resemble harmful ones, so the pair teaches intent rather than style.
    Adversarial rows (jailbreak rewrites) are left out. The TSV is read as the
    dataset card specifies: tab-delimited with keep_default_na=False.
    """
    df = pd.read_csv(raw / FILES["wildjailbreak"], sep="\t", keep_default_na=False,
                     dtype=str, usecols=["vanilla", "data_type"])
    counts = Counter(df["data_type"])
    unknown = sorted(set(counts) - set(WILDJAILBREAK_TYPES))
    if unknown:
        raise ValueError(f"WildJailbreak has unexpected data_type values {unknown[:5]}; "
                         "the TSV may have been misparsed")
    out = []
    for i, r in enumerate(df.to_dict("records")):
        label = WILDJAILBREAK_TYPES[r["data_type"]]
        if label is not None:
            out.append(_record(f"wildjailbreak_{i}", r["vanilla"], label,
                               "unsafe" if label == MITIGATE else "none", "wildjailbreak"))
    return out, {"rows_by_data_type": dict(counts), "vanilla_loaded": _label_counts(out)}


def _sample_per_label(records: list, n: int, rng: random.Random) -> list:
    """At most n records per label, sampled without replacement."""
    out = []
    for label in (ALLOW, MITIGATE):
        recs = [r for r in records if r["label"] == label]
        out += rng.sample(recs, n) if len(recs) > n else recs
    return out


# ── Near-duplicate detection ──────────────────────────────────────────────────

def _shingles(norm: str) -> set:
    if len(norm) <= SHINGLE:
        return {norm}
    return {norm[i:i + SHINGLE] for i in range(len(norm) - SHINGLE + 1)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


class NearDupIndex:
    """
    MinHash LSH over character shingles.  Items sharing any band are candidates;
    a candidate only counts once its exact Jaccard reaches the threshold.
    """

    def __init__(self, num_perm: int, bands: int, seed: int):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2 ** 63, num_perm, dtype=np.uint64) | np.uint64(1)
        self._b = rng.integers(0, 2 ** 63, num_perm, dtype=np.uint64)
        self._bands = bands
        self._width = num_perm // bands
        self._buckets = defaultdict(list)
        self._shingles = []

    def _signature(self, sh: set) -> np.ndarray:
        h = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
             for s in sh),
            dtype=np.uint64, count=len(sh),
        )
        # (a*h + b) mod 2^64 with odd a is a permutation of uint64
        return (self._a[:, None] * h[None, :] + self._b[:, None]).min(axis=1)

    def _keys(self, sig: np.ndarray) -> list:
        w = self._width
        return [(band, sig[band * w:(band + 1) * w].tobytes()) for band in range(self._bands)]

    def add(self, norm: str) -> int:
        sh = _shingles(norm)
        idx = len(self._shingles)
        self._shingles.append(sh)
        for key in self._keys(self._signature(sh)):
            self._buckets[key].append(idx)
        return idx

    def query(self, norm: str, threshold: float) -> list:
        sh = _shingles(norm)
        candidates = set()
        for key in self._keys(self._signature(sh)):
            candidates.update(self._buckets.get(key, ()))
        return sorted(i for i in candidates if _jaccard(sh, self._shingles[i]) >= threshold)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ── Cleaning steps ────────────────────────────────────────────────────────────

def dedup_benchmark(records: list) -> tuple:
    """Drop empty prompts and exact duplicates (after normalisation) within one benchmark."""
    seen, out, dup, empty = set(), [], 0, 0
    for r in records:
        key = normalise(r["text"])
        if not key:
            empty += 1
        elif key in seen:
            dup += 1
        else:
            seen.add(key)
            out.append(r)
    return out, {"empty_dropped": empty, "duplicates_dropped": dup}


def collapse_near_duplicates(records: list, threshold: float, num_perm: int,
                             bands: int, seed: int) -> tuple:
    """Keep one prompt per near-duplicate group; drop groups with mixed labels."""
    norms = [normalise(r["text"]) for r in records]
    live = [i for i, n in enumerate(norms) if n]
    empty = Counter(records[i]["source"] for i, n in enumerate(norms) if not n)

    index = NearDupIndex(num_perm, bands, seed)
    uf = _UnionFind(len(live))
    for pos, i in enumerate(live):
        for other in index.query(norms[i], threshold):
            uf.union(pos, other)
        index.add(norms[i])

    groups = defaultdict(list)
    for pos in range(len(live)):
        groups[uf.find(pos)].append(live[pos])

    out = []
    collapsed, conflicts = Counter(), Counter()
    n_conflict_groups = 0
    for members in groups.values():
        if len({records[i]["label"] for i in members}) > 1:
            n_conflict_groups += 1
            conflicts.update(records[i]["source"] for i in members)
            continue
        out.append(records[members[0]])
        collapsed.update(records[i]["source"] for i in members[1:])

    stats = {
        "empty_dropped_by_source":          dict(empty),
        "duplicates_collapsed_by_source":   dict(collapsed),
        "label_conflict_groups":            n_conflict_groups,
        "label_conflict_dropped_by_source": dict(conflicts),
        "remaining":                        len(out),
    }
    return out, stats


def drop_eval_overlap(pool: list, benchmarks: dict, threshold: float,
                      num_perm: int, bands: int, seed: int) -> tuple:
    """Drop pool prompts that are near-duplicates of any evaluation prompt."""
    index = NearDupIndex(num_perm, bands, seed)
    owner = []
    for name, recs in benchmarks.items():
        for r in recs:
            index.add(normalise(r["text"]))
            owner.append(name)

    kept, by_source, by_benchmark = [], Counter(), Counter()
    for r in pool:
        hits = index.query(normalise(r["text"]), threshold)
        if hits:
            by_source[r["source"]] += 1
            by_benchmark.update({owner[i] for i in hits})
        else:
            kept.append(r)
    return kept, {"dropped_by_source": dict(by_source),
                  "matched_benchmarks": dict(by_benchmark),
                  "remaining": len(kept)}


def source_shortcut(records: list) -> dict:
    """
    How much of the label a prompt's source gives away.

    accuracy is what a predictor scores by guessing each source's majority
    label. 0.5 means the source says nothing; the shipped pool scored far above
    that because BeaverTails appeared only among harmful prompts and Dolly only
    among safe ones, so the classifier could learn the source instead of the harm.
    """
    counts = defaultdict(Counter)
    for r in records:
        counts[r["source"]][r["label"]] += 1
    correct = sum(max(c[ALLOW], c[MITIGATE]) for c in counts.values())
    return {
        "accuracy": round(correct / len(records), 4) if records else None,
        "by_source": {s: {"allow": c[ALLOW], "mitigate": c[MITIGATE]}
                      for s, c in sorted(counts.items())},
    }


def _by_source_label(pool: list) -> dict:
    """{source: {label: [records]}}"""
    grouped = defaultdict(lambda: {ALLOW: [], MITIGATE: []})
    for r in pool:
        grouped[r["source"]][r["label"]].append(r)
    return dict(grouped)


def _scale_to(counts: dict, total: int) -> dict:
    """Scale per-source counts down to `total`, in proportion, by largest remainder."""
    current = sum(counts.values())
    if current <= total or current == 0:
        return dict(counts)
    exact = {s: k * total / current for s, k in counts.items()}
    out = {s: int(v) for s, v in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - out[s]), s))[:total - sum(out.values())]:
        out[s] += 1
    return out


def balance(pool: list, max_per_class: int, rng: random.Random, paired_per_class: int = 0,
            mode: str = "off", single_class_cap: float = 0.25) -> tuple:
    """
    Downsample to the same number of prompts per class, capped at max_per_class.

    mode:
      off     each source in PAIRED_SOURCES contributes the same number of
              prompts to both classes (at most paired_per_class) and the rest of
              each class comes from the other sources. This built the shipped pool.
      cap     as "off" for the two-class sources, but a source that only ever
              appears in one class fills at most single_class_cap of that class.
      strict  only sources that supply both classes, each contributing the same
              number of prompts to each class, so the source of a prompt carries
              no information about its label.
    """
    if mode not in ("off", "cap", "strict"):
        raise ValueError(f"unknown source-balance mode {mode!r}")

    if mode == "off":
        paired = {ALLOW: [], MITIGATE: []}
        other = {ALLOW: [], MITIGATE: []}
        for r in pool:
            (paired if r["source"] in PAIRED_SOURCES else other)[r["label"]].append(r)
        k = min(len(paired[ALLOW]), len(paired[MITIGATE]), paired_per_class, max_per_class)
        available = {"allow": len(other[ALLOW]) + k, "mitigate": len(other[MITIGATE]) + k}
        n = min(available["allow"], available["mitigate"], max_per_class)
        if n == 0:
            raise ValueError(f"cannot balance: available per class {available}")
        limited_by = "max_per_class" if n == max_per_class else min(available, key=available.get)
        sampled = {label: rng.sample(paired[label], k) + rng.sample(other[label], n - k)
                   for label in (ALLOW, MITIGATE)}
        return sampled, {"mode": mode, "available": available, "per_class": n,
                         "paired_per_class": k,
                         "paired_available": {"allow": len(paired[ALLOW]),
                                              "mitigate": len(paired[MITIGATE])},
                         "limited_by": limited_by}

    by_source = _by_source_label(pool)
    two_class = {s: v for s, v in by_source.items() if v[ALLOW] and v[MITIGATE]}
    single = {s: v for s, v in by_source.items() if s not in two_class}
    if not two_class:
        raise ValueError("cannot balance by source: no source supplies both classes")

    # What each two-class source can give to both classes
    capacity = {}
    for s, v in two_class.items():
        k = min(len(v[ALLOW]), len(v[MITIGATE]))
        if s in PAIRED_SOURCES and paired_per_class:
            k = min(k, paired_per_class)
        capacity[s] = k
    base = sum(capacity.values())

    if mode == "strict":
        n = min(base, max_per_class)
        if n == 0:
            raise ValueError("cannot balance by source: two-class sources supply nothing")
        keep = _scale_to(capacity, n)
        fill = {ALLOW: {}, MITIGATE: {}}
    else:
        # The cap is a share of the final size, which depends on the cap; two or
        # three rounds are enough for it to settle.
        n = max_per_class
        for _ in range(10):
            allowance = int(single_class_cap * n)
            extra = {label: sum(min(len(v[label]), allowance) for v in single.values() if v[label])
                     for label in (ALLOW, MITIGATE)}
            available_by_label = {label: min(base, n) + extra[label] for label in (ALLOW, MITIGATE)}
            settled = min(max_per_class, available_by_label[ALLOW], available_by_label[MITIGATE])
            if settled == n:
                break
            n = settled
        if n == 0:
            raise ValueError(f"cannot balance: available per class {available_by_label}")
        keep = _scale_to(capacity, min(base, n))
        allowance = int(single_class_cap * n)
        fill = {ALLOW: {}, MITIGATE: {}}
        for label in (ALLOW, MITIGATE):
            need = n - sum(keep.values())
            for s in sorted(single):
                if need <= 0:
                    break
                take = min(len(single[s][label]), allowance, need)
                if take:
                    fill[label][s] = take
                    need -= take
            if need > 0:
                raise ValueError(f"cannot balance: {need} prompts short in class {label} "
                                 f"under a single-class cap of {single_class_cap}")

    sampled = {ALLOW: [], MITIGATE: []}
    for s, k in sorted(keep.items()):
        for label in (ALLOW, MITIGATE):
            sampled[label] += rng.sample(two_class[s][label], k)
    for label in (ALLOW, MITIGATE):
        for s, k in sorted(fill[label].items()):
            sampled[label] += rng.sample(single[s][label], k)

    n_final = len(sampled[ALLOW])
    stats = {"mode": mode, "per_class": n_final,
             "paired_per_class": keep.get(PAIRED_SOURCES[0], 0) if PAIRED_SOURCES else 0,
             "two_class_per_source": dict(sorted(keep.items())),
             "single_class_per_source": {"allow": dict(sorted(fill[ALLOW].items())),
                                         "mitigate": dict(sorted(fill[MITIGATE].items()))},
             "two_class_capacity": dict(sorted(capacity.items())),
             "dropped_single_class_sources": sorted(single) if mode == "strict" else [],
             "single_class_cap": single_class_cap if mode == "cap" else None,
             "limited_by": "max_per_class" if n_final == max_per_class else "sources"}
    return sampled, stats


def split(by_label: dict, val_frac: float, test_frac: float, rng: random.Random) -> dict:
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        recs = list(by_label[label])
        rng.shuffle(recs)
        n_val, n_test = int(len(recs) * val_frac), int(len(recs) * test_frac)
        splits["val"] += recs[:n_val]
        splits["test"] += recs[n_val:n_val + n_test]
        splits["train"] += recs[n_val + n_test:]
    for recs in splits.values():
        rng.shuffle(recs)
    return splits


def _count_near_dups(queries: list, targets: list, threshold: float,
                     num_perm: int, bands: int, seed: int) -> int:
    index = NearDupIndex(num_perm, bands, seed)
    for r in targets:
        index.add(normalise(r["text"]))
    return sum(1 for r in queries if index.query(normalise(r["text"]), threshold))


def overlap_checks(splits: dict, benchmarks: dict, threshold: float,
                   num_perm: int, bands: int, seed: int) -> dict:
    """Near-duplicate counts between splits and against every benchmark. All should be 0."""
    checks = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            checks[f"{a}_vs_{b}"] = _count_near_dups(splits[a], splits[b], threshold, num_perm, bands, seed)
    for name in names:
        for bench, recs in benchmarks.items():
            checks[f"{name}_vs_{bench}"] = _count_near_dups(splits[name], recs, threshold, num_perm, bands, seed)
    return checks


def _cross_benchmark_exact(benchmarks: dict) -> dict:
    keys = {name: {normalise(r["text"]) for r in recs} for name, recs in benchmarks.items()}
    names = list(keys)
    return {f"{a}_vs_{b}": len(keys[a] & keys[b])
            for i, a in enumerate(names) for b in names[i + 1:]}


def _write_jsonl(records: list, path: Path, warn: bool = True) -> None:
    """
    Write records to *path*.

    warn=False is for a staged write, where *path* is not a results file anyone
    has yet: _promote() warns about the destination when it moves it into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if warn:
        warn_if_overwriting(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _staging_dir(final: Path) -> Path:
    """A sibling of *final*, so a staged file moves into place with one rename."""
    return final.with_name(final.name + ".staging")


def _promote(staged: Path, final: Path) -> None:
    """Move a staged file over its final path, replacing whatever was there."""
    final.parent.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(final)
    os.replace(staged, final)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    run_info = collect_run_info(cfg)
    rng = random.Random(seed)

    raw = Path(args.raw_dir) if args.raw_dir else Path(cfg["data"]["raw_dir"]) / "hf"
    pool_dir = Path(args.pool_dir or cfg["data"]["distillation_pool_dir"])
    bench_dir = Path(args.benchmarks_dir or cfg["data"]["benchmarks_dir"])
    lsh = {"num_perm": args.num_perm, "bands": args.bands}
    threshold = args.near_dup_threshold

    missing = [rel for rel in FILES.values() if not (raw / rel).is_file()]
    if missing:
        logger.error(f"Missing source files under {raw}: {missing}. Run scripts/fetch_datasets.py first.")
        return 1

    # Evaluation benchmarks
    toxic_test, human_test = load_toxicchat(raw, "test")
    loaded = {
        "jailbreakbench": load_jailbreakbench(raw),
        "xstest":         load_xstest(raw),
        "harmbench":      load_harmbench(raw),
        "toxicchat":      toxic_test,
        "orbench":        load_orbench(raw),
    }
    benchmarks, bench_meta = {}, {}
    for name in BENCHMARKS:
        recs, stats = dedup_benchmark(loaded[name])
        benchmarks[name] = recs
        bench_meta[name] = {"loaded": len(loaded[name]), **stats,
                            "final": len(recs), "labels": _label_counts(recs)}
    bench_meta["toxicchat"]["human_annotated"] = human_test
    logger.info("Benchmarks: " + ", ".join(f"{n} {len(r)}" for n, r in benchmarks.items()))

    # Training pool
    dolly, dolly_with_context = load_dolly(raw)
    toxic_train, human_train = load_toxicchat(raw, "train")
    beaver, beaver_stats = load_beavertails(raw)
    wjb_all, wjb_stats = load_wildjailbreak(raw)
    wjb = _sample_per_label(wjb_all, args.wildjailbreak_candidates, rng)
    candidates = dolly + toxic_train + beaver + wjb
    logger.info(f"Pool candidates: dolly {len(dolly)}, toxicchat_train {len(toxic_train)}, "
                f"beavertails {len(beaver)}, wildjailbreak {len(wjb)} "
                f"(sampled from {len(wjb_all)} vanilla prompts)")

    pool, dedup_stats = collapse_near_duplicates(candidates, threshold, seed=seed, **lsh)
    logger.info(f"After near-duplicate collapse: {len(pool)}")
    pool, eval_stats = drop_eval_overlap(pool, benchmarks, threshold, seed=seed, **lsh)
    logger.info(f"After removing evaluation overlap: {len(pool)}")
    shortcut_before = source_shortcut(pool)
    by_label, balance_stats = balance(pool, args.max_per_class, rng, args.paired_per_class,
                                      mode=args.source_balance,
                                      single_class_cap=args.single_class_cap)
    logger.info(f"Balanced: {balance_stats['per_class']} per class, "
                f"{balance_stats['paired_per_class']} of them from WildJailbreak in each class "
                f"(limited by {balance_stats['limited_by']}, source-balance {args.source_balance})")
    if balance_stats.get("dropped_single_class_sources"):
        logger.info("Dropped single-class sources: "
                    + ", ".join(balance_stats["dropped_single_class_sources"]))
    balanced = by_label[ALLOW] + by_label[MITIGATE]
    shortcut_after = source_shortcut(balanced)
    logger.info(f"Source shortcut: a source-majority guess scores "
                f"{shortcut_before['accuracy']} on the candidates and "
                f"{shortcut_after['accuracy']} on the balanced pool (0.5 = the source says nothing)")
    splits = split(by_label, args.val_frac, args.test_frac, rng)

    logger.info("Checking overlaps with independent hash permutations...")
    checks = overlap_checks(splits, benchmarks, threshold, seed=seed + 1, **lsh)
    leaks = {k: v for k, v in checks.items() if v}

    # Everything is written to staging directories beside the real ones and only
    # moved into place once the overlap checks have passed. A rejected run then
    # leaves the previous dataset exactly as it was, rather than a complete-looking
    # one that a later training run would consume.
    stage_pool, stage_bench = _staging_dir(pool_dir), _staging_dir(bench_dir)
    for d in dict.fromkeys((stage_pool, stage_bench)):
        if d.exists():
            logger.info(f"Clearing previous staging directory {d}")
            shutil.rmtree(d)

    staged = []                      # (staged path, final path), in write order
    for name, recs in benchmarks.items():
        rel = Path(name) / "test.jsonl"
        _write_jsonl(recs, stage_bench / rel, warn=False)
        staged.append((stage_bench / rel, bench_dir / rel))
    for name, recs in splits.items():
        _write_jsonl(recs, stage_pool / f"{name}.jsonl", warn=False)
        staged.append((stage_pool / f"{name}.jsonl", pool_dir / f"{name}.jsonl"))

    manifest_path = raw / "manifest.json"
    revisions = {}
    if manifest_path.exists():
        sources = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})
        revisions = {name: {"repo": s["repo"], "revision": s["revision"]} for name, s in sources.items()}
    else:
        logger.warning(f"No fetch manifest at {manifest_path}; source revisions not recorded.")

    metadata = {
        "parameters": {"seed": seed, "near_dup_threshold": threshold, "shingle_chars": SHINGLE,
                       "max_per_class": args.max_per_class, "paired_per_class": args.paired_per_class,
                       "source_balance": args.source_balance,
                       "single_class_cap": args.single_class_cap,
                       "wildjailbreak_candidates_per_class": args.wildjailbreak_candidates,
                       "val_frac": args.val_frac,
                       "test_frac": args.test_frac, **lsh},
        "source_revisions": revisions,
        "benchmarks": bench_meta,
        "benchmark_cross_overlap_exact": _cross_benchmark_exact(benchmarks),
        "pool_sources": {
            "dolly":           {"loaded": len(dolly), "skipped_with_context": dolly_with_context},
            "toxicchat_train": {"loaded": len(toxic_train), "human_annotated": human_train,
                                "labels": _label_counts(toxic_train)},
            "beavertails":     beaver_stats,
            "wildjailbreak":   {**wjb_stats, "sampled_candidates": _label_counts(wjb)},
        },
        "near_duplicate_collapse": dedup_stats,
        "evaluation_overlap_removed": eval_stats,
        "balance": balance_stats,
        "source_shortcut": {"candidates": shortcut_before, "balanced": shortcut_after,
                            "splits": {name: source_shortcut(recs) for name, recs in splits.items()}},
        "splits": {name: {"n": len(recs), "labels": _label_counts(recs),
                          "sources": dict(Counter(r["source"] for r in recs))}
                   for name, recs in splits.items()},
        "overlap_checks": checks,
        # The checks that came back non-zero, empty on an accepted build. A staged
        # build that was rejected keeps its failing counts here.
        "overlap_checks_failed": leaks,
        "accepted": not leaks,
        "run_info": finalize_run_info(run_info),
    }
    staged_meta = stage_pool / "metadata.json"
    staged_meta.parent.mkdir(parents=True, exist_ok=True)
    staged_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    staged.append((staged_meta, pool_dir / "metadata.json"))

    if leaks:
        logger.error(f"Overlap checks failed: {leaks}")
        logger.error(
            f"Nothing was written to {pool_dir} or {bench_dir}; the previous dataset is "
            f"untouched. The rejected build is staged at {stage_pool} and {stage_bench}, "
            f"with the failing counts under overlap_checks_failed in {staged_meta}."
        )
        return 1

    for staged_path, final_path in staged:
        _promote(staged_path, final_path)
    for d in dict.fromkeys((stage_pool, stage_bench)):
        shutil.rmtree(d, ignore_errors=True)
    logger.info(f"Metadata → {pool_dir / 'metadata.json'}")
    logger.info("Splits: " + ", ".join(f"{n} {len(r)}" for n, r in splits.items())
                + " | all overlap checks 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
