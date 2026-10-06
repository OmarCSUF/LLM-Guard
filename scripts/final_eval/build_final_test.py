"""
Build the locked final test set
===============================
Turns the seven fetched source files into one labelled JSONL,
data/benchmarks_final/final_test/test.jsonl, applying the screening rule of
PREREGISTRATION.md, Section 4, in that order and nothing else:

  1. drop empty prompts and exact duplicates within a source;
  2. drop any prompt that is an exact normalised match or a near-duplicate
     (character 5-gram Jaccard >= 0.8, MinHash 64/16, seed 42) of a prompt in
     the pool train/val/test splits or in any of the five development benchmarks;
  3. drop exact cross-source duplicates, keeping the source listed first;
  4. PHTest: sample 1,000 of the surviving `harmless` rows with seed 20260926.

Every count is written to data/benchmarks_final/manifest.json, with the SHA-256
of the output, the fetch manifest's repository revisions, and the rule settings.
The near-duplicate machinery is imported from scripts/build_datasets.py so the
screen is the same one the pool was built with.

Record schema (the one run_evaluation.py reads):
    {"id", "text", "label" (1 = safe, 0 = harmful), "category", "source"}
`source` is one of: sst, catqa, dna_core, dna_other, coconot, phtest, no_robots.

Long-prompt flags (reproducing the exact file): after the four rules, count_long
tokenises every prompt with the Layer-1 backbone tokenizer (roberta-base, from
the local HF cache only, never downloaded) and adds "over_512_tokens": true to
the records that exceed 512 tokens (16 No Robots rows in the recorded build).
If the tokenizer is not cached the step is skipped: the prompts are the same but
the file hash differs from the recorded one. The skip is logged as a warning;
pass --require-tokenizer to make it an error instead, and --backbone-revision
to load a specific cached snapshot of the tokenizer.

Run from the repository root:
    python scripts/final_eval/build_final_test.py
    python scripts/final_eval/build_final_test.py --raw data/raw/hf_final --out data/benchmarks_final
    python scripts/final_eval/build_final_test.py --require-tokenizer \
        --backbone-revision e2da8e2f811d1448a5b465c236feacd80ffbac7b
"""

import sys
import os

REPO = os.environ.get("LLMGUARD_REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))  # local checkout without `pip install -e .`
sys.path.insert(0, os.path.join(REPO, "scripts"))

import argparse
import hashlib
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path

import build_datasets as bd                      # normalise, NearDupIndex, _record
from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("build_final_test")

ALLOW, MITIGATE = 1, 0
PHTEST_SAMPLE = 1000
PHTEST_SEED = 20260926
NEAR_DUP_THRESHOLD = 0.8
NUM_PERM, BANDS, MINHASH_SEED = 64, 16, 42

# Do-Not-Answer risk areas. Core = a moderation guard should block; other = the
# guard could legitimately allow (secondary set, never pooled into the primary rate).
DNA_CORE = {
    "malicious uses",
    "discrimination, exclusion, toxicity, hateful, offensive",
    "misinformation harms",
}
DNA_OTHER = {
    "information hazards",
    "human-chatbot interaction harms",
}

FILES = {
    "sst":       "Bertievidgen/SimpleSafetyTests/sst_test_cases.csv",
    "catqa":     "declare-lab/CategoricalHarmfulQA/data/catqa_english.json",
    "dna":       "LibrAI/do-not-answer/data_en.csv",
    "coconot":   "allenai/coconot/contrast/test-00000-of-00001.parquet",
    "phtest":    "furonghuang-lab/PHTest/PHTest.csv",
    "no_robots": "HuggingFaceH4/no_robots/data/test-00000-of-00001.parquet",
}

# Precedence for cross-source exact duplicates (rule 3): first listed wins.
SOURCE_ORDER = ["sst", "catqa", "dna_core", "dna_other", "coconot", "phtest", "no_robots"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the locked final test set")
    p.add_argument("--raw", default=None, help="Fetched sources (default: <data.raw_dir>/hf_final)")
    p.add_argument("--out", default="data/benchmarks_final", help="Output directory")
    p.add_argument("--pool", default=None, help="Pool directory (default: data.distillation_pool_dir)")
    p.add_argument("--benchmarks-root", default=None,
                   help="Directory holding <name>/test.jsonl for the five development benchmarks "
                        "(default: each benchmark's path in config)")
    p.add_argument("--require-tokenizer", action="store_true",
                   help="Fail if the roberta-base tokenizer is not in the local HF cache, instead "
                        "of skipping the over_512_tokens flags (default: skip with a warning, the "
                        "recorded behaviour)")
    p.add_argument("--backbone-revision", default=None, metavar="SHA",
                   help="Load this cached revision of the roberta-base tokenizer for the long-prompt "
                        "count (default: whatever snapshot the cache resolves)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(df, *names):
    """First column of df whose lower-cased name matches one of names."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def _read_json_or_jsonl(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        # dict-of-lists (pandas "columns" orient) or {"data": [...]}
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            keys = list(data)
            n = len(data[keys[0]])
            data = [{k: data[k][i] for k in keys} for i in range(n)]
    if not isinstance(data, list):
        raise ValueError(f"{path}: unexpected JSON shape {type(data).__name__}")
    return data


def _read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Loaders (one list of records per source) ──────────────────────────────────

def load_sst(raw: Path) -> list:
    import pandas as pd
    df = pd.read_csv(raw / FILES["sst"], keep_default_na=False)
    c_id, c_prompt = _col(df, "id"), _col(df, "prompt")
    c_cat = _col(df, "harm_area", "category")
    return [bd._record(f"sst_{r[c_id]}", r[c_prompt], MITIGATE, r[c_cat], "sst")
            for r in df.to_dict("records")]


def load_catqa(raw: Path) -> list:
    rows = _read_json_or_jsonl(raw / FILES["catqa"])
    out = []
    for i, r in enumerate(rows):
        keys = {k.lower(): k for k in r}
        q = r[keys["question"]]
        cat = r.get(keys.get("category", ""), "none")
        sub = r.get(keys.get("subcategory", ""), "")
        category = f"{cat} / {sub}" if sub else cat
        out.append(bd._record(f"catqa_{i}", q, MITIGATE, category, "catqa"))
    return out


def load_dna(raw: Path) -> tuple:
    """Do-Not-Answer split into core (primary harmful) and other (secondary)."""
    import pandas as pd
    df = pd.read_csv(raw / FILES["dna"], keep_default_na=False)
    c_id, c_q, c_area = _col(df, "id"), _col(df, "question"), _col(df, "risk_area")
    c_type = _col(df, "types_of_harm")
    core, other, unknown = [], [], Counter()
    for r in df.to_dict("records"):
        area = str(r[c_area]).strip().lower()
        category = f"{r[c_area]} / {r[c_type]}"
        if area in DNA_CORE:
            core.append(bd._record(f"dna_{r[c_id]}", r[c_q], MITIGATE, category, "dna_core"))
        elif area in DNA_OTHER:
            other.append(bd._record(f"dna_{r[c_id]}", r[c_q], MITIGATE, category, "dna_other"))
        else:
            unknown[area] += 1
    if unknown:
        raise ValueError(f"Do-Not-Answer has risk areas the rule does not name: {dict(unknown)}. "
                         "Stop: the split into core/other must be decided before use, not here.")
    return core, other


def load_coconot(raw: Path) -> list:
    import pandas as pd
    df = pd.read_parquet(raw / FILES["coconot"])
    c_id, c_prompt = _col(df, "id"), _col(df, "prompt")
    c_cat = _col(df, "category")
    c_sub = _col(df, "subcategory") if "subcategory" in {c.lower() for c in df.columns} else None
    out = []
    for i, r in enumerate(df.to_dict("records")):
        cat = f"{r[c_cat]} / {r[c_sub]}" if c_sub else r[c_cat]
        out.append(bd._record(f"coconot_{r[c_id]}", r[c_prompt], ALLOW, cat, "coconot"))
    return out


def load_phtest(raw: Path) -> tuple:
    """Returns (harmless records, number of controversial rows dropped)."""
    import pandas as pd
    df = pd.read_csv(raw / FILES["phtest"], keep_default_na=False)
    c_id, c_req = _col(df, "ID", "id"), _col(df, "Request", "prompt", "request")
    c_label = _col(df, "Harmfulness", "harmfulness", "label")
    labels = {str(v).strip().lower() for v in df[c_label]}
    if not labels <= {"harmless", "controversial"}:
        raise ValueError(f"PHTest label values {sorted(labels)}; expected harmless/controversial")
    out, dropped = [], 0
    for r in df.to_dict("records"):
        if str(r[c_label]).strip().lower() == "harmless":
            out.append(bd._record(f"phtest_{r[c_id]}", r[c_req], ALLOW, "harmless", "phtest"))
        else:
            dropped += 1
    return out, dropped


def load_no_robots(raw: Path) -> list:
    import pandas as pd
    df = pd.read_parquet(raw / FILES["no_robots"])
    c_id = _col(df, "prompt_id")
    c_cat = _col(df, "category")
    c_msgs = _col(df, "messages") if "messages" in {c.lower() for c in df.columns} else None
    c_prompt = _col(df, "prompt") if "prompt" in {c.lower() for c in df.columns} else None
    out = []
    for r in df.to_dict("records"):
        text = None
        if c_msgs is not None:
            msgs = r[c_msgs]
            try:
                msgs = list(msgs)
            except TypeError:
                msgs = []
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    text = m.get("content")
                    break
        if text is None and c_prompt is not None:
            text = r[c_prompt]
        out.append(bd._record(f"nr_{r[c_id]}", text, ALLOW, r[c_cat], "no_robots"))
    return out


# ── Screening ─────────────────────────────────────────────────────────────────

def dedup_within(records: list) -> tuple:
    return bd.dedup_benchmark(records)


class Screen:
    """Exact-normalised and near-duplicate lookup against the reference prompts."""

    def __init__(self, reference: dict):
        self.exact = {}
        self.index = bd.NearDupIndex(NUM_PERM, BANDS, MINHASH_SEED)
        self.owner = []
        for name, recs in reference.items():
            for r in recs:
                norm = bd.normalise(r["text"])
                if not norm:
                    continue
                self.exact.setdefault(norm, name)
                self.index.add(norm)
                self.owner.append(name)

    def hit(self, text: str):
        norm = bd.normalise(text)
        if norm in self.exact:
            return "exact", self.exact[norm]
        hits = self.index.query(norm, NEAR_DUP_THRESHOLD)
        if hits:
            return "near", self.owner[hits[0]]
        return None


def screen_against_reference(records: list, screen: Screen) -> tuple:
    kept, dropped = [], Counter()
    for r in records:
        h = screen.hit(r["text"])
        if h is None:
            kept.append(r)
        else:
            dropped[f"{h[0]}:{h[1]}"] += 1
    return kept, dict(dropped)


def drop_cross_source(by_source: "OrderedDict[str, list]") -> tuple:
    seen, dropped = {}, Counter()
    out = OrderedDict()
    for name in SOURCE_ORDER:
        out[name] = []
        for r in by_source.get(name, []):
            key = bd.normalise(r["text"])
            if key in seen:
                dropped[f"{name}<-{seen[key]}"] += 1
            else:
                seen[key] = name
                out[name].append(r)
    return out, dict(dropped)


def count_long(records: list, tokenizer_name: str = "roberta-base", limit: int = 512,
               require: bool = False, revision: str = None) -> dict:
    """
    How many prompts exceed Layer 1's token limit; flags them with over_512_tokens.

    The tokenizer is loaded from the local HF cache only (never downloaded), at
    `revision` when given. If it is not cached the step is skipped with a
    warning and the output file's hash will differ from the recorded build;
    with require=True the skip becomes an error instead.
    """
    kwargs = {"local_files_only": True}
    if revision:
        kwargs["revision"] = revision
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except Exception as exc:                                       # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if require:
            raise SystemExit(f"--require-tokenizer: cannot load '{tokenizer_name}' from the local "
                             f"HF cache ({reason}). Download it first, e.g. "
                             f"`hf download {tokenizer_name}`, then rerun.") from exc
        logger.warning(f"Long-prompt count SKIPPED: tokenizer '{tokenizer_name}' is not in the "
                       f"local HF cache ({reason}). No over_512_tokens flags are written, so the "
                       f"output hash will differ from the recorded build. Pass --require-tokenizer "
                       f"to make this an error.")
        return {"skipped": reason}
    over = Counter()
    for r in records:
        n = len(tok(r["text"], add_special_tokens=True, truncation=False)["input_ids"])
        if n > limit:
            over[r["source"]] += 1
            r["over_512_tokens"] = True
    logger.info(f"Long-prompt count: {sum(over.values())} prompt(s) over {limit} tokens "
                f"flagged over_512_tokens {dict(over)}")
    return {"limit": limit, "over_by_source": dict(over)}


# ── Main ──────────────────────────────────────────────────────────────────────

def build(raw: Path, pool_dir: Path, benchmark_files: dict, out_dir: Path,
          require_tokenizer: bool = False, backbone_revision: str = None) -> dict:
    """Apply the four screening rules and write the final set; returns the manifest body."""
    stages = OrderedDict()

    # Load
    sst = load_sst(raw)
    catqa = load_catqa(raw)
    dna_core, dna_other = load_dna(raw)
    coconot = load_coconot(raw)
    phtest, ph_controversial = load_phtest(raw)
    no_robots = load_no_robots(raw)
    loaded = OrderedDict([("sst", sst), ("catqa", catqa), ("dna_core", dna_core),
                          ("dna_other", dna_other), ("coconot", coconot),
                          ("phtest", phtest), ("no_robots", no_robots)])
    stages["loaded"] = {k: len(v) for k, v in loaded.items()}
    stages["phtest_controversial_dropped"] = ph_controversial

    # Rule 1
    step1, step1_stats = OrderedDict(), {}
    for k, v in loaded.items():
        step1[k], step1_stats[k] = dedup_within(v)
    stages["after_rule1_within_source"] = {k: len(v) for k, v in step1.items()}
    stages["rule1_dropped"] = step1_stats

    # Rule 2
    reference = OrderedDict()
    for split in ("train", "val", "test"):
        reference[f"pool_{split}"] = _read_jsonl(pool_dir / f"{split}.jsonl")
    for name, path in benchmark_files.items():
        reference[f"bench_{name}"] = _read_jsonl(path)
    stages["reference_sizes"] = {k: len(v) for k, v in reference.items()}
    screen = Screen(reference)
    step2, step2_stats = OrderedDict(), {}
    for k, v in step1.items():
        step2[k], step2_stats[k] = screen_against_reference(v, screen)
    stages["after_rule2_reference_screen"] = {k: len(v) for k, v in step2.items()}
    stages["rule2_dropped_by_reference"] = step2_stats

    # Rule 3
    step3, cross = drop_cross_source(step2)
    stages["after_rule3_cross_source"] = {k: len(v) for k, v in step3.items()}
    stages["rule3_dropped"] = cross

    # Rule 4
    ph = step3["phtest"]
    if len(ph) > PHTEST_SAMPLE:
        rng = random.Random(PHTEST_SEED)
        ph = sorted(rng.sample(ph, PHTEST_SAMPLE), key=lambda r: r["id"])
    step3["phtest"] = ph
    stages["after_rule4_phtest_sample"] = {k: len(v) for k, v in step3.items()}

    final = [r for name in SOURCE_ORDER for r in step3[name]]
    ids = [r["id"] for r in final]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate ids in the final set")
    stages["long_prompts"] = count_long(final, require=require_tokenizer, revision=backbone_revision)

    out_file = out_dir / "final_test" / "test.jsonl"
    bd._write_jsonl(final, out_file)
    by_label = Counter((r["source"], r["label"]) for r in final)
    summary = {
        "output": str(out_file),
        "output_sha256": sha256_file(out_file),
        "n_total": len(final),
        "n_by_source": {s: len(step3[s]) for s in SOURCE_ORDER},
        "n_by_source_label": {f"{s}:{l}": n for (s, l), n in sorted(by_label.items())},
        "n_safe": sum(1 for r in final if r["label"] == ALLOW),
        "n_harmful_primary": sum(1 for r in final if r["label"] == MITIGATE and r["source"] != "dna_other"),
        "n_harmful_secondary": len(step3["dna_other"]),
        "stages": stages,
        "rules": {
            "normalisation": "NFKC, lower-case, whitespace collapse (build_datasets.normalise)",
            "near_dup": {"shingle": bd.SHINGLE, "jaccard_threshold": NEAR_DUP_THRESHOLD,
                         "num_perm": NUM_PERM, "bands": BANDS, "seed": MINHASH_SEED},
            "phtest_sample": {"n": PHTEST_SAMPLE, "seed": PHTEST_SEED},
            "source_precedence": SOURCE_ORDER,
            "dna_core_risk_areas": sorted(DNA_CORE),
            "dna_other_risk_areas": sorted(DNA_OTHER),
        },
    }
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config()
    run_info = collect_run_info(cfg)
    raw = Path(args.raw) if args.raw else Path(cfg["data"]["raw_dir"]) / "hf_final"
    pool_dir = Path(args.pool) if args.pool else Path(cfg["data"]["distillation_pool_dir"])
    out_dir = Path(args.out)

    benches = {}
    for b in cfg.get("evaluation", {}).get("benchmarks", []):
        if not b.get("enabled", True):
            continue
        path = (Path(args.benchmarks_root) / b["name"] / "test.jsonl" if args.benchmarks_root
                else Path(b["path"]) / "test.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"development benchmark missing: {path}")
        benches[b["name"]] = path
    expected = {"jailbreakbench", "xstest", "harmbench", "toxicchat", "orbench"}
    if set(benches) != expected:
        raise SystemExit(f"Expected exactly the five development benchmarks enabled, got {sorted(benches)}")

    missing = [k for k, f in FILES.items() if not (raw / f).exists()]
    if missing:
        raise SystemExit(f"Source files missing under {raw}: {missing}. Run fetch_final_sources.py first.")

    fetch_manifest = raw / "manifest.json"
    revisions = {}
    if fetch_manifest.exists():
        src = json.loads(fetch_manifest.read_text(encoding="utf-8")).get("sources", {})
        revisions = {k: {"repo": v["repo"], "revision": v["revision"],
                         "files": [{"path": f["path"], "digest": f["digest"]} for f in v["files"]]}
                     for k, v in src.items()}

    summary = build(raw, pool_dir, benches, out_dir,
                    require_tokenizer=args.require_tokenizer,
                    backbone_revision=args.backbone_revision)
    summary["source_revisions"] = revisions
    summary["reference_files"] = {**{f"pool_{s}": sha256_file(pool_dir / f"{s}.jsonl") for s in ("train", "val", "test")},
                                  **{f"bench_{k}": sha256_file(p) for k, p in benches.items()}}
    summary["run_info"] = finalize_run_info(run_info)

    manifest = out_dir / "manifest.json"
    warn_if_overwriting(manifest)
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Final test set built")
    for s in SOURCE_ORDER:
        logger.info(f"  {s:<10} loaded {summary['stages']['loaded'][s]:>5} -> final {summary['n_by_source'][s]:>5}")
    logger.info(f"  total {summary['n_total']} | safe {summary['n_safe']} | harmful primary "
                f"{summary['n_harmful_primary']} | harmful secondary {summary['n_harmful_secondary']}")
    logger.info(f"  output sha256 {summary['output_sha256']} (recorded build: "
                f"39425a51fc671079dc341f57ec6a4821b1fb2732b50ae23cc95ccda3135e11a0)")
    logger.info(f"Manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
