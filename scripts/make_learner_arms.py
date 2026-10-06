"""
Build the RQ4 learner-English arms from W&I+LOCNESS (BEA-2019).

Real non-native English, matched for content: every sentence in the corpus
comes with an expert correction, so the sentence as the learner wrote it and
the same sentence in native-like form can be run as a pair. Levels A, B and C
are learners (CEFR); N is native (LOCNESS). Writes, under --root:

    <root>/original/<name>/test.jsonl     the sentence as written
    <root>/corrected/<name>/test.jsonl    annotator 0's edits applied

with <name> one of wi_a, wi_b, wi_c, locness_n. Ids, order and the safe label
are identical across the two arms, so scripts/parity_attribution.py and
scripts/summarise_results.py --compare can pair them sentence by sentence.
Each row also carries the number of edits and their ERRANT categories, which
is what the attribution reads.

The pre-registration for the study this serves is docs/rq4_learner_prereg.md.

Source: the M2 files of wi+locness_v2.1.bea19.tar.gz, found by name anywhere
under --source ([ABCN].{train,dev}.gold.bea19.m2; the combined ABC* and ABCN*
files are ignored so nothing is counted twice). Licence: non-commercial use
only; nothing from the corpus is copied into this repository.

Usage:
    python scripts/make_learner_arms.py --source data/raw/wi_locness
    python scripts/make_learner_arms.py --source data/raw/wi_locness --train-per-level 1500
    python scripts/make_learner_arms.py --source data/raw/wi_locness --all
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info

logger = get_logger("make_learner_arms")

SAFE_LABEL = 1                       # evaluation/pipeline_evaluator.py: 1 = safe
ARMS = ("original", "corrected")
LEVELS = {"A": "wi_a", "B": "wi_b", "C": "wi_c", "N": "locness_n"}
FILE_RE = re.compile(r"^(?P<level>[ABCN])\.(?P<split>train|dev)\.gold\.bea19\.m2$")
MIN_WORD_TOKENS = 3
REVIEW_SAMPLE = 30
DEFAULT_TRAIN_PER_LEVEL = 1500


# ── M2 ───────────────────────────────────────────────────────────────────────

def parse_m2(path: Path) -> List[Tuple[List[str], Dict[int, list]]]:
    """
    Every sentence in an M2 file as (tokens, {annotator: [(start, end, correction,
    type), ...]}). The S line is space-tokenised by the corpus; the offsets in the
    A lines index those tokens. A noop annotation is kept so that an annotator who
    changed nothing still counts as having annotated the sentence.
    """
    sentences = []
    tokens, edits = None, None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "S" or line.startswith("S "):
                if tokens is not None:
                    sentences.append((tokens, dict(edits)))
                tokens = line[2:].split(" ") if len(line) > 2 else []
                edits = defaultdict(list)
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) < 6 or tokens is None:
                    continue
                start, end = (int(x) for x in parts[0].split())
                edits[int(parts[5])].append((start, end, parts[2], parts[1]))
            elif not line.strip():
                if tokens is not None:
                    sentences.append((tokens, dict(edits)))
                tokens, edits = None, None
    if tokens is not None:
        sentences.append((tokens, dict(edits)))
    return sentences


def real_edits(edits: list) -> list:
    """The annotator's edits that change something: noop and negative spans dropped."""
    return [e for e in edits if e[3] != "noop" and e[0] >= 0 and e[1] >= e[0]]


def apply_edits(tokens: List[str], edits: list) -> List[str]:
    """
    One annotator's edits applied to the token list. Edits are applied left to
    right; an insertion has start == end; a deletion has an empty correction.
    """
    out, pos = [], 0
    for start, end, corr, _ in sorted(real_edits(edits), key=lambda e: (e[0], e[1])):
        if start < pos:                      # overlapping annotation: keep the first
            continue
        out.extend(tokens[pos:start])
        if corr and corr != "-NONE-":
            out.extend(corr.split(" "))
        pos = end
    out.extend(tokens[pos:])
    return out


def choose_annotator(edits: Dict[int, list]) -> int:
    """Annotator 0 when present, else the lowest id; -1 when nobody annotated."""
    if not edits:
        return -1
    return 0 if 0 in edits else min(edits)


def category(etype: str) -> str:
    """'R:VERB:SVA' -> 'VERB:SVA', 'M:DET' -> 'DET'; 'UNK' and 'noop' stay as they are."""
    if ":" in etype and etype.split(":", 1)[0] in ("M", "R", "U"):
        return etype.split(":", 1)[1]
    return etype


def proxy_like(etype: str) -> bool:
    """
    Whether an edit falls in one of the seven feature types the first study's
    rules imitate: determiners, prepositions, subject-verb agreement, noun
    number, a missing verb (the dropped copula), an inserted pronoun (the
    resumptive), and word order (the uninverted question).
    """
    op = etype.split(":", 1)[0] if ":" in etype else ""
    cat = category(etype)
    if cat in ("DET", "PREP", "VERB:SVA", "NOUN:NUM", "WO"):
        return True
    if cat == "VERB" and op == "M":
        return True
    if cat == "PRON" and op == "U":
        return True
    return False


# ── Detokenisation ───────────────────────────────────────────────────────────

_NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
_NO_SPACE_AFTER = frozenset("([{$")
_CLITICS = frozenset(["n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "N'T", "'S"])


def detokenise(tokens: List[str]) -> str:
    """
    Space-tokenised corpus text back to running prose: no space before closing
    punctuation or clitics, none after an opening bracket, quotes paired. Both
    arms go through this, so whatever it gets wrong it gets wrong in both.
    """
    out = ""
    prev = ""
    quote_open = False
    for tok in tokens:
        if tok == "":
            continue
        if not out:
            out, prev = tok, tok
            if tok == '"':
                quote_open = True
            continue
        if tok in _CLITICS or tok[0] in _NO_SPACE_BEFORE or tok.startswith("..."):
            out += tok
        elif prev and prev[-1] in _NO_SPACE_AFTER:
            out += tok
        elif tok == '"':
            if quote_open:
                out += tok
                quote_open = False
            else:
                out += " " + tok
                quote_open = True
        elif prev == '"' and quote_open:
            out += tok
        else:
            out += " " + tok
        prev = tok
    return out


def word_tokens(tokens: List[str]) -> int:
    return sum(1 for t in tokens if any(ch.isalpha() for ch in t))


# ── Files and sampling ───────────────────────────────────────────────────────

def find_m2_files(source: Path) -> Dict[Tuple[str, str], Path]:
    """{(level, split): path} for the per-level gold files under source."""
    found = {}
    for path in sorted(source.rglob("*.m2")):
        m = FILE_RE.match(path.name)
        if m:
            found[(m.group("level"), m.group("split"))] = path
    return found


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_level(level: str, files: Dict[str, Path], seed: int, train_cap, keep_all: bool):
    """
    Rows for both arms of one level, plus the counts the manifest reports.
    Development sentences are all kept; training sentences are sampled to
    train_cap with a generator seeded from the project seed and the level, so
    the sample is the same on every machine.
    """
    name = LEVELS[level]
    rows = {arm: [] for arm in ARMS}
    stats = {"source_files": {}, "n_source": 0, "n_dropped_short": 0, "n_sampled_out": 0,
             "n_kept": 0, "n_edited": 0, "edits_total": 0, "tokens_total": 0,
             "by_n_edits": Counter(), "categories": Counter(), "types": Counter(),
             "annotators": Counter(), "proxy_like_edits": 0, "n_overlapping_skipped": 0}
    review = []

    for split in ("dev", "train"):
        path = files.get(split)
        if path is None:
            continue
        stats["source_files"][split] = {"path": str(path), "sha256": sha256_of(path),
                                        "bytes": path.stat().st_size}
        sentences = parse_m2(path)
        stats["n_source"] += len(sentences)

        keep = []
        for i, (tokens, edits) in enumerate(sentences):
            if word_tokens(tokens) < MIN_WORD_TOKENS:
                stats["n_dropped_short"] += 1
                continue
            keep.append((i, tokens, edits))

        if split == "train" and not keep_all and train_cap is not None and len(keep) > train_cap:
            rng = random.Random(f"{seed}:{level}:{split}")
            chosen = sorted(rng.sample(range(len(keep)), train_cap))
            stats["n_sampled_out"] += len(keep) - train_cap
            keep = [keep[j] for j in chosen]

        for i, tokens, edits in keep:
            ann = choose_annotator(edits)
            mine = real_edits(edits.get(ann, [])) if ann >= 0 else []
            corrected = apply_edits(tokens, edits.get(ann, [])) if ann >= 0 else list(tokens)
            # Edits that overlap are skipped by apply_edits; count them so the
            # manifest says so rather than hiding it.
            applied = 0
            pos = 0
            for start, end, _, _ in sorted(mine, key=lambda e: (e[0], e[1])):
                if start < pos:
                    stats["n_overlapping_skipped"] += 1
                    continue
                applied += 1
                pos = end
            n_edits = applied
            types = [e[3] for e in mine]
            cats = [category(t) for t in types]

            text_orig = detokenise(tokens)
            text_corr = detokenise(corrected)
            if n_edits == 0:
                assert text_corr == text_orig, f"{name}: zero edits but the arms differ"

            rid = f"{name}-{split}-{i:05d}"
            base = {"id": rid, "label": SAFE_LABEL, "level": level, "split": split,
                    "n_edits": n_edits, "edit_types": types, "edit_cats": cats,
                    "annotator": ann, "n_tokens": len(tokens)}
            rows["original"].append({**base, "text": text_orig, "arm": "original"})
            rows["corrected"].append({**base, "text": text_corr, "arm": "corrected"})

            stats["n_kept"] += 1
            stats["n_edited"] += 1 if n_edits else 0
            stats["edits_total"] += n_edits
            stats["tokens_total"] += len(tokens)
            stats["by_n_edits"][min(n_edits, 3)] += 1
            stats["categories"].update(cats)
            stats["types"].update(types)
            stats["annotators"][ann] += 1
            stats["proxy_like_edits"] += sum(1 for t in types if proxy_like(t))
            if n_edits and len(review) < 400:
                review.append({"name": name, "id": rid, "original": text_orig, "corrected": text_corr,
                               "edits": types})

    return name, rows, stats, review


def finish_stats(stats: dict) -> dict:
    total_edits = sum(stats["categories"].values())
    out = dict(stats)
    out["by_n_edits"] = {("3+" if k == 3 else str(k)): v for k, v in sorted(stats["by_n_edits"].items())}
    out["categories"] = dict(stats["categories"].most_common())
    out["types"] = dict(stats["types"].most_common(40))
    out["annotators"] = {str(k): v for k, v in sorted(stats["annotators"].items())}
    out["edits_per_100_tokens"] = (100.0 * stats["edits_total"] / stats["tokens_total"]
                                   if stats["tokens_total"] else 0.0)
    out["edited_share"] = stats["n_edited"] / stats["n_kept"] if stats["n_kept"] else 0.0
    out["proxy_like_share_of_edits"] = (stats["proxy_like_edits"] / total_edits) if total_edits else 0.0
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True,
                   help="Directory holding the extracted corpus (searched recursively for the M2 files)")
    p.add_argument("--root", default="data/benchmarks_learner", help="Where the two arms are written")
    p.add_argument("--train-per-level", type=int, default=DEFAULT_TRAIN_PER_LEVEL,
                   help="Training sentences kept per learner level (default %(default)s); dev is kept in full")
    p.add_argument("--all", action="store_true", help="Keep every training sentence")
    p.add_argument("--review-sample", type=int, default=REVIEW_SAMPLE,
                   help="Pairs written to review_sample.md for a meaning check")
    p.add_argument("--seed", type=int, default=None, help="Default: project.seed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    run_info = collect_run_info(cfg)
    seed = args.seed if args.seed is not None else int(cfg["project"]["seed"])
    source, root = Path(args.source), Path(args.root)

    files = find_m2_files(source)
    if not files:
        raise SystemExit(f"no [ABCN].{{train,dev}}.gold.bea19.m2 files under {source}")
    licence = [str(p) for p in source.rglob("*") if p.is_file() and "licen" in p.name.lower()]

    manifest, review_all = {}, []
    print()
    print(f"  {'arm':<11}{'level':>6}{'source':>8}{'kept':>7}{'edited':>8}{'edits/100tok':>13}{'proxy-like':>11}")
    print("  " + "-" * 64)
    for level in "ABCN":
        per_split = {split: path for (lv, split), path in files.items() if lv == level}
        if not per_split:
            continue
        name, rows, stats, review = build_level(level, per_split, seed, args.train_per_level, args.all)
        ids = [r["id"] for r in rows["original"]]
        assert ids == [r["id"] for r in rows["corrected"]], f"{name}: ids differ between arms"
        assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
        assert all(r["label"] == SAFE_LABEL for arm in ARMS for r in rows[arm]), f"{name}: label"
        for arm in ARMS:
            write_jsonl(root / arm / name / "test.jsonl", rows[arm])
        st = finish_stats(stats)
        manifest[name] = st
        review_all.extend(review)
        print(f"  {name:<11}{level:>6}{st['n_source']:>8}{st['n_kept']:>7}{st['n_edited']:>8}"
              f"{st['edits_per_100_tokens']:>13.2f}{st['proxy_like_share_of_edits']:>11.2f}")

    if not manifest:
        raise SystemExit("nothing built")

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "run_info": finalize_run_info(run_info),
        "corpus": "W&I+LOCNESS v2.1 (BEA-2019), non-commercial licence; not redistributed",
        "licence_files_seen": licence,
        "safe_label": SAFE_LABEL, "arms": list(ARMS),
        "min_word_tokens": MIN_WORD_TOKENS,
        "train_per_level": None if args.all else args.train_per_level,
        "seed": seed,
        "levels": manifest,
    }, indent=2), encoding="utf-8")

    rng = random.Random(seed)
    rng.shuffle(review_all)
    review_all = review_all[:args.review_sample]
    lines = ["# Meaning-preservation sample (learner arms)", "",
             "Each block is one sentence as the learner wrote it and as the annotator corrected it.",
             "The pair is only usable if both say the same thing.", ""]
    for item in review_all:
        lines.append(f"## {item['name']} / {item['id']}  ({', '.join(item['edits'])})")
        lines.append(f"- **original**:  {item['original']}")
        lines.append(f"- **corrected**: {item['corrected']}")
        lines.append("")
    (root / "review_sample.md").write_text("\n".join(lines), encoding="utf-8")

    print("  " + "-" * 64)
    print(f"  written to: {root}/{{original,corrected}}/<name>/test.jsonl")
    print(f"  manifest:   {root / 'manifest.json'}")
    print(f"  review:     {root / 'review_sample.md'} ({len(review_all)} pairs)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
