"""
Build the RQ4 over-refusal parity sets.

For every benchmark with safe prompts, writes three copies of the *safe*
prompts under --root:

    <root>/original/<benchmark>/test.jsonl   the prompts unchanged
    <root>/general/<benchmark>/test.jsonl    general L2 grammatical features
    <root>/arabic/<benchmark>/test.jsonl     Arabic-L1 grammatical features

Every copy keeps each prompt's id and label, so scripts/summarise_results.py
can pair them prompt by prompt and report a paired bootstrap, and
scripts/compare_flips.py can count decisions that change.

Harmful prompts are never written or rewritten: this measures whether the
guard over-blocks ordinary requests phrased in non-native English, and a
harmful prompt has no place in that comparison. The rule is asserted, not
just intended.

Usage:
    python scripts/make_l2_benchmarks.py --root data/benchmarks_parity
    python scripts/make_l2_benchmarks.py --root data/benchmarks_parity --benchmarks xstest orbench
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from llmguard.evaluation.l2_variants import PROFILES, variant
from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info

logger = get_logger("make_l2_benchmarks")

SAFE_LABEL = 1        # evaluation/pipeline_evaluator.py: 0 = harmful, 1 = safe
REVIEW_SAMPLE = 30


def read_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_rows(rows: list, name: str) -> list:
    """
    The safe prompts, each with an id, and nothing else.

    Harmful prompts are dropped here and the result is checked, so no later
    step can put one into a parity set.
    """
    out = []
    for i, row in enumerate(rows):
        if int(row.get("label", SAFE_LABEL)) != SAFE_LABEL:
            continue
        row = dict(row)
        row.setdefault("id", f"{name}-{i}")
        out.append(row)

    assert all(int(r.get("label", SAFE_LABEL)) == SAFE_LABEL for r in out), \
        f"{name}: a harmful prompt reached the parity set"
    ids = [r["id"] for r in out]
    assert len(set(ids)) == len(ids), f"{name}: duplicate prompt ids"
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="data/benchmarks_parity",
                   help="Directory to write the three arms into")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Named benchmarks from config; default is every enabled one")
    p.add_argument("--review-sample", type=int, default=REVIEW_SAMPLE,
                   help="Prompts to write to review_sample.md for a meaning check")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    run_info = collect_run_info(cfg)
    root = Path(args.root)
    seed = cfg["project"]["seed"]

    bench_list = [b for b in cfg.get("evaluation", {}).get("benchmarks", [])
                  if b.get("enabled", True)]
    if args.benchmarks:
        wanted = set(args.benchmarks)
        missing = wanted - {b["name"] for b in bench_list}
        if missing:
            raise SystemExit(f"not in config: {', '.join(sorted(missing))}")
        bench_list = [b for b in bench_list if b["name"] in wanted]

    manifest, review = {}, []
    print()
    print(f"  {'benchmark':<16}{'prompts':>9}{'safe':>7}" +
          "".join(f"{p + ' edited':>16}" for p in PROFILES))
    print("  " + "-" * 64)

    for bench in bench_list:
        name = bench["name"]
        source = Path(bench["path"]) / "test.jsonl"
        if not source.exists():
            logger.warning(f"{name}: {source} not found; skipped")
            continue

        rows = read_jsonl(source)
        safe = safe_rows(rows, name)

        if not safe:
            print(f"  {name:<16}{len(rows):>9}{0:>7}" + "".join(f"{'-':>16}" for _ in PROFILES))
            manifest[name] = {"n_prompts": len(rows), "n_safe": 0,
                              "note": "no safe prompts; nothing to compare"}
            continue

        ids = [r["id"] for r in safe]
        write_jsonl(root / "original" / name / "test.jsonl", safe)
        entry = {"n_prompts": len(rows), "n_safe": len(safe), "profiles": {}}

        for profile in PROFILES:
            out_rows, rules, unchanged = [], Counter(), 0
            for row in safe:
                text, applied = variant(row.get("text", ""), str(row["id"]), profile)
                if not applied:
                    unchanged += 1
                rules.update(applied)
                new = dict(row)
                new["text"] = text
                new["l2_profile"] = profile
                new["l2_rules"] = applied
                out_rows.append(new)

            assert [r["id"] for r in out_rows] == ids, f"{name}/{profile}: ids moved"
            assert all(int(r["label"]) == SAFE_LABEL for r in out_rows), \
                f"{name}/{profile}: label changed"
            write_jsonl(root / profile / name / "test.jsonl", out_rows)
            entry["profiles"][profile] = {
                "edited": len(safe) - unchanged,
                "unchanged": unchanged,
                "rule_counts": dict(rules.most_common()),
            }

        manifest[name] = entry
        print(f"  {name:<16}{len(rows):>9}{len(safe):>7}" +
              "".join(f"{entry['profiles'][p]['edited']:>16}" for p in PROFILES))

        rng = random.Random(f"{seed}:{name}")
        for row in rng.sample(safe, min(6, len(safe))):
            review.append({
                "benchmark": name,
                "id": row["id"],
                "original": row.get("text", ""),
                **{p: variant(row.get("text", ""), str(row["id"]), p)[0] for p in PROFILES},
            })

    if not manifest:
        raise SystemExit("no benchmarks built")

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(
        {"run_info": finalize_run_info(run_info), "safe_label": SAFE_LABEL,
         "profiles": list(PROFILES), "benchmarks": manifest}, indent=2), encoding="utf-8")

    rng = random.Random(seed)
    rng.shuffle(review)
    review = review[:args.review_sample]
    lines = ["# Meaning-preservation sample", "",
             "Each block is one safe prompt in the three arms. The rewrite is only",
             "meaning-preserving if the request is still the same request.", ""]
    for item in review:
        lines.append(f"## {item['benchmark']} / {item['id']}")
        lines.append(f"- **original**: {item['original']}")
        for p in PROFILES:
            lines.append(f"- **{p}**: {item[p]}")
        lines.append("")
    (root / "review_sample.md").write_text("\n".join(lines), encoding="utf-8")

    total_safe = sum(e.get("n_safe", 0) for e in manifest.values())
    print("  " + "-" * 64)
    print(f"  safe prompts per arm: {total_safe}")
    print(f"  written to: {root}/{{original,{','.join(PROFILES)}}}/<benchmark>/test.jsonl")
    print(f"  manifest:   {root / 'manifest.json'}")
    print(f"  review:     {root / 'review_sample.md'} ({len(review)} prompts)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
