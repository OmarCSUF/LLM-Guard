"""
Count decisions that change between two runs on the same prompts.

scripts/summarise_results.py reports how a rate moves and whether the move
survives a paired bootstrap. This reports the thing behind the rate: how many
individual prompts changed side, and in which direction. For the RQ4 parity
probe the two numbers to read are how many safe prompts went from allowed to
blocked, and how many went the other way.

The test is McNemar's, computed exactly. It looks only at prompts the two runs
disagree about, which is the right comparison for paired decisions on the same
prompt: if grammatical form made no difference, flips should fall either way
with equal probability.

A prompt counts as blocked when it was not allowed, so an escalated prompt is
blocked, matching summarise_results.py.

Usage:
    python scripts/compare_flips.py \\
        --original data/processed/parity_original.json \\
        --variant  data/processed/parity_general.json \\
        --out data/processed/parity_flips_general.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
from math import comb
from pathlib import Path

from llmguard.utils.logger import get_logger

logger = get_logger("compare_flips")


def predictions_dir_for(results_path: Path) -> Path:
    """Where run_evaluation.py writes predictions for --output results_path."""
    return results_path.with_name(results_path.stem + "_predictions")


def load_rows(results_path: Path) -> dict:
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    pred_dir = predictions_dir_for(results_path)
    out = {}
    for bench in saved:
        path = pred_dir / f"{bench}.jsonl"
        if not path.exists():
            logger.warning(f"no predictions for {bench} at {path}; skipped")
            continue
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        by_id = {}
        for r in rows:
            if r.get("id") is None:
                raise ValueError(f"{path}: every prediction row needs an id")
            by_id[str(r["id"])] = r
        out[bench] = by_id
    if not out:
        raise FileNotFoundError(f"no prediction files in {pred_dir}")
    return out


def blocked(row: dict) -> bool:
    return int(row["final_pred"]) == 0 or bool(row.get("escalated"))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value from the discordant counts.

    Under the null, each of the b + c flips is equally likely to go either way,
    so the count on one side is Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--original", type=Path, required=True, help="results JSON for the unchanged prompts")
    p.add_argument("--variant", type=Path, required=True, help="results JSON for the rewritten prompts")
    p.add_argument("--out", type=Path, default=None, help="write the numbers here as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base, var = load_rows(args.original), load_rows(args.variant)
    shared = [b for b in base if b in var]
    if not shared:
        raise SystemExit("the two runs have no benchmark in common")

    report = {}
    print()
    print(f"  {'benchmark':<14}{'n':>6}{'blocked':>9}{'blocked':>9}"
          f"{'allow>block':>13}{'block>allow':>13}{'p':>9}")
    print(f"  {'':<14}{'':>6}{'before':>9}{'after':>9}{'':>13}{'':>13}{'':>9}")
    print("  " + "-" * 73)

    for bench in shared:
        ids = [i for i in base[bench] if i in var[bench]]
        missing = len(base[bench]) - len(ids)
        if missing:
            logger.warning(f"{bench}: {missing} prompts are not in both runs; ignored")

        labels = {i for i in ids if int(base[bench][i]["label"]) != int(var[bench][i]["label"])}
        if labels:
            raise SystemExit(f"{bench}: labels differ between runs for {len(labels)} prompts")

        b0 = {i: blocked(base[bench][i]) for i in ids}
        b1 = {i: blocked(var[bench][i]) for i in ids}
        to_blocked = sorted(i for i in ids if not b0[i] and b1[i])
        to_allowed = sorted(i for i in ids if b0[i] and not b1[i])
        p = mcnemar_exact(len(to_blocked), len(to_allowed))

        shift = [float(var[bench][i]["harm_score"]) - float(base[bench][i]["harm_score"]) for i in ids]
        mean_shift = sum(shift) / len(shift) if shift else 0.0
        routing = {}
        for i in ids:
            before, after = base[bench][i].get("l1_decision"), var[bench][i].get("l1_decision")
            if before != after:
                routing[f"{before}->{after}"] = routing.get(f"{before}->{after}", 0) + 1

        report[bench] = {
            "n": len(ids),
            "blocked_before": sum(b0.values()),
            "blocked_after": sum(b1.values()),
            "allowed_to_blocked": len(to_blocked),
            "blocked_to_allowed": len(to_allowed),
            "mcnemar_p": p,
            "mean_harm_score_shift": mean_shift,
            "l1_routing_changes": routing,
            "ids_allowed_to_blocked": to_blocked[:50],
            "ids_blocked_to_allowed": to_allowed[:50],
        }
        print(f"  {bench:<14}{len(ids):>6}{sum(b0.values()):>9}{sum(b1.values()):>9}"
              f"{len(to_blocked):>13}{len(to_allowed):>13}{p:>9.3f}")

    total = {
        "n": sum(r["n"] for r in report.values()),
        "allowed_to_blocked": sum(r["allowed_to_blocked"] for r in report.values()),
        "blocked_to_allowed": sum(r["blocked_to_allowed"] for r in report.values()),
    }
    total["mcnemar_p"] = mcnemar_exact(total["allowed_to_blocked"], total["blocked_to_allowed"])
    print("  " + "-" * 73)
    print(f"  {'all':<14}{total['n']:>6}{'':>9}{'':>9}"
          f"{total['allowed_to_blocked']:>13}{total['blocked_to_allowed']:>13}"
          f"{total['mcnemar_p']:>9.3f}")
    print()
    print("  A prompt is blocked when it was not allowed; escalated counts as blocked.")
    print("  p is a two-sided exact McNemar test on the prompts that changed side.")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"benchmarks": report, "all": total}, indent=2),
                            encoding="utf-8")
        print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
