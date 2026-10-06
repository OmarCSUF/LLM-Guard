"""
Paired parity analysis with attribution: bound, dose and cause.

scripts/compare_flips.py counts the decisions that change side between two
runs on the same prompts. This goes further, for the RQ4 studies:

  * a paired bootstrap interval on the difference in blocked share and a
    non-inferiority verdict at a stated margin, which is the claim RQ4 needs
    ("the non-native form is not blocked more often, within delta");
  * the determinism check the paired design rests on: an item with zero
    edits is identical in both arms and must get the same decision and score;
  * dose-response by number of edits, with a rank correlation;
  * attribution: which rules (proxy study) or error categories (learner
    study) the flips travel with.

The variant arm's test.jsonl supplies the per-item attributes, since the
prediction files carry only id, text, label and the layers' decisions.

Direction: every difference is variant minus baseline, where the variant is
the non-native form. For the proxy study the baseline is the `original` arm
and the variant a rewritten arm; for the learner study the baseline is the
`corrected` arm and the variant the `original` (as written) arm.

Usage:
    python scripts/parity_attribution.py \\
        --baseline data/processed/parity_original.json \\
        --variant  data/processed/parity_general.json \\
        --variant-arm data/benchmarks_parity/general --by l2_rules \\
        --out data/processed/parity_attr_general.json

    python scripts/parity_attribution.py \\
        --baseline results/learner_corrected.json \\
        --variant  results/learner_original.json \\
        --variant-arm data/benchmarks_learner/original --by edit_cats --count n_edits \\
        --out results/learner_attr.json

Exit status 1 when the determinism check fails.
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                       # for `from scripts.<module> import ...`
sys.path.insert(0, os.path.join(_ROOT, "src"))  # local checkout without `pip install -e .`

import argparse
import json
import zlib
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

from scripts.compare_flips import load_rows, blocked, mcnemar_exact
from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger

logger = get_logger("parity_attribution")

DEFAULT_DELTA = 0.02
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 2000


def read_arm(arm_root: Path, bench: str, by: str, count: str) -> dict:
    """{id: (attributes list, edit count)} from <arm_root>/<bench>/test.jsonl."""
    path = arm_root / bench / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no arm file for {bench} at {path}")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            attrs = row.get(by, [])
            if attrs is None:
                attrs = []
            if not isinstance(attrs, list):
                attrs = [attrs]
            n = int(row.get(count)) if count and row.get(count) is not None else len(attrs)
            out[str(row["id"])] = ([str(a) for a in attrs], n)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm(pvals: dict) -> dict:
    """Holm step-down adjustment over a {name: p} dict."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = running
    return out


def spearman_perm(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, permutations: int) -> dict:
    """Spearman rho with a permutation p-value; None when there is nothing to rank."""
    if len(x) < 10 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p_perm": None, "n": int(len(x))}

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        # average ties
        vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    hits = 0
    for _ in range(permutations):
        rp = rng.permutation(ry)
        if abs(np.corrcoef(rx, rp)[0, 1]) >= abs(rho) - 1e-12:
            hits += 1
    return {"rho": rho, "p_perm": (hits + 1) / (permutations + 1), "n": int(len(x))}


def analyse(bench: str, base: dict, var: dict, attrs: dict, delta: float,
            b_resamples: int, permutations: int, seed: int) -> dict:
    ids = [i for i in base if i in var]
    if len(ids) != len(base) or len(ids) != len(var):
        logger.warning(f"{bench}: {len(base)} baseline, {len(var)} variant, {len(ids)} in both")
    for i in ids:
        if int(base[i]["label"]) != int(var[i]["label"]):
            raise SystemExit(f"{bench}: label differs for id {i}")
    missing_attr = [i for i in ids if i not in attrs]
    if missing_attr:
        raise SystemExit(f"{bench}: {len(missing_attr)} ids have no row in the variant arm file")

    b0 = np.array([blocked(base[i]) for i in ids], dtype=float)
    b1 = np.array([blocked(var[i]) for i in ids], dtype=float)
    shift = np.array([float(var[i]["harm_score"]) - float(base[i]["harm_score"]) for i in ids])
    counts = np.array([attrs[i][1] for i in ids], dtype=int)
    edited = counts > 0

    # determinism on the items the arms share byte for byte
    zero = ~edited
    det_changed = int(np.sum((b0 != b1) & zero))
    det_max_shift = float(np.max(np.abs(shift[zero]))) if zero.any() else 0.0
    determinism_ok = det_changed == 0 and det_max_shift == 0.0

    def paired(mask, tag):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        x0, x1 = b0[mask], b1[mask]
        to_blocked = int(np.sum((x0 == 0) & (x1 == 1)))
        to_allowed = int(np.sum((x0 == 1) & (x1 == 0)))
        diff = float(x1.mean() - x0.mean())
        rng = np.random.default_rng([seed, zlib.crc32(f"{bench}:{tag}".encode("utf-8"))])
        idx = rng.integers(0, n, size=(b_resamples, n))
        diffs = (x1[idx].mean(axis=1) - x0[idx].mean(axis=1))
        lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
        return {"n": n, "blocked_baseline": int(x0.sum()), "blocked_variant": int(x1.sum()),
                "rate_baseline": float(x0.mean()), "rate_variant": float(x1.mean()),
                "diff": diff, "ci_low": lo, "ci_high": hi,
                "non_inferior": bool(hi < delta), "delta": delta,
                "allowed_to_blocked": to_blocked, "blocked_to_allowed": to_allowed,
                "mcnemar_p": mcnemar_exact(to_blocked, to_allowed),
                "mean_shift": float(shift[mask].mean())}

    out = {
        "n": len(ids),
        "determinism": {"zero_edit_items": int(zero.sum()), "decisions_changed": det_changed,
                        "max_abs_shift": det_max_shift, "ok": determinism_ok},
        "edited": paired(edited, "edited"),
        "all": paired(np.ones(len(ids), dtype=bool), "all"),
        "wilson_variant_blocked": wilson(int(b1.sum()), len(ids)),
        "wilson_baseline_blocked": wilson(int(b0.sum()), len(ids)),
    }

    # dose-response
    bins = {}
    for label, mask in (("0", counts == 0), ("1", counts == 1), ("2", counts == 2), ("3+", counts >= 3)):
        n = int(mask.sum())
        bins[label] = {"n": n,
                       "allowed_to_blocked": int(np.sum((b0 == 0) & (b1 == 1) & mask)),
                       "blocked_to_allowed": int(np.sum((b0 == 1) & (b1 == 0) & mask)),
                       "mean_shift": float(shift[mask].mean()) if n else None}
    rng = np.random.default_rng([seed, zlib.crc32(f"{bench}:spearman".encode("utf-8"))])
    out["dose"] = {"bins": bins, "spearman_count_vs_shift": spearman_perm(counts.astype(float), shift, rng, permutations)}

    # attribution by rule or category
    per = defaultdict(lambda: {"n": 0, "allowed_to_blocked": 0, "blocked_to_allowed": 0, "shift_sum": 0.0})
    for k, i in enumerate(ids):
        for a in set(attrs[i][0]):
            d = per[a]
            d["n"] += 1
            d["allowed_to_blocked"] += int(b0[k] == 0 and b1[k] == 1)
            d["blocked_to_allowed"] += int(b0[k] == 1 and b1[k] == 0)
            d["shift_sum"] += float(shift[k])
    attribution = {}
    for a, d in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        attribution[a] = {"n": d["n"], "allowed_to_blocked": d["allowed_to_blocked"],
                          "blocked_to_allowed": d["blocked_to_allowed"],
                          "flip_rate": (d["allowed_to_blocked"] + d["blocked_to_allowed"]) / d["n"],
                          "mean_shift": d["shift_sum"] / d["n"]}
    out["attribution"] = attribution
    return out


def pooled(results: dict, names: list, delta: float, b_resamples: int, seed: int,
           base_all: dict, var_all: dict, attrs_all: dict) -> dict:
    """Edited items pooled over the named benchmarks, resampled within benchmark."""
    parts = []
    for bench in names:
        ids = [i for i in base_all[bench] if i in var_all[bench] and attrs_all[bench][i][1] > 0]
        if ids:
            parts.append((np.array([blocked(base_all[bench][i]) for i in ids], dtype=float),
                          np.array([blocked(var_all[bench][i]) for i in ids], dtype=float)))
    if not parts:
        return {"n": 0}
    n = sum(len(p[0]) for p in parts)
    x0 = np.concatenate([p[0] for p in parts])
    x1 = np.concatenate([p[1] for p in parts])
    rng = np.random.default_rng([seed, zlib.crc32("pooled".encode("utf-8"))])
    diffs = np.zeros(b_resamples)
    for a0, a1 in parts:
        idx = rng.integers(0, len(a0), size=(b_resamples, len(a0)))
        diffs += (a1[idx].sum(axis=1) - a0[idx].sum(axis=1))
    diffs /= n
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    to_blocked = int(np.sum((x0 == 0) & (x1 == 1)))
    to_allowed = int(np.sum((x0 == 1) & (x1 == 0)))
    return {"n": n, "benchmarks": names, "rate_baseline": float(x0.mean()), "rate_variant": float(x1.mean()),
            "diff": float(x1.mean() - x0.mean()), "ci_low": lo, "ci_high": hi,
            "non_inferior": bool(hi < delta), "delta": delta,
            "allowed_to_blocked": to_blocked, "blocked_to_allowed": to_allowed,
            "mcnemar_p": mcnemar_exact(to_blocked, to_allowed)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", type=Path, required=True, help="results JSON of the native-like arm")
    p.add_argument("--variant", type=Path, required=True, help="results JSON of the non-native arm")
    p.add_argument("--variant-arm", type=Path, required=True,
                   help="arm root holding <benchmark>/test.jsonl with the per-item attributes")
    p.add_argument("--by", default="l2_rules", help="list field naming the rules or categories (default %(default)s)")
    p.add_argument("--count", default=None,
                   help="integer field with the number of edits; default is the length of --by")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="non-inferiority margin (default %(default)s)")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    p.add_argument("--pool", nargs="+", default=None,
                   help="benchmarks to pool for the headline (default: every one with 'wi_' in its name, else all)")
    p.add_argument("--seed", type=int, default=None, help="default: project.seed")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    seed = args.seed if args.seed is not None else int(cfg["project"]["seed"])

    base_all, var_all = load_rows(args.baseline), load_rows(args.variant)
    names = [b for b in base_all if b in var_all]
    if not names:
        raise SystemExit("the two runs have no benchmark in common")
    attrs_all = {b: read_arm(args.variant_arm, b, args.by, args.count) for b in names}

    report = {"direction": "variant minus baseline (non-native minus native-like)",
              "delta": args.delta, "bootstrap": args.bootstrap, "seed": seed,
              "by": args.by, "count": args.count or f"len({args.by})", "benchmarks": {}}
    print()
    print(f"  {'benchmark':<12}{'n':>6}{'edited':>7}{'blk base':>9}{'blk var':>8}{'diff':>8}"
          f"{'95% CI':>18}{'<delta':>7}{'a>b':>5}{'b>a':>5}{'p':>7}{'det':>5}")
    print("  " + "-" * 97)
    failed = False
    for bench in names:
        r = analyse(bench, base_all[bench], var_all[bench], attrs_all[bench], args.delta,
                    args.bootstrap, args.permutations, seed)
        report["benchmarks"][bench] = r
        e = r["edited"]
        if e.get("n"):
            print(f"  {bench:<12}{r['n']:>6}{e['n']:>7}{e['blocked_baseline']:>9}{e['blocked_variant']:>8}"
                  f"{e['diff']:>+8.3f}  [{e['ci_low']:>+6.3f}, {e['ci_high']:>+6.3f}]"
                  f"{'yes' if e['non_inferior'] else 'no':>7}{e['allowed_to_blocked']:>5}"
                  f"{e['blocked_to_allowed']:>5}{e['mcnemar_p']:>7.3f}{'ok' if r['determinism']['ok'] else 'FAIL':>5}")
        else:
            print(f"  {bench:<12}{r['n']:>6}{0:>7}  (no edited items)")
        failed = failed or not r["determinism"]["ok"]

    pool_names = args.pool or [b for b in names if "wi_" in b] or names
    report["pooled"] = pooled(report["benchmarks"], pool_names, args.delta, args.bootstrap, seed,
                              base_all, var_all, attrs_all)
    pl = report["pooled"]
    if pl.get("n"):
        print("  " + "-" * 97)
        print(f"  {'pooled':<12}{pl['n']:>6}{pl['n']:>7}{'':>9}{'':>8}{pl['diff']:>+8.3f}  "
              f"[{pl['ci_low']:>+6.3f}, {pl['ci_high']:>+6.3f}]{'yes' if pl['non_inferior'] else 'no':>7}"
              f"{pl['allowed_to_blocked']:>5}{pl['blocked_to_allowed']:>5}{pl['mcnemar_p']:>7.3f}")
        print(f"  pooled over: {', '.join(pool_names)}")
    report["holm_mcnemar_edited"] = holm({b: report["benchmarks"][b]["edited"]["mcnemar_p"]
                                         for b in pool_names if report["benchmarks"][b]["edited"].get("n")})

    print()
    print(f"  diff is variant minus baseline on the edited items; CI is a {args.bootstrap}-resample paired")
    print(f"  percentile bootstrap; '<delta' says whether the upper limit is below {args.delta:.3f}.")
    print("  a>b: allowed in baseline, blocked in variant. det: zero-edit items unchanged.")
    print()
    for bench in names:
        r = report["benchmarks"][bench]
        d = r["dose"]
        sp = d["spearman_count_vs_shift"]
        rho = f"{sp['rho']:+.3f} (perm p={sp['p_perm']:.3f})" if sp["rho"] is not None else "n/a"
        bins = "  ".join(f"{k}: n={v['n']} shift={v['mean_shift']:+.4f} a>b={v['allowed_to_blocked']} b>a={v['blocked_to_allowed']}"
                         for k, v in d["bins"].items() if v["n"])
        print(f"  {bench}: dose  {bins}")
        print(f"  {bench}: spearman(edits, shift) = {rho}")
        top = list(r["attribution"].items())[:12]
        if top:
            print(f"  {bench}: by {args.by}: " + "; ".join(
                f"{a} n={v['n']} a>b={v['allowed_to_blocked']} b>a={v['blocked_to_allowed']} shift={v['mean_shift']:+.4f}"
                for a, v in top))
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  written to {args.out}")
    if failed:
        print("  DETERMINISM CHECK FAILED: an item identical in both arms changed decision or score.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
