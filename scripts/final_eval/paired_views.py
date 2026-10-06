"""
Paired effect of the leading-clause payload view on the HarmBench attack suite
==============================================================================
Reads the two per-variant red-team files written with the view disabled and
enabled (3,600 variants each, paired by request, attack type and variant text),
and reports the change in bypass rate with a 95% percentile interval from a
cluster bootstrap that resamples the 200 seed requests. The bootstrap stream is
derived from the study seed and a fixed name, exactly as analyse_final.py does,
so the interval is reproducible and independent of anything else analysed.

    python scripts/final_eval/paired_views.py [--views-off FILE] [--views-on FILE] [--out FILE]

Writes results/final_eval_20260926/analysis/paired_views.json.
"""
import json, zlib, collections
from pathlib import Path
import numpy as np

BOOT_B, BOOT_SEED, ALPHA = 2000, 20260926, 0.05
OFF = Path("results/development/redteam_all200_variants.jsonl")        # views off; --views-off
ON = Path("results/development/redteam_all200_views_variants.jsonl")    # views on; --views-on
OUT = Path("results/final_eval_20260926/analysis/paired_views.json")   # --out


def load(p: Path) -> dict:
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return {(r["original"], r["attack_type"], r["variant"]): bool(r["bypassed"]) for r in rows}


def main() -> int:
    off, on = load(OFF), load(ON)
    keys = [k for k in off if k in on]
    assert len(keys) == len(off) == len(on), (len(keys), len(off), len(on))
    by_req = collections.defaultdict(list)
    for k in keys:
        by_req[k[0]].append(float(on[k]) - float(off[k]))
    reqs = sorted(by_req)
    arrs = [np.array(by_req[r]) for r in reqs]
    n = sum(len(a) for a in arrs)
    point = sum(a.sum() for a in arrs) / n
    rng = np.random.default_rng([BOOT_SEED, zlib.crc32(b"redteam|paired_views")])
    reps = []
    for _ in range(BOOT_B):
        idx = rng.integers(0, len(reqs), len(reqs))
        reps.append(sum(arrs[i].sum() for i in idx) / sum(len(arrs[i]) for i in idx))
    lo, hi = np.percentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    by_type = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        t = by_type[k[1]]; t[0] += 1; t[1] += off[k]; t[2] += on[k]
    out = {
        "n_variants": n, "n_requests": len(reqs),
        "bypassed_views_off": int(sum(off.values())), "bypassed_views_on": int(sum(on.values())),
        "rate_views_off": sum(off.values()) / n, "rate_views_on": sum(on.values()) / n,
        "paired_diff_points": 100 * point,
        "ci95_request_clustered_points": [100 * float(lo), 100 * float(hi)],
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "stream": "redteam|paired_views", "cluster": "seed request"},
        "by_attack_type": {t: {"n": v[0], "bypassed_off": v[1], "bypassed_on": v[2],
                               "rate_off": v[1] / v[0], "rate_on": v[2] / v[0]} for t, v in sorted(by_type.items())},
        "inputs": {"views_off": str(OFF), "views_on": str(ON)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"paired diff {out['paired_diff_points']:+.2f} pp, 95% CI [{100*lo:+.2f}, {100*hi:+.2f}] -> {OUT}")
    return 0




if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    _ap.add_argument("--views-off", default=str(OFF), help="per-variant file of the views-off run (default: %(default)s)")
    _ap.add_argument("--views-on", default=str(ON), help="per-variant file of the views-on run (default: %(default)s)")
    _ap.add_argument("--out", default=str(OUT), help="output JSON (default: %(default)s)")
    _a = _ap.parse_args()
    OFF, ON, OUT = Path(_a.views_off), Path(_a.views_on), Path(_a.out)
    raise SystemExit(main())
