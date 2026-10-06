"""
Operating-point frontiers on the final test set, from stored scores
===================================================================
No model is loaded. For every arm that stores a graded harm score, the decision
threshold is swept and two rates are computed at each threshold, each a
macro-average over sources so that no source is weighted by its size:
  safe blocked     : mean over CoCoNot, PHTest, No Robots of the share blocked
  harmful allowed  : mean over SimpleSafetyTests, CatQA, Do-Not-Answer core of the share allowed
Do-Not-Answer "other" is excluded as secondary, as everywhere else.

The cascade's routing band is also swept offline: with the stored Layer-1 score
of each checkpoint and the stored judge decision on every prompt, a symmetric
band [0.5-w, 0.5+w] routes the prompts inside it to the judge and decides the
rest by Layer 1. w = 0.2 is the configured band. This is a post hoc sensitivity
analysis on the final test set; nothing was selected from it.

    python scripts/final_eval/frontiers.py [--study DIR] [--out FILE]
Writes results/final_eval_20260926/analysis/frontiers.json
"""
import json
from pathlib import Path
import numpy as np

STUDY = Path("results/final_eval_20260926")   # overridden by --study
OUT = None                                      # overridden by --out; None = <study>/analysis/frontiers.json
SAFE = ("coconot", "phtest", "no_robots")
HARM = ("sst", "catqa", "dna_core")
ARMS = {  # arm: (predictions file, score field)
    "layer1": ("final_cascade", "l1_score"),
    "layer1_s42": ("final_cascade_s42", "l1_score"),
    "layer1_s43": ("final_cascade_s43", "l1_score"),
    "layer1_s44": ("final_cascade_s44", "l1_score"),
    "judge": ("final_judge", "l2_p_mitigate"),
    "judge_llama_v1": ("final_judge_llama_v1", "l2_p_mitigate"),
    "granite": ("final_granite", "p_unsafe"),
    "llamaguard": ("final_llamaguard", "p_unsafe"),
    "shieldgemma": ("final_shieldgemma", "p_unsafe"),
}
NATIVE = {"cascade": "final_cascade", "layer1": "final_layer1", "judge": "final_judge",
          "judge_llama_v1": "final_judge_llama_v1", "granite": "final_granite",
          "llamaguard": "final_llamaguard", "shieldgemma": "final_shieldgemma"}


def load(name):
    return {r["id"]: r for r in (json.loads(l) for l in open(STUDY / f"{name}_predictions/test.jsonl"))}


def macro_rates(block, src):
    """block: dict id -> bool blocked. Returns (safe_blocked, harmful_allowed), macro over sources."""
    sb = np.mean([np.mean([block[i] for i in src[s]]) for s in SAFE])
    ha = np.mean([np.mean([not block[i] for i in src[s]]) for s in HARM])
    return float(sb), float(ha)


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(STUDY / "final_test.jsonl")}
    src = {}
    for i, r in meta.items():
        src.setdefault(r["source"], []).append(i)
    out = {"macro_sources": {"safe": SAFE, "harmful": HARM}, "frontiers": {}, "native": {}, "band_sweep": {}}

    for arm, (fname, field) in ARMS.items():
        rows = load(fname)
        score = {i: float(rows[i][field]) for i in meta}
        grid = np.unique(np.quantile(list(score.values()), np.linspace(0, 1, 401)))
        grid = np.concatenate(([-1e-9], grid, [1 + 1e-9]))
        pts = []
        for t in grid:
            block = {i: score[i] >= t for i in meta}
            sb, ha = macro_rates(block, src)
            pts.append((float(t), sb, ha))
        out["frontiers"][arm] = pts

    for arm, fname in NATIVE.items():
        rows = load(fname)
        block = {i: int(rows[i]["final_pred"]) == 0 for i in meta}
        out["native"][arm] = macro_rates(block, src)

    judge = load("final_judge")
    jblock = {i: int(judge[i]["final_pred"]) == 0 for i in meta}
    for ck, fname in (("historical", "final_cascade"), ("s42", "final_cascade_s42"),
                      ("s43", "final_cascade_s43"), ("s44", "final_cascade_s44")):
        rows = load(fname)
        s = {i: float(rows[i]["l1_score"]) for i in meta}
        sweep = []
        for w in np.round(np.arange(0.0, 0.5001, 0.01), 2):
            lo, hi = 0.5 - w, 0.5 + w
            routed = [i for i in meta if lo <= s[i] <= hi]
            block = {i: (jblock[i] if lo <= s[i] <= hi else s[i] > hi) for i in meta}
            sb, ha = macro_rates(block, src)
            sweep.append({"w": float(w), "band": [round(lo, 2), round(hi, 2)],
                          "routed_share": len(routed) / len(meta), "safe_blocked": sb, "harmful_allowed": ha})
        out["band_sweep"][ck] = sweep

    (Path(OUT) if OUT else STUDY / "analysis" / "frontiers.json").write_text(json.dumps(out, indent=1))
    # console summary
    print("native macro rates (safe_blocked, harmful_allowed):")
    for a, v in out["native"].items():
        print(f"  {a:16s} {100*v[0]:5.1f}  {100*v[1]:5.1f}")
    print("band sweep, historical (w, routed%, SB, HA):")
    for p in out["band_sweep"]["historical"]:
        if round(p["w"] * 100) % 5 == 0:
            print(f"  w={p['w']:.2f} [{p['band'][0]:.2f},{p['band'][1]:.2f}] routed {100*p['routed_share']:5.1f}%  SB {100*p['safe_blocked']:5.1f}  HA {100*p['harmful_allowed']:5.1f}")
    for ck in ("s42", "s43", "s44"):
        p0 = out["band_sweep"][ck][20]; p5 = out["band_sweep"][ck][50]
        print(f"  {ck}: w=0.20 routed {100*p0['routed_share']:.1f}% SB {100*p0['safe_blocked']:.1f} HA {100*p0['harmful_allowed']:.1f} | w=0.50 routed {100*p5['routed_share']:.1f}% SB {100*p5['safe_blocked']:.1f} HA {100*p5['harmful_allowed']:.1f}")
    # frontier readouts: harmful allowed at the safe-blocked level of the configured cascade and of Layer 1's native point
    sb_c = out["native"]["cascade"][0]
    print(f"\nharmful allowed on each frontier at safe_blocked <= {100*sb_c:.1f}% (cascade's macro level):")
    for arm, pts in out["frontiers"].items():
        ok = [p for p in pts if p[1] <= sb_c]
        best = min(ok, key=lambda p: p[2]) if ok else None
        print(f"  {arm:16s} HA {100*best[2]:5.1f} at t={best[0]:.3f} (SB {100*best[1]:.1f})" if best else f"  {arm}: none")
    return 0




def _cli(default_out: str, extra=None):
    """Study/output locations. Defaults reproduce the recorded run exactly; the globals are
    module-level so the computation itself is unchanged."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--study", default=str(STUDY), help="study folder (default: %(default)s)")
    ap.add_argument("--out", default=None, help=f"output file (default: <study>/analysis/{default_out})")
    for name, default, help_ in (extra or []):
        ap.add_argument(name, default=default, help=help_ + " (default: %(default)s)")
    return ap.parse_args()


if __name__ == "__main__":
    _a = _cli("frontiers.json")
    STUDY, OUT = Path(_a.study), _a.out
    raise SystemExit(main())
