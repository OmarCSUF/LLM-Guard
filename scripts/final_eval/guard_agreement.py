"""
Native verdict versus token-probability harm score, per external guard
======================================================================
Each guard arm stores, per prompt, the guard's own generated verdict (the
native decision), the probability of the unsafe answer renormalised over the
first-token spellings of the two answers at the step at which the verdict is
emitted (the harm score), the probability mass those answer tokens hold at that
step, and for Llama Guard the step index. This script reports, on the final
test set and on the pool validation split: the parse rate of the native output;
the agreement between the native decision and the harm score thresholded at
0.5, with the direction of every disagreement; the answer-mass distribution;
and for ShieldGemma the agreement between "any policy answers Yes" and the
maximum per-policy probability at 0.5.

    python scripts/final_eval/guard_agreement.py [--study DIR] [--out FILE]
Writes results/final_eval_20260926/analysis/guard_agreement.json
"""
import json, collections
from pathlib import Path
import numpy as np

STUDY = Path("results/final_eval_20260926")   # overridden by --study
OUT = None                                      # overridden by --out; None = <study>/analysis/guard_agreement.json
GUARDS = ("granite", "llamaguard", "shieldgemma")


def load(prefix, g):
    p = STUDY / f"{prefix}_{g}_predictions"
    files = sorted(p.glob("*.jsonl"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f)]
    return rows


def summarise(rows):
    n = len(rows)
    parse_ok = sum(1 for r in rows if r.get("native_parse_ok"))
    native_unsafe = [int(r["final_pred"]) == 0 for r in rows]
    score_unsafe = [float(r["p_unsafe"]) >= 0.5 for r in rows]
    agree = sum(a == b for a, b in zip(native_unsafe, score_unsafe))
    dis = collections.Counter()
    for r, a, b in zip(rows, native_unsafe, score_unsafe):
        if a != b:
            dis["native unsafe, score < 0.5" if a else "native safe, score >= 0.5"] += 1
    mass = np.array([r.get("mass_on_answers") for r in rows if r.get("mass_on_answers") is not None], float)
    out = {"n": n, "native_parse_ok": parse_ok, "agreement": agree / n, "n_disagree": n - agree,
           "disagreements": dict(dis),
           "answer_mass": {"median": float(np.median(mass)), "min": float(mass.min()),
                           "share_below_0.5": float((mass < 0.5).mean()), "share_below_0.9": float((mass < 0.9).mean())} if len(mass) else None}
    steps = collections.Counter(r.get("answer_step") for r in rows if r.get("answer_step") is not None)
    if steps:
        out["answer_step"] = dict(sorted(steps.items()))
    # scores of the disagreeing prompts, to see how close to 0.5 they sit
    ds = [float(r["p_unsafe"]) for r, a, b in zip(rows, native_unsafe, score_unsafe) if a != b]
    if ds:
        out["disagree_scores"] = {"min": min(ds), "max": max(ds), "median": float(np.median(ds))}
    if rows and rows[0].get("per_policy"):
        pol = collections.Counter(); agree_pol = 0
        for r in rows:
            pp = r["per_policy"]
            any_yes = any(v["native_unsafe"] for v in pp.values())
            mx = max(v["p_unsafe"] for v in pp.values()) >= 0.5
            agree_pol += any_yes == mx
            for k, v in pp.items():
                pol[k] += int(v["native_unsafe"] == (v["p_unsafe"] >= 0.5))
        out["shieldgemma"] = {"any_yes_vs_max_p": agree_pol / n,
                              "per_policy_agreement": {k: v / n for k, v in pol.items()},
                              "worst_policy": dict(collections.Counter(r.get("worst_policy") for r in rows))}
    return out


def main():
    out = {}
    for g in GUARDS:
        out[g] = {"final_test_set": summarise(load("final", g))}
        try:
            out[g]["pool_validation"] = summarise(load("val", g))
        except Exception as e:
            out[g]["pool_validation"] = f"missing: {e}"
    (Path(OUT) if OUT else STUDY / "analysis" / "guard_agreement.json").write_text(json.dumps(out, indent=1))
    for g, v in out.items():
        for split, s in v.items():
            if isinstance(s, dict):
                print(f"{g:12s} {split:16s} n={s['n']} parse_ok={s['native_parse_ok']} agree={100*s['agreement']:.2f}% "
                      f"disagree={s['n_disagree']} {s['disagreements']} mass={s['answer_mass']} step={s.get('answer_step')} dscores={s.get('disagree_scores')}")
                if "shieldgemma" in s:
                    print("   ", s["shieldgemma"])
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
    _a = _cli("guard_agreement.json")
    STUDY, OUT = Path(_a.study), _a.out
    raise SystemExit(main())
