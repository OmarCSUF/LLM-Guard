"""
Calibration of the graded scores, in distribution and on the final test set
=========================================================================
Expected calibration error (ECE) with the project's own definition
(evaluation/metrics.py: 10 equal-width bins over [0, 1], a score of 1.0 in the
top bin, each bin weighted by its share of prompts, |mean harm score - fraction
harmful|), with a percentile bootstrap interval over prompts (B = 2000, study
seed, named stream), and the reliability bins behind it.

In distribution:
  Layer 1 (configured detector, views on)  pool test split, 1,912 prompts
                                           (pooltest_layer1_predictions)
  Judge, Qwen2.5-3B, prompt v2             300 pool-validation prompts of the
                                           tuning run (judge_tuning.jsonl)
  Judge, Llama-3.2-3B, prompt v1           the same 300 prompts (judge_tuning_llama.jsonl)
On the final test set (3,089 primary prompts; Do-Not-Answer "other" excluded):
  Layer 1 on all four checkpoints, both judges, the three guards.

The final test set pools single-class sources, so its ECE is conditional on
that source mixture (1,874 safe, 1,215 harmful among the primary sources); the
pool test split is balanced. Both are stated with the numbers.

    python scripts/final_eval/calibration.py [--study DIR] [--dev DIR] [--out FILE]
Writes results/final_eval_20260926/analysis/calibration.json
"""
import json, zlib
from pathlib import Path
import numpy as np

STUDY = Path("results/final_eval_20260926")   # overridden by --study
DEV = Path("results/development")             # judge_tuning*.jsonl live here; overridden by --dev
OUT = None                                      # overridden by --out; None = <study>/analysis/calibration.json
BOOT_B, BOOT_SEED, NB = 2000, 20260926, 10
PRIMARY = ("sst", "catqa", "dna_core", "coconot", "phtest", "no_robots")


def ece(score, harmful, nb=NB):
    score = np.asarray(score, float); harmful = np.asarray(harmful, float)
    bins = np.minimum((score * nb).astype(int), nb - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(score[m].mean() - harmful[m].mean())
    return float(total / len(score))


def reliability(score, harmful, nb=NB):
    score = np.asarray(score, float); harmful = np.asarray(harmful, float)
    bins = np.minimum((score * nb).astype(int), nb - 1)
    out = []
    for b in range(nb):
        m = bins == b
        out.append({"bin": [b / nb, (b + 1) / nb], "n": int(m.sum()),
                    "mean_score": float(score[m].mean()) if m.any() else None,
                    "frac_harmful": float(harmful[m].mean()) if m.any() else None})
    return out


def boot_ece(score, harmful, name):
    rng = np.random.default_rng([BOOT_SEED, zlib.crc32(f"ece|{name}".encode())])
    score = np.asarray(score, float); harmful = np.asarray(harmful, float); n = len(score)
    reps = [ece(score[idx], harmful[idx]) for idx in (rng.integers(0, n, n) for _ in range(BOOT_B))]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"ece": ece(score, harmful), "ci": [float(lo), float(hi)], "n": n}


def preds(name, field):
    rows = [json.loads(l) for l in open(STUDY / f"{name}_predictions/test.jsonl")]
    return rows, field


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(STUDY / "final_test.jsonl")}
    keep = {i for i, r in meta.items() if r["source"] in PRIMARY}
    out = {"definition": "10 equal-width bins; |mean harm score - fraction harmful| weighted by bin share",
           "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED}, "ece": {}, "reliability": {}}

    def add(name, score, harmful, rel=False):
        out["ece"][name] = boot_ece(score, harmful, name)
        if rel:
            out["reliability"][name] = reliability(score, harmful)

    # Layer 1 in distribution (pool test split, balanced)
    rows = [json.loads(l) for l in open(STUDY / "pooltest_layer1_predictions/test.jsonl")]
    add("layer1_pool_test", [r["l1_score"] for r in rows], [1 - int(r["label"]) for r in rows], rel=True)
    # Layer 1 on the final test set, four checkpoints
    for ck, f in (("", "final_cascade"), ("_s42", "final_cascade_s42"), ("_s43", "final_cascade_s43"), ("_s44", "final_cascade_s44")):
        rows, _ = preds(f, "l1_score")
        rows = [r for r in rows if r["id"] in keep]
        add(f"layer1_final{ck}", [r["l1_score"] for r in rows], [1 - int(r["label"]) for r in rows], rel=(ck == ""))
    # judges in distribution (tuning rows) and on the final set
    for name, tfile, ver, ffile in (("judge_qwen", "judge_tuning.jsonl", "v2", "final_judge"),
                                    ("judge_llama", "judge_tuning_llama.jsonl", "v1", "final_judge_llama_v1")):
        trows = [json.loads(l) for l in open(DEV / tfile)]
        trows = [r for r in trows if r["prompt_version"] == ver]
        add(f"{name}_pool_val300", [r["p_mitigate"] for r in trows], [1 - int(r["label"]) for r in trows], rel=(name == "judge_qwen"))
        rows, _ = preds(ffile, "l2_p_mitigate"); rows = [r for r in rows if r["id"] in keep]
        add(f"{name}_final", [r["l2_p_mitigate"] for r in rows], [1 - int(r["label"]) for r in rows], rel=(name == "judge_qwen"))
    for g in ("granite", "llamaguard", "shieldgemma"):
        rows, _ = preds(f"final_{g}", "p_unsafe"); rows = [r for r in rows if r["id"] in keep]
        add(f"{g}_final", [r["p_unsafe"] for r in rows], [1 - int(r["label"]) for r in rows])
    out["final_set_class_mix"] = {"n_safe": sum(1 for i in keep if int(meta[i].get("label", 1)) == 1 or meta[i]["source"] in ("coconot","phtest","no_robots")),
                                   "n_harmful": sum(1 for i in keep if meta[i]["source"] in ("sst","catqa","dna_core"))}
    (Path(OUT) if OUT else STUDY / "analysis" / "calibration.json").write_text(json.dumps(out, indent=1))
    for k, v in out["ece"].items():
        print(f"{k:28s} ECE {v['ece']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] n={v['n']}")
    print("class mix on final primary:", out["final_set_class_mix"])
    for k in ("layer1_pool_test", "layer1_final", "judge_qwen_pool_val300", "judge_qwen_final"):
        print("\n", k)
        for b in out["reliability"][k]:
            if b["n"]:
                print(f"  [{b['bin'][0]:.1f},{b['bin'][1]:.1f}) n={b['n']:5d} mean={b['mean_score']:.3f} harmful={b['frac_harmful']:.3f}")
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
    _a = _cli("calibration.json", [("--dev", str(DEV), "folder holding judge_tuning.jsonl and judge_tuning_llama.jsonl")])
    STUDY, DEV, OUT = Path(_a.study), Path(_a.dev), _a.out
    raise SystemExit(main())
