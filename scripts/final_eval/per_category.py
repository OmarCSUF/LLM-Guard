"""
Per-category rates inside each source of the final test set
===========================================================
The pre-registration (Section 6, secondary outcomes) names per-category rates
inside CatQA and Do-Not-Answer, to be read as a ranking because the cells are
small. This script computes them for every source that carries a category
label, for every arm, from the stored per-prompt predictions: harmful allowed
on harmful sources, safe blocked on safe sources, with a 95% Wilson interval
per cell. CatQA, Do-Not-Answer and CoCoNot labels are "area / subcategory";
CatQA is grouped by its 11 areas, Do-Not-Answer and CoCoNot are kept at the
subcategory level because their areas are few and large. PHTest carries a
single label ("harmless") and cannot be broken down.

    python scripts/final_eval/per_category.py [--study DIR] [--out FILE]
Writes results/final_eval_20260926/analysis/per_category.json and .md
"""
import json, math, collections
from pathlib import Path

STUDY = Path("results/final_eval_20260926")   # overridden by --study
OUT = None                                      # overridden by --out; None = <study>/analysis/per_category.json
ARMS = [("cascade", "final_cascade"), ("layer1", "final_layer1"), ("judge", "final_judge"),
        ("judge_llama_v1", "final_judge_llama_v1"), ("granite", "final_granite"),
        ("llamaguard", "final_llamaguard"), ("shieldgemma", "final_shieldgemma")]
SAFE = {"coconot", "phtest", "no_robots"}
Z = 1.959964


def wilson(k, n):
    if n == 0: return (None, None)
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (c - h, c + h)


def group(src, cat):
    if src == "catqa":
        return cat.split(" / ")[0]
    return cat


def main():
    meta = [json.loads(l) for l in open(STUDY / "final_test.jsonl")]
    preds = {a: {r["id"]: int(r["final_pred"]) for r in (json.loads(l) for l in open(STUDY / f"{f}_predictions/test.jsonl"))}
             for a, f in ARMS}
    cells = collections.OrderedDict()
    for r in meta:
        if r["source"] in ("phtest", "dna_other"):
            continue
        cells.setdefault((r["source"], group(r["source"], r["category"])), []).append(r["id"])
    out, md = [], ["# Per-category rates by arm (harmful sources: harmful allowed; safe sources: safe blocked), %\n",
                   "| Source | Category | n | " + " | ".join(a for a, _ in ARMS) + " |", "|---|---|---:|" + "---:|" * len(ARMS)]
    for (src, cat), ids in cells.items():
        row = {"source": src, "category": cat, "n": len(ids), "rates": {}}
        for a, _ in ARMS:
            if src in SAFE:
                k = sum(1 for i in ids if preds[a][i] == 0)   # blocked
            else:
                k = sum(1 for i in ids if preds[a][i] == 1)   # allowed
            lo, hi = wilson(k, len(ids))
            row["rates"][a] = {"count": k, "rate": k / len(ids), "ci": [lo, hi]}
        out.append(row)
        md.append(f"| {src} | {cat} | {len(ids)} | " + " | ".join(f"{100*row['rates'][a]['rate']:.1f}" for a, _ in ARMS) + " |")
    (Path(OUT) if OUT else STUDY / "analysis" / "per_category.json").write_text(json.dumps(out, indent=1))
    ((Path(OUT).with_suffix(".md")) if OUT else STUDY / "analysis" / "per_category.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
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
    _a = _cli("per_category.json")
    STUDY, OUT = Path(_a.study), _a.out
    raise SystemExit(main())
