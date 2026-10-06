"""
Thesis Results with Confidence Intervals
========================================
Turns the per-prompt predictions written by scripts/run_evaluation.py into
thesis tables with bootstrap confidence intervals, without re-running a model.

For every run and benchmark it:
  1. recomputes the metrics from the predictions and checks them against the
     run's results JSON (a mismatch means the two files do not belong together);
  2. draws B stratified bootstrap resamples (prompts drawn with replacement
     within each class, so class counts stay fixed and AUROC stays defined)
     and reports percentile confidence intervals;
  3. counts outcomes and layer routing.

--compare A:B reports A - B on the same resamples (a paired bootstrap). An
interval that excludes 0 marks a difference unlikely to be resampling noise.

Operational rates count what a user experiences. An escalated prompt is
refused by Layer 3, so it counts as blocked:
  safe_blocked    : share of safe prompts that were not allowed
  harmful_allowed : share of harmful prompts that were allowed (jailbreak success rate)
The other metrics follow evaluation/metrics.py (accuracy, precision, recall and
F1 leave escalated prompts out; AUROC, AUPRC, FPR@95%TPR and ECE use the harm score).

Outputs: <out>.json (everything), <out>.md (tables to paste), <out>.csv (long format).

Usage:
    python scripts/summarise_results.py \\
        --run layer1=data/processed/eval_layer1_only_v2.json \\
        --run pipeline=data/processed/eval_pipeline_v2.json \\
        --compare pipeline:layer1 --out data/processed/thesis_results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import csv
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation import metrics as M

logger = get_logger("summarise_results")

METRICS = ("auroc", "auprc", "fpr_at_95tpr", "safe_blocked", "harmful_allowed",
           "accuracy", "precision", "recall", "f1", "ece")
TABLE_METRICS = (("auroc", "AUROC"), ("auprc", "AUPRC"), ("fpr_at_95tpr", "FPR@95%TPR"),
                 ("safe_blocked", "Safe blocked"), ("harmful_allowed", "Harmful allowed"),
                 ("f1", "F1"), ("ece", "ECE"))
# ASCII headers: the tables are also printed, and some consoles cannot encode a Greek delta.
DELTA_METRICS = (("auroc", "AUROC diff"), ("safe_blocked", "Safe blocked diff"),
                 ("harmful_allowed", "Harmful allowed diff"), ("f1", "F1 diff"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap confidence intervals and thesis tables from saved predictions")
    p.add_argument("--run", action="append", required=True, metavar="NAME=RESULTS_JSON",
                   help="A run_evaluation.py results file; predictions are read from <stem>_predictions/")
    p.add_argument("--compare", action="append", default=[], metavar="A:B",
                   help="Report run A minus run B with paired bootstrap intervals")
    p.add_argument("--bootstrap", type=int, default=None,
                   help="Resamples (default: evaluation.fairness.bootstrap_samples)")
    p.add_argument("--alpha", type=float, default=None,
                   help="1 - confidence level (default: evaluation.fairness.alpha)")
    p.add_argument("--seed", type=int, default=None, help="Default: project.seed")
    p.add_argument("--out", default="data/processed/thesis_results",
                   help="Output path without extension")
    return p.parse_args()


# ── Loading ───────────────────────────────────────────────────────────────────

def predictions_dir_for(results_path: Path) -> Path:
    """Where run_evaluation.py writes predictions for --output results_path."""
    return results_path.with_name(results_path.stem + "_predictions")


def _read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_run(spec: str) -> dict:
    """NAME=RESULTS_JSON -> {name, results_path, predictions_dir, saved, rows: {benchmark: rows}}."""
    if "=" not in spec:
        raise ValueError(f"--run expects NAME=RESULTS_JSON, got {spec!r}")
    name, path = spec.split("=", 1)
    results_path = Path(path)
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    pred_dir = predictions_dir_for(results_path)
    rows = {}
    for bench in saved:
        f = pred_dir / f"{bench}.jsonl"
        if f.exists():
            rows[bench] = _read_jsonl(f)
        else:
            logger.warning(f"{name}: no predictions for {bench} at {f}; skipped")
    if not rows:
        raise FileNotFoundError(f"{name}: no prediction files in {pred_dir}")
    return {"name": name, "results_path": results_path, "predictions_dir": pred_dir,
            "saved": saved, "rows": rows}


def sort_by_id(rows: list, where: str) -> list:
    """Rows ordered by id, so runs on the same benchmark line up prompt by prompt."""
    ids = [r.get("id") for r in rows]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{where}: every prediction row needs a unique id")
    return sorted(rows, key=lambda r: str(r["id"]))


def arrays(rows: list) -> tuple:
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["harm_score"]) for r in rows])
    p = np.array([int(r["final_pred"]) for r in rows])
    esc = np.array([bool(r.get("escalated")) for r in rows])
    return y, s, p, esc


# ── Metrics ───────────────────────────────────────────────────────────────────

def metric_values(y, s, p, esc) -> dict:
    """Unrounded metrics for one set of predictions; None where undefined."""
    safe, decided = y == 1, ~esc
    blocked = (p == 0) | esc
    prf = M.precision_recall_f1(y[decided], p[decided])
    return {
        "auroc":           M.auroc(y, s),
        "auprc":           M.auprc(y, s),
        "fpr_at_95tpr":    M.fpr_at_tpr(y, s, 0.95),
        "safe_blocked":    float(blocked[safe].mean()) if safe.any() else None,
        "harmful_allowed": M.jailbreak_success_rate(y, p),
        "accuracy":        M.accuracy(y[decided], p[decided]),
        "precision":       prf["precision"],
        "recall":          prf["recall"],
        "f1":              prf["f1"],
        "ece":             M.ece(y, s),
    }


def outcome_counts(rows: list, y, p, esc) -> dict:
    harm, safe = y == 0, y == 1
    blocked = (p == 0) | esc
    return {
        "n": int(len(y)), "harmful": int(harm.sum()), "safe": int(safe.sum()),
        "harmful_blocked": int((blocked & harm).sum()), "harmful_allowed": int((~blocked & harm).sum()),
        "safe_blocked": int((blocked & safe).sum()), "safe_allowed": int((~blocked & safe).sum()),
        "escalated": int(esc.sum()), "errors": sum(1 for r in rows if r.get("error")),
    }


def routing(rows: list) -> dict:
    c = Counter()
    for r in rows:
        c[f"l1_{r.get('l1_decision')}"] += 1
        if r.get("l2_decision") is not None:
            c[f"l2_{r['l2_decision']}"] += 1
        if r.get("l3_mode") is not None:
            c[f"l3_{r['l3_mode']}"] += 1
    return {
        "l1_allow": c["l1_allow"], "l1_gray_zone": c["l1_gray_zone"], "l1_mitigate": c["l1_mitigate"],
        "to_judge_share": round(c["l1_gray_zone"] / len(rows), 4) if rows else None,
        "l2_allow": c["l2_allow"], "l2_mitigate": c["l2_mitigate"], "l2_uncertain": c["l2_uncertain"],
        "l3_refuse": c["l3_refuse"], "l3_redact": c["l3_redact"], "l3_safe_rewrite": c["l3_safe_rewrite"],
    }


def check_saved(rows: list, saved_metrics: dict) -> list:
    """Metric names whose value in the results JSON differs from the one recomputed from predictions."""
    y, s, p, esc = arrays(rows)
    recomputed = M.compute_all(y, s, p, escalated=esc)
    bad = []
    for key, value in recomputed.items():
        if key not in saved_metrics:
            continue
        stored = saved_metrics[key]
        if value is None or stored is None:
            if not (value is None and stored is None):
                bad.append(key)
        elif abs(float(value) - float(stored)) > 1e-9:
            bad.append(key)
    return bad


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One resample: indices drawn with replacement within each class."""
    parts = [rng.choice(idx, size=len(idx), replace=True)
             for idx in (np.flatnonzero(y == 0), np.flatnonzero(y == 1)) if len(idx)]
    return np.concatenate(parts)


def interval(values: list, alpha: float) -> dict:
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return {"ci_low": None, "ci_high": None, "undefined_resamples": len(values)}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi), "undefined_resamples": len(values) - len(vals)}


def summarise_benchmark(bench: str, runs: list, comparisons: list, b: int, alpha: float, seed: int) -> dict:
    """Point estimates, intervals, counts and routing for each run, plus paired differences."""
    present = [r for r in runs if bench in r["rows"]]
    ordered = {r["name"]: sort_by_id(r["rows"][bench], f"{r['name']}/{bench}") for r in present}
    ref_ids = [row["id"] for row in ordered[present[0]["name"]]]
    for name, rows in ordered.items():
        if [row["id"] for row in rows] != ref_ids:
            raise ValueError(f"{bench}: run {name!r} has different prompt ids from run {present[0]['name']!r}")
    data = {name: arrays(rows) for name, rows in ordered.items()}
    y_ref = data[present[0]["name"]][0]
    for name, (y, *_rest) in data.items():
        if not np.array_equal(y, y_ref):
            raise ValueError(f"{bench}: run {name!r} has different labels for the same prompt ids")

    rng = np.random.default_rng([seed, zlib.crc32(bench.encode("utf-8"))])
    samples = {name: {m: [] for m in METRICS} for name in data}
    for _ in range(b):
        idx = stratified_indices(y_ref, rng)
        for name, arr in data.items():
            vals = metric_values(*(a[idx] for a in arr))
            for m in METRICS:
                samples[name][m].append(vals[m])

    out = {"runs": {}, "comparisons": {}}
    for r in present:
        name = r["name"]
        point = metric_values(*data[name])
        saved = r["saved"].get(bench, {}).get("metrics", {})
        out["runs"][name] = {
            "metrics": {m: {"value": point[m], **interval(samples[name][m], alpha)} for m in METRICS},
            "counts": outcome_counts(ordered[name], *[data[name][i] for i in (0, 2, 3)]),
            "routing": routing(ordered[name]),
            "saved_metric_mismatches": check_saved(ordered[name], saved),
        }
    for a, bname in comparisons:
        if a not in data or bname not in data:
            continue
        pa, pb = metric_values(*data[a]), metric_values(*data[bname])
        deltas = {}
        for m in METRICS:
            diffs = [x - z if x is not None and z is not None else None
                     for x, z in zip(samples[a][m], samples[bname][m])]
            value = pa[m] - pb[m] if pa[m] is not None and pb[m] is not None else None
            ci = interval(diffs, alpha)
            excludes_zero = (ci["ci_low"] is not None and (ci["ci_low"] > 0 or ci["ci_high"] < 0))
            deltas[m] = {"value": value, **ci, "excludes_zero": bool(excludes_zero)}
        out["comparisons"][f"{a} - {bname}"] = deltas
    return out


# ── Reporting ─────────────────────────────────────────────────────────────────

def _cell(entry: dict, digits: int = 3, mark: bool = False) -> str:
    if entry["value"] is None:
        return "n/a"
    text = f"{entry['value']:.{digits}f}"
    if entry["ci_low"] is not None:
        text += f" [{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]"
    if mark and entry.get("excludes_zero"):
        text += " *"
    return text


def to_markdown(summary: dict) -> str:
    p = summary["parameters"]
    level = round(100 * (1 - p["alpha"]))
    lines = [f"# Results with {level}% bootstrap confidence intervals", ""]
    run_names = list(summary["runs"])
    for name in run_names:
        info = summary["runs"][name]
        lines += [f"## Run `{name}`", "",
                  f"Results: `{info['results']}` (commit `{info['commit']}`)", "",
                  "| Benchmark | Harmful / safe | " + " | ".join(label for _, label in TABLE_METRICS) + " |",
                  "|---|---|" + "---|" * len(TABLE_METRICS)]
        for bench, entry in summary["benchmarks"].items():
            if name not in entry["runs"]:
                continue
            run = entry["runs"][name]
            c = run["counts"]
            cells = [_cell(run["metrics"][m]) for m, _ in TABLE_METRICS]
            lines.append(f"| {bench} | {c['harmful']} / {c['safe']} | " + " | ".join(cells) + " |")
        lines += ["", "| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |",
                  "|---|---|---|---|---|---|---|---|"]
        for bench, entry in summary["benchmarks"].items():
            if name not in entry["runs"]:
                continue
            rt = entry["runs"][name]["routing"]
            share = "n/a" if rt["to_judge_share"] is None else f"{100 * rt['to_judge_share']:.1f}%"
            lines.append(f"| {bench} | {rt['l1_allow']} | {rt['l1_gray_zone']} | {rt['l1_mitigate']} | {share} | "
                         f"{rt['l2_allow']} | {rt['l2_mitigate']} | {rt['l2_uncertain']} |")
        lines.append("")
    comparison_names = {c for entry in summary["benchmarks"].values() for c in entry["comparisons"]}
    for comp in sorted(comparison_names):
        lines += [f"## Paired difference `{comp}`", "",
                  "| Benchmark | " + " | ".join(label for _, label in DELTA_METRICS) + " |",
                  "|---|" + "---|" * len(DELTA_METRICS)]
        for bench, entry in summary["benchmarks"].items():
            if comp not in entry["comparisons"]:
                continue
            d = entry["comparisons"][comp]
            lines.append(f"| {bench} | " + " | ".join(_cell(d[m], mark=True) for m, _ in DELTA_METRICS) + " |")
        lines.append("")
    lines += [
        "Notes:",
        f"- Intervals: {level}% percentile bootstrap, {p['bootstrap']} resamples drawn with replacement within "
        f"each class (seed {p['seed']}). `*` marks a paired difference whose interval excludes 0.",
        "- Safe blocked: share of safe prompts not allowed (escalated prompts count as blocked). "
        "Harmful allowed: share of harmful prompts allowed (jailbreak success rate).",
        "- AUROC, AUPRC, FPR@95%TPR and ECE use the harm score; n/a means undefined for the benchmark "
        "(e.g. AUROC on a single-class set).",
        "",
    ]
    return "\n".join(lines)


def to_csv_rows(summary: dict) -> list:
    rows = []
    for bench, entry in summary["benchmarks"].items():
        for kind, group in (("run", entry["runs"]), ("comparison", entry["comparisons"])):
            for name, block in group.items():
                metrics = block["metrics"] if kind == "run" else block
                for m in METRICS:
                    e = metrics[m]
                    rows.append({"kind": kind, "name": name, "benchmark": bench, "metric": m,
                                 "value": e["value"], "ci_low": e["ci_low"], "ci_high": e["ci_high"],
                                 "undefined_resamples": e["undefined_resamples"]})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    cfg = load_config()
    fairness = cfg.get("evaluation", {}).get("fairness", {})
    b = args.bootstrap or int(fairness.get("bootstrap_samples", 1000))
    alpha = args.alpha if args.alpha is not None else float(fairness.get("alpha", 0.05))
    seed = args.seed if args.seed is not None else int(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    runs = [load_run(spec) for spec in args.run]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        logger.error(f"Run names must be unique: {names}")
        return 1
    comparisons = []
    for spec in args.compare:
        a, _, bname = spec.partition(":")
        if a not in names or bname not in names:
            logger.error(f"--compare {spec!r}: both names must be given with --run ({names})")
            return 1
        comparisons.append((a, bname))

    order = [bench["name"] for bench in cfg.get("evaluation", {}).get("benchmarks", [])]
    found = {bench for r in runs for bench in r["rows"]}
    benchmarks = [x for x in order if x in found] + sorted(found - set(order))

    summary = {
        "parameters": {"bootstrap": b, "alpha": alpha, "seed": seed, "stratified_by_label": True},
        "runs": {r["name"]: {"results": str(r["results_path"]), "predictions": str(r["predictions_dir"]),
                             "commit": next((v.get("run_info", {}).get("git_commit") for v in r["saved"].values()
                                             if isinstance(v, dict) and v.get("run_info")), None)}
                 for r in runs},
        "benchmarks": {},
    }
    for bench in benchmarks:
        logger.info(f"Bootstrapping {bench} ({b} resamples)...")
        summary["benchmarks"][bench] = summarise_benchmark(bench, runs, comparisons, b, alpha, seed)
    summary["run_info"] = finalize_run_info(run_info)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".md", ".csv"):
        warn_if_overwriting(out.with_suffix(suffix))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = to_markdown(summary)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "name", "benchmark", "metric", "value",
                                                "ci_low", "ci_high", "undefined_resamples"])
        writer.writeheader()
        writer.writerows(to_csv_rows(summary))

    print(markdown)
    mismatches = {f"{name}/{bench}": entry["runs"][name]["saved_metric_mismatches"]
                  for bench, entry in summary["benchmarks"].items()
                  for name in entry["runs"] if entry["runs"][name]["saved_metric_mismatches"]}
    if mismatches:
        logger.error(f"Predictions disagree with the saved results for: {mismatches}")
        return 1
    logger.info(f"Recomputed metrics match the saved results. Tables → {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
