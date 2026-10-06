"""
GaurdLLM — Full Pipeline Evaluation Script
============================================
Evaluates the GaurdLLM pipeline (Layer 1 + Layer 2 + Layer 3) on one or
more benchmark JSONL files and prints standardised metrics.

Input JSONL format:
    {"text": "...", "label": 0}   # 0 = harmful (mitigate), 1 = safe (allow)

Metrics reported per benchmark:
    accuracy, AUROC, AUPRC, FPR@95TPR, ECE, F1,
    jailbreak_success_rate, escalation_rate
A metric that is undefined for a benchmark (e.g. AUROC on harmful-only
HarmBench) prints as n/a and is saved as null.
Per-prompt predictions (id, text, label, each layer's decision, the final
prediction, any error) go to <output stem>_predictions/<benchmark>.jsonl when
--output is given, or to --predictions-dir.
scripts/summarise_results.py turns those files into tables with bootstrap
confidence intervals.  scripts/run_ablation.py writes the same two formats
for the classifier-only, no-curvature and judge-only variants.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --benchmarks harmbench xstest
    python scripts/run_evaluation.py --input data/benchmarks/harmbench/test.jsonl
    python scripts/run_evaluation.py --limit 200 --no-l2  # L1-only quick eval
    python scripts/run_evaluation.py --output results/eval_results.jsonl
    python scripts/run_evaluation.py --benchmark-root data/benchmarks_parity/arabic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.data_utils import set_seed
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation import PipelineEvaluator
from llmguard.evaluation.metrics import format_metric

logger = get_logger("run_evaluation")

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GaurdLLM Pipeline Evaluation")
    p.add_argument("--input",      type=str, default=None,
                   help="Path to a single JSONL benchmark file")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Named benchmarks from config (e.g. harmbench xstest). A benchmark "
                        "the config disables still runs when named here")
    p.add_argument("--limit",      type=int, default=None,
                   help="Evaluate only the first N samples per benchmark")
    p.add_argument("--no-l2",      action="store_true",
                   help="Skip Layer-2 (L1-only evaluation, fastest)")
    p.add_argument("--no-l3",      action="store_true",
                   help="Skip Layer-3 mitigator")
    p.add_argument("--classifier-only", action="store_true",
                   help="Force Layer 1 onto the distilled classifier alone, whatever "
                        "the config says: loads the one-feature stacker and skips "
                        "curvature and log-rank")
    p.add_argument("--full-stacker", action="store_true",
                   help="Force the three-signal Layer 1, for reproducing the older "
                        "numbers once classifier_only is the configured default")
    p.add_argument("--payload-views", dest="payload_views", action="store_true", default=None,
                   help="Score the text behind a short leading instruction frame on its "
                        "own and keep the higher score (layer1.payload_views: true)")
    p.add_argument("--no-payload-views", dest="payload_views", action="store_false",
                   help="Force the leading-clause view off, whatever the config says")
    p.add_argument("--normalise-input", dest="normalise_input", action="store_true", default=None,
                   help="Score decoded base64 and de-leeted recoveries beside the raw text "
                        "(layer1.normalise_input: true)")
    p.add_argument("--no-normalise-input", dest="normalise_input", action="store_false",
                   help="Force input normalisation off, whatever the config says. Added for "
                        "the post hoc component ablation of 2026-09-30; absent, nothing changes")
    p.add_argument("--layer1-artifacts", type=str, default=None, metavar="SUBDIR",
                   help="Load Layer 1 from artifacts/SUBDIR instead of layer1.artifact_subdir "
                        "(a retrained model kept apart from the shipped one)")
    p.add_argument("--guard-llm",  action="store_true",
                   help="Load GuardLLM so Layer 3 serves real rewrites instead of falling "
                        "back to refusals (shares the judge's model)")
    p.add_argument("--benchmark-root", type=str, default=None,
                   help="Read each named benchmark from ROOT/<name>/test.jsonl instead of "
                        "its configured path, keeping the benchmark names so the runs can "
                        "be paired (used for the RQ4 parity sets)")
    p.add_argument("--output",     type=str, default=None,
                   help="Write JSON results to this path")
    p.add_argument("--predictions-dir", type=str, default=None,
                   help="Write per-prompt predictions to DIR/<benchmark>.jsonl "
                        "(default with --output: <output stem>_predictions/)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(name: str, m: dict, ls: dict) -> None:
    print()
    print(c(f"  Benchmark: {name}", CYAN, BOLD))
    print(c("  +------------------------+--------+", DIM))
    print(c("  |  Metric                |  Value |", DIM))
    print(c("  +------------------------+--------+", DIM))

    rows = [
        ("Accuracy",          m.get("accuracy",               "?")),
        ("AUROC",             m.get("auroc",                  "?")),
        ("AUPRC",             m.get("auprc",                  "?")),
        ("FPR @ 95% TPR",     m.get("fpr_at_95tpr",          "?")),
        ("ECE",               m.get("ece",                    "?")),
        ("F1",                m.get("f1",                     "?")),
        ("Jailbreak Success", m.get("jailbreak_success_rate", "?")),
        ("Escalation Rate",   m.get("escalation_rate",        "?")),
        ("Precision",         m.get("precision",              "?")),
        ("Recall",            m.get("recall",                 "?")),
    ]
    for label, val in rows:
        if val is None:
            val_str, colour = "n/a", GRAY
        elif isinstance(val, float):
            val_str = f"{val:.4f}"
            colour  = GREEN if label in ("Accuracy", "AUROC", "AUPRC", "F1", "Recall", "Precision") and val >= 0.8 else WHITE
            if label in ("FPR @ 95% TPR", "ECE", "Jailbreak Success", "Escalation Rate") and val <= 0.1:
                colour = GREEN
        else:
            val_str, colour = str(val), WHITE
        print(f"  |  {label:<22}|  {c(val_str, colour, BOLD):<14}|")

    print(c("  +------------------------+--------+", DIM))

    # Routing breakdown
    print(c("  Layer routing:", DIM))
    print(f"    L1: allow={ls.get('l1_allow',0)}  gray={ls.get('l1_gray_zone',0)}  mitigate={ls.get('l1_mitigate',0)}")
    if ls.get("l2_allow", 0) + ls.get("l2_mitigate", 0) + ls.get("l2_uncertain", 0) > 0:
        print(f"    L2: allow={ls.get('l2_allow',0)}  mitigate={ls.get('l2_mitigate',0)}  uncertain={ls.get('l2_uncertain',0)}")
    if ls.get("l3_refuse", 0) + ls.get("l3_redact", 0) + ls.get("l3_safe_rewrite", 0) > 0:
        print(f"    L3: refuse={ls.get('l3_refuse',0)}  redact={ls.get('l3_redact',0)}  safe_rewrite={ls.get('l3_safe_rewrite',0)}")
    print()


def benchmark_file(bench: dict, root: Optional[str]) -> Path:
    """The test file for a benchmark: under --benchmark-root if given, else its own path."""
    if root:
        return Path(root) / bench["name"] / "test.jsonl"
    return Path(bench["path"]) / "test.jsonl"


def predictions_dir(args) -> Optional[Path]:
    """--predictions-dir, else <output stem>_predictions next to --output, else None."""
    if args.predictions_dir:
        return Path(args.predictions_dir)
    if args.output:
        out = Path(args.output)
        return out.with_name(out.stem + "_predictions")
    return None


def write_predictions(result, directory: Path) -> Path:
    """One JSON line per prompt, in benchmark order."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.benchmark}.jsonl"
    warn_if_overwriting(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in result.predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def print_summary(all_results: dict) -> None:
    """One row per benchmark; undefined metrics print as n/a."""
    columns = ["accuracy", "auroc", "auprc", "fpr_at_95tpr", "f1", "jailbreak_success_rate"]
    print(c("  ── Cross-Benchmark Summary ─────────────────────────", DIM))
    print(f"  {'Benchmark':<20} " + "  ".join(f"{name[:8]:<8}" for name in columns))
    print("  " + "-" * 79)
    for bname, res in all_results.items():
        m = res["metrics"]
        vals = "  ".join(f"{format_metric(m.get(name)):<8}" for name in columns)
        print(f"  {bname:<20} {vals}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config()
    # None means "whatever the config says"; a flag overrides it either way,
    # before run_info snapshots the config.
    if args.payload_views is not None:
        cfg.setdefault("layer1", {})["payload_views"] = args.payload_views
    if args.normalise_input is not None:
        cfg.setdefault("layer1", {})["normalise_input"] = args.normalise_input
    if args.layer1_artifacts:
        cfg.setdefault("layer1", {})["artifact_subdir"] = args.layer1_artifacts
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    load_l2 = not args.no_l2
    load_l3 = not args.no_l3 and not args.no_l2
    if args.guard_llm and not load_l3:
        raise SystemExit("--guard-llm needs Layer 3; drop --no-l3 (and --no-l2)")
    if args.classifier_only and args.full_stacker:
        raise SystemExit("--classifier-only and --full-stacker cannot both be set")
    # None means "whatever the config says"; a flag overrides it either way.
    classifier_only = True if args.classifier_only else (False if args.full_stacker else None)
    pred_dir = predictions_dir(args)

    print()
    print(c("  Loading pipeline...", CYAN, BOLD))
    evaluator = PipelineEvaluator.from_config(
        cfg,
        load_l2=load_l2,
        load_l3=load_l3,
        load_guard_llm=args.guard_llm,
        classifier_only=classifier_only,
    )
    print(c("  Pipeline ready.\n", GREEN))

    all_results = {}

    # Single file mode
    if args.input:
        result = evaluator.evaluate_file(
            args.input, limit=args.limit
        )
        print_metrics_table(result.benchmark, result.metrics, result.layer_stats)
        all_results[result.benchmark] = result.to_dict()
        if pred_dir:
            write_predictions(result, pred_dir)

    # Named benchmarks from config. Naming one runs it even when the config
    # disables it: the RQ4 learner arms are disabled so the default run does
    # not include them, and are run by name with --benchmark-root.
    elif args.benchmarks:
        bench_configs = {
            b["name"]: b
            for b in cfg.get("evaluation", {}).get("benchmarks", [])
        }
        for name in args.benchmarks:
            if name not in bench_configs:
                print(c(f"  !! Benchmark '{name}' not found in config.", YELLOW))
                continue
            test_file = benchmark_file(bench_configs[name], args.benchmark_root)
            if not test_file.exists():
                print(c(f"  !! File not found: {test_file}", RED))
                continue
            result = evaluator.evaluate_file(
                str(test_file), benchmark=name, limit=args.limit
            )
            print_metrics_table(result.benchmark, result.metrics, result.layer_stats)
            all_results[name] = result.to_dict()
            if pred_dir:
                write_predictions(result, pred_dir)

    # All enabled benchmarks from config
    else:
        bench_list = cfg.get("evaluation", {}).get("benchmarks", [])
        enabled = [b for b in bench_list if b.get("enabled", True)]
        if not enabled:
            print(c("  No benchmarks enabled in config. Use --input for a custom file.", YELLOW))
            return

        for bench in enabled:
            name      = bench["name"]
            test_file = benchmark_file(bench, args.benchmark_root)
            if not test_file.exists():
                print(c(f"  Skipping {name}: {test_file} not found.", YELLOW))
                continue
            result = evaluator.evaluate_file(
                str(test_file), benchmark=name, limit=args.limit
            )
            print_metrics_table(result.benchmark, result.metrics, result.layer_stats)
            all_results[name] = result.to_dict()
            if pred_dir:
                write_predictions(result, pred_dir)

    if not all_results:
        print(c("  No benchmarks evaluated. Use --input <file.jsonl>", YELLOW))
        return

    # Multi-benchmark summary
    if len(all_results) > 1:
        print_summary(all_results)

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        warn_if_overwriting(out_path)
        # Stored per benchmark so the file stays a {benchmark: result} mapping.
        finished = finalize_run_info(run_info)
        for res in all_results.values():
            res["run_info"] = finished
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(c(f"  Results saved to: {args.output}", GREEN))
    if pred_dir:
        print(c(f"  Predictions saved to: {pred_dir}/", GREEN))


if __name__ == "__main__":
    main()
