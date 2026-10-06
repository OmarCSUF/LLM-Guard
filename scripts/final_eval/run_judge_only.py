"""
Judge-only arm
==============
The Layer-2 judge alone on every prompt of a labelled JSONL, with no Layer 1 in
front of it: one forward pass per prompt reading P(mitigate) from the token
distribution after "DECISION:", the 0.40/0.60 band from the configuration
(uncertain counts as blocked, as the pipeline treats escalation), and
P(mitigate) as the harm score. This is the "judge" variant of
scripts/run_ablation.py, run directly on a file so that Layer 1 is never loaded.

Writes the same two products as scripts/run_evaluation.py --output, so that
scripts/summarise_results.py and analyse_final.py read them unchanged:
    <output>                          {benchmark: {metrics, layer_stats, run_info, ...}}
    <output stem>_predictions/<benchmark>.jsonl    one row per prompt

Run from the repository root with the guard environment active:
    python scripts/final_eval/run_judge_only.py \
        --input data/benchmarks_final/final_test/test.jsonl \
        --output data/processed/final_eval/final_judge.json
"""

import sys
import os

REPO = os.environ.get("LLMGUARD_REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))  # local checkout without `pip install -e .`

import argparse
import json
import time
from pathlib import Path

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation.metrics import compute_all

logger = get_logger("run_judge_only")

SCORE_DIGITS = 4     # layer2/consistency.py rounds P(mitigate) to 4 decimals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 judge alone on every prompt of a file")
    p.add_argument("--input", required=True, help="Labelled JSONL: {id, text, label}")
    p.add_argument("--output", required=True, help="Results JSON; predictions go beside it")
    p.add_argument("--benchmark", default=None, help="Name recorded for the file (default: its stem)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prompt-version", default=None,
                   help="Override layer2.judge.prompt_version. Absent, nothing changes.")
    p.add_argument("--judge-model", default=None,
                   help="Override layer2.judge.model_name. Used to score an identical "
                        "protocol with a different judge; absent, nothing changes.")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def load_records(path: Path, limit=None) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def decide(p: float, allow_below: float, mitigate_above: float) -> tuple:
    """(decision, final_pred, escalated) with the pipeline's own band."""
    if p > mitigate_above:
        return "mitigate", 0, False
    if p < allow_below:
        return "allow", 1, False
    return "uncertain", 0, True


def main() -> int:
    args = parse_args()
    cfg = load_config()
    if args.judge_model:
        cfg["layer2"]["judge"]["model_name"] = args.judge_model
        logger.info(f"judge model overridden: {args.judge_model}")
    if args.prompt_version:
        cfg["layer2"]["judge"]["prompt_version"] = args.prompt_version
        logger.info(f"judge prompt version overridden: {args.prompt_version}")
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    consistency = cfg["layer2"]["consistency"]
    if consistency.get("mode") != "logprob":
        raise SystemExit("run_judge_only needs layer2.consistency.mode: logprob")
    allow_below = float(consistency.get("allow_below", 0.4))
    mitigate_above = float(consistency.get("mitigate_above", 0.6))

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    name = args.benchmark or in_path.stem
    records = load_records(in_path, args.limit)
    logger.info(f"{len(records)} prompts from {in_path} as benchmark '{name}'")

    from llmguard.layer2 import Layer2Judge
    logger.info("Loading Layer 2 (no Layer 1)...")
    judge = Layer2Judge.from_config(cfg)
    llm = judge.checker.judge
    if getattr(llm, "backend", None) != "huggingface":
        raise SystemExit("p_mitigate needs the huggingface backend (Ollama exposes no logits)")

    out_path = Path(args.output)
    pred_dir = out_path.with_name(out_path.stem + "_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{name}.jsonl"
    warn_if_overwriting(pred_path)

    y_true, y_score, y_pred, y_esc = [], [], [], []
    stats = {"l2_allow": 0, "l2_mitigate": 0, "l2_uncertain": 0, "errors": 0}
    errors, times = [], []
    t_start = time.perf_counter()
    with open(pred_path, "w", encoding="utf-8", newline="\n") as fh:
        for i, rec in enumerate(records, start=1):
            text = rec.get("text", "")
            label = int(rec.get("label", 1))
            row = {"id": rec.get("id", f"{name}_{i - 1}"), "text": text, "label": label,
                   "l1_decision": None, "l1_score": None,
                   "l2_decision": None, "l2_confidence": None, "l2_severity": None,
                   "l2_category": None, "l2_votes": None, "l2_p_mitigate": None,
                   "l3_mode": None, "final_pred": 1 - label, "harm_score": float(label),
                   "escalated": False, "t_ms": None, "error": None}
            try:
                t0 = time.perf_counter()
                p = llm.p_mitigate(text)
                dt = (time.perf_counter() - t0) * 1000.0
                if p is None:
                    raise RuntimeError("p_mitigate returned None")
                p = round(float(p), SCORE_DIGITS)
                decision, final, esc = decide(p, allow_below, mitigate_above)
                row.update({"l2_decision": decision, "l2_confidence": round(max(p, 1 - p), 4),
                            "l2_severity": p, "l2_p_mitigate": p,
                            "final_pred": final, "harm_score": p, "escalated": esc,
                            "t_ms": round(dt, 2)})
                stats[f"l2_{decision}"] += 1
                times.append(dt)
            except Exception as exc:                                   # noqa: BLE001
                # A crash is counted as a misclassification, as run_evaluation.py does.
                logger.warning(f"{name}/{row['id']}: {exc}")
                row["error"] = str(exc)
                errors.append(str(exc))
                stats["errors"] += 1
            y_true.append(label)
            y_score.append(row["harm_score"])
            y_pred.append(row["final_pred"])
            y_esc.append(row["escalated"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % args.log_every == 0 or i == len(records):
                spent = time.perf_counter() - t_start
                logger.info(f"  {i}/{len(records)} | {spent / i:.2f}s/prompt")

    elapsed = time.perf_counter() - t_start
    metrics = compute_all(y_true, y_score, y_pred, escalated=y_esc) if y_true else {}
    times_sorted = sorted(times)
    result = {
        "benchmark": name,
        "n_samples": len(y_true),
        "elapsed_s": round(elapsed, 2),
        "metrics": metrics,
        "layer_stats": stats,
        "errors": errors,
        "arm": "judge_only",
        "decision_band": {"allow_below": allow_below, "mitigate_above": mitigate_above},
        "latency_ms": {"median": round(times_sorted[len(times_sorted) // 2], 2) if times_sorted else None,
                       "mean": round(sum(times) / len(times), 2) if times else None,
                       "n": len(times)},
        "run_info": finalize_run_info(run_info),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(out_path)
    out_path.write_text(json.dumps({name: result}, indent=2), encoding="utf-8")
    logger.info(f"Results -> {out_path} | predictions -> {pred_path}")
    logger.info("  " + json.dumps({k: metrics.get(k) for k in ("auroc", "safe_blocked", "harmful_allowed",
                                                                  "accuracy", "escalation_rate")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
