"""
Phase 3 — Run Layer-2 LLM-as-Judge on Gray-Zone Samples
=========================================================
Steps:
  1. Load gray-zone records (output of Layer-1 with decision="gray_zone")
  2. Initialise Layer2Judge (loads Qwen2.5-3B-Instruct with 4-bit quantisation)
  3. Run judge_batch() over all gray-zone texts
  4. Save JudgeResult records to the output JSONL file
  5. Print summary: allow / mitigate / escalated counts, mean confidence

Prerequisites:
  - Layer-1 training complete (scripts/train_layer1.py)
  - Gray-zone records available (produced by a Layer-1 batch inference pass)
  - pip install bitsandbytes>=0.43.0 accelerate>=0.27.0 requests

Usage (activate your conda env first):
    source <your venv>/bin/activate
    python scripts/run_layer2.py [--input PATH] [--output PATH] [--dry-run]

Flags:
    --input   PATH : gray-zone JSONL (default: data/processed/layer1_gray_zone.jsonl)
    --output  PATH : results JSONL   (default: data/processed/layer2_results.jsonl)
    --dry-run      : process only first 10 samples (smoke-test)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
import time
import torch
from collections import Counter
from pathlib import Path

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, save_jsonl, set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.layer2 import Layer2Judge

logger = get_logger("run_layer2")


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers  (same pattern as train_layer1.py)
# ─────────────────────────────────────────────────────────────────────────────

def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
        logger.info(
            f"[GPU mem{' ' + tag if tag else ''}] "
            f"allocated={allocated:.2f}GB  reserved={reserved:.2f}GB"
        )


def clear_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────

def config_snapshot(cfg: dict) -> dict:
    """
    The Layer-2 settings a stored result cannot be read without.

    Beyond the model and K: consistency.mode decides whether the decision comes
    from K sampled votes or from P(mitigate) in one forward pass, and
    judge.prompt_version selects between two materially different system
    prompts, so the same K and c_min mean different things under each. The
    logprob bands are recorded only in that mode, where they are what decides
    allow / mitigate / escalate. Defaults match layer2/judge.py and
    layer2/consistency.py, so a config that omits a key records what the code
    will actually use.
    """
    judge_cfg = cfg["layer2"]["judge"]
    cons_cfg  = cfg["layer2"]["consistency"]
    mode = cons_cfg.get("mode", "vote")
    snapshot = {
        "model":          judge_cfg["model_name"],
        "backend":        judge_cfg["backend"],
        "prompt_version": judge_cfg.get("prompt_version", "v1"),
        "K":              cons_cfg["K"],
        "c_min":          cons_cfg["c_min"],
        "mode":           mode,
        "load_in_4bit":   judge_cfg.get("load_in_4bit", False),
    }
    if mode == "logprob":
        snapshot["allow_below"]    = cons_cfg.get("allow_below", 0.4)
        snapshot["mitigate_above"] = cons_cfg.get("mitigate_above", 0.6)
    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Layer-2 LLM Judge on gray-zone samples")
    p.add_argument(
        "--input",
        default="data/processed/layer1_gray_zone.jsonl",
        help="Path to gray-zone JSONL produced by Layer-1 batch inference",
    )
    p.add_argument(
        "--output",
        default="data/processed/layer2_results.jsonl",
        help="Path to write Layer-2 JudgeResult records",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 10 samples (for smoke-testing)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config()
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    logger.info("=" * 60)
    logger.info("LLM-Guard — Layer-2 Judge Run")
    snapshot = config_snapshot(cfg)
    logger.info(f"  Model  : {snapshot['model']}")
    logger.info(f"  Backend: {snapshot['backend']}")
    logger.info(f"  Prompt : {snapshot['prompt_version']}")
    logger.info(f"  Mode   : {snapshot['mode']}"
                + (f" (allow < {snapshot['allow_below']}, "
                   f"mitigate > {snapshot['mitigate_above']})"
                   if snapshot["mode"] == "logprob" else ""))
    logger.info(f"  K      : {snapshot['K']}")
    logger.info(f"  Input  : {args.input}")
    logger.info(f"  Output : {args.output}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # ── 1. Load gray-zone records ─────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(
            f"Input file not found: {input_path}\n"
            "Run Layer-1 batch inference first to produce gray-zone records.\n"
            "Example: use Layer1Detector.detect_batch() and save records "
            "where result.decision == 'gray_zone' to this path."
        )
        sys.exit(1)

    records = load_jsonl(str(input_path))
    logger.info(f"Loaded {len(records)} gray-zone records from {input_path}")

    if args.dry_run:
        records = records[:10]
        logger.info(f"[DRY RUN] Limited to {len(records)} samples.")

    if not records:
        logger.warning("No records to process. Exiting.")
        return

    # ── 2. Check bitsandbytes is available (HuggingFace backend) ─────────────
    if cfg["layer2"]["judge"]["backend"] == "huggingface":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            logger.error(
                "bitsandbytes is required for the HuggingFace backend.\n"
                "Install with:  pip install bitsandbytes>=0.43.0 accelerate>=0.27.0"
            )
            sys.exit(1)

    # ── 3. Initialise Layer2Judge ─────────────────────────────────────────────
    logger.info("Initialising Layer2Judge (loading model — may take a minute)...")
    judge = Layer2Judge.from_config(cfg)
    log_gpu_memory("after judge model load")

    # ── 4. Run judge_batch ────────────────────────────────────────────────────
    texts = [r["text"] for r in records]
    logger.info(f"Running judge on {len(texts)} texts (K={cfg['layer2']['consistency']['K']} per text)...")
    t_start = time.perf_counter()
    results = judge.judge_batch(texts, log_every=50)
    elapsed_s = time.perf_counter() - t_start
    log_gpu_memory("after judge_batch")

    # ── 5. Save per-sample JSONL ──────────────────────────────────────────────
    output_records = []
    for rec, result in zip(records, results):
        out = rec.copy()
        out.update(result.to_dict())
        output_records.append(out)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_records, args.output)
    logger.info(f"Results saved → {Path(args.output).resolve()}")

    # Flush any remaining buffered distillation records
    flushed = judge.buffer.flush()
    if flushed > 0:
        logger.info(f"Final distillation flush: {flushed} records written.")

    clear_gpu()

    # ── 6. Compute summary metrics ────────────────────────────────────────────
    allow_n     = sum(1 for r in results if r.is_safe())
    mitigate_n  = sum(1 for r in results if r.is_harmful())
    escalated_n = sum(1 for r in results if r.needs_escalation())
    mean_conf   = sum(r.confidence for r in results) / max(len(results), 1)
    mean_sev    = sum(r.severity   for r in results) / max(len(results), 1)
    cat_counts  = Counter(r.category for r in results)

    logger.info("=" * 60)
    logger.info("Layer-2 Run Complete")
    logger.info(f"  Total processed : {len(results)}")
    logger.info(f"  Allow           : {allow_n}  ({100*allow_n/max(len(results),1):.1f}%)")
    logger.info(f"  Mitigate        : {mitigate_n}  ({100*mitigate_n/max(len(results),1):.1f}%)")
    logger.info(f"  Escalated       : {escalated_n}  ({100*escalated_n/max(len(results),1):.1f}%)")
    logger.info(f"  Mean confidence : {mean_conf:.4f}")
    logger.info(f"  Mean severity   : {mean_sev:.4f}")
    logger.info(f"  Elapsed         : {elapsed_s:.1f}s")
    logger.info(f"  Output          → {Path(args.output).resolve()}")
    logger.info("=" * 60)

    # ── 7. Save results.json (structured summary for reports) ─────────────────
    artifact_dir = Path("artifacts/layer2")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_samples":    len(results),
        "allow":            allow_n,
        "mitigate":         mitigate_n,
        "escalated":        escalated_n,
        "allow_pct":        round(100 * allow_n    / max(len(results), 1), 2),
        "mitigate_pct":     round(100 * mitigate_n / max(len(results), 1), 2),
        "escalated_pct":    round(100 * escalated_n/ max(len(results), 1), 2),
        "mean_confidence":  round(mean_conf, 4),
        "mean_severity":    round(mean_sev,  4),
        "elapsed_s":        round(elapsed_s, 2),
        "top_categories":   dict(cat_counts.most_common(10)),
        "config_snapshot": snapshot,
        "input_file":  str(Path(args.input).resolve()),
        "output_file": str(Path(args.output).resolve()),
        "run_info":    finalize_run_info(run_info),
    }

    results_path = artifact_dir / "results.json"
    warn_if_overwriting(results_path)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved  → {results_path.resolve()}")


if __name__ == "__main__":
    main()
