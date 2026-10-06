"""
Extract Gray-Zone Samples from Layer-1 Output
==============================================
Runs Layer-1 batch inference on a dataset and saves records for Layer-2.

By default saves only decision=="gray_zone" samples.
Use --score-range to override and capture samples by calibrated score band
(useful when the test set is heavily skewed toward one class).

Layer 1 is built by Layer1Detector.from_artifacts_for(cfg), the shared loader,
so the scores written here come from the same model the evaluation and the red
team measure: which stacker (layer1.classifier_only), which directory
(layer1.artifact_subdir), and whether normalisation and the payload views are
on. All four are logged at load time.

Usage:
    source <your venv>/bin/activate
    python scripts/extract_gray_zone.py [options]

Flags:
    --input   PATH       : source JSONL (default: data/distillation_pool/test.jsonl)
    --output  PATH       : output JSONL (default: data/processed/layer1_gray_zone.jsonl)
    --dry-run            : process only first 100 samples
    --score-range LO HI  : capture samples with score_calib in [LO, HI] instead of
                           using the decision label (e.g. --score-range 0.4 0.95)
    --include-mitigate N : also grab N top-scoring mitigate samples to ensure
                           Layer-2 has something to judge when gray-zone is empty
    --layer1-artifacts SUBDIR : load Layer 1 from artifacts/SUBDIR instead of
                           layer1.artifact_subdir
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import torch
from pathlib import Path

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, save_jsonl, set_seed
from llmguard.utils.logger import get_logger
from llmguard.layer1 import Layer1Detector
from llmguard.layer1.detector import layer1_artifact_dir

logger = get_logger("extract_gray_zone")


def artifact_subdir(value: str) -> str:
    """
    layer1.artifact_subdir names a directory under project.artifact_dir, so
    "artifacts/layer1_v3", "artifacts/layer1_v3/" and "layer1_v3" all mean the
    same subdirectory.
    """
    return Path(value).name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract gray-zone (or score-range) samples using Layer-1")
    p.add_argument(
        "--input",
        default="data/distillation_pool/test.jsonl",
        help="Input JSONL file with text records",
    )
    p.add_argument(
        "--output",
        default="data/processed/layer1_gray_zone.jsonl",
        help="Output JSONL path for gray-zone records",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 100 samples",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N samples from the input file",
    )
    p.add_argument(
        "--score-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Capture samples with score_calib in [LOW, HIGH] (overrides decision filter)",
    )
    p.add_argument(
        "--include-mitigate",
        type=int,
        default=0,
        metavar="N",
        help="Also include up to N highest-scoring mitigate samples "
             "(use when gray zone is empty but you still want Layer-2 input)",
    )
    p.add_argument(
        "--layer1-artifacts",
        type=str,
        default=None,
        metavar="SUBDIR",
        help="Load Layer 1 from artifacts/SUBDIR instead of layer1.artifact_subdir "
             "(a retrained model kept apart from the shipped one)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()
    # None means "whatever the config says"; the flag overrides the subdirectory
    # before Layer 1 is built, the same way run_evaluation.py does it.
    if args.layer1_artifacts:
        cfg.setdefault("layer1", {})["artifact_subdir"] = artifact_subdir(args.layer1_artifacts)
    set_seed(cfg["project"]["seed"])

    logger.info("=" * 60)
    logger.info("Layer-1 — Gray-Zone / Score-Range Extraction")
    logger.info(f"  Input        : {args.input}")
    logger.info(f"  Output       : {args.output}")
    if args.score_range:
        logger.info(f"  Score range  : [{args.score_range[0]}, {args.score_range[1]}]")
    else:
        logger.info(f"  Mode         : decision == gray_zone")
    if args.include_mitigate:
        logger.info(f"  +Mitigate    : up to {args.include_mitigate} samples")
    logger.info("=" * 60)

    # ── Load records ──────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    records = load_jsonl(str(input_path))
    logger.info(f"Loaded {len(records)} records from {input_path}")

    if args.dry_run:
        records = records[:100]
        logger.info(f"[DRY RUN] Limited to {len(records)} samples.")
    elif args.limit:
        records = records[:args.limit]
        logger.info(f"[LIMIT] Truncated to {len(records)} samples.")

    # ── Load Layer-1 detector (from trained artifacts) ───────────────────────
    # from_artifacts_for is the shared loader: it follows layer1.classifier_only,
    # layer1.artifact_subdir, layer1.normalise_input and layer1.payload_views, so
    # the gray zone is cut from the same model the evaluation and the red team
    # measure. Hard-coding fusion_v2.joblib here scored the pool with the
    # three-signal stacker while the shipped system uses the one-feature one.
    logger.info("Loading Layer-1 detector from saved artifacts...")
    detector = Layer1Detector.from_artifacts_for(cfg)

    classifier_only = not getattr(detector.fusion, "needs_lm_features", True)
    logger.info(f"  Artifacts    : {layer1_artifact_dir(cfg)}")
    logger.info(
        "  Stacker      : "
        + ("classifier only (curvature and log-rank skipped)" if classifier_only
           else "three signals (curvature x log-rank x classifier)")
    )
    logger.info(f"  Normalisation: {'on' if detector.normalise_input else 'off'}")
    logger.info(f"  Payload views: {'on' if detector.payload_views else 'off'}")
    logger.info(
        f"  Calibration  : T={detector.scaler.T:.4f}  thresholds "
        f"{detector.router.tau_low}/{detector.router.tau_high}"
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    texts = [r["text"] for r in records]
    logger.info(f"Running Layer-1 on {len(texts)} texts...")
    results = detector.detect_batch(texts, log_every=500)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Collect results ───────────────────────────────────────────────────────
    allow_n = mitigate_n = gray_n = 0
    selected = []
    mitigate_pool = []  # for --include-mitigate fallback

    for rec, result in zip(records, results):
        out = rec.copy()
        out["l1_decision"]    = result.decision
        out["l1_score_raw"]   = round(result.raw_score, 4)
        out["l1_score_calib"] = round(result.calibrated_score, 4)

        if result.decision == "allow":
            allow_n += 1
        elif result.decision == "mitigate":
            mitigate_n += 1
            mitigate_pool.append(out)
        else:
            gray_n += 1

        # Selection logic
        if args.score_range:
            lo, hi = args.score_range
            if lo <= result.calibrated_score <= hi:
                selected.append(out)
        elif result.decision == "gray_zone":
            selected.append(out)

    logger.info(f"  Allow    : {allow_n}")
    logger.info(f"  Gray zone: {gray_n}")
    logger.info(f"  Mitigate : {mitigate_n}")

    # ── Fallback: include top-N mitigate if gray zone is empty ───────────────
    if args.include_mitigate > 0 and len(selected) == 0:
        logger.info(
            f"  No gray-zone samples found. Including top {args.include_mitigate} "
            f"mitigate samples by score as Layer-2 input."
        )
        mitigate_pool.sort(key=lambda x: x["l1_score_raw"])  # lowest first = most borderline
        selected = mitigate_pool[: args.include_mitigate]
    elif args.include_mitigate > 0:
        # Append N borderline mitigate samples to augment gray zone
        mitigate_pool.sort(key=lambda x: x["l1_score_raw"])
        extra = mitigate_pool[: args.include_mitigate]
        selected.extend(extra)
        logger.info(f"  +{len(extra)} borderline mitigate samples added.")

    logger.info(f"  Selected for Layer-2: {len(selected)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(selected, args.output)
    logger.info(f"Saved {len(selected)} records → {Path(args.output).resolve()}")
    logger.info("=" * 60)
    logger.info("Next step: python scripts/run_layer2.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
