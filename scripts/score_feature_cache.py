"""
Backfill the trained classifier score into the Layer-1 feature cache
=====================================================================
`scripts/train_layer1.py` caches curvature and log-rank to
`data/processed/layer1_features/<split>_features.jsonl`, but writes `g = 0.5`
into every row, because that cache is built at step 3, before the classifier
exists. Step 7 recomputes the real `g` in memory and never writes it back. So
every row of every split carries the placeholder:

    train 15,296 rows, val 1,912, test 1,912 — one distinct g value, 0.5

That is correct for the stage the cache serves and wrong for everything
downstream. Anything that reads `g` from it and fits or plots a classifier-only
stacker gets a model with one input value, which scores every prompt identically:
AUROC 0.5000 and the whole split routed to the gray zone. Those are properties of
the cache, not of the classifier.

This script fills the gap without retraining. It loads the classifier checkpoint
the configuration selects, scores the cached rows in batches, and writes
`<split>_scored.jsonl` beside the original. The original is never modified,
because step 4 of training is supposed to see the placeholder; a cache with real
scores in it would silently change what the first stacker is fitted on.

`g` is P(allow), the same quantity `ClassifierFeature.compute_batch` returns and
`update_classifier_scores` assigns, so a scored row is what training holds in
memory at step 7.

A `<split>_scored.fingerprint` records the classifier checkpoint and the source
cache it was built from, so a consumer can tell a stale scored file from a
current one.

Usage:
    python scripts/score_feature_cache.py                      # all three splits
    python scripts/score_feature_cache.py --splits test val
    python scripts/score_feature_cache.py --layer1-artifacts layer1_v3
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
import time
from pathlib import Path

from llmguard.layer1.detector import layer1_artifact_dir
from llmguard.layer1.features import ClassifierFeature, resolve_device
from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, save_jsonl, set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("score_feature_cache")

SPLITS = ("train", "val", "test")
PLACEHOLDER = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_fingerprint(checkpoint: Path) -> dict:
    """Name, size and mtime of each checkpoint file, so staleness is visible."""
    if not checkpoint.is_dir():
        return {}
    return {p.name: [p.stat().st_size, int(p.stat().st_mtime)]
            for p in sorted(checkpoint.iterdir()) if p.is_file()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", default="data/processed/layer1_features",
                   help="Directory holding <split>_features.jsonl")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                   help="Which splits to score (default: all three)")
    p.add_argument("--layer1-artifacts", default=None,
                   help="artifacts/<subdir> holding the classifier to score with; "
                        "defaults to layer1.artifact_subdir from config")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size (default 64, as in train_layer1.py)")
    p.add_argument("--force", action="store_true",
                   help="Rescore even when a current scored file already exists")
    return p.parse_args()


def already_current(scored: Path, fingerprint: dict) -> bool:
    """True when a scored file exists and its fingerprint still matches."""
    stamp = scored.with_name(scored.stem + ".fingerprint")
    if not (scored.exists() and stamp.exists()):
        return False
    try:
        return json.loads(stamp.read_text(encoding="utf-8")) == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    if args.layer1_artifacts:
        cfg.setdefault("layer1", {})["artifact_subdir"] = Path(args.layer1_artifacts).name

    artifact_dir = layer1_artifact_dir(cfg)
    checkpoint = artifact_dir / "classifier" / "checkpoint_best"
    if not checkpoint.is_dir():
        logger.error(f"No classifier checkpoint at {checkpoint}. Train Layer 1 first.")
        return 1

    features_dir = Path(args.features_dir)
    clf_cfg = cfg["layer1"]["classifier"]
    device = resolve_device(cfg["project"]["device"])

    logger.info(f"Classifier : {checkpoint}")
    logger.info(f"Device     : {device}")
    logger.info(f"Cache      : {features_dir}")

    clf = ClassifierFeature(
        checkpoint_path=str(checkpoint),
        backbone=clf_cfg["backbone"],
        backbone_revision=clf_cfg.get("backbone_revision"),
        max_length=clf_cfg["max_length"],
        device=device,
        fp16=cfg["project"].get("fp16", True),
    )

    ckpt_stamp = checkpoint_fingerprint(checkpoint)
    written = []
    for split in args.splits:
        source = features_dir / f"{split}_features.jsonl"
        if not source.exists():
            logger.warning(f"{source} not found — skipping {split}")
            continue

        scored_path = features_dir / f"{split}_scored.jsonl"
        fingerprint = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "classifier_checkpoint": str(checkpoint),
            "classifier_files": ckpt_stamp,
            "backbone": clf_cfg["backbone"],
            "max_length": clf_cfg["max_length"],
        }
        if already_current(scored_path, fingerprint) and not args.force:
            logger.info(f"{split}: {scored_path} is current — skipping (use --force to rescore)")
            continue

        records = load_jsonl(str(source))
        placeholders = sum(1 for r in records if r.get("g") == PLACEHOLDER)
        logger.info(f"{split}: {len(records)} rows, {placeholders} carrying the "
                    f"g={PLACEHOLDER} placeholder — scoring...")

        t0 = time.perf_counter()
        texts = [r.get("text", "") for r in records]
        scores = clf.compute_batch(texts, batch_size=args.batch_size)
        elapsed = time.perf_counter() - t0

        updated = []
        for rec, g in zip(records, scores):
            rec = dict(rec)
            rec["g_placeholder"] = rec.get("g")
            rec["g"] = float(g)
            updated.append(rec)

        distinct = len({round(r["g"], 6) for r in updated})
        safe = [r["g"] for r in updated if r.get("label") == 1]
        harmful = [r["g"] for r in updated if r.get("label") == 0]
        warn_if_overwriting(scored_path)
        save_jsonl(updated, str(scored_path))
        scored_path.with_name(scored_path.stem + ".fingerprint").write_text(
            json.dumps(fingerprint, indent=2), encoding="utf-8")

        logger.info(
            f"{split}: wrote {scored_path} in {elapsed:.1f}s — {distinct} distinct g values"
            + (f"; mean g (P(allow)) safe {sum(safe)/len(safe):.4f} "
               f"vs harmful {sum(harmful)/len(harmful):.4f}"
               if safe and harmful else "")
        )
        written.append({"split": split, "path": str(scored_path), "n": len(updated),
                        "distinct_g": distinct, "elapsed_s": round(elapsed, 2)})

    if written:
        manifest = features_dir / "scored_manifest.json"
        manifest.write_text(json.dumps(
            {"written": written, "classifier_checkpoint": str(checkpoint),
             "run_info": finalize_run_info(run_info)}, indent=2), encoding="utf-8")
        logger.info(f"Manifest → {manifest}")

    logger.info(
        "Point a consumer at a scored file with --features-path, for example:\n"
        f"  python scripts/plot_layer1.py --features-path {features_dir}/test_scored.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
