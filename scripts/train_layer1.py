"""
Phase 2 — Train Layer-1 Classifier
=====================================
Steps:
  1. Load the distillation pool (train / val / test splits)
  2. Compute curvature + log-rank features (cached to avoid recomputation)
  3. Fit LogisticFusion stacker on train features (placeholder g=0.5)
  4. Fine-tune RoBERTa+LoRA classifier on train records (GPU + AMP)
  5. Re-compute g(x) with trained classifier; re-fit fusion
  6. Calibrate TemperatureScaler on validation set
  7. Fit the classifier-only stacker and calibrate it (what the shipped
     configuration, layer1.classifier_only, loads)
  8. Evaluate BOTH stackers on the test set: AUROC, AUPRC, FPR@95%TPR, accuracy
  9. Save all artifacts to --artifact-dir (default artifacts/layer1/)

results.json records both stackers, because the two artifact pairs this script
writes are not interchangeable:

    test_metrics                   fusion_v2 + scaler.json (three signals), by
                                   Layer1Detector.evaluate — unchanged meaning
    test_metrics_clf_only          fusion_clf_only + scaler_clf_only.json
    test_metrics_fusion_v2_cached  fusion_v2 scored exactly the way the
                                   classifier-only row is, so the two stackers
                                   are comparable on identical inputs
    stacker_selected_by_config     which pair layer1.classifier_only makes the
                                   shipped model (from_artifacts_for)

test_metrics keeps its historical key and its historical meaning, and now
carries an explicit "stacker" field so it cannot be mistaken for the shipped
configuration's numbers.

A retrain with a different recipe goes to its own directory and is selected
at load time with layer1.artifact_subdir or --layer1-artifacts, so the shipped
model is never overwritten (the script refuses unless --force is given):

    python scripts/train_layer1.py --artifact-dir artifacts/layer1_openers --augment-openers
    python scripts/train_layer1.py --artifact-dir artifacts/layer1_control

--augment-openers adds opener-augmented copies of the training records for the
classifier only (layer1/augment.py); the stackers, the calibration and the test
evaluation use the original splits, so they stay comparable across runs.

GPU notes:
  - Device auto-detected from config ("auto" → cuda if available)
  - Mixed-precision training via AMP (float16 on GPU)
  - GPU memory logged at key steps via torch.cuda.memory_allocated()
  - torch.cuda.empty_cache() called between heavy phases
  - Feature computation cached so reruns skip the slow step

--seed overrides project.seed for one run and is recorded in run_info, so a
seed-repeat study is three invocations of this script rather than three copies
of it:

    for s in 42 43 44; do
        python scripts/train_layer1.py --seed "$s" \
            --artifact-dir "artifacts/layer1_repeat_s$s"
    done

Usage (venv active, from the repository root):
    python scripts/train_layer1.py [--artifact-dir DIR] [--seed INT]
                                   [--augment-openers] [--augment-rate R]
                                   [--force]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, set_seed, save_jsonl, save_json
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

from llmguard.layer1.features   import CurvatureFeature, LogRankFeature, ClassifierFeature, resolve_device
from llmguard.layer1.fusion     import (
    DECISION_ALLOW, ClassifierOnlyFusion, LogisticFusion, TemperatureScaler,
    ThresholdRouter,
)
from llmguard.layer1.augment    import augment_with_openers, summarise as summarise_augmentation
from llmguard.layer1.classifier import build_lora_classifier, ClassifierTrainer
from llmguard.layer1.detector   import Layer1Detector

logger = get_logger("train_layer1")


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
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
# Feature computation with caching
# ─────────────────────────────────────────────────────────────────────────────

def _features_fingerprint(records: list, settings: dict) -> str:
    """Identify what a feature cache was built from: every record and the feature settings."""
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for r in records:
        h.update(json.dumps([r.get("id"), r["text"], r["label"]], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_features(
    records: list,
    curv_feat: CurvatureFeature,
    rank_feat: LogRankFeature,
    cache_path: str,
    settings: dict = None,
) -> list:
    """
    Compute curvature + log-rank features for all records.
    Results are cached to a .jsonl file with a .fingerprint beside it. The cache
    is reloaded only when the fingerprint matches the current records and
    settings, so a rebuilt pool is never trained on stale features.
    """
    fingerprint = _features_fingerprint(records, settings or {})
    stamp = Path(cache_path).with_suffix(".fingerprint")
    if Path(cache_path).exists():
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == fingerprint:
            logger.info(f"Feature cache matches current records, loading: {cache_path}")
            return load_jsonl(cache_path)
        logger.warning(
            f"Feature cache {cache_path} was built from different records or settings "
            "(or predates fingerprints) — recomputing."
        )

    logger.info(f"Computing features for {len(records)} records → {cache_path}")
    enriched = []

    for i, rec in enumerate(tqdm(records, desc="Features", unit="sample")):
        rec = rec.copy()
        try:
            rec["fcurv"] = curv_feat.compute(rec["text"])
            rec["frank"] = rank_feat.compute(rec["text"])
        except Exception as e:
            logger.warning(f"  Feature error on record {i}: {e} — using 0.5 fallback")
            rec["fcurv"] = 0.5
            rec["frank"] = 0.5
        rec["g"] = 0.5   # placeholder; updated after classifier is trained
        enriched.append(rec)

        # Periodic GPU cache clear to avoid OOM on large datasets
        if (i + 1) % 500 == 0:
            clear_gpu()
            log_gpu_memory(f"after {i+1} samples")

    save_jsonl(enriched, cache_path)
    stamp.write_text(fingerprint, encoding="utf-8")   # written last: a crash leaves no stamp
    logger.info(f"Features cached → {cache_path}")
    return enriched


def update_classifier_scores(
    records: list,
    clf_feat: ClassifierFeature,
    batch_size: int = 64,
) -> list:
    """
    Replace the g=0.5 placeholder in each record with the trained
    classifier score. Batched for GPU efficiency.
    """
    texts = [r["text"] for r in records]
    scores = clf_feat.compute_batch(texts, batch_size=batch_size)
    updated = []
    for rec, g in zip(records, scores):
        rec = rec.copy()
        rec["g"] = g
        updated.append(rec)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Test metrics from cached features
# ─────────────────────────────────────────────────────────────────────────────

def metrics_from_features(
    records: list,
    fusion,
    scaler: TemperatureScaler,
    router: ThresholdRouter,
) -> dict:
    """
    The metrics Layer1Detector.evaluate reports, computed from features that are
    already in memory.

    Same arithmetic and the same rounding as layer1/detector.py::evaluate —
    fusion → temperature scaling → router, mitigate as the positive class, a
    gray-zone prompt counted as not-allowed — but the features are taken from
    the records instead of being recomputed, so scoring a second stacker costs
    no model inference. Records need "fcurv", "frank", "g" and "label"; a
    one-feature stacker ignores the first two.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    if not records:
        logger.warning("metrics_from_features() called with empty records.")
        return {}

    raw_scores = [fusion.raw_score(r["fcurv"], r["frank"], r["g"]) for r in records]
    pred_scores = [scaler.scale(x) for x in raw_scores]
    decisions = [router.route(s) for s in pred_scores]

    true_labels = [r["label"] for r in records]
    # mitigate = positive class, as in detector.evaluate
    mitigate_labels = [1 - l for l in true_labels]
    pred_labels = [0 if d == DECISION_ALLOW else 1 for d in decisions]

    threshold, fpr95 = router.compute_fpr_at_tpr(pred_scores, true_labels, target_tpr=0.95)
    return {
        "accuracy":           round(accuracy_score(mitigate_labels, pred_labels), 4),
        "auroc":              round(roc_auc_score(mitigate_labels, pred_scores), 4),
        "auprc":              round(average_precision_score(mitigate_labels, pred_scores), 4),
        "fpr@95tpr":          round(fpr95, 4),
        "threshold_at_95tpr": round(threshold, 4),
        "n_samples":          len(records),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARTIFACT_DIR = "artifacts/layer1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Layer 1: features, classifier, stackers, calibration")
    p.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR,
                   help="Where the artifacts go (default: the shipped Layer 1). A retrain with a "
                        "different recipe belongs in its own directory, selected at load time with "
                        "layer1.artifact_subdir or --layer1-artifacts")
    p.add_argument("--augment-openers", action="store_true",
                   help="Add opener-augmented copies of the training records for the classifier "
                        "(layer1/augment.py); stackers and calibration still use the original splits")
    p.add_argument("--augment-rate", type=float, default=0.5,
                   help="Share of training records that get an augmented copy (default 0.5)")
    p.add_argument("--seed", type=int, default=None,
                   help="Training random seed (default: project.seed from config). The value used "
                        "is written into the run_info snapshot, and the RNG is reset again right "
                        "before the classifier is built so initialisation does not depend on "
                        "whether the feature caches were reused")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an artifact directory that already holds a trained classifier")
    return p.parse_args(argv)


def check_artifact_dir(artifact_dir: Path, force: bool) -> None:
    """Refuse to train over an existing classifier unless --force is given."""
    if (artifact_dir / "classifier" / "checkpoint_best").exists() and not force:
        raise SystemExit(
            f"{artifact_dir} already holds a trained Layer 1. Pass --artifact-dir "
            f"artifacts/<name> to train a model alongside it, or --force to overwrite."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args   = parse_args(argv)
    cfg    = load_config()
    # --seed overrides project.seed for this run. The config dict is updated
    # before run_info is collected so the snapshot records the seed that was
    # actually used rather than the one in config/config.yaml.
    seed   = cfg["project"]["seed"] if args.seed is None else args.seed
    cfg["project"]["seed"] = seed
    set_seed(seed)
    run_info = collect_run_info(cfg)

    device = resolve_device(cfg["project"]["device"])
    fp16   = cfg["project"].get("fp16", True)
    l1_cfg = cfg["layer1"]
    clf_cfg = l1_cfg["classifier"]

    artifact_dir = Path(args.artifact_dir)
    check_artifact_dir(artifact_dir, args.force)
    cache_dir    = Path("data/processed/layer1_features")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = Path(cfg["data"]["distillation_pool_dir"])

    logger.info("=" * 60)
    logger.info(f"LLM-Guard — Layer-1 Training")
    logger.info(f"  Device : {device}")
    logger.info(f"  FP16   : {fp16 and device == 'cuda'}")
    logger.info(f"  Output : {artifact_dir}")
    logger.info(f"  Seed   : {seed}" + ("" if args.seed is None else " (--seed overrides project.seed)"))
    logger.info(f"  Openers: {'augmented, rate ' + str(args.augment_rate) if args.augment_openers else 'not augmented'}")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading distillation pool...")
    train_records = load_jsonl(str(pool_dir / "train.jsonl"))
    val_records   = load_jsonl(str(pool_dir / "val.jsonl"))
    test_records  = load_jsonl(str(pool_dir / "test.jsonl"))
    logger.info(
        f"  Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}"
    )

    # The classifier alone may see opener-augmented copies of the training
    # records (layer1/augment.py). The features, the stackers, the calibration
    # and the test evaluation below keep using the original splits.
    clf_train_records = train_records
    augmentation = None
    if args.augment_openers:
        extra = augment_with_openers(train_records, rate=args.augment_rate, seed=seed)
        clf_train_records = train_records + extra
        augmentation = summarise_augmentation(train_records, extra, args.augment_rate, seed)
        save_json(augmentation, str(artifact_dir / "augmentation.json"))
        logger.info(
            f"  Opener augmentation: {len(extra)} copies added to {len(train_records)} "
            f"training records (by label {augmentation['augmented_by_label']}); "
            f"recorded in {artifact_dir / 'augmentation.json'}"
        )

    # ── 2. Load scoring LM (shared between curvature + log-rank) ─────────────
    scoring_lm = l1_cfg.get("scoring_lm", {"model_name": "distilgpt2"})   # pinned revision from config (same weights)
    logger.info(f"Loading scoring LM ({scoring_lm['model_name']})...")
    curv_feat = CurvatureFeature(
        model_name=scoring_lm["model_name"],
        revision=scoring_lm.get("revision"),
        num_perturbations=l1_cfg["curvature"]["num_perturbations"],
        temperature_range=tuple(l1_cfg["curvature"]["temperature_range"]),
        device=device,
        fp16=fp16,
    )
    rank_feat = LogRankFeature(
        shared_model=curv_feat.model,
        shared_tokenizer=curv_feat.tokenizer,
    )
    log_gpu_memory("after scoring LM load")

    # ── 3. Compute curvature + log-rank features (cached) ────────────────────
    feature_settings = {"scoring_model": scoring_lm["model_name"], **l1_cfg["curvature"]}   # unchanged fingerprint input
    train_enriched = compute_features(
        train_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "train_features.jsonl"),
        settings=feature_settings,
    )
    val_enriched = compute_features(
        val_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "val_features.jsonl"),
        settings=feature_settings,
    )
    test_enriched = compute_features(
        test_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "test_features.jsonl"),
        settings=feature_settings,
    )

    # ── 4. Fit initial logistic fusion (g=0.5 placeholder) ───────────────────
    logger.info("Fitting initial logistic fusion (g=placeholder)...")
    fusion = LogisticFusion()
    train_feats  = np.array([[r["fcurv"], r["frank"], r["g"]] for r in train_enriched])
    train_labels = np.array([r["label"] for r in train_enriched])
    fusion.fit(train_feats, train_labels)
    fusion.save(str(artifact_dir / "fusion_v1.joblib"))

    # ── 5. Train RoBERTa+LoRA Classifier ─────────────────────────────────────
    # Free the scoring LM from GPU before loading the classifier
    del curv_feat.model
    clear_gpu()
    log_gpu_memory("after scoring LM freed")

    logger.info("Building RoBERTa+LoRA classifier...")
    # Reset the RNG here: computing the features above consumes random numbers
    # but loading them from cache does not, so without this the classifier's
    # LoRA initialisation would depend on whether the caches were warm.
    set_seed(seed)
    model, tokenizer = build_lora_classifier(
        backbone=clf_cfg["backbone"],
        revision=clf_cfg.get("backbone_revision"),
        lora_r=clf_cfg["lora_r"],
        lora_alpha=clf_cfg["lora_alpha"],
        lora_dropout=clf_cfg["lora_dropout"],
        device=device,
        fp16=fp16,
    )
    log_gpu_memory("after classifier model load")

    trainer = ClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        backbone=clf_cfg["backbone"],
        backbone_revision=clf_cfg.get("backbone_revision"),
        device=device,
        fp16=fp16,
        lr=clf_cfg["learning_rate"],
        batch_size=clf_cfg["batch_size"],
        num_workers=clf_cfg.get("num_workers", 0),
        pin_memory=clf_cfg.get("pin_memory", False),
        max_epochs=clf_cfg["epochs"],
        patience=clf_cfg["early_stopping_patience"],
        max_length=clf_cfg["max_length"],
        output_dir=str(artifact_dir / "classifier"),
    )

    logger.info(f"Training classifier on {len(clf_train_records)} records...")
    history = trainer.train(clf_train_records, val_records)
    trainer.load_best()
    log_gpu_memory("after classifier training")

    # ── 6. Re-compute g(x) with trained classifier ───────────────────────────
    logger.info("Re-computing g(x) with trained classifier...")
    clf_feature = ClassifierFeature(
        checkpoint_path=str(artifact_dir / "classifier" / "checkpoint_best"),
        backbone=clf_cfg["backbone"],
        backbone_revision=clf_cfg.get("backbone_revision"),
        max_length=clf_cfg["max_length"],
        device=device,
        fp16=fp16,
    )

    bs = clf_cfg["batch_size"]
    train_final = update_classifier_scores(train_enriched, clf_feature, batch_size=bs)
    val_final   = update_classifier_scores(val_enriched,   clf_feature, batch_size=bs)
    test_final  = update_classifier_scores(test_enriched,  clf_feature, batch_size=bs)

    # ── 7. Re-fit fusion with real g(x) scores ───────────────────────────────
    logger.info("Re-fitting logistic fusion with trained classifier scores...")
    train_feats_v2 = np.array([[r["fcurv"], r["frank"], r["g"]] for r in train_final])
    fusion.fit(train_feats_v2, train_labels)
    fusion.save(str(artifact_dir / "fusion_v2.joblib"))

    # ── 8. Calibrate temperature scaler on validation set ────────────────────
    logger.info("Calibrating temperature scaler...")
    val_raw_scores = [
        fusion.raw_score(r["fcurv"], r["frank"], r["g"]) for r in val_final
    ]
    val_labels = [r["label"] for r in val_final]
    scaler = TemperatureScaler()
    T_star = scaler.calibrate(val_raw_scores, val_labels)
    scaler.save(str(artifact_dir / "scaler.json"))

    # ── 8b. Classifier-only stacker and its calibration ──────────────────────
    # The shipped configuration (layer1.classifier_only: true) loads these two
    # files. They used to be written only by scripts/run_ablation.py, as a side
    # effect of its own refit; that script now keeps its stacker in its own
    # directory and only ever reads these. Same data and order as the
    # three-feature stacker: fitted on the original train split's g(x),
    # calibrated on val.
    logger.info("Fitting the classifier-only stacker...")
    fusion_clf = ClassifierOnlyFusion()
    fusion_clf.fit(train_feats_v2, train_labels)
    fusion_clf.save(str(artifact_dir / "fusion_clf_only.joblib"))
    scaler_clf = TemperatureScaler()
    T_clf = scaler_clf.calibrate(
        [fusion_clf.raw_score(0.5, 0.5, r["g"]) for r in val_final], val_labels
    )
    scaler_clf.save(str(artifact_dir / "scaler_clf_only.json"))

    # ── 9. Evaluate on test set ───────────────────────────────────────────────
    # Both stackers are scored, because the two artifact pairs written above are
    # not interchangeable and the config picks one of them. The classifier-only
    # numbers, and the comparable three-signal numbers beside them, come from
    # test_final — the features are already in memory, so the second and third
    # evaluation cost no model inference.
    logger.info("Evaluating on test set...")

    # Reload scoring LM for evaluation (needed by detector.detect())
    curv_feat_eval = CurvatureFeature(
        model_name=scoring_lm["model_name"],
        revision=scoring_lm.get("revision"),
        num_perturbations=l1_cfg["curvature"]["num_perturbations"],
        temperature_range=tuple(l1_cfg["curvature"]["temperature_range"]),
        device=device,
        fp16=fp16,
    )
    rank_feat_eval = LogRankFeature(
        shared_model=curv_feat_eval.model,
        shared_tokenizer=curv_feat_eval.tokenizer,
    )

    router = ThresholdRouter(
        tau_low=l1_cfg["thresholds"]["low"],
        tau_high=l1_cfg["thresholds"]["high"],
    )
    detector = Layer1Detector(
        curvature_feature=curv_feat_eval,
        logrank_feature=rank_feat_eval,
        classifier_feature=clf_feature,
        fusion=fusion,
        scaler=scaler,
        router=router,
    )

    test_metrics = detector.evaluate(test_records)
    # Historical key, historical meaning — now says so out loud.
    test_metrics["stacker"] = "fusion_v2"
    test_metrics["stacker_files"] = ["fusion_v2.joblib", "scaler.json"]
    test_metrics["evaluation"] = (
        "Layer1Detector.evaluate on the pool test split; the detector recomputes "
        "curvature, log-rank and g(x) from the raw text"
    )
    clear_gpu()

    # The shipped stacker (layer1.classifier_only) and, on exactly the same
    # inputs, the three-signal stacker it replaces.
    test_metrics_clf_only = metrics_from_features(test_final, fusion_clf, scaler_clf, router)
    test_metrics_clf_only["stacker"] = "fusion_clf_only"
    test_metrics_clf_only["stacker_files"] = ["fusion_clf_only.joblib", "scaler_clf_only.json"]
    test_metrics_clf_only["evaluation"] = (
        "cached pool test features (fcurv and frank present but unused by this "
        "one-feature stacker)"
    )
    test_metrics_v2_cached = metrics_from_features(test_final, fusion, scaler, router)
    test_metrics_v2_cached["stacker"] = "fusion_v2"
    test_metrics_v2_cached["stacker_files"] = ["fusion_v2.joblib", "scaler.json"]
    test_metrics_v2_cached["evaluation"] = (
        "cached pool test features — the same inputs as test_metrics_clf_only, so "
        "the two stackers are directly comparable"
    )

    classifier_only = bool(l1_cfg.get("classifier_only", False))
    selected = {
        "layer1.classifier_only": classifier_only,
        "stacker": "fusion_clf_only" if classifier_only else "fusion_v2",
        "test_metrics_key": "test_metrics_clf_only" if classifier_only else "test_metrics",
        "loaded_by": "layer1.detector.Layer1Detector.from_artifacts_for",
    }

    # ── 10. Save results ──────────────────────────────────────────────────────
    results = {
        "test_metrics": test_metrics,
        "test_metrics_clf_only": test_metrics_clf_only,
        "test_metrics_fusion_v2_cached": test_metrics_v2_cached,
        "stacker_selected_by_config": selected,
        "temperature_T": T_star,
        "temperature_T_clf_only": T_clf,
        "artifact_dir": str(artifact_dir),
        "seed": seed,
        "n_classifier_training_records": len(clf_train_records),
        "augmentation": augmentation,
        "training_history": history,
        "config_snapshot": {
            "device": device,
            "fp16": fp16 and device == "cuda",
            "thresholds": l1_cfg["thresholds"],
            "classifier_backbone": clf_cfg["backbone"],
            "lora_r": clf_cfg["lora_r"],
            "batch_size": clf_cfg["batch_size"],
        },
        "run_info": finalize_run_info(run_info),
    }
    warn_if_overwriting(artifact_dir / "results.json")
    with open(artifact_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Layer-1 Training Complete")
    logger.info(f"  Seed        : {seed}")
    logger.info("  Test metrics, fusion_v2 (three signals), Layer1Detector.evaluate:")
    logger.info(f"    AUROC={test_metrics.get('auroc', 'N/A')}  "
                f"AUPRC={test_metrics.get('auprc', 'N/A')}  "
                f"FPR@95%TPR={test_metrics.get('fpr@95tpr', 'N/A')}  "
                f"Accuracy={test_metrics.get('accuracy', 'N/A')}")
    for label, block in (("fusion_clf_only (shipped stacker when layer1.classifier_only)",
                          test_metrics_clf_only),
                         ("fusion_v2, same cached features", test_metrics_v2_cached)):
        logger.info(f"  Test metrics, {label}:")
        logger.info(f"    AUROC={block.get('auroc', 'N/A')}  "
                    f"AUPRC={block.get('auprc', 'N/A')}  "
                    f"FPR@95%TPR={block.get('fpr@95tpr', 'N/A')}  "
                    f"Accuracy={block.get('accuracy', 'N/A')}")
    logger.info(f"  Config selects: {selected['stacker']} "
                f"(layer1.classifier_only={selected['layer1.classifier_only']}) → "
                f"results.json[{selected['test_metrics_key']!r}]")
    logger.info(f"  T*          : {T_star:.4f}  (classifier-only stacker: {T_clf:.4f})")
    logger.info(f"  Artifacts   → {artifact_dir.resolve()}")
    if augmentation:
        logger.info(f"  Classifier trained on {augmentation['n_training']} records "
                    f"({augmentation['n_augmented']} opener-augmented); select this model with "
                    f"--layer1-artifacts {artifact_dir.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
