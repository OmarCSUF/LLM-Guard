"""
Layer-1 Detector — Main Entry Point
=====================================
Ties together the three feature signals, logistic fusion,
temperature calibration, and threshold routing.

                          ┌─────────────────────────┐
      text x ──────────── │   CurvatureFeature       │──► fcurv(x)
                          │   LogRankFeature          │──► frank(x)     ┐
                          │   ClassifierFeature       │──► g(x)         │
                          └─────────────────────────┘                  │
                                                                        ▼
                                                           ┌────────────────────┐
                                                           │  LogisticFusion     │
                                                           │  s_raw = σ(z(x))   │
                                                           └────────┬───────────┘
                                                                    │
                                                           ┌────────▼───────────┐
                                                           │  TemperatureScaler  │
                                                           │  s = σ(z/T)         │
                                                           └────────┬───────────┘
                                                                    │
                                                    ┌───────────────▼──────────────┐
                                                    │        ThresholdRouter        │
                                                    │  allow / gray_zone / mitigate │
                                                    └───────────────────────────────┘

GPU changes vs. CPU version:
  - Device resolved via resolve_device("auto") — uses CUDA if available
  - fp16 flag threaded through to all sub-components
  - GPU memory cleared after batch evaluation via torch.cuda.empty_cache()
  - from_config reads fp16 from config["project"]["fp16"]

Public API:
    detector = Layer1Detector.from_config(cfg)
    result   = detector.detect(text)           # → DetectionResult
    results  = detector.detect_batch(texts)    # → List[DetectionResult]
"""

from __future__ import annotations

import json
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict

from llmguard.layer1.features import (
    CurvatureFeature, LogRankFeature, ClassifierFeature, resolve_device,
    DEFAULT_SCORING_LM,
)
from llmguard.layer1.normalise import normalise, payloads, leading_clause_views
from llmguard.layer1.fusion import (
    ClassifierOnlyFusion,
    LogisticFusion, TemperatureScaler, ThresholdRouter,
    DECISION_ALLOW, DECISION_GRAY_ZONE, DECISION_MITIGATE,
)
from llmguard.utils.logger import get_logger

logger = get_logger("layer1.detector")


def layer1_artifact_dir(cfg: dict):
    """
    The directory Layer 1 loads its artifacts from.

    project.artifact_dir / layer1.artifact_subdir, which is "layer1" unless the
    config or --layer1-artifacts names a retrained model. Every loader goes
    through here, so the evaluation, the red team, the ablation and the framing
    diagnostic cannot end up on different artifacts.
    """
    from pathlib import Path as _Path
    sub = cfg.get("layer1", {}).get("artifact_subdir") or "layer1"
    return _Path(cfg["project"]["artifact_dir"]) / sub


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Output of Layer-1 for a single text.

    Fields:
        text_preview    : first 100 chars of input
        fcurv           : curvature feature score   ∈ [0, 1]
        frank           : log-rank feature score    ∈ [0, 1]
        g_classifier    : classifier feature score  ∈ [0, 1]
        raw_score       : fused score before calibration
        calibrated_score: fused score after temperature scaling
        decision        : "allow" | "gray_zone" | "mitigate"
        normalised      : transforms whose recovered form scored higher than the
                          text as given, joined by "+", or "base64-payload" /
                          "leading-clause" when a payload view won; empty when
                          the scores come from the text exactly as received
    """
    text_preview:     str
    fcurv:            float
    frank:            float
    g_classifier:     float
    raw_score:        float
    calibrated_score: float
    decision:         str
    normalised:       str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_safe(self) -> bool:
        return self.decision == DECISION_ALLOW

    def needs_judge(self) -> bool:
        return self.decision == DECISION_GRAY_ZONE

    def is_harmful(self) -> bool:
        return self.decision == DECISION_MITIGATE


# ─────────────────────────────────────────────────────────────────────────────
# Layer-1 Detector
# ─────────────────────────────────────────────────────────────────────────────

class Layer1Detector:
    """
    Full Layer-1 pipeline: features → fusion → calibration → routing.

    Construction:
        Use Layer1Detector.from_config(cfg_dict) for standard builds.
        Use Layer1Detector(...) for unit tests / custom builds.
    """

    def __init__(
        self,
        curvature_feature:  CurvatureFeature,
        logrank_feature:    LogRankFeature,
        classifier_feature: ClassifierFeature,
        fusion:  LogisticFusion,
        scaler:  TemperatureScaler,
        router:  ThresholdRouter,
        normalise_input: bool = False,
        payload_views:   bool = False,
        audit = None,
    ):
        self.curvature_feature  = curvature_feature
        self.logrank_feature    = logrank_feature
        self.classifier_feature = classifier_feature
        self.fusion = fusion
        self.scaler = scaler
        self.router = router
        # Off unless a config asks for it, so hand-built detectors in tests and
        # in the ablation keep scoring exactly what they are handed.
        self.normalise_input = normalise_input
        # The text behind a short leading instruction frame is scored on its own
        # and the higher score kept (layer1/normalise.py). The constructor default
        # is off, so hand-built detectors score what they are handed; the config
        # default is on since patch 34.
        self.payload_views = payload_views
        # An audit logger (utils/logger.py), when one is attached. Only the
        # serving entry points attach one (app.py, scripts/demo.py), so the
        # evaluation, the ablation and the red team never write their prompts
        # into the audit log.
        self.audit = audit

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "Layer1Detector":
        """
        Build a fresh (untrained) Layer1Detector from config.yaml dict.
        Call fit_fusion() and fit_scaler() before using for inference.

        The scoring LM is layer1.scoring_lm.model_name at its pinned revision
        (distilgpt2, unpinned, when the block is absent); the classifier
        backbone is layer1.classifier.backbone at backbone_revision.
        """
        l1 = cfg["layer1"]
        device = resolve_device(cfg["project"]["device"])
        fp16   = cfg["project"].get("fp16", True)

        logger.info(f"Initialising Layer-1 Detector (device={device}, fp16={fp16})...")

        # Shared scoring LM — loaded once, used by both curvature and log-rank
        lm_cfg = l1.get("scoring_lm") or {}
        curv = CurvatureFeature(
            model_name=lm_cfg.get("model_name") or DEFAULT_SCORING_LM,
            revision=lm_cfg.get("revision"),
            num_perturbations=l1["curvature"]["num_perturbations"],
            temperature_range=tuple(l1["curvature"]["temperature_range"]),
            device=device,
            fp16=fp16,
        )
        rank = LogRankFeature(
            shared_model=curv.model,
            shared_tokenizer=curv.tokenizer,
        )

        clf_cfg = l1["classifier"]
        clf = ClassifierFeature(
            checkpoint_path=None,   # not trained yet
            backbone=clf_cfg["backbone"],
            backbone_revision=clf_cfg.get("backbone_revision"),
            max_length=clf_cfg["max_length"],
            device=device,
            fp16=fp16,
        )

        th = l1["thresholds"]
        logger.info(f"Layer-1 thresholds: τ_low={th['low']}, τ_high={th['high']}")

        normalise_input = l1.get("normalise_input", False)
        if normalise_input:
            logger.info("Layer-1 input normalisation: on (base64, leetspeak)")
        payload_views = bool(l1.get("payload_views", False))
        if payload_views:
            logger.info("Layer-1 payload views: on (text behind a leading instruction frame)")

        return cls(
            curvature_feature=curv,
            logrank_feature=rank,
            classifier_feature=clf,
            fusion=LogisticFusion(),
            scaler=TemperatureScaler(T=l1["calibration"]["temperature"]),
            router=ThresholdRouter(tau_low=th["low"], tau_high=th["high"]),
            normalise_input=normalise_input,
            payload_views=payload_views,
        )

    @classmethod
    def from_artifacts(
        cls,
        cfg: dict,
        fusion_path: str,
        scaler_path: str,
        classifier_checkpoint: str,
        classifier_only: bool = False,
    ) -> "Layer1Detector":
        """
        Load a fully-fitted Layer-1 from saved artifacts.
        Use this for inference after training is complete.

        classifier_only swaps in the one-feature stacker, so curvature and
        log-rank are neither read nor computed. The scoring LM is still loaded,
        because the same object is shared with the routing code; it simply goes
        unused.
        """
        device = resolve_device(cfg["project"]["device"])
        fp16   = cfg["project"].get("fp16", True)
        clf_cfg = cfg["layer1"]["classifier"]

        detector = cls.from_config(cfg)
        if classifier_only:
            detector.fusion = ClassifierOnlyFusion()
            logger.info("Layer-1 stacker: classifier only (curvature and log-rank skipped)")
        detector.fusion.load(fusion_path)
        detector.scaler.load(scaler_path)
        detector.classifier_feature = ClassifierFeature(
            checkpoint_path=classifier_checkpoint,
            backbone=clf_cfg["backbone"],
            backbone_revision=clf_cfg.get("backbone_revision"),
            max_length=clf_cfg["max_length"],
            device=device,
            fp16=fp16,
        )
        return detector

    @classmethod
    def from_artifacts_for(cls, cfg: dict) -> "Layer1Detector":
        """
        Load Layer 1 with the artifacts the config asks for.

        Every entry point goes through here, so the evaluation and the red team
        cannot end up testing different versions of Layer 1. Which stacker is
        loaded follows layer1.classifier_only; which directory follows
        layer1.artifact_subdir (layer1_artifact_dir).
        """
        classifier_only = bool(cfg.get("layer1", {}).get("classifier_only", False))
        artifact_dir = layer1_artifact_dir(cfg)
        fusion_file = "fusion_clf_only.joblib" if classifier_only else "fusion_v2.joblib"
        scaler_file = "scaler_clf_only.json"   if classifier_only else "scaler.json"
        return cls.from_artifacts(
            cfg=cfg,
            fusion_path=str(artifact_dir / fusion_file),
            scaler_path=str(artifact_dir / scaler_file),
            classifier_checkpoint=str(artifact_dir / "classifier" / "checkpoint_best"),
            classifier_only=classifier_only,
        )

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        """
        Run Layer-1 on a single text.

        Steps:
          1. Compute three feature scores (curvature, log-rank, classifier)
          2. Fuse via logistic stacker → raw_score
          3. Calibrate via temperature scaling → calibrated_score
          4. Route to allow / gray_zone / mitigate

        With normalise_input on, steps 1 to 3 also run on the recovered text
        whenever normalisation changes it, and on each decoded base64 payload
        by itself. With payload_views on they also run on the text behind each
        short leading instruction frame. The highest calibrated score is the
        one routed. See layer1/normalise.py.
        """
        if not text or not text.strip():
            return self._audit(text, DetectionResult(
                text_preview="",
                fcurv=0.5, frank=0.5, g_classifier=0.5,
                raw_score=0.5, calibrated_score=0.5,
                decision=DECISION_GRAY_ZONE,
            ))

        fcurv, frank, g, raw_score, calibrated = self._score(text)
        applied = ""

        if self.normalise_input or self.payload_views:
            forms = []
            if self.normalise_input:
                recovered, transforms = normalise(text)
                if transforms:
                    forms.append((recovered, "+".join(transforms)))
                # The payload alone, away from the instruction wrapped around it:
                # the wrapper can carry the score on its own (layer1/normalise.py).
                for part in payloads(text):
                    forms.append((part, "base64-payload"))
            if self.payload_views:
                # The text behind a short leading instruction frame, scored on
                # its own: a plain-text frame carries the score the way a base64
                # wrapper does (layer1/normalise.py, leading_clause_views).
                for part in leading_clause_views(text):
                    forms.append((part, "leading-clause"))

            for form, tag in forms:
                alt = self._score(form)
                # Keep the highest score. Normalisation is allowed to raise a
                # score, never to lower one, so an obfuscated prompt cannot
                # score better than the same prompt written plainly.
                if alt[4] > calibrated:
                    fcurv, frank, g, raw_score, calibrated = alt
                    applied = tag

        decision = self.router.route(calibrated)

        logger.debug(
            f"L1 | fcurv={fcurv:.3f} frank={frank:.3f} g={g:.3f} "
            f"raw={raw_score:.3f} cal={calibrated:.3f} → {decision}"
            + (f" (normalised: {applied})" if applied else "")
        )
        return self._audit(text, DetectionResult(
            text_preview=text[:100],
            fcurv=round(fcurv, 4),
            frank=round(frank, 4),
            g_classifier=round(g, 4),
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated, 4),
            decision=decision,
            normalised=applied,
        ))

    def _score(self, text: str) -> tuple:
        """
        Features, fusion and calibration for one string, without routing.

        A stacker that does not read the two scoring-LM features has them
        skipped rather than computed and discarded, which is where the drop
        from 38 ms to 6 ms per prompt comes from. The reported fcurv and frank
        are then 0.5 placeholders and mean "not computed".
        """
        if getattr(self.fusion, "needs_lm_features", True):
            fcurv = self.curvature_feature.compute(text)
            frank = self.logrank_feature.compute(text)
        else:
            fcurv = frank = 0.5
        g = self.classifier_feature.compute(text)
        raw_score = self.fusion.raw_score(fcurv, frank, g)
        return fcurv, frank, g, raw_score, self.scaler.scale(raw_score)

    def _audit(self, text: str, result: DetectionResult) -> DetectionResult:
        """
        Write the decision to the audit log when a logger is attached, and hand
        the result back unchanged. The scores logged are the ones routed, so a
        record shows the normalised or payload view when that is what decided.
        """
        if self.audit is not None:
            self.audit.log_layer1(
                text=text or "",
                scores={
                    "fcurv": result.fcurv,
                    "frank": result.frank,
                    "g_classifier": result.g_classifier,
                    "raw_score": result.raw_score,
                    "calibrated_score": result.calibrated_score,
                },
                decision=result.decision,
                normalised=result.normalised,
            )
        return result

    def detect_batch(self, texts: List[str], log_every: int = 200) -> List[DetectionResult]:
        """
        Run Layer-1 on a list of texts.
        Clears GPU cache every `log_every` items to prevent OOM on large batches.
        """
        results = []
        for i, text in enumerate(texts):
            if i > 0 and i % log_every == 0:
                logger.info(f"  Layer-1: {i}/{len(texts)} processed...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(self.detect(text))

        # Final cache clear after batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    # ── Calibration helpers ───────────────────────────────────────────────────

    def fit_fusion(self, records: list) -> None:
        """
        Fit the logistic stacker on labelled records.
        Expects each record to have 'fcurv', 'frank', 'g' pre-computed.
        If missing, recomputes them (slow — prefer caching first).
        """
        import numpy as np

        features, labels = [], []
        for rec in records:
            fc = rec.get("fcurv") if rec.get("fcurv") is not None else self.curvature_feature.compute(rec["text"])
            fr = rec.get("frank") if rec.get("frank") is not None else self.logrank_feature.compute(rec["text"])
            g  = rec.get("g")     if rec.get("g")     is not None else self.classifier_feature.compute(rec["text"])
            features.append([fc, fr, g])
            labels.append(rec["label"])

        self.fusion.fit(np.array(features), np.array(labels))

    def fit_scaler(self, records: list) -> float:
        """
        Fit temperature scaler on validation records.
        Expects 'fcurv', 'frank', 'g' to be pre-computed in each record.
        """
        raw_scores, labels = [], []
        for rec in records:
            fc = rec.get("fcurv") if rec.get("fcurv") is not None else self.curvature_feature.compute(rec["text"])
            fr = rec.get("frank") if rec.get("frank") is not None else self.logrank_feature.compute(rec["text"])
            g  = rec.get("g")     if rec.get("g")     is not None else self.classifier_feature.compute(rec["text"])
            raw_scores.append(self.fusion.raw_score(fc, fr, g))
            labels.append(rec["label"])

        return self.scaler.calibrate(raw_scores, labels)

    def save_fusion_artifacts(
        self,
        fusion_path: str = "artifacts/layer1/fusion.joblib",
        scaler_path: str = "artifacts/layer1/scaler.json",
    ) -> None:
        self.fusion.save(fusion_path)
        self.scaler.save(scaler_path)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, records: list) -> Dict:
        """
        Evaluate Layer-1 on labelled records.
        Returns: accuracy, AUROC, AUPRC, FPR@95%TPR.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

        if not records:
            logger.warning("evaluate() called with empty records.")
            return {}

        texts = [r["text"] for r in records]
        results = self.detect_batch(texts)

        true_labels  = [r["label"] for r in records]
        pred_scores  = [r.calibrated_score for r in results]
        pred_labels  = [0 if r.is_safe() else 1 for r in results]

        # mitigate = positive class (label 1 for metric computation)
        mitigate_labels = [1 - l for l in true_labels]

        auroc = roc_auc_score(mitigate_labels, pred_scores)
        auprc = average_precision_score(mitigate_labels, pred_scores)
        acc   = accuracy_score(mitigate_labels, pred_labels)
        threshold, fpr95 = self.router.compute_fpr_at_tpr(
            pred_scores, true_labels, target_tpr=0.95
        )

        metrics = {
            "accuracy":           round(acc,       4),
            "auroc":              round(auroc,      4),
            "auprc":              round(auprc,      4),
            "fpr@95tpr":          round(fpr95,      4),
            "threshold_at_95tpr": round(threshold,  4),
            "n_samples":          len(records),
        }
        logger.info(f"Layer-1 eval: {json.dumps(metrics)}")
        return metrics
