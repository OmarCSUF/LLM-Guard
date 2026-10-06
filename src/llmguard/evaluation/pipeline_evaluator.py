"""
Pipeline Evaluator
==================
Runs the full GaurdLLM pipeline (Layer 1 + optional Layer 2 + optional Layer 3)
on a labelled dataset and returns a standardised metrics dict.

Input record format (JSONL):
    {"text": "...", "label": 0|1}   # 0=harmful, 1=safe

Output:
    EvalResult dataclass with per-layer and composite metrics.

Public API:
    evaluator = PipelineEvaluator.from_config(cfg)
    result    = evaluator.evaluate(records)
    result    = evaluator.evaluate_file("data/benchmarks/harmbench/test.jsonl")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from llmguard.utils.logger import get_logger
from llmguard.evaluation.metrics import compute_all, format_metric

logger = get_logger("evaluation.pipeline_evaluator")


# ── EvalResult ────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Structured result from a single evaluation run.
    """
    benchmark:   str
    n_samples:   int
    elapsed_s:   float
    metrics:     dict                    # from compute_all()
    layer_stats: dict = field(default_factory=dict)  # per-layer routing counts
    errors:      List[str] = field(default_factory=list)
    predictions: List[dict] = field(default_factory=list)   # one row per prompt; not in to_dict()

    def to_dict(self) -> dict:
        return {
            "benchmark":   self.benchmark,
            "n_samples":   self.n_samples,
            "elapsed_s":   round(self.elapsed_s, 2),
            "metrics":     self.metrics,
            "layer_stats": self.layer_stats,
            "errors":      self.errors,
        }

    def summary_str(self) -> str:
        m = self.metrics
        ls = self.layer_stats
        lines = [
            f"Benchmark     : {self.benchmark}",
            f"Samples       : {self.n_samples}  ({self.elapsed_s:.1f}s)",
            f"Accuracy      : {format_metric(m.get('accuracy'))}",
            f"AUROC         : {format_metric(m.get('auroc'))}",
            f"AUPRC         : {format_metric(m.get('auprc'))}",
            f"FPR@95TPR     : {format_metric(m.get('fpr_at_95tpr'))}",
            f"ECE           : {format_metric(m.get('ece'))}",
            f"JSR           : {format_metric(m.get('jailbreak_success_rate'))}",
            f"F1            : {format_metric(m.get('f1'))}",
            f"Escalation    : {format_metric(m.get('escalation_rate'))}",
            "",
            f"L1 allow      : {ls.get('l1_allow', 0)}",
            f"L1 gray_zone  : {ls.get('l1_gray_zone', 0)}",
            f"L1 mitigate   : {ls.get('l1_mitigate', 0)}",
            f"L2 allow      : {ls.get('l2_allow', 0)}",
            f"L2 mitigate   : {ls.get('l2_mitigate', 0)}",
            f"L2 uncertain  : {ls.get('l2_uncertain', 0)}",
            f"L3 refuse     : {ls.get('l3_refuse', 0)}",
            f"L3 redact     : {ls.get('l3_redact', 0)}",
            f"L3 safe_rw    : {ls.get('l3_safe_rewrite', 0)}",
        ]
        return "\n".join(lines)


# ── PipelineEvaluator ─────────────────────────────────────────────────────────

class PipelineEvaluator:
    """
    Evaluates the full GaurdLLM pipeline on a labelled dataset.

    Accepts any combination of layers: L1-only, L1+L2, L1+L2+L3.
    Escalated samples (Layer-2 uncertain) are excluded from accuracy/F1
    but tracked separately in layer_stats and counted in escalation_rate.
    """

    def __init__(self, detector, judge=None, mitigator=None):
        self.detector  = detector
        self.judge     = judge
        self.mitigator = mitigator

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg:            dict,
        load_l2:        bool = True,
        load_l3:        bool = True,
        load_guard_llm: bool = False,
        classifier_only: Optional[bool] = None,
    ) -> "PipelineEvaluator":
        """
        Build a PipelineEvaluator from config.yaml dict.

        Args:
            load_l2        : load Layer-2 judge (requires GPU + Qwen weights)
            load_l3        : load Layer-3 mitigator (refuse/redact, no extra GPU)
            load_guard_llm : also load GuardLLM in L3 for safe_rewrite
            classifier_only: use the one-feature stacker and skip curvature and
                             log-rank entirely. None follows the config; True or
                             False overrides it, which is how the command line
                             forces either Layer 1 for a comparison run.
        """
        from llmguard.layer1.detector import Layer1Detector, layer1_artifact_dir

        logger.info("Loading Layer-1 detector...")
        if classifier_only is None:
            detector = Layer1Detector.from_artifacts_for(cfg)
        else:
            # An explicit choice from the caller, so the config is bypassed;
            # the directory still follows layer1.artifact_subdir.
            artifact_dir = layer1_artifact_dir(cfg)
            fusion_file = "fusion_clf_only.joblib" if classifier_only else "fusion_v2.joblib"
            scaler_file = "scaler_clf_only.json"   if classifier_only else "scaler.json"
            detector = Layer1Detector.from_artifacts(
                cfg=cfg,
                fusion_path=str(artifact_dir / fusion_file),
                scaler_path=str(artifact_dir / scaler_file),
                classifier_checkpoint=str(artifact_dir / "classifier" / "checkpoint_best"),
                classifier_only=classifier_only,
            )

        judge = None
        if load_l2:
            try:
                from llmguard.layer2 import Layer2Judge
                logger.info("Loading Layer-2 judge...")
                judge = Layer2Judge.from_config(cfg)
            except Exception as exc:
                logger.warning(f"Layer-2 failed to load: {exc}")

        mitigator = None
        if load_l3:
            try:
                from llmguard.layer3 import Layer3Mitigator
                logger.info("Loading Layer-3 mitigator...")
                mitigator = Layer3Mitigator.from_config(
                    cfg, load_guard_llm=load_guard_llm, share_from=judge
                )
            except Exception as exc:
                logger.warning(f"Layer-3 failed to load: {exc}")

        return cls(detector=detector, judge=judge, mitigator=mitigator)

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _run_one(self, text: str) -> dict:
        """
        Run the pipeline on a single text.
        Returns a dict with final_pred (0/1), harm_score (float),
        escalated (bool), and per-layer routing info.
        """
        info = {
            "l1_decision":   None,
            "l1_score":      0.5,
            "l2_decision":   None,
            "l2_confidence": None,
            "l2_severity":   None,
            "l2_category":   None,
            "l2_votes":      None,
            "l2_p_mitigate": None,
            "l3_mode":       None,
            "final_pred":    1,       # default = allow (safe)
            "harm_score":    0.0,
            "escalated":     False,
        }

        # Layer 1
        l1 = self.detector.detect(text)
        info["l1_decision"] = l1.decision
        info["l1_score"]    = l1.calibrated_score
        info["harm_score"]  = l1.calibrated_score

        if l1.decision == "allow":
            info["final_pred"] = 1
            return info

        if l1.decision == "mitigate":
            info["final_pred"] = 0
            if self.mitigator is not None:
                l3 = self.mitigator.mitigate_from_l1(text, l1, record=False)
                info["l3_mode"] = l3.mode
            return info

        # gray_zone -> Layer 2
        if self.judge is None:
            info["final_pred"] = 1   # unresolved gray zone = allow (conservative)
            return info

        l2 = self.judge.judge(text, record=False)   # keep benchmark prompts out of the distillation pool
        info["l2_decision"]   = l2.decision
        info["l2_confidence"] = l2.confidence
        info["l2_severity"]   = getattr(l2, "severity", None)
        info["l2_category"]   = getattr(l2, "category", None)
        info["l2_votes"]      = [v.get("decision") for v in (getattr(l2, "votes", None) or [])]
        info["l2_p_mitigate"] = getattr(l2, "p_mitigate", None)
        # Blend L1 score with L2 severity for a composite harm score
        info["harm_score"]    = 0.5 * l1.calibrated_score + 0.5 * l2.severity

        if l2.escalated:
            info["escalated"]  = True
            info["final_pred"] = 0    # conservative: treat uncertain as harmful
            if self.mitigator is not None:
                l3 = self.mitigator.mitigate_from_l2(text, l2, record=False)
                info["l3_mode"] = l3.mode
            return info

        if l2.decision == "allow":
            info["final_pred"] = 1
            return info

        # mitigate
        info["final_pred"] = 0
        if self.mitigator is not None:
            l3 = self.mitigator.mitigate_from_l2(text, l2, record=False)
            info["l3_mode"] = l3.mode
        return info

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        records:   List[dict],
        benchmark: str = "custom",
        log_every: int = 50,
    ) -> EvalResult:
        """
        Evaluate on a list of labelled records.

        Args:
            records   : list of {"text": str, "label": 0|1, ...}
            benchmark : name for the EvalResult
            log_every : progress log interval

        Returns:
            EvalResult
        """
        t0 = time.perf_counter()

        y_true, y_score, y_pred, y_esc = [], [], [], []
        layer_stats = {
            "l1_allow":     0, "l1_gray_zone": 0, "l1_mitigate": 0,
            "l2_allow":     0, "l2_mitigate":  0, "l2_uncertain": 0,
            "l3_refuse":    0, "l3_redact":     0, "l3_safe_rewrite": 0,
            "errors":       0,
        }
        errors = []
        predictions = []   # one row per prompt, for error analysis

        for i, rec in enumerate(records):
            if i > 0 and i % log_every == 0:
                logger.info(f"  Evaluating: {i}/{len(records)}...")

            text  = rec.get("text", "")
            label = int(rec.get("label", 1))

            try:
                info = self._run_one(text)
            except Exception as exc:
                # Score a crash as a misclassification whatever the label, so an
                # input that breaks the pipeline can never improve the metrics.
                logger.warning(f"Error on sample {i} (counted as misclassified): {exc}")
                errors.append(str(exc))
                layer_stats["errors"] += 1
                y_true.append(label)
                y_score.append(float(label))
                y_pred.append(1 - label)
                y_esc.append(False)
                predictions.append({"id": rec.get("id"), "text": text, "label": label,
                                    "final_pred": 1 - label, "harm_score": float(label),
                                    "escalated": False, "error": str(exc)})
                continue

            y_true.append(label)
            y_score.append(info["harm_score"])
            y_pred.append(info["final_pred"])
            y_esc.append(info["escalated"])
            predictions.append({"id": rec.get("id"), "text": text, "label": label,
                                **info, "error": None})

            # Layer stats
            d1 = info["l1_decision"]
            if d1 == "allow":
                layer_stats["l1_allow"] += 1
            elif d1 == "gray_zone":
                layer_stats["l1_gray_zone"] += 1
            elif d1 == "mitigate":
                layer_stats["l1_mitigate"] += 1

            d2 = info["l2_decision"]
            if d2 == "allow":
                layer_stats["l2_allow"] += 1
            elif d2 == "mitigate":
                layer_stats["l2_mitigate"] += 1
            elif d2 == "uncertain" or info["escalated"]:
                layer_stats["l2_uncertain"] += 1

            m3 = info["l3_mode"]
            if m3 == "refuse":
                layer_stats["l3_refuse"] += 1
            elif m3 == "redact":
                layer_stats["l3_redact"] += 1
            elif m3 == "safe_rewrite":
                layer_stats["l3_safe_rewrite"] += 1

        elapsed = time.perf_counter() - t0

        if not y_true:
            return EvalResult(
                benchmark=benchmark,
                n_samples=0,
                elapsed_s=elapsed,
                metrics={},
                layer_stats=layer_stats,
                errors=errors,
                predictions=predictions,
            )

        metrics = compute_all(y_true, y_score, y_pred, escalated=y_esc)
        return EvalResult(
            benchmark=benchmark,
            n_samples=len(y_true),
            elapsed_s=elapsed,
            metrics=metrics,
            layer_stats=layer_stats,
            errors=errors,
            predictions=predictions,
        )

    def evaluate_file(
        self,
        path:      str,
        benchmark: str = None,
        limit:     int = None,
        log_every: int = 50,
    ) -> EvalResult:
        """
        Evaluate from a JSONL file.  Each line: {"text": ..., "label": 0|1}.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        name = benchmark or p.stem
        records = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if limit and len(records) >= limit:
                    break

        logger.info(f"Evaluating on {name}: {len(records)} records from {path}")
        return self.evaluate(records, benchmark=name, log_every=log_every)
