"""
Layer-2 Judge — Main Entry Point
===================================
Orchestrates the full Layer-2 pipeline: LLMJudge → ConsistencyChecker →
DistillationBuffer → AuditLogger.

                          ┌────────────────────────────────────┐
  gray_zone text ─────── │  LLMJudge  (Qwen2.5-3B-Instruct)   │
                          │    ×K queries at sample_temperature  │
                          └──────────────┬─────────────────────┘
                                         │  K × RawJudgeResponse
                          ┌──────────────▼─────────────────────┐
                          │  ConsistencyChecker                  │
                          │    majority vote + confidence         │
                          │    escalate if confidence < c_min    │
                          └──────────────┬─────────────────────┘
                                         │  JudgeResult
                          ┌──────────────▼─────────────────────┐
                          │  DistillationBuffer                  │
                          │    accumulate → flush → signal L1    │
                          └──────────────┬─────────────────────┘
                                         │
                          ┌──────────────▼─────────────────────┐
                          │  AuditLogger  (JSONL audit trail)    │
                          └────────────────────────────────────┘

Public API (mirrors Layer1Detector):
    judge  = Layer2Judge.from_config(cfg)
    result = judge.judge(text)           # → JudgeResult
    results = judge.judge_batch(texts)   # → List[JudgeResult]
"""

from __future__ import annotations

import torch
from typing import Callable, List, Optional

from llmguard.layer1.fusion import DECISION_ALLOW
from llmguard.layer2.judge import LLMJudge, DECISION_UNCERTAIN
from llmguard.layer2.consistency import ConsistencyChecker, JudgeResult
from llmguard.layer2.distillation import DistillationBuffer
from llmguard.utils.logger import get_logger, AuditLogger

logger = get_logger("layer2.layer2_judge")


# ─────────────────────────────────────────────────────────────────────────────
# Layer-2 Judge
# ─────────────────────────────────────────────────────────────────────────────

class Layer2Judge:
    """
    Full Layer-2 pipeline: ConsistencyChecker → DistillationBuffer → AuditLogger.

    Construction:
        Use Layer2Judge.from_config(cfg) for standard builds.

    Usage:
        judge   = Layer2Judge.from_config(cfg)
        result  = judge.judge(text)            # → JudgeResult
        results = judge.judge_batch(texts)     # → List[JudgeResult]
    """

    def __init__(
        self,
        checker: ConsistencyChecker,
        buffer:  DistillationBuffer,
        audit:   AuditLogger,
    ):
        self.checker = checker
        self.buffer  = buffer
        self.audit   = audit

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg:        dict,
        on_retrain: Optional[Callable[[], None]] = None,
    ) -> "Layer2Judge":
        """
        Build a Layer2Judge from config.yaml dict.

        Construction order:
          LLMJudge → ConsistencyChecker → DistillationBuffer → AuditLogger

        Args:
            cfg        : full config.yaml dict (from load_config())
            on_retrain : optional callback invoked when distillation threshold
                         is crossed (useful for automated Layer-1 retraining)
        """
        logger.info("Initialising Layer2Judge...")

        llm_judge = LLMJudge.from_config(cfg)
        checker   = ConsistencyChecker.from_config(cfg, llm_judge)
        buffer    = DistillationBuffer.from_config(cfg, on_retrain=on_retrain)
        audit     = AuditLogger.from_config(cfg)

        logger.info("Layer2Judge ready.")
        return cls(checker=checker, buffer=buffer, audit=audit)

    # ── Model sharing ─────────────────────────────────────────────────────────

    def shared_hf_model(self):
        """
        Expose the loaded HuggingFace judge so Layer-3 can reuse its weights.

        Returns:
            (model_name, model, tokenizer), or (None, None, None) when the judge
            runs on a non-HuggingFace backend and holds no local weights.
        """
        llm = self.checker.judge
        if llm.backend != "huggingface" or llm._model is None:
            return None, None, None
        return llm.model_name, llm._model, llm._tokenizer

    # ── Core detection ────────────────────────────────────────────────────────

    def judge(self, text: str, record: bool = True) -> JudgeResult:
        """
        Run Layer-2 on a single gray-zone text.

        Steps:
          1. Empty-string fast-return (allow by default)
          2. ConsistencyChecker fires K queries → JudgeResult
          3. DistillationBuffer records the result (skips escalated)
          4. AuditLogger writes a JSONL audit entry
          5. Returns JudgeResult

        Args:
            text   : the prompt to evaluate (should be a gray_zone output from L1)
            record : False skips the distillation buffer and audit log. Use it for
                     evaluation, so test prompts never enter Layer-1 training data.

        Returns:
            JudgeResult with decision, confidence, severity, category, rationale
        """
        if not text or not text.strip():
            return JudgeResult(
                text_preview="",
                decision=DECISION_ALLOW,
                confidence=1.0,
                severity=0.0,
                category="none",
                rationale="Empty input — allowed by default.",
                votes=[],
                query_count=0,
                escalated=False,
            )

        result = self.checker.check(text)
        if not record:
            return result

        retrain_triggered = self.buffer.add(text, result)
        if retrain_triggered:
            logger.warning(
                f"Distillation threshold reached "
                f"(cumulative={self.buffer.cumulative_count()}). "
                "The labels are in judge_labels.jsonl. Nothing retrains on them in "
                "this version: scripts/train_layer1.py reads the pool splits only."
            )

        self.audit.log_layer2(
            text=text,
            judge_decision=result.decision,
            severity=result.severity,
            rationale=result.rationale,
            confidence=result.confidence,
        )

        logger.debug(
            f"L2 | decision={result.decision} conf={result.confidence:.2f} "
            f"severity={result.severity:.2f} category={result.category} "
            f"escalated={result.escalated}"
        )
        return result

    def judge_batch(
        self,
        texts:     List[str],
        log_every: int = 50,
    ) -> List[JudgeResult]:
        """
        Run Layer-2 on a list of gray-zone texts.

        Clears GPU cache every `log_every` items to prevent OOM on large batches.

        Args:
            texts     : list of gray-zone texts from Layer-1
            log_every : progress log interval

        Returns:
            List[JudgeResult] in the same order as `texts`
        """
        results: List[JudgeResult] = []
        for i, text in enumerate(texts):
            if i > 0 and i % log_every == 0:
                logger.info(f"  Layer-2: {i}/{len(texts)} processed...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(self.judge(text))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, records: list) -> dict:
        """
        Evaluate Layer-2 on labelled records.

        Expects each record: {"text": str, "label": 0|1}
          label 1 = allow (safe), label 0 = mitigate (harmful)

        Returns:
            Dict with accuracy, escalation_rate, mitigate_precision, mitigate_recall
            (a rate is None when its denominator is empty)
        """
        if not records:
            logger.warning("evaluate() called with empty records.")
            return {}

        texts       = [r["text"] for r in records]
        true_labels = [r["label"] for r in records]
        results     = self.judge_batch(texts)

        correct       = 0
        escalated_n   = 0
        tp = fp = fn = tn = 0

        for true_label, res in zip(true_labels, results):
            if res.escalated:
                escalated_n += 1
                continue   # exclude from accuracy (sent to Layer-3)

            pred_label = 1 if res.is_safe() else 0
            if pred_label == true_label:
                correct += 1

            # For precision / recall on mitigate class (positive = harmful = 0)
            if true_label == 0 and pred_label == 0:
                tp += 1
            elif true_label == 1 and pred_label == 0:
                fp += 1
            elif true_label == 0 and pred_label == 1:
                fn += 1
            else:
                tn += 1

        non_escalated = len(records) - escalated_n
        accuracy      = correct / non_escalated if non_escalated else None
        precision     = tp / (tp + fp) if tp + fp else None   # nothing predicted mitigate
        recall        = tp / (tp + fn) if tp + fn else None   # no harmful prompt decided
        esc_rate      = escalated_n / len(records)

        def _r(x):
            return None if x is None else round(x, 4)

        metrics = {
            "accuracy":            _r(accuracy),
            "mitigate_precision":  _r(precision),
            "mitigate_recall":     _r(recall),     # recall on harmful prompts
            "escalation_rate":     round(esc_rate,  4),
            "n_samples":           len(records),
            "n_escalated":         escalated_n,
        }
        import json
        logger.info(f"Layer-2 eval: {json.dumps(metrics)}")
        return metrics
