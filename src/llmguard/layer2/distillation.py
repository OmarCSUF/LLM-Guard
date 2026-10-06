"""
Layer-2 Distillation Buffer
=============================
Accumulates Judge-labelled records and flushes them to the distillation pool
when the buffer reaches `retrain_every_n` samples.  Escalated / uncertain
results are excluded (they would add noise to Layer-1 training data).

Persistence:
  State survives restarts via two files:
    data/distillation_pool/judge_labels.jsonl    — appended on each flush
    data/distillation_pool/distillation_meta.json — cumulative count tracker

Record schema (compatible with Layer-1 PromptDataset):
  {
    "text":               str,
    "label":              0 | 1,      # 0=mitigate, 1=allow  (Layer-1 convention)
    "judge_decision":     str,
    "judge_confidence":   float,
    "judge_severity":     float,
    "judge_category":     str,
    "judge_rationale":    str,
    "source":             "layer2_judge",
    "timestamp":          str,        # ISO 8601 UTC
  }

Retraining signal:
  DistillationBuffer.add() returns True when a flush was triggered.
  The caller (Layer2Judge) logs a warning and optionally invokes on_retrain().
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Dict, Optional

from llmguard.layer1.fusion import DECISION_ALLOW
from llmguard.layer2.judge import DECISION_UNCERTAIN
from llmguard.utils.logger import get_logger

logger = get_logger("layer2.distillation")


# ─────────────────────────────────────────────────────────────────────────────
# Distillation Buffer
# ─────────────────────────────────────────────────────────────────────────────

class DistillationBuffer:
    """
    Accumulates Judge-labelled records and flushes them to the distillation
    pool when the buffer reaches retrain_every_n samples.

    Uncertain / escalated results are silently skipped — they would add noise
    to Layer-1 training data.

    Usage:
        buf = DistillationBuffer.from_config(cfg)
        triggered = buf.add(text, judge_result)
        if triggered:
            # Layer-1 retraining recommended
            ...
    """

    def __init__(
        self,
        retrain_every_n: int,
        pool_path:       str,
        on_retrain:      Optional[Callable[[], None]] = None,
    ):
        self.retrain_every_n = retrain_every_n
        self.pool_path       = Path(pool_path)
        self.on_retrain      = on_retrain   # optional callback on threshold crossing

        self._buffer:    List[Dict] = []
        self._meta_path  = self.pool_path / "distillation_meta.json"
        self._labels_path = self.pool_path / "judge_labels.jsonl"

        self._cumulative_count: int = self._load_meta()

        self.pool_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DistillationBuffer: retrain_every_n={retrain_every_n}, "
            f"pool={pool_path}, cumulative_so_far={self._cumulative_count}"
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg:        dict,
        on_retrain: Optional[Callable[[], None]] = None,
    ) -> "DistillationBuffer":
        """Build DistillationBuffer from config.yaml dict."""
        d = cfg["layer2"]["distillation"]
        return cls(
            retrain_every_n=d["retrain_every_n"],
            pool_path=d["pool_path"],
            on_retrain=on_retrain,
        )

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_meta(self) -> int:
        """Read cumulative label count from meta file (0 if file absent)."""
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f).get("cumulative_count", 0)
        return 0

    def _save_meta(self) -> None:
        """Persist cumulative count so restarts resume correctly."""
        self.pool_path.mkdir(parents=True, exist_ok=True)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"cumulative_count": self._cumulative_count}, f)

    # ── Core API ──────────────────────────────────────────────────────────────

    def add(self, text: str, result) -> bool:
        """
        Add one Judge result to the in-memory buffer.

        Skips escalated / uncertain results (they are too noisy for training).
        Returns True if the retrain_every_n threshold was crossed and a flush
        was performed (Layer-1 retraining is recommended).

        Args:
            text   : original input text
            result : JudgeResult from ConsistencyChecker

        Returns:
            True  → flush occurred; Layer-1 retraining suggested
            False → still accumulating
        """
        if result.escalated or result.decision == DECISION_UNCERTAIN:
            logger.debug(
                "Skipping escalated/uncertain result — "
                "not adding to distillation pool."
            )
            return False

        label = 1 if result.decision == DECISION_ALLOW else 0
        record: Dict = {
            "text":             text,
            "label":            label,
            "judge_decision":   result.decision,
            "judge_confidence": round(result.confidence, 4),
            "judge_severity":   round(result.severity, 4),
            "judge_category":   result.category,
            "judge_rationale":  result.rationale,
            "source":           "layer2_judge",
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._buffer.append(record)

        if len(self._buffer) >= self.retrain_every_n:
            n_written = self.flush()
            logger.info(
                f"Distillation threshold reached: flushed {n_written} records. "
                f"Cumulative pool size: {self._cumulative_count}."
            )
            if self.on_retrain is not None:
                self.on_retrain()
            return True

        return False

    def flush(self) -> int:
        """
        Append all buffered records to judge_labels.jsonl and clear the buffer.

        Returns:
            Number of records written in this flush.
        """
        if not self._buffer:
            return 0

        self.pool_path.mkdir(parents=True, exist_ok=True)

        with open(self._labels_path, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")

        n = len(self._buffer)
        self._cumulative_count += n
        self._buffer.clear()
        self._save_meta()

        logger.info(
            f"Flushed {n} records → {self._labels_path}. "
            f"Cumulative: {self._cumulative_count}."
        )
        return n

    # ── Introspection ─────────────────────────────────────────────────────────

    def buffer_size(self) -> int:
        """Number of records currently in the in-memory buffer (not yet flushed)."""
        return len(self._buffer)

    def cumulative_count(self) -> int:
        """Total records written to the pool across all flushes (from meta file)."""
        return self._cumulative_count
