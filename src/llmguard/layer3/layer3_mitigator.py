"""
Layer-3 Mitigator — Main Entry Point
======================================
Orchestrates the full Layer-3 pipeline:

              harmful / uncertain text
                        |
          ┌─────────────▼──────────────────┐
          │  ConstraintEngine               │
          │    scan input, record triggers  │
          └─────────────┬──────────────────┘
                        │  triggered categories
          ┌─────────────▼──────────────────┐
          │  ModeSelector                   │
          │    category + severity          │
          │    → refuse / redact /          │
          │      safe_rewrite               │
          └─────────────┬──────────────────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
     refuse│      redact│  safe_     │rewrite
           ▼            ▼  ─────────▼──────────────
     RefuseHandler  RedactHandler   GuardLLM
     (template)     (span masking)  (K_prime candidates
                                     → re-rank → accept
                                     or fallback refuse)
           │            │            │
          ┌▼────────────▼────────────▼┐
          │  AuditLogger              │
          │    JSONL audit entry      │
          └───────────────────────────┘
                        │
                 MitigationResult

Public API (mirrors Layer2Judge):
    mitigator = Layer3Mitigator.from_config(cfg)
    mitigator = Layer3Mitigator.from_config(cfg, share_from=judge)  # reuse L2 weights
    result    = mitigator.mitigate(text, category, severity)
    result    = mitigator.mitigate_from_l2(text, judge_result)
    result    = mitigator.mitigate_from_l1(text, detection_result)
    results   = mitigator.mitigate_batch(records)
"""
from __future__ import annotations

from typing import List, Optional

from llmguard.utils.logger import get_logger, AuditLogger
from llmguard.utils.config_loader import judge_model
from llmguard.layer3.constraints import ConstraintEngine
from llmguard.layer3.mitigator import (
    MitigationResult,
    ModeSelector,
    RefuseHandler,
    RedactHandler,
)

logger = get_logger("layer3.layer3_mitigator")


class Layer3Mitigator:
    """
    Full Layer-3 pipeline: ConstraintEngine → ModeSelector → Handler → AuditLogger.

    Construction:
        Use Layer3Mitigator.from_config(cfg) for standard builds.

    Usage:
        mitigator = Layer3Mitigator.from_config(cfg)
        result    = mitigator.mitigate(text)
        result    = mitigator.mitigate_from_l2(text, judge_result)
        result    = mitigator.mitigate_from_l1(text, l1_result)
    """

    def __init__(
        self,
        engine:          ConstraintEngine,
        mode_selector:   ModeSelector,
        refuse_handler:  RefuseHandler,
        redact_handler:  RedactHandler,
        guard_llm,                        # GuardLLM | None
        audit:           AuditLogger,
        available_modes: List[str] = None,
        rewrite_check    = None,          # FulfilmentCheck | None (layer3/fulfilment.py)
    ):
        self.rewrite_check   = rewrite_check
        self.engine          = engine
        self.mode_selector   = mode_selector
        self.refuse_handler  = refuse_handler
        self.redact_handler  = redact_handler
        self.guard_llm       = guard_llm
        self.audit           = audit
        self.available_modes = available_modes or ["refuse", "redact", "safe_rewrite"]

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg:            dict,
        load_guard_llm: bool = True,
        share_from=None,                  # Layer2Judge | None
    ) -> "Layer3Mitigator":
        """
        Build a Layer3Mitigator from a config.yaml dict.

        Args:
            cfg            : full config dict (from load_config())
            load_guard_llm : whether to load Qwen2.5-3B for safe_rewrite mode.
                             Set False for lightweight deployments that only need
                             refuse + redact (no GPU required).
            share_from     : a loaded Layer2Judge. If it runs the same HuggingFace
                             model as layer3.guard_llm, GuardLLM reuses those
                             weights instead of loading a second copy.
        """
        logger.info("Initialising Layer3Mitigator...")

        engine         = ConstraintEngine.from_config(cfg)
        mode_selector  = ModeSelector.from_config(cfg)
        refuse_handler = RefuseHandler.from_config(cfg)
        redact_handler = RedactHandler.from_config(cfg, engine)
        audit          = AuditLogger.from_config(cfg)

        guard_llm = None
        if load_guard_llm:
            shared_model, shared_tokenizer = cls._shared_weights(cfg, share_from)
            try:
                from llmguard.layer3.guard_llm import GuardLLM
                guard_llm = GuardLLM.from_config(
                    cfg, model=shared_model, tokenizer=shared_tokenizer
                )
            except Exception as exc:
                logger.warning(
                    f"GuardLLM failed to load ({exc}). "
                    "safe_rewrite mode will fall back to refuse."
                )

        # The request-conditioned check of served rewrites, on the same weights.
        # Built only when the config asks for it (layer3.rewrite_check.mode) and a
        # Guard-LLM is loaded, since without one nothing is ever rewritten.
        rewrite_check = None
        if guard_llm is not None:
            from llmguard.layer3.fulfilment import FulfilmentCheck
            rewrite_check = FulfilmentCheck.from_config(
                cfg, model=guard_llm.model, tokenizer=guard_llm.tokenizer
            )

        available_modes = cfg["layer3"]["mitigation"].get(
            "modes", ["refuse", "redact", "safe_rewrite"]
        )
        logger.info("Layer3Mitigator ready.")
        return cls(
            engine=engine,
            mode_selector=mode_selector,
            refuse_handler=refuse_handler,
            redact_handler=redact_handler,
            guard_llm=guard_llm,
            audit=audit,
            available_modes=available_modes,
            rewrite_check=rewrite_check,
        )

    @staticmethod
    def _shared_weights(cfg: dict, share_from):
        """
        Return (model, tokenizer) from a Layer2Judge when they can be reused.

        Returns (None, None), so GuardLLM loads its own copy, when no judge was
        given, the judge holds no HuggingFace weights, or it runs a different
        model from layer3.guard_llm.model_name. The names are compared as
        strings, both resolved through utils/config_loader.judge_model, so an
        LLMGUARD_JUDGE_MODEL override applies to both sides alike.
        """
        if share_from is None:
            return None, None

        name, model, tokenizer = share_from.shared_hf_model()
        wanted, _ = judge_model(cfg["layer3"]["guard_llm"])
        if model is None:
            logger.warning(
                "Layer-2 judge holds no HuggingFace weights to share; "
                "GuardLLM will load its own copy."
            )
            return None, None
        if name != wanted:
            logger.warning(
                f"Layer-2 judge runs {name!r} but layer3.guard_llm.model_name is "
                f"{wanted!r}; GuardLLM will load its own copy."
            )
            return None, None

        logger.info(f"Layer-3 will share the Layer-2 model instance ({name}).")
        return model, tokenizer

    # ── Core mitigation ───────────────────────────────────────────────────────

    def mitigate(
        self,
        text:          str,
        category:      str   = "other",
        severity:      float = 0.5,
        rationale:     str   = "",
        escalated:     bool  = False,
        mode_override: Optional[str] = None,
        categorised:   bool  = True,
        record:        bool  = True,
    ) -> MitigationResult:
        """
        Run Layer-3 mitigation on a single harmful or uncertain text.

        Args:
            text          : the original prompt to mitigate
            category      : violation category from Layer-2 (or "other")
            severity      : 0.0–1.0 severity score from Layer-2
            rationale     : Layer-2 rationale string (passed through to audit)
            escalated     : True when Layer-2 was uncertain (escalated flag)
            mode_override : force a specific mode, bypassing ModeSelector
            categorised   : False when *category* is a placeholder rather than a
                            real verdict, as on the Layer-1 route
            record        : False skips the audit log. The evaluation and the red
                            team pass it, as they do to the judge, so benchmark and
                            attack prompts never enter the log

        Returns:
            MitigationResult
        """
        if not text or not text.strip():
            return MitigationResult(
                text_preview="",
                decision="mitigate",
                mode="refuse",
                output=self.refuse_handler.refuse("default"),
                category=category,
                severity=severity,
                constraints_triggered=[],
                escalated_to_human=False,
                rationale="Empty input.",
                requested_mode="refuse",
            )

        # Step 1: scan input for triggered constraint categories
        triggered = self.engine.categories(text)

        # Step 2: choose mitigation mode. The constraint categories from step 1
        # are the only category evidence the Layer-1 route has.
        mode = self.mode_selector.select(
            category=category,
            severity=severity,
            escalated=escalated,
            override=mode_override,
            triggered=triggered,
            categorised=categorised,
        )

        # Step 3: execute chosen handler (it may fall back to another mode)
        output, escalated_to_human, served_mode, p_leak, vetoed, candidate = self._execute_mode(
            mode=mode,
            text=text,
            category=category,
            severity=severity,
        )

        # Step 4: write audit log entry, unless this is an evaluation call
        if record:
            self.audit.log_layer3(
                text=text,
                mode=served_mode,
                output=output,
                constraints_applied=triggered,
            )

        logger.debug(
            f"L3 | mode={served_mode} requested={mode} category={category} severity={severity:.2f} "
            f"constraints={triggered} escalated={escalated_to_human}"
        )

        return MitigationResult(
            text_preview=text[:100],
            decision="mitigate",
            mode=served_mode,
            output=output,
            category=category,
            severity=severity,
            constraints_triggered=triggered,
            escalated_to_human=escalated_to_human,
            rationale=rationale,
            requested_mode=mode,
            rewrite_p_leak=p_leak,
            rewrite_vetoed=vetoed,
            rewrite_candidate=candidate,
        )

    def _execute_mode(
        self,
        mode:     str,
        text:     str,
        category: str,
        severity: float,
    ) -> tuple:
        """
        Execute the chosen mitigation mode.

        Returns (output_str, escalated_bool, served_mode, p_leak, vetoed, candidate).
        served_mode is the mode that actually produced the output: "refuse"
        whenever safe_rewrite falls back, including when every rewrite candidate
        fails validation or the rewrite check withholds the survivor. p_leak is
        the check's score when it ran, vetoed says whether it withheld the
        rewrite, and candidate is the withheld text (layer3/fulfilment.py).
        """

        if mode == "refuse":
            return self.refuse_handler.refuse(category), False, "refuse", None, False, None

        if mode == "redact":
            return self.redact_handler.redact(text, category), False, "redact", None, False, None

        if mode == "safe_rewrite":
            if self.guard_llm is None:
                logger.warning(
                    "safe_rewrite requested but GuardLLM is not loaded — "
                    "falling back to refuse."
                )
                return self.refuse_handler.refuse(category), False, "refuse", None, False, None

            output = self.guard_llm.rewrite(
                text, category=category, severity=severity
            )
            if output is None:
                # All K_prime candidates violated constraints.
                logger.warning(
                    "safe_rewrite: all candidates violated constraints — "
                    "escalating to human review."
                )
                return (
                    "This content could not be safely rewritten and has been "
                    "flagged for human review.",
                    True,
                    "refuse",
                    None, False, None,
                )

            # The surviving candidate is clean by the constraint engine's four
            # regexes. The request-conditioned check asks the other question,
            # whether it still supplies what the request asked for.
            p_leak = None
            if self.rewrite_check is not None:
                result = self.rewrite_check.check(text, output)
                p_leak = result.p_leak
                if result.leak and self.rewrite_check.veto:
                    logger.info(
                        f"safe_rewrite: rewrite withheld, P(leak)={p_leak:.3f} >= "
                        f"{self.rewrite_check.threshold:.2f}; serving a refusal."
                    )
                    return self.refuse_handler.refuse(category), False, "refuse", p_leak, True, output
            return output, False, "safe_rewrite", p_leak, False, None

        logger.warning(f"Unknown mitigation mode '{mode}' — falling back to refuse.")
        return self.refuse_handler.refuse(category), False, "refuse", None, False, None

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def mitigate_from_l2(self, text: str, judge_result, record: bool = True) -> MitigationResult:
        """
        Build mitigate() arguments directly from a Layer-2 JudgeResult.
        Use when routing a mitigate/uncertain result from Layer 2 into Layer 3.
        """
        return self.mitigate(
            text=text,
            category=getattr(judge_result, "category", "other") or "other",
            severity=getattr(judge_result, "severity",  0.5),
            rationale=getattr(judge_result, "rationale", ""),
            escalated=getattr(judge_result, "escalated", False),
            record=record,
        )

    def mitigate_from_l1(self, text: str, detection_result, record: bool = True) -> MitigationResult:
        """
        Build mitigate() arguments from a Layer-1 DetectionResult.
        Use when Layer 1 routes directly to mitigate (high-confidence block,
        calibrated_score > high_threshold) without passing through Layer 2.
        """
        score = getattr(detection_result, "calibrated_score", 0.9)
        return self.mitigate(
            text=text,
            # Layer 1 produces no category. "other" is a placeholder the
            # handlers need, not a verdict, so it is flagged as such.
            category="other",
            categorised=False,
            severity=score,
            rationale=f"Layer-1 calibrated score: {score:.3f}",
            escalated=False,
            record=record,
        )

    def mitigate_batch(self, records: list) -> List[MitigationResult]:
        """
        Run Layer-3 on a list of record dicts.

        Expected record keys (all optional with defaults):
            text      : str   — the prompt to mitigate
            category  : str   — violation category  (default "other")
            severity  : float — 0.0–1.0              (default 0.5)
            rationale : str   — Layer-2 rationale    (default "")
            escalated : bool  — Layer-2 escalated    (default False)

        Returns:
            List[MitigationResult] in the same order as *records*.
        """
        results: List[MitigationResult] = []
        for i, rec in enumerate(records):
            if i > 0 and i % 100 == 0:
                logger.info(f"  Layer-3: {i}/{len(records)} processed...")
            results.append(self.mitigate(
                text=rec.get("text",      ""),
                category=rec.get("category",  "other"),
                severity=rec.get("severity",  0.5),
                rationale=rec.get("rationale", ""),
                escalated=rec.get("escalated", False),
            ))
        return results
