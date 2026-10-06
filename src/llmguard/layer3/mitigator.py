"""
Layer-3 Mitigation Components
==============================
Contains:
  MitigationResult  — dataclass returned by Layer3Mitigator (the Layer-3 output)
  ModeSelector      — maps (category, severity, escalated) to a mitigation mode
  RefuseHandler     — returns a category-specific refusal from templates
  RedactHandler     — masks harmful spans with [REDACTED] markers

These classes are composed inside Layer3Mitigator (layer3_mitigator.py).
GuardLLM (safe_rewrite mode) lives separately in guard_llm.py.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import List, Optional

from llmguard.utils.logger import get_logger
from llmguard.layer3.constraints import ConstraintEngine

logger = get_logger("layer3.mitigator")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_templates(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── MitigationResult ──────────────────────────────────────────────────────────

@dataclass
class MitigationResult:
    """
    Output of Layer3Mitigator.mitigate().

    Fields mirror JudgeResult (Layer 2) for a consistent pipeline API.
    """
    text_preview:          str
    decision:              str          # always "mitigate"
    mode:                  str          # served: "refuse" | "redact" | "safe_rewrite"
    output:                str          # the safe response / refusal text
    category:              str          # violation category
    severity:              float        # 0.0–1.0
    constraints_triggered: List[str]    # which pattern categories were triggered
    escalated_to_human:    bool = False # True when L3 also cannot resolve safely
    rationale:             str  = ""    # inherited from Layer-2 (for audit trail)
    requested_mode:        str  = ""    # chosen by ModeSelector; differs from mode on fallback
    rewrite_p_leak:        Optional[float] = None  # request-conditioned check, when it ran (patch 36)
    rewrite_vetoed:        bool = False            # the check withheld the rewrite; mode is then refuse
    rewrite_candidate:     Optional[str] = None    # the withheld rewrite, in full, for review

    def to_dict(self) -> dict:
        return {
            "text_preview":          self.text_preview,
            "decision":              self.decision,
            "mode":                  self.mode,
            "output":                self.output,
            "category":              self.category,
            "severity":              self.severity,
            "constraints_triggered": self.constraints_triggered,
            "escalated_to_human":    self.escalated_to_human,
            "rationale":             self.rationale,
            "requested_mode":        self.requested_mode,
            "rewrite_p_leak":        self.rewrite_p_leak,
            "rewrite_vetoed":        self.rewrite_vetoed,
            "rewrite_candidate":     self.rewrite_candidate,
        }

    def is_safe(self) -> bool:
        return False  # Layer-3 output is always a mitigation action

    def is_escalated(self) -> bool:
        return self.escalated_to_human


# ── ModeSelector ──────────────────────────────────────────────────────────────

# Categories so dangerous that no rewrite attempt should be made.
_REFUSE_ALWAYS = {"jailbreak", "self_harm", "malware", "violence"}

# Severity threshold above which we always refuse regardless of category.
# Overridable from config; see layer3.mitigation.severity_refuse_threshold.
_SEVERITY_REFUSE_THRESHOLD = 0.99


class ModeSelector:
    """
    Maps (category, severity, escalated) to a mitigation mode string.

    Priority rules (applied in order):
      1. category in REFUSE_ALWAYS                      -> "refuse"
      2. a constraint-engine category in REFUSE_ALWAYS  -> "refuse"
      3. severity >= severity_refuse_threshold          -> "refuse"
      4. escalated == True (uncertain from L2)          -> "refuse"
      5. category unknown, and refuse_when_uncategorised -> "refuse"
      6. else                                           -> default_mode (config)

    An explicit override string bypasses all rules.

    Rule 2 exists because Layer 1 routes to Layer 3 with a score and nothing
    else, so the caller has no category to offer and rule 1 could never fire on
    that route. The constraint engine has already scanned the same text by the
    time this runs, so its categories are used as the evidence rule 1 lacks.

    Rule 5 is off by default. Turning it on stops Layer 1's blocks being
    rewritten blind: GuardLLM is otherwise asked to rewrite a prompt whose
    category it has been told is "other", which is how a request to forge an
    opioid prescription came back as instructions for forging a non-opioid one.
    """

    def __init__(
        self,
        default_mode:              str   = "refuse",
        severity_refuse_threshold: float = _SEVERITY_REFUSE_THRESHOLD,
        refuse_when_uncategorised: bool  = False,
    ):
        self.default_mode              = default_mode
        self.severity_refuse_threshold = severity_refuse_threshold
        self.refuse_when_uncategorised = refuse_when_uncategorised

    @classmethod
    def from_config(cls, cfg: dict) -> "ModeSelector":
        mit = cfg["layer3"]["mitigation"]
        return cls(
            default_mode=mit.get("default_mode", "refuse"),
            severity_refuse_threshold=mit.get(
                "severity_refuse_threshold", _SEVERITY_REFUSE_THRESHOLD),
            refuse_when_uncategorised=mit.get("refuse_when_uncategorised", False),
        )

    def select(
        self,
        category:    str,
        severity:    float,
        escalated:   bool = False,
        override:    Optional[str] = None,
        triggered:   Optional[List[str]] = None,
        categorised: bool = True,
    ) -> str:
        if override is not None:
            return override
        if category in _REFUSE_ALWAYS:
            return "refuse"
        if any(t in _REFUSE_ALWAYS for t in (triggered or ())):
            return "refuse"
        if severity >= self.severity_refuse_threshold:
            return "refuse"
        if escalated:
            return "refuse"
        if not categorised and self.refuse_when_uncategorised:
            return "refuse"
        return self.default_mode


# ── RefuseHandler ─────────────────────────────────────────────────────────────

class RefuseHandler:
    """
    Returns a category-specific refusal message from safety_templates.yaml
    under the refusal_templates key.
    Falls back to the "default" template if the category has no entry.
    """

    def __init__(self, refusals: dict):
        self._refusals = refusals

    @classmethod
    def from_config(cls, cfg: dict) -> "RefuseHandler":
        path = cfg["layer3"]["constraints"]["templates_path"]
        templates = _load_templates(path)
        return cls(templates.get("refusal_templates", {}))

    def refuse(self, category: str) -> str:
        """Return the refusal string for *category*, stripped of whitespace."""
        msg = (
            self._refusals.get(category)
            or self._refusals.get("default",
                "I'm sorry, but I cannot assist with that request.")
        )
        return msg.strip()


# ── RedactHandler ─────────────────────────────────────────────────────────────

class RedactHandler:
    """
    Replaces spans matched by ConstraintEngine with [REDACTED] markers.

    Algorithm:
      1. Run ConstraintEngine on the text to find all pattern matches.
      2. Sort matches by start offset descending so substitutions do not
         shift the offsets of earlier matches.
      3. Replace each matched span with the placeholder + reason annotation.
      4. If no specific spans were matched (category-level block only),
         return a single-line redaction notice.
    """

    def __init__(self, templates: dict, engine: ConstraintEngine):
        markers = templates.get("redaction_markers", {})
        self._placeholder   = markers.get("placeholder",   "[REDACTED]")
        self._reason_prefix = markers.get("reason_prefix", "[Content removed:")
        self._reason_suffix = markers.get("reason_suffix", "]")
        self._engine        = engine

    @classmethod
    def from_config(cls, cfg: dict, engine: ConstraintEngine) -> "RedactHandler":
        path = cfg["layer3"]["constraints"]["templates_path"]
        templates = _load_templates(path)
        return cls(templates, engine)

    def redact(self, text: str, category: str) -> str:
        """Return a redacted version of *text*, or a blanket notice if no spans hit."""
        matches = self._engine.check(text)

        if not matches:
            reason = (f"{self._reason_prefix} {category}{self._reason_suffix}")
            return f"{self._placeholder} {reason}"

        # Sort descending by start so in-place replacements stay accurate.
        sorted_matches = sorted(matches, key=lambda m: m.span[0], reverse=True)
        result = text
        for match in sorted_matches:
            start, end = match.span
            reason = f"{self._reason_prefix} {match.category}{self._reason_suffix}"
            result = result[:start] + self._placeholder + " " + reason + result[end:]
        return result
