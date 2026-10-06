"""
Layer-3 Constraint Engine
=========================
Loads banned regex patterns from safety_templates.yaml and checks text
against them.  Used in two places:
  (a) scan the incoming harmful prompt to record which rules triggered
  (b) validate GuardLLM safe_rewrite candidates before accepting them

Public API:
    engine = ConstraintEngine.from_config(cfg)
    matches   = engine.check(text)       -> List[ConstraintMatch]
    cats      = engine.categories(text)  -> List[str]  (triggered categories)
    clean     = engine.is_clean(text)    -> bool
"""
from __future__ import annotations

import re
import yaml
from dataclasses import dataclass
from typing import List, Tuple

from llmguard.utils.logger import get_logger

logger = get_logger("layer3.constraints")


@dataclass
class ConstraintMatch:
    pattern:  str
    category: str
    span:     Tuple[int, int]


class ConstraintEngine:
    """
    Checks text against a set of compiled banned regex patterns loaded from
    safety_templates.yaml → banned_patterns.

    Each entry in banned_patterns must have:
        pattern  : str   — a Python regex string
        category : str   — violation category label (violence / phishing / …)
    """

    def __init__(self, patterns: List[dict]):
        self._compiled: List[dict] = []
        for entry in patterns:
            try:
                self._compiled.append({
                    "regex":    re.compile(entry["pattern"]),
                    "category": entry["category"],
                    "pattern":  entry["pattern"],
                })
            except re.error as exc:
                logger.warning(
                    f"Skipping invalid regex pattern '{entry['pattern']}': {exc}"
                )
        logger.debug(f"ConstraintEngine: {len(self._compiled)} patterns compiled.")

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "ConstraintEngine":
        templates_path = cfg["layer3"]["constraints"]["templates_path"]
        with open(templates_path, "r", encoding="utf-8") as fh:
            templates = yaml.safe_load(fh)
        patterns = templates.get("banned_patterns", [])
        logger.info(f"ConstraintEngine loaded {len(patterns)} banned patterns "
                    f"from {templates_path}.")
        return cls(patterns)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, text: str) -> List[ConstraintMatch]:
        """Return every constraint match found in *text*."""
        matches: List[ConstraintMatch] = []
        for entry in self._compiled:
            for m in entry["regex"].finditer(text):
                matches.append(ConstraintMatch(
                    pattern=entry["pattern"],
                    category=entry["category"],
                    span=m.span(),
                ))
        return matches

    def categories(self, text: str) -> List[str]:
        """Return deduplicated list of triggered violation categories."""
        return list({m.category for m in self.check(text)})

    def is_clean(self, text: str) -> bool:
        """True if no banned patterns are triggered in *text*."""
        return not any(True for entry in self._compiled
                       if entry["regex"].search(text))
