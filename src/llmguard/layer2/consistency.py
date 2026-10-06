"""
Layer-2 Self-Consistency Checker
==================================
Queries the LLM Judge K times at a higher sample temperature and aggregates
the results via majority vote to produce a confident, stable final decision.

Algorithm:
  1. Fire K judge calls at sample_temperature (e.g. 0.7)
  2. Count votes: allow vs mitigate
  3. Majority wins; ties (even K) break to "mitigate" (conservative)
  4. confidence = winner_count / K
  5. If confidence < c_min → escalated=True, decision = DECISION_UNCERTAIN
  6. Category = most common among majority votes
  7. Severity  = mean of severity scores from majority votes
  8. Rationale = from the vote whose category matches the majority category

Invalid (unparseable) judge outputs are not votes, but confidence is still
winner_count / K, so failures push a result toward escalation; if every
output is invalid the result is escalated.

mode="logprob" replaces the K sampled votes with one forward pass
(LLMJudge.p_mitigate) plus one greedy call for category, severity and
rationale: mitigate if P(mitigate) > mitigate_above, allow if it is below
allow_below, otherwise uncertain (escalated).

References:
  Wang et al., 2023 — Self-Consistency Improves Chain of Thought Reasoning
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from llmguard.layer1.fusion import DECISION_ALLOW, DECISION_MITIGATE
from llmguard.layer2.judge import LLMJudge, RawJudgeResponse, DECISION_UNCERTAIN, DECISION_INVALID
from llmguard.utils.logger import get_logger

logger = get_logger("layer2.consistency")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    """
    Final output of Layer-2 for a single text.

    Fields:
        text_preview : first 100 chars of input
        decision     : "allow" | "mitigate" | "uncertain"
        confidence   : fraction of K votes agreeing with majority ∈ [0, 1]
        severity     : mean severity across majority-agreeing votes ∈ [0.0, 1.0]
        category     : most common violation category among majority votes
        rationale    : rationale from the most representative majority vote
        votes        : list of all K RawJudgeResponse.to_dict() entries
        query_count  : number of judge queries issued (= K)
        escalated    : True when confidence < c_min (needs human / Layer-3 review)
        p_mitigate   : P(mitigate) from the next-token distribution (logprob mode)
        invalid_votes: judge outputs that could not be parsed
    """
    text_preview: str
    decision:     str
    confidence:   float
    severity:     float
    category:     str
    rationale:    str
    votes:        List[Dict]
    query_count:  int
    escalated:    bool
    p_mitigate:    Optional[float] = None   # logprob mode only
    invalid_votes: int = 0                  # unparseable judge outputs

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_safe(self) -> bool:
        return self.decision == DECISION_ALLOW

    def is_harmful(self) -> bool:
        return self.decision == DECISION_MITIGATE

    def needs_escalation(self) -> bool:
        return self.escalated


# ─────────────────────────────────────────────────────────────────────────────
# Consistency Checker
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyChecker:
    """
    Self-consistency wrapper around LLMJudge.

    Queries the judge K times at sample_temperature and applies majority voting
    to produce a final decision with a confidence score.

    If confidence < c_min, the result is flagged as escalated
    (decision = DECISION_UNCERTAIN) for human review or Layer-3 handling.

    This class is stateless per-call: no state is retained between check()
    invocations, making it thread-safe.
    """

    def __init__(
        self,
        judge:              LLMJudge,
        K:                  int   = 5,
        sample_temperature: float = 0.7,
        c_min:              float = 0.6,
        mode:               str   = "vote",
        allow_below:        float = 0.4,
        mitigate_above:     float = 0.6,
    ):
        if mode not in ("vote", "logprob"):
            raise ValueError(f"Unknown consistency mode {mode!r}; choose 'vote' or 'logprob'")
        if not 0.0 <= allow_below <= mitigate_above <= 1.0:
            raise ValueError("need 0 <= allow_below <= mitigate_above <= 1")
        self.judge              = judge
        self.K                  = K
        self.sample_temperature = sample_temperature
        self.c_min              = c_min
        self.mode               = mode
        self.allow_below        = allow_below
        self.mitigate_above     = mitigate_above

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict, judge: LLMJudge) -> "ConsistencyChecker":
        """Build ConsistencyChecker from config.yaml dict."""
        c = cfg["layer2"]["consistency"]
        return cls(
            judge=judge,
            K=c["K"],
            sample_temperature=c["sample_temperature"],
            c_min=c["c_min"],
            mode=c.get("mode", "vote"),
            allow_below=c.get("allow_below", 0.4),
            mitigate_above=c.get("mitigate_above", 0.6),
        )

    # ── Core logic ────────────────────────────────────────────────────────────

    def check(self, text: str) -> JudgeResult:
        """
        Run K judge queries and return an aggregated JudgeResult.

        Steps:
          1. Fire K calls at sample_temperature
          2. Tally allow vs mitigate votes
          3. Compute confidence = majority_count / K
          4. If confidence < c_min → escalated=True, decision=DECISION_UNCERTAIN
          5. Aggregate category (most common among majority), severity (mean),
             rationale (from vote whose category best represents the majority)
        """
        if self.mode == "logprob":
            return self._check_logprob(text)

        logger.debug(f"ConsistencyChecker: firing {self.K} judge queries...")

        raw_votes: List[RawJudgeResponse] = []
        for i in range(self.K):
            resp = self.judge.judge(text, temperature=self.sample_temperature)
            raw_votes.append(resp)
            logger.debug(
                f"  Vote {i + 1}/{self.K}: decision={resp.decision} "
                f"severity={resp.severity:.2f} category={resp.category}"
            )

        return self._aggregate(text, raw_votes)

    def _aggregate(
        self,
        text:      str,
        raw_votes: List[RawJudgeResponse],
    ) -> JudgeResult:
        """
        Aggregate K raw votes into a single JudgeResult.

        Tie-breaking: if allow and mitigate split exactly 50/50 (only possible
        with even K), mitigate wins (conservative choice).
        """
        votes_serialised = [v.to_dict() for v in raw_votes]

        # Count decisions; invalid (unparseable) outputs are not votes
        decision_counts = Counter(v.decision for v in raw_votes)
        allow_count    = decision_counts.get(DECISION_ALLOW,    0)
        mitigate_count = decision_counts.get(DECISION_MITIGATE, 0)
        invalid_count  = decision_counts.get(DECISION_INVALID,  0)

        if allow_count + mitigate_count == 0:
            logger.warning(f"ConsistencyChecker: all {len(raw_votes)} judge outputs invalid — escalating.")
            return JudgeResult(
                text_preview=text[:100],
                decision=DECISION_UNCERTAIN,
                confidence=0.0,
                severity=0.5,
                category="unknown",
                rationale="Every judge output was unparseable.",
                votes=votes_serialised,
                query_count=len(raw_votes),
                escalated=True,
                invalid_votes=invalid_count,
            )

        if allow_count > mitigate_count:
            majority_decision = DECISION_ALLOW
            majority_count    = allow_count
        else:
            # Ties break to mitigate (conservative)
            majority_decision = DECISION_MITIGATE
            majority_count    = mitigate_count

        confidence = majority_count / max(self.K, 1)
        escalated  = confidence < self.c_min

        # Filter to only votes that agree with the majority
        majority_votes = [v for v in raw_votes if v.decision == majority_decision]

        # Category: most common among majority votes
        category_counts = Counter(v.category for v in majority_votes)
        majority_category = category_counts.most_common(1)[0][0] if category_counts else "unknown"

        # Severity: mean of majority-agreeing votes
        severity = sum(v.severity for v in majority_votes) / max(len(majority_votes), 1)
        severity = round(max(0.0, min(1.0, severity)), 4)

        # Rationale: from the vote whose category matches the majority category;
        # fall back to any majority-agreeing vote if none match category
        rationale_vote = next(
            (v for v in majority_votes if v.category == majority_category),
            majority_votes[0] if majority_votes else raw_votes[0],
        )
        rationale = rationale_vote.rationale

        final_decision = DECISION_UNCERTAIN if escalated else majority_decision

        logger.info(
            f"Consistency result: decision={final_decision} "
            f"confidence={confidence:.2f} severity={severity:.2f} "
            f"category={majority_category} escalated={escalated} "
            f"(allow={allow_count} mitigate={mitigate_count} invalid={invalid_count} K={self.K})"
        )

        return JudgeResult(
            text_preview=text[:100],
            decision=final_decision,
            confidence=round(confidence, 4),
            severity=severity,
            category=majority_category,
            rationale=rationale,
            votes=votes_serialised,
            query_count=len(raw_votes),
            escalated=escalated,
            invalid_votes=invalid_count,
        )

    def _check_logprob(self, text: str) -> JudgeResult:
        """P(mitigate) from one forward pass decides; one greedy call supplies the details."""
        p = self.judge.p_mitigate(text)
        if p is None:
            raise RuntimeError("consistency mode 'logprob' needs the huggingface judge backend")
        resp = self.judge.judge(text, temperature=0.0)

        if p > self.mitigate_above:
            decision = DECISION_MITIGATE
        elif p < self.allow_below:
            decision = DECISION_ALLOW
        else:
            decision = DECISION_UNCERTAIN
        escalated = decision == DECISION_UNCERTAIN

        valid = resp.decision in (DECISION_ALLOW, DECISION_MITIGATE)
        agrees = valid and resp.decision == decision
        severity = resp.severity if agrees else round(p, 4)
        if decision == DECISION_ALLOW:
            category = "none"
        elif decision == DECISION_MITIGATE:
            category = resp.category if agrees and resp.category not in ("none", "unknown") else "other"
        else:
            category = resp.category if valid else "unknown"

        logger.info(
            f"Logprob result: decision={decision} p_mitigate={p:.3f} "
            f"greedy={resp.decision} escalated={escalated}"
        )
        return JudgeResult(
            text_preview=text[:100],
            decision=decision,
            confidence=round(max(p, 1.0 - p), 4),
            severity=severity,
            category=category,
            rationale=resp.rationale if valid else "",
            votes=[resp.to_dict()],
            query_count=1,
            escalated=escalated,
            p_mitigate=round(p, 4),
            invalid_votes=0 if valid else 1,
        )
