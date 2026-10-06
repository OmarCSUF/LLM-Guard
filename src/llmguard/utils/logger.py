import hashlib
import logging
import json
import time
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ── Audit events ─────────────────────────────────────────────────────────────
# One name per layer. config/nist_ai_rmf_mapping.yaml keys its runtime tags on
# these, and scripts/nist_mapping_report.py checks that every one has an entry.
EVENT_L1 = "l1_decision"
EVENT_L2 = "l2_judgement"
EVENT_L3 = "l3_mitigation"


def text_sha256(text: str) -> str:
    """
    The key that joins one prompt's records across the layers. Records keep only
    a 100-character preview of the text, and two prompts can share a preview, so
    the hash of the full text travels with each record instead.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class AuditLogger:
    """
    Logs every L1/L2/L3 decision as a JSON line for auditability.

    Each record carries the layer, an event name, a preview of the text and the
    SHA-256 of the full text, so the records one prompt leaves at each layer can
    be joined without storing the prompt three times.

    With logging.nist_mapping on, each record also carries the NIST AI RMF 1.0
    subcategories that config/nist_ai_rmf_mapping.yaml assigns to its event. The
    mapping file, not this class, says which those are and how far each is met;
    a tag means "this record is evidence bearing on these subcategories", not
    that the subcategory is satisfied.

    Which decisions reach the log is decided by the callers. Layer 1 logs only
    when a logger is attached, which the serving entry points do and the
    evaluation does not. Layers 2 and 3 log unless called with record=False,
    which the evaluation and the red team pass. Benchmark and attack prompts
    therefore never enter the log.
    """

    DEFAULT_MAPPING_PATH = "config/nist_ai_rmf_mapping.yaml"

    def __init__(
        self,
        log_path: str = "logs/audit.jsonl",
        nist_mapping: bool = False,
        mapping_path: str = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.nist_mapping = bool(nist_mapping)
        self.mapping_path = Path(mapping_path or self.DEFAULT_MAPPING_PATH)
        self._tags = self._load_tags() if self.nist_mapping else {}

    @classmethod
    def from_config(cls, cfg: dict) -> "AuditLogger":
        """
        Build from config.yaml: logging.audit_log_path, logging.nist_mapping and
        logging.nist_mapping_path. Every layer builds its logger here, so the
        flag means the same thing at each of them.
        """
        log_cfg = cfg.get("logging", {}) or {}
        return cls(
            log_path=log_cfg.get("audit_log_path", "logs/audit.jsonl"),
            nist_mapping=log_cfg.get("nist_mapping", False),
            mapping_path=log_cfg.get("nist_mapping_path"),
        )

    def _load_tags(self) -> dict:
        """
        Event name -> subcategory ids, from the mapping file's runtime_events.
        A flag that promises tags the file cannot supply fails here, at
        construction, rather than writing untagged records quietly.
        """
        import yaml
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                "logging.nist_mapping is on but the mapping file is missing: "
                f"{self.mapping_path}"
            )
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        events = doc.get("runtime_events") or {}
        return {str(event): [str(i) for i in (ids or [])] for event, ids in events.items()}

    def tags_for(self, event: str) -> list:
        """The subcategory ids written into records of this event; empty when unmapped."""
        return list(self._tags.get(event, []))

    def log(self, record: dict) -> None:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.nist_mapping and "event" in record:
            record["nist_ai_rmf"] = self.tags_for(record["event"])
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_layer1(self, text: str, scores: dict, decision: str,
                   normalised: str = "") -> None:
        text = text or ""
        self.log({
            "layer": 1,
            "event": EVENT_L1,
            "text_preview": text[:100],
            "text_sha256": text_sha256(text),
            "scores": scores,
            "decision": decision,
            "normalised": normalised,
        })

    def log_layer2(self, text: str, judge_decision: str,
                   severity: float, rationale: str, confidence: float) -> None:
        self.log({
            "layer": 2,
            "event": EVENT_L2,
            "text_preview": text[:100],
            "text_sha256": text_sha256(text),
            "judge_decision": judge_decision,
            "severity": severity,
            "rationale": rationale,
            "confidence": confidence,
        })

    def log_layer3(self, text: str, mode: str, output: str,
                   constraints_applied: list) -> None:
        self.log({
            "layer": 3,
            "event": EVENT_L3,
            "text_preview": text[:100],
            "text_sha256": text_sha256(text),
            "mitigation_mode": mode,
            "output_preview": output[:100],
            "constraints_applied": constraints_applied,
        })
