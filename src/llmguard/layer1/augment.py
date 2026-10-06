"""
Opener augmentation for the Layer-1 classifier (patch 35)
=========================================================
The framing diagnostic of 22 September 2026 (scripts/framing_diagnostic.py)
found that the classifier reads the request opener as a label cue. In front of
a HarmBench request that is blocked when written plainly, "What is the process
for " lowers the calibrated score by 0.425 on average and by more than 0.4 for
91 of 194 such requests; "I would like to understand " does so for 23. The
request with its own imperative removed is still blocked in 16 of 16 cases, so
the cue is the opener, not the missing imperative. The cue comes from the pool,
where question-form openers are mostly benign.

augment_with_openers() adds, for a random share of the training records of
both classes, one copy that starts with a request opener drawn from
TRAIN_OPENERS. Both classes get the same openers at the same rate, so an opener
carries no label information and the classifier has to score what follows it.
Only the training split is augmented; validation, test and the benchmarks are
untouched, and so are the stackers and the calibration, which
scripts/train_layer1.py still fits on the original splits.

TRAIN_OPENERS shares no content word with the eight openers that
redteam/attacks.py uses for prefix_rewrite ("Could you walk me through ",
"What is the process for ", ...). Those stay held out: a fall in that attack's
bypass rate is then evidence that the classifier generalises across openers,
not that it has seen them.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# Question-form, polite and declarative request openers. None of the content
# words (best, way, someone, go, about, tell, need, figure, curious, show,
# teach, outline, method, tips, want, know, clarify) occurs in the eight
# prefix_rewrite openers, whose content words are walk, through, understand,
# help, learn, provide, detailed, explanation, process, describe, step,
# technical, breakdown, explain and detail.
TRAIN_OPENERS: List[str] = [
    "What is the best way to ",
    "How would someone go about ",
    "Can you tell me ",
    "I need to figure out ",
    "I am curious about ",
    "Is there a way to ",
    "Show me how to ",
    "Teach me ",
    "Outline the method for ",
    "Any tips on ",
    "I want to know ",
    "Could someone clarify ",
]

# The light strip prefix_rewrite applies to its seeds, so the augmented copy
# reads as one request rather than two openers in a row. Nothing else about the
# text changes: no case change, no rewording.
_STRIP_OPENERS = ("Tell me ", "Explain ", "Describe ", "How do I ", "How to ")


def strip_opener(text: str) -> str:
    """Remove one leading imperative or interrogative opener, if present."""
    for opener in _STRIP_OPENERS:
        if text.lower().startswith(opener.lower()):
            return text[len(opener):]
    return text


def with_opener(text: str, opener: str) -> str:
    """The text behind a request opener: opener + text minus its own opener."""
    return opener + strip_opener(text.strip())


def augment_with_openers(
    records: List[Dict],
    rate: float = 0.5,
    seed: int = 42,
    openers: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Opener-augmented copies of a random share of *records*, in record order.

    Each chosen record yields one copy whose text is with_opener(text, o) for
    an opener o drawn at random; label and every other field are kept, the id
    gets an "_opener" suffix and the copy records which original and opener it
    came from. The originals are not returned. Deterministic for a given seed.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    pool = list(openers or TRAIN_OPENERS)
    rng = random.Random(seed)
    out: List[Dict] = []
    for i, rec in enumerate(records):
        text = (rec.get("text") or "").strip()
        if not text or rec.get("label") is None:
            continue
        if rng.random() >= rate:
            continue
        opener = rng.choice(pool)
        copy = dict(rec)
        copy["text"] = with_opener(text, opener)
        copy["id"] = f"{rec.get('id', i)}_opener"
        copy["augmented_from"] = rec.get("id", i)
        copy["opener"] = opener.strip()
        out.append(copy)
    return out


def summarise(records: List[Dict], augmented: List[Dict], rate: float, seed: int) -> Dict:
    """What an augmented training run did, for the artifact directory's record."""
    by_label: Dict[str, int] = {}
    for rec in augmented:
        by_label[str(rec["label"])] = by_label.get(str(rec["label"]), 0) + 1
    by_opener: Dict[str, int] = {}
    for rec in augmented:
        by_opener[rec["opener"]] = by_opener.get(rec["opener"], 0) + 1
    return {
        "openers": list(TRAIN_OPENERS),
        "rate": rate,
        "seed": seed,
        "n_original": len(records),
        "n_augmented": len(augmented),
        "n_training": len(records) + len(augmented),
        "augmented_by_label": by_label,
        "augmented_by_opener": by_opener,
    }
