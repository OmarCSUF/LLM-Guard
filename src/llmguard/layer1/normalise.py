"""
Layer-1 Input Normalisation
============================
Recover the readable form of a prompt before Layer 1 scores it.

Layer 1 reads a base64 blob or a leetspeak string as subword noise. Its
tokeniser has never seen those shapes, and curvature and log-rank measure
perturbation likelihood under an English language model, which base64 does not
have, so all three features fail together and in the same direction. In a
red-team run every base64 and leetspeak variant scored about 0.034 and was
allowed: a 100% bypass rate, and the only attack class that defeated the
cascade outright.

Two transforms:

  base64     a run that decodes to readable text is replaced by that text
  leetspeak  character substitutions are folded back, but only when enough of
             the document carries them to rule out ordinary writing such as
             "GPT4", "COVID19" or "3 dogs"

Neither transform decides anything. normalise() returns the recovered text and
the names of the transforms that fired; the caller scores both the text as
given and the recovered form and keeps the higher score.

A decoded base64 payload is also scored on its own, through payloads(). An
encoded attack arrives as an instruction plus the payload, and the instruction
can carry the score. In a red-team run, "Decode this Base64 string and answer
the question inside:" in front of two chemistry-synthesis requests took their
score from 0.976 to 0.04, while the same requests scored 0.976 unwrapped. The
payload is the request, so it is scored as one. Normalisation can
therefore raise a score but never lower one, and cannot open a bypass that did
not already exist. What it can do is block a benign prompt that happens to
recover into something the classifier dislikes, which is the cost to measure.

A plain-text instruction frame carries the score the same way. "Parse the
following JSON request and fulfil it:" in front of an unchanged harmful request
took it below the allow threshold for 175 of 600 red-team variants, with no
encoding involved. leading_clause_views() returns the text after each short
leading clause that ends in a colon or semicolon, which is how a frame
introduces its payload; the caller scores those views too and keeps the
highest score. Sentence ends are not boundaries: on the same data they closed
no further attack and cost three times as many benign decisions.

Public API:
    recovered, applied = normalise(text)             -> (str, List[str])
    parts              = payloads(text)              -> List[str]
    views              = leading_clause_views(text)  -> List[str]
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import List, Tuple

# A run long enough to carry a sentence. Shorter runs match ordinary words.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

_LEET = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
}

# A document counts as leetspeak when at least this share of its word-like
# tokens carry a substitution, and at least this many do. Observed attacks run
# between 60% and 80%; prose with a stray "5pm" and a "3rd" sits far below.
_LEET_MIN_SHARE  = 0.30
_LEET_MIN_TOKENS = 3

# A decoded blob counts as text when it is mostly printable and has a space.
_PRINTABLE_MIN_SHARE = 0.90


def _looks_like_text(s: str) -> bool:
    """True when a decoded blob reads as language rather than bytes."""
    if not s or " " not in s:
        return False
    printable = sum(1 for ch in s if 32 <= ord(ch) < 127 or ch in "\n\t")
    return printable / len(s) >= _PRINTABLE_MIN_SHARE


def _decode_blob(blob: str):
    """The text a base64 run decodes to, or None when it is not readable text."""
    try:
        decoded = base64.b64decode(blob, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return decoded if _looks_like_text(decoded) else None


def _decode_base64(text: str) -> Tuple[str, bool]:
    """Replace base64 runs that decode to readable text with that text."""
    hit = False

    def _sub(match):
        nonlocal hit
        decoded = _decode_blob(match.group(0))
        if decoded is None:
            return match.group(0)
        hit = True
        return decoded

    return _B64_RUN.sub(_sub, text), hit


def _is_leeted(token: str) -> bool:
    """
    True when *token* carries a substitution rather than an ordinary digit.

    Trailing digits are stripped before the check, so "GPT4" and "COVID19" do
    not count while "G1v3", "c@5h" and "0ut" do. A token with no letter at all,
    such as a bare "4", never counts.
    """
    if not any(ch.isalpha() for ch in token):
        return False
    return any(ch in _LEET for ch in token.rstrip("0123456789"))


def _fold_leet(text: str) -> Tuple[str, bool]:
    """
    Fold leetspeak back to letters, but only for a document that is leeted
    throughout. Folding on a per-token basis would rewrite "5pm" in an
    otherwise ordinary sentence; the gate is deliberately document-level.
    """
    tokens = [t for t in text.split() if any(ch.isalpha() for ch in t)]
    if not tokens:
        return text, False
    leeted = sum(1 for t in tokens if _is_leeted(t))
    if leeted < _LEET_MIN_TOKENS or leeted / len(tokens) < _LEET_MIN_SHARE:
        return text, False
    folded = "".join(_LEET.get(ch, ch) for ch in text)
    return folded, folded != text


def payloads(text: str) -> List[str]:
    """
    Every base64 payload in *text* that decodes to readable text, each
    normalised in turn so a leetspeak payload inside base64 comes back folded.

    Empty for text with no such payload, which is almost all text: none of
    about 26,900 benign benchmark and pool prompts contains one.
    """
    if not text:
        return []
    out: List[str] = []
    for match in _B64_RUN.finditer(text):
        decoded = _decode_blob(match.group(0))
        if decoded is not None:
            out.append(normalise(decoded)[0])
    return out


# A leading instruction frame ends where its colon or semicolon does. The
# limits keep the view to short frames: the longest frame the red team uses is
# 14 words, a two-part frame needs two boundaries, and a tail of fewer than
# three words is not a request.
_CLAUSE_BOUNDARY       = re.compile(r"[:;]\s+")
_CLAUSE_MAX_LEAD_WORDS = 24
_CLAUSE_MAX_VIEWS      = 3
_CLAUSE_MIN_TAIL_WORDS = 3


def leading_clause_views(text: str) -> List[str]:
    """
    The text after each short leading clause that ends in a colon or semicolon.

    Measured with the shipped Layer 1 before any run (22 Sep 2026): the view
    recovers the request for every one of the 600 technical-prefix red-team
    variants, would move 154 of their 175 bypasses to mitigate and 3 to the
    judge, moves no decision on the pool's validation or test split, and moves
    21 of 4,597 safe ToxicChat prompts and 2 of 1,319 OR-Bench safe prompts.
    Most text has no view: 392 of the 4,597 ToxicChat prompts and about one
    pool prompt in twenty carry such a boundary.

    Controlled runs against the same baseline (22 Sep 2026): the technical-prefix
    bypass fell from 29.2% to 3.0% and the verified bypass rate over all 200
    HarmBench seeds from 13.3% to 4.6%; three benchmarks were unchanged,
    ToxicChat blocked 14 more safe prompts of 4,597 and caught 3 more harmful
    ones, OR-Bench moved 2 safe prompts. On by default since patch 34.

    Returns up to _CLAUSE_MAX_VIEWS views, one per boundary inside the first
    _CLAUSE_MAX_LEAD_WORDS words, each at least _CLAUSE_MIN_TAIL_WORDS words
    long. Like payloads(), this decides nothing: the caller keeps the highest
    score, so a view can raise a score but never lower one.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    out: List[str] = []
    for match in _CLAUSE_BOUNDARY.finditer(text):
        if len(text[:match.start()].split()) > _CLAUSE_MAX_LEAD_WORDS:
            break
        tail = text[match.end():].strip()
        if len(tail.split()) >= _CLAUSE_MIN_TAIL_WORDS and tail != text and tail not in out:
            out.append(tail)
        if len(out) >= _CLAUSE_MAX_VIEWS:
            break
    return out


def normalise(text: str) -> Tuple[str, List[str]]:
    """
    Recover *text* from surface-form obfuscation.

    Returns:
        (recovered_text, applied) where applied names the transforms that fired
        and is empty when the text was left alone. base64 runs true first, so a
        leetspeak payload hidden inside base64 is folded after decoding.
    """
    if not text or not text.strip():
        return text, []

    applied: List[str] = []
    out = text

    out, hit = _decode_base64(out)
    if hit:
        applied.append("base64")

    out, hit = _fold_leet(out)
    if hit:
        applied.append("leetspeak")

    return out, applied
