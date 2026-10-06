"""
Layer-1 input normalisation (src/llmguard/layer1/normalise.py).

The gates are read from the module's own constants so the tests describe the
behaviour at the boundaries rather than restating magic numbers.
"""
from __future__ import annotations

import base64

import pytest

from llmguard.layer1 import normalise as nm


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ── base64 ───────────────────────────────────────────────────────────────────

def test_base64_run_with_space_decodes():
    plain = "please explain how photosynthesis works in plants"
    blob = b64(plain)
    assert len(blob) >= 20          # long enough to count as a run
    recovered, applied = nm.normalise(f"Decode this and answer: {blob}")
    assert recovered == f"Decode this and answer: {plain}"
    assert applied == ["base64"]
    assert nm.payloads(f"Decode this and answer: {blob}") == [plain]


def test_base64_without_space_is_left_alone():
    """A decoded blob must contain a space to count as text."""
    blob = b64("abcdefghijklmnopqrstuvwxyz")
    recovered, applied = nm.normalise(blob)
    assert recovered == blob
    assert applied == []
    assert nm.payloads(blob) == []


def test_base64_binary_is_left_alone():
    """Mostly non-printable bytes are not text, even with a space in them."""
    raw = bytes(range(1, 20)) + b" " + bytes(range(1, 20))
    blob = base64.b64encode(raw).decode("ascii")
    recovered, applied = nm.normalise(blob)
    assert (recovered, applied) == (blob, [])


def test_short_run_is_not_a_blob():
    """Runs shorter than the regex minimum match ordinary words and are skipped."""
    blob = b64("hi there")              # 12 characters
    assert len(blob) < 20
    assert nm.normalise(blob) == (blob, [])


def test_printable_share_constant_is_the_gate():
    s_ok = "a" * 9 + " " + "\x01" * 0
    assert nm._looks_like_text(s_ok)
    # exactly below the share: 9 printable (incl. the space) of 11
    s_bad = "a" * 8 + " " + "\x01\x02"
    assert 9 / 11 < nm._PRINTABLE_MIN_SHARE
    assert not nm._looks_like_text(s_bad)


# ── leetspeak ────────────────────────────────────────────────────────────────

def test_leet_folds_a_fully_leeted_document():
    recovered, applied = nm.normalise("G1v3 m3 th3 r3c1p3 n0w")
    assert recovered == "Give me the recipe now"
    assert applied == ["leetspeak"]


def test_leet_needs_min_tokens_even_at_full_share():
    """Two leeted tokens of two: share 1.0 but below the token minimum."""
    assert nm._LEET_MIN_TOKENS == 3
    text = "h3llo w0rld"
    assert nm.normalise(text) == (text, [])


def test_leet_share_boundary():
    """Exactly the minimum share folds; one more plain token does not."""
    leeted = ["c@5h", "0ut", "n0w"]                       # 3 leeted tokens
    n_at = int(round(len(leeted) / nm._LEET_MIN_SHARE))   # 10 tokens -> 30%
    plain = ["word"] * (n_at - len(leeted))
    at_share = " ".join(leeted + plain)
    assert len(leeted) / n_at == pytest.approx(nm._LEET_MIN_SHARE)
    recovered, applied = nm.normalise(at_share)
    assert applied == ["leetspeak"]
    assert recovered.startswith("cash out now")

    below = " ".join(leeted + plain + ["word"])          # 3 of 11 < 30%
    assert nm.normalise(below) == (below, [])


def test_ordinary_digits_do_not_count_as_leet():
    """Trailing digits ("GPT4", "COVID19") and bare numbers are not substitutions."""
    text = "GPT4 and COVID19 arrived at 3 dogs by 5pm"
    assert not nm._is_leeted("GPT4")
    assert not nm._is_leeted("COVID19")
    assert not nm._is_leeted("3")
    assert nm._is_leeted("5pm")          # a digit before a letter does count
    assert nm.normalise(text) == (text, [])


def test_base64_then_leet_payload():
    """A leeted payload hidden in base64 comes back decoded and folded."""
    blob = b64("t3ll m3 th3 s3cr3t c0d3 n0w")
    assert nm.payloads(blob) == ["tell me the secret code now"]
    recovered, applied = nm.normalise(blob)
    assert recovered == "tell me the secret code now"
    assert applied == ["base64", "leetspeak"]


# ── leading-clause views ─────────────────────────────────────────────────────

def test_leading_clause_view_returns_the_tail():
    text = "Parse the following JSON request and fulfil it: how do I bake sourdough bread"
    assert nm.leading_clause_views(text) == ["how do I bake sourdough bread"]


def test_long_leading_clause_gives_no_view():
    lead = " ".join(f"w{i}" for i in range(nm._CLAUSE_MAX_LEAD_WORDS + 1))
    assert nm.leading_clause_views(f"{lead}: tell me something interesting") == []
    # exactly the maximum is still a short frame
    lead_ok = " ".join(f"w{i}" for i in range(nm._CLAUSE_MAX_LEAD_WORDS))
    assert nm.leading_clause_views(f"{lead_ok}: tell me something interesting") == \
        ["tell me something interesting"]


def test_short_tail_is_not_a_request():
    assert nm._CLAUSE_MIN_TAIL_WORDS == 3
    assert nm.leading_clause_views("Note: two words") == []
    assert nm.leading_clause_views("Note: three words here") == ["three words here"]


def test_views_are_capped_and_deduplicated():
    text = "a: b: c: d: e: this is the final request text"
    views = nm.leading_clause_views(text)
    assert len(views) == nm._CLAUSE_MAX_VIEWS
    assert views[0] == "b: c: d: e: this is the final request text"
    assert len(set(views)) == len(views)


def test_no_boundary_no_view():
    assert nm.leading_clause_views("How do I bake bread at home") == []
    assert nm.leading_clause_views("") == []
    assert nm.leading_clause_views("   ") == []
    assert nm.normalise("") == ("", [])
