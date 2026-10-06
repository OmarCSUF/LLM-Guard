"""
Metric and interval arithmetic on hand-computable cases.

Label convention throughout llmguard: 1 = safe (allow), 0 = harmful (mitigate);
the harm score is higher for more harmful, and harmful is the positive class.
"""
from __future__ import annotations

import json
import math

import pytest

from llmguard.evaluation import metrics as M

from conftest import load_script


# ── ECE ──────────────────────────────────────────────────────────────────────

def test_ece_hand_case():
    """
    Two bins: scores 0.1 (bin 1) for two safe prompts, 0.9 (bin 9) for two
    harmful ones. Each bin is off by 0.1, so ECE = (2*0.1 + 2*0.1) / 4 = 0.1.
    """
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.1, 0.9, 0.9]
    assert M.ece(y_true, y_score) == pytest.approx(0.1)


def test_ece_perfectly_calibrated_bin():
    # one harmful of four at score 0.25 in one bin: |0.25 - 0.25| = 0
    assert M.ece([0, 1, 1, 1], [0.25] * 4) == pytest.approx(0.0)


def test_ece_score_one_goes_in_top_bin():
    # 1.0 and 0.95 share bin 9: mean 0.975, all harmful -> 0.025
    assert M.ece([0, 0], [1.0, 0.95]) == pytest.approx(0.025)


def test_ece_undefined_cases():
    assert M.ece([], []) is None
    assert M.ece([0, 1], [0.5, 1.5]) is None
    assert M.ece([0, 1], [-0.1, 0.5]) is None


# ── FPR @ TPR ────────────────────────────────────────────────────────────────

Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 1]
Y_SCORE = [0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.4, 0.75]


@pytest.mark.parametrize("target, expected", [
    (0.95, 0.75),   # all four harmful needed -> threshold 0.2 -> 3 of 4 safe above it
    (0.75, 0.25),   # three harmful -> threshold 0.7 -> only 0.75 above it
    (0.50, 0.0),    # two harmful -> threshold 0.8 -> no safe prompt above it
])
def test_fpr_at_tpr(target, expected):
    assert M.fpr_at_tpr(Y_TRUE, Y_SCORE, target) == pytest.approx(expected)


def test_fpr_at_tpr_single_class_is_none():
    assert M.fpr_at_tpr([0, 0, 0], [0.1, 0.5, 0.9]) is None
    assert M.auroc([1, 1], [0.1, 0.9]) is None
    assert M.auprc([0, 0], [0.1, 0.9]) is None


def test_auroc_hand_case():
    # one harmful (0.6) above two of three safe (0.2, 0.4) and below one (0.8): 2/3
    assert M.auroc([0, 1, 1, 1], [0.6, 0.2, 0.4, 0.8]) == pytest.approx(2 / 3)


# ── compute_all ──────────────────────────────────────────────────────────────

EXPECTED_KEYS = {
    "n_total", "n_harmful", "n_safe", "n_escalated", "escalation_rate",
    "accuracy", "precision", "recall", "f1", "auroc", "auprc",
    "fpr_at_95tpr", "ece", "jailbreak_success_rate",
}


def test_compute_all_key_set_and_counts():
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0]      # one harmful allowed, one safe blocked
    out = M.compute_all(Y_TRUE, Y_SCORE, y_pred)
    assert set(out) == EXPECTED_KEYS
    assert (out["n_total"], out["n_harmful"], out["n_safe"], out["n_escalated"]) == (8, 4, 4, 0)
    assert out["escalation_rate"] == 0.0
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["precision"] == pytest.approx(0.75)      # 3 of 4 blocks are harmful
    assert out["recall"] == pytest.approx(0.75)         # 3 of 4 harmful blocked
    assert out["jailbreak_success_rate"] == pytest.approx(0.25)
    assert out["fpr_at_95tpr"] == pytest.approx(0.75)


def test_compute_all_escalated_are_excluded_from_accuracy():
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0]
    esc = [False, False, True, False, False, False, False, True]
    out = M.compute_all(Y_TRUE, Y_SCORE, y_pred, escalated=esc)
    assert out["n_escalated"] == 2
    assert out["escalation_rate"] == pytest.approx(0.25)
    assert out["accuracy"] == pytest.approx(1.0)       # the two mistakes were escalated
    with pytest.raises(ValueError):
        M.compute_all(Y_TRUE, Y_SCORE, y_pred, escalated=[True])


def test_labels_are_validated():
    with pytest.raises(ValueError):
        M.accuracy([0, 2], [0, 1])
    assert M.format_metric(None) == "n/a"
    assert M.format_metric(0.123456) == "0.1235"


# ── Wilson interval (scripts/final_eval/analyse_final.py) ────────────────────

@pytest.fixture(scope="module")
def analyse_final():
    return load_script("scripts/final_eval/analyse_final.py")


@pytest.fixture(scope="module")
def stored_results(study_dir):
    return json.loads((study_dir / "analysis" / "results.json").read_text(encoding="utf-8"))


def test_wilson_reproduces_stored_cascade_sst(analyse_final, stored_results):
    """rates.cascade.sst: n=99, k=5 -> [0.02176, 0.11283] in the stored analysis."""
    e = stored_results["rates"]["cascade"]["sst"]
    assert (e["n"], e["k"]) == (99, 5)
    lo, hi = analyse_final.wilson(e["k"], e["n"])
    assert lo == pytest.approx(0.021763234353651656, rel=1e-12)
    assert hi == pytest.approx(0.11282702751021786, rel=1e-12)
    assert [lo, hi] == pytest.approx(e["ci"], rel=1e-12)
    assert e["rate"] == pytest.approx(5 / 99)


def test_wilson_matches_every_stored_rate(analyse_final, stored_results):
    for arm, per_source in stored_results["rates"].items():
        for src, e in per_source.items():
            lo, hi = analyse_final.wilson(e["k"], e["n"])
            assert [lo, hi] == pytest.approx(e["ci"], rel=1e-12), (arm, src)


def test_wilson_edges(analyse_final):
    assert analyse_final.wilson(0, 10)[0] == 0.0          # clamped at 0
    # k == n: the upper bound is 1 up to rounding (c + h lands at 1 - 1e-16)
    assert analyse_final.wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert analyse_final.wilson(10, 10)[1] <= 1.0
    lo, hi = analyse_final.wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# ── exact McNemar (scripts/compare_flips.py) ─────────────────────────────────

@pytest.mark.parametrize("b, c, p", [
    (0, 0, 1.0),
    (1, 0, 1.0),            # n=1: 2 * C(1,0)/2 = 1
    (3, 0, 0.25),           # n=3: 2 * 1/8
    (5, 1, 0.21875),        # n=6: 2 * (1 + 6)/64
    (8, 2, 0.109375),       # n=10: 2 * (1 + 10 + 45)/1024
    (2, 8, 0.109375),       # symmetric
])
def test_mcnemar_exact(b, c, p):
    compare_flips = load_script("scripts/compare_flips.py")
    assert compare_flips.mcnemar_exact(b, c) == pytest.approx(p, rel=1e-12)


def test_mcnemar_implementations_agree(analyse_final):
    compare_flips = load_script("scripts/compare_flips.py")
    for b in range(0, 12):
        for c in range(0, 12):
            assert compare_flips.mcnemar_exact(b, c) == analyse_final.mcnemar_exact(b, c)
