"""
Layer-1 Visualisation
======================
Generates diagnostic plots from the trained Layer-1 artifacts.

Which stacker is plotted
------------------------
Layer 1 ships two stackers in the same artifact directory and the config
chooses between them (layer1.classifier_only):

  fusion_v2.joblib        + scaler.json            three signals: fcurv, frank, g
  fusion_clf_only.joblib  + scaler_clf_only.json   the distilled classifier alone

They are different models and they produce different figures. `--stacker`
selects one; the default, `config`, follows layer1.classifier_only exactly as
layer1/detector.py does, so the plots describe the model that actually serves.
Every figure title and every output filename names the stacker it came from,
so two runs never overwrite or impersonate each other.

Plots produced (saved to <artifact-dir>/plots/, <name>_<stacker>.png):
  1. training_curves     — train/val loss + val accuracy over epochs
  2. roc_pr_curves       — ROC curve + Precision-Recall curve on test set
  3. score_distribution  — calibrated score histogram: allow vs mitigate
  4. feature_scatter     — fcurv vs frank scatter coloured by label
  5. feature_weights     — logistic stacker coefficients (feature importance)
  6. decision_routing    — allow / gray_zone / mitigate counts on test set

Reads:
  config/config.yaml                  (thresholds, artifact_subdir, classifier_only)
  <artifact-dir>/results.json
  <artifact-dir>/<stacker fusion .joblib>
  <artifact-dir>/<stacker scaler .json>
  data/processed/layer1_features/test_features.jsonl

Calibration and routing are not reimplemented here: layer1.fusion's
TemperatureScaler and ThresholdRouter are the ones used on the serving path.

Usage:
  python scripts/plot_layer1.py [--stacker {config,full,clf_only}] [--out-dir PATH]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from llmguard.layer1.detector import layer1_artifact_dir
from llmguard.layer1.fusion import (
    ClassifierOnlyFusion, LogisticFusion, TemperatureScaler, ThresholdRouter,
    DECISION_ALLOW, DECISION_GRAY_ZONE, DECISION_MITIGATE,
)
from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl
from llmguard.utils.logger import get_logger

logger = get_logger("plot_layer1")

# ── Colour palette ────────────────────────────────────────────────────────────
C_ALLOW    = "#2ecc71"   # green
C_MITIGATE = "#e74c3c"   # red
C_GRAY     = "#f39c12"   # amber
C_PRIMARY  = "#3498db"   # blue
C_DARK     = "#2c3e50"   # dark navy
C_NOTE     = "#7f8c8d"   # grey, for footnotes

DEFAULT_CONFIG_PATH  = "config/config.yaml"
# Decimals the pipeline serves a calibrated score to (layer1/detector.py rounds
# to 4, run_ablation.py fixes SCORE_DIGITS = 4). Figures quantise to the same
# precision so a plotted AUROC is the one the router would produce.
SCORE_DIGITS = 4

DEFAULT_FEATURES_PATH = "data/processed/layer1_features/test_features.jsonl"
# The scored cache carries the trained classifier's real g; the plain one carries
# the 0.5 placeholder train_layer1.py writes at step 3 and never updates, which
# makes any classifier-only figure degenerate. scripts/score_feature_cache.py
# writes the scored file, and it is preferred when present.
SCORED_FEATURES_PATH = "data/processed/layer1_features/test_scored.jsonl"


def default_features_path() -> str:
    """The scored feature cache when it exists, else the placeholder one."""
    return (SCORED_FEATURES_PATH if Path(SCORED_FEATURES_PATH).exists()
            else DEFAULT_FEATURES_PATH)

DECISION_ORDER = (DECISION_ALLOW, DECISION_GRAY_ZONE, DECISION_MITIGATE)


# ─────────────────────────────────────────────────────────────────────────────
# Which stacker
# ─────────────────────────────────────────────────────────────────────────────

STACKER_CONFIG   = "config"
STACKER_FULL     = "full"
STACKER_CLF_ONLY = "clf_only"


@dataclass(frozen=True)
class Stacker:
    """One of the two stackers Layer 1 ships, and how to name what it produced."""
    key:         str          # "full" | "clf_only"
    fusion_file: str          # joblib inside the artifact directory
    scaler_file: str          # temperature-scaler JSON beside it
    slug:        str          # filename tag, e.g. "clf_only"
    label:       str          # for figure titles
    features:    Tuple[str, ...]        # the stacker's own feature columns
    axis_labels: Tuple[str, ...]        # the same, spelled out for a bar chart

    def is_clf_only(self) -> bool:
        return self.key == STACKER_CLF_ONLY

    def new_fusion(self):
        """An unfitted wrapper of the right class, ready for .load()."""
        return ClassifierOnlyFusion() if self.is_clf_only() else LogisticFusion()


STACKERS: Dict[str, Stacker] = {
    STACKER_FULL: Stacker(
        key=STACKER_FULL,
        fusion_file="fusion_v2.joblib",
        scaler_file="scaler.json",
        slug="fusion_v2",
        label="three-signal stacker (fusion_v2)",
        features=("fcurv", "frank", "g"),
        axis_labels=("fcurv\n(curvature)", "frank\n(log-rank)", "g\n(classifier, P(allow))"),
    ),
    STACKER_CLF_ONLY: Stacker(
        key=STACKER_CLF_ONLY,
        fusion_file="fusion_clf_only.joblib",
        scaler_file="scaler_clf_only.json",
        slug="clf_only",
        label="classifier-only stacker (fusion_clf_only)",
        features=("g",),
        axis_labels=("g\n(classifier, P(allow))",),
    ),
}

#: Figure base names, in the order main() draws them.
FIGURE_NAMES = (
    "training_curves",
    "roc_pr_curves",
    "score_distribution",
    "feature_scatter",
    "feature_weights",
    "decision_routing",
)


def resolve_stacker(choice: str, cfg: dict) -> Stacker:
    """
    Turn --stacker into a Stacker.

    "config" reads layer1.classifier_only, the same switch
    Layer1Detector.from_artifacts_for uses, so the figures describe whichever
    model the shipped configuration loads.
    """
    if choice == STACKER_CONFIG:
        clf_only = bool(cfg.get("layer1", {}).get("classifier_only", False))
        choice = STACKER_CLF_ONLY if clf_only else STACKER_FULL
        logger.info(
            f"--stacker config → layer1.classifier_only={clf_only} → {choice}"
        )
    if choice not in STACKERS:
        raise ValueError(f"Unknown stacker {choice!r}; expected one of "
                         f"{[STACKER_CONFIG, *STACKERS]}")
    return STACKERS[choice]


def plot_path(out_dir: Path, name: str, stacker: Stacker) -> Path:
    """<out_dir>/<name>_<stacker slug>.png — the stacker is part of the filename."""
    return Path(out_dir) / f"{name}_{stacker.slug}.png"


def plot_paths(out_dir: Path, stacker: Stacker) -> Dict[str, Path]:
    return {name: plot_path(out_dir, name, stacker) for name in FIGURE_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Loading and scoring
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(artifact_dir: Path, stacker: Stacker):
    """
    Load results.json, the stacker for `stacker`, and its temperature scaler.

    Returns (results, fusion, scaler) where `fusion` is a fitted
    LogisticFusion / ClassifierOnlyFusion and `scaler` is a
    layer1.fusion.TemperatureScaler — the same classes the detector uses.
    """
    artifact_dir = Path(artifact_dir)
    fusion_path  = artifact_dir / stacker.fusion_file
    scaler_path  = artifact_dir / stacker.scaler_file

    for path in (artifact_dir / "results.json", fusion_path, scaler_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. The {stacker.label} cannot be plotted from "
                f"{artifact_dir}; train it first or pass --stacker for the other one."
            )

    results = json.loads((artifact_dir / "results.json").read_text())

    fusion = stacker.new_fusion()
    fusion.load(str(fusion_path))

    scaler = TemperatureScaler()
    scaler.load(str(scaler_path))

    return results, fusion, scaler


def _stacker_estimator(fusion):
    """
    The sklearn estimator inside a fusion wrapper.

    The wrapper only exposes a one-row raw_score(); reaching the estimator is
    what lets 1,912 rows go through predict_proba in a single call. Read-only,
    and the result is checked against raw_score() below.
    """
    clf = getattr(fusion, "_clf", None)
    if clf is None or not hasattr(clf, "predict_proba"):
        raise TypeError(
            f"{type(fusion).__name__} has no fitted sklearn estimator to batch "
            "through; load it with .load() before computing scores."
        )
    return clf


def feature_matrix(records: List[dict], fusion) -> np.ndarray:
    """
    (N, k) matrix in the column order this stacker was fitted on.

    The layout comes from fusion.feature_vector, so the three-signal and
    classifier-only stackers are each fed exactly what they expect.
    """
    rows = [
        fusion.feature_vector(
            float(r.get("fcurv", 0.5)),
            float(r.get("frank", 0.5)),
            float(r.get("g", 0.5)),
        )
        for r in records
    ]
    return np.asarray(rows, dtype=float)


def compute_scores(records: List[dict], fusion, scaler: TemperatureScaler):
    """
    Raw and calibrated scores for every record, in one batched predict_proba.

    Uses the cached fcurv / frank / g values from the feature JSONL, the loaded
    stacker, and layer1.fusion.TemperatureScaler for calibration.

    Returns (raw_scores, cal_scores, labels) as numpy arrays.
    """
    if not records:
        return np.array([]), np.array([]), np.array([])

    feats = feature_matrix(records, fusion)
    clf   = _stacker_estimator(fusion)

    expected = getattr(clf, "n_features_in_", feats.shape[1])
    if feats.shape[1] != expected:
        raise ValueError(
            f"{type(fusion).__name__} builds {feats.shape[1]} feature column(s) but "
            f"the loaded estimator was fitted on {expected}. The fusion file and the "
            "stacker class disagree — check --stacker."
        )

    allow_idx  = list(clf.classes_).index(1)
    raw_scores = 1.0 - clf.predict_proba(feats)[:, allow_idx]

    # Guard the batching shortcut: the batched column must equal the shipped
    # per-row path on a sample of rows.
    for i in range(min(8, len(records))):
        r = records[i]
        one = fusion.raw_score(
            float(r.get("fcurv", 0.5)), float(r.get("frank", 0.5)), float(r.get("g", 0.5))
        )
        if not np.isclose(one, raw_scores[i], atol=1e-9):
            raise AssertionError(
                f"Batched score {raw_scores[i]!r} disagrees with "
                f"{type(fusion).__name__}.raw_score {one!r} on record {i}."
            )

    # Quantised to the precision the pipeline actually serves. DetectionResult
    # rounds calibrated_score to 4 decimals and run_ablation.py fixes
    # SCORE_DIGITS = 4 for the same reason: at full precision the ties that a
    # saturated score distribution produces are broken arbitrarily, which moves
    # AUROC. On the pool test split that difference is 0.9937 unrounded against
    # 0.9914 as routed, so an unrounded figure would flatter the model by more
    # than several of the comparisons this project reports.
    cal_scores = np.round(
        np.asarray(scaler.scale_batch(raw_scores.tolist()), dtype=float), SCORE_DIGITS)
    labels     = np.asarray([int(r["label"]) for r in records], dtype=int)
    return raw_scores, cal_scores, labels


def placeholder_caveat(records: List[dict], stacker: Stacker) -> Optional[str]:
    """
    Warn when the feature cache cannot support the figures.

    scripts/train_layer1.py writes fcurv and frank to
    data/processed/layer1_features/*.jsonl with g fixed at the 0.5 placeholder,
    and the trained classifier score is never written back. A cache in that
    state carries no classifier signal, so the classifier-only stacker scores
    every prompt identically and the three-signal stacker is missing its
    strongest feature. Say so on the figure rather than let it be read as a
    result.
    """
    if not records:
        return None
    g_values = {round(float(r.get("g", 0.5)), 9) for r in records}
    if len(g_values) > 1:
        return None
    only = next(iter(g_values))
    if only != 0.5:
        return None
    if stacker.is_clf_only():
        return ("CAVEAT: g is the constant 0.5 placeholder in this feature cache, so "
                "this stacker scores every prompt identically. Not a result.")
    return ("CAVEAT: g is the constant 0.5 placeholder in this feature cache, so the "
            "classifier signal is absent and only fcurv and frank vary. Not a result.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _title(fig, base: str, stacker: Stacker) -> None:
    """Suptitle that always names the stacker the figure came from."""
    fig.suptitle(f"{base} — {stacker.label}", fontsize=13,
                 fontweight="bold", color=C_DARK)


def _footnote(fig, *lines: Optional[str]) -> None:
    text = "\n".join(l for l in lines if l)
    if not text:
        return
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=8,
             style="italic", color=C_NOTE)


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_path: Path, stacker: Stacker) -> None:
    """
    Train/val loss and val accuracy across epochs.

    These curves come from the distilled classifier's own training history, not
    from either stacker; the stacker is named so the file cannot be mixed up
    with the other run's.
    """
    epochs     = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    val_acc    = history["val_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Layer-1 Classifier Training Curves", fontsize=13,
                 fontweight="bold", color=C_DARK)

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", color=C_PRIMARY,  label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss,   marker="s", color=C_MITIGATE, label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    ax = axes[1]
    ax.plot(epochs, val_acc, marker="D", color=C_ALLOW, linewidth=2, label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.set_xticks(epochs)
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    _footnote(
        fig,
        f"Classifier training history from results.json. Filed with the "
        f"{stacker.label} figures; the stacker does not affect these curves.",
    )
    _save(fig, out_path)


def plot_roc_pr(cal_scores: np.ndarray, labels: np.ndarray, out_path: Path,
                stacker: Stacker, caveat: Optional[str] = None) -> None:
    """ROC and Precision-Recall curves for this stacker on the test set."""
    # Flip: mitigate = positive class (label=0 → positive = 1-label)
    pos_labels = 1 - labels

    fpr, tpr, _  = roc_curve(pos_labels, cal_scores)
    auroc        = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(pos_labels, cal_scores)
    auprc        = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _title(fig, "Layer-1 Test Set — ROC & Precision-Recall", stacker)

    ax = axes[0]
    ax.plot(fpr, tpr, color=C_PRIMARY, linewidth=2.5,
            label=f"ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C_PRIMARY)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    ax = axes[1]
    ax.plot(rec, prec, color=C_MITIGATE, linewidth=2.5,
            label=f"PR Curve (AUPRC = {auprc:.4f})")
    ax.fill_between(rec, prec, alpha=0.08, color=C_MITIGATE)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    _footnote(
        fig,
        f"Recomputed from the cached test features through {stacker.fusion_file} "
        f"and {stacker.scaler_file}.",
        caveat,
    )
    _save(fig, out_path)


def plot_score_distribution(cal_scores: np.ndarray, labels: np.ndarray,
                            tau_low: float, tau_high: float, out_path: Path,
                            stacker: Stacker, caveat: Optional[str] = None) -> None:
    """Histogram of calibrated scores split by true class."""
    allow_scores    = cal_scores[labels == 1]
    mitigate_scores = cal_scores[labels == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    _title(fig, "Layer-1 Score Distribution (Calibrated)", stacker)

    bins = np.linspace(0, 1, 50)
    ax.hist(allow_scores,    bins=bins, alpha=0.65, color=C_ALLOW,
            label=f"Allow    (n={len(allow_scores):,})", edgecolor="white")
    ax.hist(mitigate_scores, bins=bins, alpha=0.65, color=C_MITIGATE,
            label=f"Mitigate (n={len(mitigate_scores):,})", edgecolor="white")

    # Threshold bands — from config, via ThresholdRouter
    ax.axvline(tau_low,  color=C_GRAY, linewidth=2, linestyle="--", label=f"τ_low  = {tau_low}")
    ax.axvline(tau_high, color=C_DARK, linewidth=2, linestyle="--", label=f"τ_high = {tau_high}")
    ax.axvspan(tau_low, tau_high, alpha=0.06, color=C_GRAY, label="Gray Zone")

    ax.set_xlabel("Calibrated Score  (higher → more harmful)")
    ax.set_ylabel("Count")
    ax.set_title("Calibrated Score Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    _footnote(
        fig,
        f"Calibrated with {stacker.scaler_file}; thresholds from "
        f"layer1.thresholds in config.",
        caveat,
    )
    _save(fig, out_path)


def plot_feature_scatter(records: List[dict], out_path: Path,
                         stacker: Stacker, sample: int = 2000) -> None:
    """2D scatter of fcurv vs frank coloured by true label."""
    allow_fc = [r["fcurv"] for r in records if r["label"] == 1]
    allow_fr = [r["frank"] for r in records if r["label"] == 1]
    mit_fc   = [r["fcurv"] for r in records if r["label"] == 0]
    mit_fr   = [r["frank"] for r in records if r["label"] == 0]

    rng = np.random.default_rng(42)

    def _sample(lst, n):
        if not lst:
            return []
        idx = rng.choice(len(lst), size=min(n, len(lst)), replace=False)
        return [lst[i] for i in idx]

    n_allow = min(sample, len(allow_fc))
    n_mit   = min(sample, len(mit_fc))

    fig, ax = plt.subplots(figsize=(8, 6))
    _title(fig, "Layer-1 Feature Space — fcurv vs frank", stacker)

    ax.scatter(_sample(allow_fc, sample), _sample(allow_fr, sample),
               alpha=0.25, s=12, color=C_ALLOW,    label=f"Allow (n={n_allow:,})")
    ax.scatter(_sample(mit_fc,   sample), _sample(mit_fr, sample),
               alpha=0.25, s=12, color=C_MITIGATE, label=f"Mitigate (n={n_mit:,})")

    ax.set_xlabel("fcurv  (curvature / perplexity signal)")
    ax.set_ylabel("frank  (log-rank signal)")
    ax.set_title("Feature Scatter (cached test features)")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    if stacker.is_clf_only():
        note = ("Both axes are inputs the classifier-only stacker does not read: "
                "it scores on g alone.")
    else:
        note = "Two of the three inputs the three-signal stacker reads."

    fig.tight_layout()
    _footnote(fig, note)
    _save(fig, out_path)


def plot_feature_weights(fusion, out_path: Path, stacker: Stacker) -> None:
    """Bar chart of the loaded stacker's logistic coefficients."""
    clf       = _stacker_estimator(fusion)
    coef      = np.asarray(clf.coef_[0], dtype=float)
    intercept = float(clf.intercept_[0])

    if len(coef) == len(stacker.axis_labels):
        names = list(stacker.axis_labels)
    else:
        # A fusion file with an unexpected width: label the columns honestly
        # rather than borrow this stacker's names for them.
        names = [f"w{i + 1}" for i in range(len(coef))]
        logger.warning(
            f"{stacker.fusion_file} has {len(coef)} coefficient(s) but "
            f"{stacker.key} expects {len(stacker.axis_labels)}; labelling generically."
        )

    # sklearn was fitted with 1=allow, so a positive coefficient pushes toward
    # allow. Flip the sign for a chart whose axis reads "toward harm".
    toward_harm = -coef
    colours     = [C_MITIGATE if c > 0 else C_ALLOW for c in toward_harm]

    fig, ax = plt.subplots(figsize=(7, 5))
    _title(fig, "Layer-1 Logistic Stacker — Feature Weights", stacker)

    bars = ax.bar(names, toward_harm, color=colours, edgecolor="white",
                  width=0.5 if len(names) > 1 else 0.3)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Coefficient  (positive → more harmful)")
    ax.set_title(f"Logistic Weights  (intercept toward harm = {-intercept:.3f})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fa")

    span = float(np.max(np.abs(toward_harm))) if len(toward_harm) else 1.0
    ax.set_ylim(min(0.0, float(np.min(toward_harm))) - 0.22 * span,
                max(0.0, float(np.max(toward_harm))) + 0.22 * span)
    for bar, val in zip(bars, toward_harm):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.03 * span if val >= 0 else -0.03 * span),
                f"{val:.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color=C_MITIGATE, lw=8, label="Positively correlated with harm"),
        Line2D([0], [0], color=C_ALLOW,    lw=8, label="Negatively correlated with harm"),
    ]
    ax.legend(handles=legend_elements, loc="best", framealpha=0.92)

    fig.tight_layout()
    _footnote(
        fig,
        f"Read from {stacker.fusion_file}. sklearn fits toward label 1 = allow, so the "
        f"signs are flipped here to make up mean 'toward harm'; g is itself P(allow), so "
        f"its bar pointing down is the expected direction, not a weak signal.",
    )
    _save(fig, out_path)


def plot_decision_routing(cal_scores: np.ndarray, labels: np.ndarray,
                          tau_low: float, tau_high: float, out_path: Path,
                          stacker: Stacker, caveat: Optional[str] = None) -> None:
    """Stacked bar chart of routing decisions per true class."""
    router    = ThresholdRouter(tau_low=tau_low, tau_high=tau_high)
    decisions = router.route_batch([float(s) for s in cal_scores])

    for_allow    = [d for d, l in zip(decisions, labels) if l == 1]
    for_mitigate = [d for d, l in zip(decisions, labels) if l == 0]

    def counts(lst):
        return {k: lst.count(k) for k in DECISION_ORDER}

    ca = counts(for_allow)
    cm = counts(for_mitigate)

    cats   = ["Allow (true)", "Mitigate (true)"]
    allows = [ca[DECISION_ALLOW],     cm[DECISION_ALLOW]]
    grays  = [ca[DECISION_GRAY_ZONE], cm[DECISION_GRAY_ZONE]]
    mits   = [ca[DECISION_MITIGATE],  cm[DECISION_MITIGATE]]
    totals = [len(for_allow), len(for_mitigate)]

    x = np.arange(len(cats))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    _title(fig, "Layer-1 Decision Routing on Test Set", stacker)

    ax.bar(x, allows, w, label="→ Allow",     color=C_ALLOW,    edgecolor="white")
    ax.bar(x, grays,  w, label="→ Gray Zone", color=C_GRAY,     edgecolor="white",
           bottom=allows)
    ax.bar(x, mits,   w, label="→ Mitigate",  color=C_MITIGATE, edgecolor="white",
           bottom=[a + g for a, g in zip(allows, grays)])

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Routing Distribution per True Class  "
                 f"(τ_low={tau_low}, τ_high={tau_high})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fa")

    # Percentage labels
    for xi, (a, g, m, tot) in enumerate(zip(allows, grays, mits, totals)):
        if tot <= 0:
            continue
        for base, size in ((0, a), (a, g), (a + g, m)):
            if size / tot < 0.04:      # too thin to hold a label
                continue
            ax.text(xi, base + size / 2, f"{100 * size / tot:.0f}%", ha="center",
                    va="center", fontsize=9, fontweight="bold", color="white")

    fig.tight_layout()
    _footnote(
        fig,
        f"Routed by layer1.fusion.ThresholdRouter on {stacker.scaler_file}-calibrated "
        f"scores; thresholds from layer1.thresholds in config.",
        caveat,
    )
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — shared with scripts/generate_report.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlotRun:
    """Everything one plotting run loaded, computed and wrote."""
    stacker:       Stacker
    artifact_dir:  Path
    features_path: Path
    out_dir:       Path
    config_path:   Path
    results:       dict
    fusion:        object
    scaler:        TemperatureScaler
    tau_low:       float
    tau_high:      float
    records:       List[dict]      = field(default_factory=list)
    raw_scores:    np.ndarray      = field(default_factory=lambda: np.array([]))
    cal_scores:    np.ndarray      = field(default_factory=lambda: np.array([]))
    labels:        np.ndarray      = field(default_factory=lambda: np.array([]))
    caveat:        Optional[str]   = None
    paths:         Dict[str, Path] = field(default_factory=dict)

    def routing_counts(self) -> Dict[str, int]:
        """allow / gray_zone / mitigate counts over the whole test set."""
        router    = ThresholdRouter(tau_low=self.tau_low, tau_high=self.tau_high)
        decisions = router.route_batch([float(s) for s in self.cal_scores])
        return {k: decisions.count(k) for k in DECISION_ORDER}


def resolve_paths(cfg: dict, artifact_dir: Optional[str],
                  out_dir: Optional[str]) -> Tuple[Path, Path]:
    """
    Where the artifacts are and where the plots go.

    --artifact-dir overrides; otherwise layer1.artifact_subdir decides, through
    layer1.detector.layer1_artifact_dir, so this script and the detector never
    read different models. --out-dir defaults to <artifact-dir>/plots.
    """
    resolved_artifacts = Path(artifact_dir) if artifact_dir else layer1_artifact_dir(cfg)
    resolved_out       = Path(out_dir) if out_dir else resolved_artifacts / "plots"
    return resolved_artifacts, resolved_out


def prepare_run(cfg: dict, stacker: Stacker, artifact_dir: Path, out_dir: Path,
                features_path: Path, config_path: Path) -> PlotRun:
    """Load artifacts + features and score them. Draws nothing."""
    results, fusion, scaler = load_artifacts(artifact_dir, stacker)

    thresholds = cfg["layer1"]["thresholds"]
    tau_low    = float(thresholds["low"])
    tau_high   = float(thresholds["high"])

    snap = (results.get("config_snapshot") or {}).get("thresholds") or {}
    if snap and (float(snap.get("low", tau_low))  != tau_low
                 or float(snap.get("high", tau_high)) != tau_high):
        logger.warning(
            f"config thresholds ({tau_low}, {tau_high}) differ from the ones recorded "
            f"in results.json ({snap.get('low')}, {snap.get('high')}). Plotting the "
            "config's, which is what the detector routes on."
        )

    logger.info(f"  Stacker : {stacker.label}")
    logger.info(f"  T* = {scaler.T:.6f} | τ_low = {tau_low} | τ_high = {tau_high}")

    records = load_jsonl(str(features_path))
    logger.info(f"  {len(records)} test feature records loaded from {features_path}")

    raw_scores, cal_scores, labels = compute_scores(records, fusion, scaler)

    caveat = placeholder_caveat(records, stacker)
    if caveat:
        logger.warning(caveat)

    return PlotRun(
        stacker=stacker, artifact_dir=Path(artifact_dir),
        features_path=Path(features_path), out_dir=Path(out_dir),
        config_path=Path(config_path), results=results, fusion=fusion, scaler=scaler,
        tau_low=tau_low, tau_high=tau_high, records=records,
        raw_scores=raw_scores, cal_scores=cal_scores, labels=labels,
        caveat=caveat, paths=plot_paths(out_dir, stacker),
    )


def draw_all(run: PlotRun) -> Dict[str, Path]:
    """Draw all six figures for `run` and return {figure name: path}."""
    run.out_dir.mkdir(parents=True, exist_ok=True)
    s, cav = run.stacker, run.caveat

    logger.info("Plotting training curves...")
    plot_training_curves(run.results["training_history"], run.paths["training_curves"], s)

    logger.info("Plotting ROC and PR curves...")
    plot_roc_pr(run.cal_scores, run.labels, run.paths["roc_pr_curves"], s, cav)

    logger.info("Plotting score distribution...")
    plot_score_distribution(run.cal_scores, run.labels, run.tau_low, run.tau_high,
                            run.paths["score_distribution"], s, cav)

    logger.info("Plotting feature scatter...")
    plot_feature_scatter(run.records, run.paths["feature_scatter"], s)

    logger.info("Plotting feature weights...")
    plot_feature_weights(run.fusion, run.paths["feature_weights"], s)

    logger.info("Plotting decision routing...")
    plot_decision_routing(run.cal_scores, run.labels, run.tau_low, run.tau_high,
                          run.paths["decision_routing"], s, cav)

    return dict(run.paths)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate Layer-1 diagnostic plots")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="config to read thresholds, artifact_subdir and classifier_only from")
    p.add_argument("--artifact-dir", default=None,
                   help="override layer1.artifact_subdir (default: artifacts/<layer1.artifact_subdir>)")
    p.add_argument("--features-path", default=None,
                   help="Cached feature records to plot. Defaults to "
                        f"{SCORED_FEATURES_PATH} when it exists, else "
                        f"{DEFAULT_FEATURES_PATH}, whose g is a constant placeholder.")
    p.add_argument("--out-dir", default=None,
                   help="where the PNGs go (default: <artifact-dir>/plots)")
    p.add_argument("--stacker", choices=[STACKER_CONFIG, STACKER_FULL, STACKER_CLF_ONLY],
                   default=STACKER_CONFIG,
                   help="which stacker to plot; 'config' follows layer1.classifier_only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg  = load_config(args.config)

    stacker = resolve_stacker(args.stacker, cfg)
    artifact_dir, out_dir = resolve_paths(cfg, args.artifact_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("Layer-1 Visualisation")
    logger.info(f"  Config    : {Path(args.config).resolve()}")
    logger.info(f"  Artifacts : {artifact_dir.resolve()}")
    features_path = args.features_path or default_features_path()
    logger.info(f"  Features  : {features_path}"
                + ("  (trained classifier scores)"
                   if str(features_path) == SCORED_FEATURES_PATH else ""))
    logger.info(f"  Output    : {out_dir.resolve()}")
    logger.info("=" * 55)

    run = prepare_run(cfg, stacker, artifact_dir, out_dir,
                      Path(features_path), Path(args.config))

    metrics = run.results.get("test_metrics", {})
    if metrics:
        logger.info(
            "  results.json test_metrics (three-signal fusion_v2 at training time): "
            f"AUROC={metrics.get('auroc')} AUPRC={metrics.get('auprc')} "
            f"Acc={metrics.get('accuracy')} n={metrics.get('n_samples')}"
        )

    draw_all(run)

    logger.info("=" * 55)
    logger.info(f"All plots for the {stacker.label}:")
    for name in FIGURE_NAMES:
        logger.info(f"  {run.paths[name]}")
    logger.info("=" * 55)
    return run


if __name__ == "__main__":
    main()
