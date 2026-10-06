"""
GaurdLLM — Evaluation Plots
============================
Figures for Layers 1, 2, 3 and the red team, built only from measurements on disk.

Every number in every figure is read at run time from a results file, a per-prompt
predictions file, or config/config.yaml. Nothing is simulated, approximated or
hard-coded. When the data a figure needs is absent the figure is SKIPPED and named
in the summary; it is never drawn from invented numbers.

That rule is the point of this script. An earlier version drew a "score
distribution" from two Gaussians it made up, derived a routing pie by multiplying
accuracy by 0.55, asserted a 87/10/3 vote split from nothing, and hard-coded
n=10,000, T*=9.88, c_min=0.6, a 0.85 refusal threshold and the counts 106/94 into
figure labels. Several of those contradicted the shipped configuration and none of
them could be traced to a run. They are gone.

Data sources, and which figures need them:

  --layer1 RESULTS       artifacts/<subdir>/results.json from train_layer1.py
                         training curves, test-metric bars
  --layer2 RESULTS       artifacts/layer2/results.json from run_layer2.py
                         judge decision pie, judge metric bars, category bars
  --layer3 RESULTS       artifacts/layer3/results.json from run_layer3.py
                         mitigation mode pie, summary bars, category bars
  --layer3-jsonl FILE    per-sample Layer-3 output written by run_layer3.py --output
                         severity histogram
  --predictions PATH     a *_predictions/<benchmark>.jsonl file or directory from
                         run_evaluation.py
                         Layer-1 score distribution, Layer-1 routing, judge
                         confidence distribution
  --eval-results FILE    a run_evaluation.py results JSON, read for layer_stats
                         pipeline flow diagrams
  --redteam FILE         a run_red_team.py report JSON
                         bypass rates, attack volume, red-team flow

A manifest.json is written beside the figures recording, for each figure, the
source files it was built from with their SHA-256, and the config values used, so
a reader can retrace any figure to the run that produced it.

Usage:
    python scripts/plot_evaluation.py \
        --predictions data/processed/final_eval_20260926/final_cascade_predictions \
        --eval-results data/processed/final_eval_20260926/final_cascade.json \
        --out artifacts/evaluation_plots
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

PALETTE = {
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "red":    "#dc2626",
    "amber":  "#d97706",
    "purple": "#7c3aed",
    "slate":  "#475569",
    "sky":    "#0ea5e9",
    "teal":   "#0d9488",
}


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────

class Provenance:
    """
    Records which files fed which figure, and which figures were skipped for
    want of data. Written to manifest.json beside the figures.

    A figure that cannot name a source file is a figure that should not exist,
    so every draw call registers its inputs here first.
    """

    def __init__(self):
        self.figures: dict = {}
        self.skipped: list = []
        self.settings: dict = {}
        self._digests: dict = {}

    def digest(self, path) -> str:
        path = Path(path)
        key = str(path)
        if key not in self._digests:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    for block in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(block)
                self._digests[key] = h.hexdigest()
            except OSError:
                self._digests[key] = None
        return self._digests[key]

    def record(self, figure: str, sources, note: str = "") -> None:
        if isinstance(sources, (str, Path)):
            sources = [sources]
        self.figures[figure] = {
            "sources": [{"path": str(p), "sha256": self.digest(p)} for p in sources],
            "note": note,
        }

    def skip(self, figure: str, reason: str) -> None:
        self.skipped.append({"figure": figure, "reason": reason})
        print(f"  [SKIP] {figure}: {reason}")

    def save(self, out: Path) -> Path:
        path = out / "manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "figures": self.figures,
                "skipped": self.skipped,
                "settings_used": self.settings,
                "rule": ("Every plotted value is read from the listed sources or from "
                         "config at run time. No figure is drawn from simulated, "
                         "approximated or hard-coded values."),
            }, fh, indent=2)
        print(f"  Saved: {path}")
        return path


PROV = Provenance()


def savefig(fig, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def source_caption(*paths) -> str:
    """A one-line 'built from' note for a figure footer."""
    names = " , ".join(str(p) for p in paths if p)
    return f"Source: {names}"


def stamp(fig, *paths) -> None:
    """Print the data source along the bottom of a figure."""
    fig.text(0.5, -0.02, source_caption(*paths), ha="center", va="top",
             fontsize=7, color=PALETTE["slate"])


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def cfg_get(cfg: dict, dotted: str, default=None):
    """Read a dotted config key, returning default when any level is missing."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def collect_predictions(path) -> tuple:
    """
    Per-prompt prediction rows and the files they came from.

    *path* is either a single JSONL written by run_evaluation.py or a
    *_predictions directory holding one JSONL per benchmark.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path] if path.exists() else []
    rows = []
    for f in files:
        rows.extend(load_jsonl(f))
    return rows, files


def metric_of(metrics: dict, *names):
    """
    First present metric among *names*, or None.

    train_layer1.py writes "fpr@95tpr" while evaluation/metrics.py writes
    "fpr_at_95tpr", so both spellings are accepted rather than one being assumed.
    """
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    return None


def fmt(value, digits: int = 4) -> str:
    """A metric for a label, or 'n/a' when the metric is undefined for the data."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def warn_if_duplicate_sibling(path: Path) -> None:
    """
    Warn when a results file is byte-identical to another in the same directory.

    artifacts/red_team/ holds results.json and results_fullpipeline.json with the
    same SHA-256, one of which is described as a full three-layer run and the
    other as a Layer-1 baseline. Whichever is passed here, the reader is told.
    """
    path = Path(path)
    if not path.exists():
        return
    mine = PROV.digest(path)
    twins = [p for p in sorted(path.parent.glob("*.json"))
             if p != path and PROV.digest(p) == mine]
    if twins:
        others = ", ".join(p.name for p in twins)
        print(f"  [WARNING] {path} is byte-identical to: {others}")
        print("            These files claim to describe different runs. Confirm which "
              "experiment produced this one before citing the figure.")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer1(data: dict, src: Path, out: Path, cfg: dict, stacker_label: str):
    print("\n[Layer 1] Generating plots...")

    history = data.get("training_history") or {}
    metrics = data.get("test_metrics") or {}
    T       = data.get("temperature_T")
    n       = metric_of(metrics, "n_samples")

    # ── 1a. Training curves ───────────────────────────────────────────────────
    if history.get("train_loss"):
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        backbone = cfg_get(cfg, "layer1.classifier.backbone", "classifier")
        batch = cfg_get(cfg, "layer1.classifier.batch_size")
        lr = cfg_get(cfg, "layer1.classifier.learning_rate")
        fig.suptitle(f"Layer 1 — Training Curves ({backbone} + LoRA, "
                     f"batch {batch}, lr {lr})",
                     fontsize=14, fontweight="bold")

        ax = axes[0]
        ax.plot(epochs, history["train_loss"], "o-", color=PALETTE["blue"],
                label="Train Loss", linewidth=2)
        if history.get("val_loss"):
            ax.plot(epochs, history["val_loss"], "s--", color=PALETTE["red"],
                    label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per Epoch")
        ax.legend()
        ax.set_xticks(epochs)

        ax = axes[1]
        if history.get("val_acc"):
            ax.plot(epochs, history["val_acc"], "D-", color=PALETTE["green"],
                    label="Val Accuracy", linewidth=2, markersize=8)
            best = max(history["val_acc"])
            ax.axhline(best, color=PALETTE["slate"], linestyle=":", alpha=0.6,
                       label=f"Best = {best:.4f}")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Validation Accuracy per Epoch")
        ax.legend()
        ax.set_xticks(epochs)

        stamp(fig, src)
        PROV.record("layer1_training_curves.png", src,
                    "epoch losses and validation accuracy as recorded by train_layer1.py")
        savefig(fig, out, "layer1_training_curves.png")
    else:
        PROV.skip("layer1_training_curves.png", f"no training_history in {src}")

    # ── 1b. Test metrics bar chart ─────────────────────────────────────────────
    auroc = metric_of(metrics, "auroc")
    if auroc is not None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        n_label = f"n={n:,}" if n else "n unrecorded"
        fig.suptitle(f"Layer 1 — Test Set Metrics ({n_label}, {stacker_label} stacker)",
                     fontsize=14, fontweight="bold")

        fpr = metric_of(metrics, "fpr@95tpr", "fpr_at_95tpr")
        pairs = [
            ("AUROC", auroc),
            ("AUPRC", metric_of(metrics, "auprc")),
            ("Accuracy", metric_of(metrics, "accuracy")),
            ("1 - FPR@95%TPR", None if fpr is None else 1.0 - fpr),
        ]
        pairs = [(k, v) for k, v in pairs if v is not None]
        labels = [k for k, _ in pairs]
        values = [v for _, v in pairs]
        colors = [PALETTE["green"] if v >= 0.8 else PALETTE["amber"] for v in values]

        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Score")
        ax.axvline(1.0, color=PALETTE["slate"], linestyle="--", alpha=0.4)
        ax.set_title(f"Temperature T* = {fmt(T, 4)}")

        stamp(fig, src)
        PROV.record("layer1_test_metrics.png", src,
                    f"test_metrics and temperature_T; describes the {stacker_label} stacker")
        savefig(fig, out, "layer1_test_metrics.png")
    else:
        PROV.skip("layer1_test_metrics.png", f"no test_metrics.auroc in {src}")


def plot_layer1_scores(rows: list, files: list, out: Path, cfg: dict):
    """
    Layer-1 calibrated score distribution by true label, from per-prompt rows.

    Replaces a figure that plotted two invented Gaussians. Rows without an
    l1_score or a label cannot contribute and are counted as excluded.
    """
    print("\n[Layer 1] Score distribution from per-prompt predictions...")

    safe = [r["l1_score"] for r in rows
            if r.get("l1_score") is not None and r.get("label") in (1, "1")]
    harm = [r["l1_score"] for r in rows
            if r.get("l1_score") is not None and r.get("label") in (0, "0")]
    if not safe and not harm:
        PROV.skip("layer1_score_distribution.png",
                  "no rows carrying both l1_score and label 0/1")
        return

    tau_low  = cfg_get(cfg, "layer1.thresholds.low")
    tau_high = cfg_get(cfg, "layer1.thresholds.high")
    if tau_low is None or tau_high is None:
        PROV.skip("layer1_score_distribution.png",
                  "layer1.thresholds.low/high absent from config")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Layer 1 — Calibrated Score Distribution and Routing Bands",
                 fontsize=14, fontweight="bold")

    bins = np.linspace(0, 1, 41)
    if safe:
        ax.hist(safe, bins=bins, color=PALETTE["green"], alpha=0.55,
                edgecolor="white", label=f"Safe prompts (n={len(safe):,})")
    if harm:
        ax.hist(harm, bins=bins, color=PALETTE["red"], alpha=0.55,
                edgecolor="white", label=f"Harmful prompts (n={len(harm):,})")

    ax.axvline(tau_low,  color=PALETTE["slate"], linestyle="--", linewidth=1.5,
               label=f"tau_low = {tau_low}")
    ax.axvline(tau_high, color=PALETTE["slate"], linestyle=":",  linewidth=1.5,
               label=f"tau_high = {tau_high}")

    top = ax.get_ylim()[1]
    ax.fill_betweenx([0, top], 0,        tau_low,  alpha=0.07, color=PALETTE["green"])
    ax.fill_betweenx([0, top], tau_low,  tau_high, alpha=0.07, color=PALETTE["amber"])
    ax.fill_betweenx([0, top], tau_high, 1.0,      alpha=0.07, color=PALETTE["red"])
    ax.set_ylim(0, top)

    for centre, text, colour in ((tau_low / 2, "ALLOW", PALETTE["green"]),
                                 ((tau_low + tau_high) / 2, "GRAY ZONE", PALETTE["amber"]),
                                 ((tau_high + 1) / 2, "MITIGATE", PALETTE["red"])):
        ax.text(centre, top * 0.96, text, ha="center", color=colour,
                fontweight="bold", fontsize=10)

    in_band = sum(1 for v in safe + harm if tau_low <= v <= tau_high)
    total = len(safe) + len(harm)
    ax.set_xlabel("Calibrated Score (higher = more likely harmful)")
    ax.set_ylabel("Number of Prompts")
    ax.set_title(f"{in_band:,} of {total:,} prompts "
                 f"({100 * in_band / max(total, 1):.1f}%) fall inside the gray zone")
    # Centred, because the distribution is bimodal at the edges and the middle is
    # the only empty region; upper-centre would cover the GRAY ZONE band label.
    ax.legend(loc="center", fontsize=9, framealpha=0.95)

    stamp(fig, *files)
    PROV.record("layer1_score_distribution.png", files,
                "observed l1_score by true label; bands from layer1.thresholds in config")
    savefig(fig, out, "layer1_score_distribution.png")


def plot_layer1_routing(layer_stats: dict, src: Path, out: Path):
    """Layer-1 routing counts as recorded, replacing a pie inferred from accuracy."""
    print("\n[Layer 1] Routing distribution from recorded layer_stats...")

    counts = [layer_stats.get("l1_allow", 0),
              layer_stats.get("l1_gray_zone", 0),
              layer_stats.get("l1_mitigate", 0)]
    total = sum(counts)
    if total == 0:
        PROV.skip("layer1_routing_distribution.png",
                  f"layer_stats in {src} records no Layer-1 decisions")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.suptitle("Layer 1 — Routing Distribution", fontsize=14, fontweight="bold")

    names = ["Allow", "Gray Zone", "Mitigate"]
    colors = [PALETTE["green"], PALETTE["amber"], PALETTE["red"]]
    keep = [(c, f"{n}\n{c:,} ({100 * c / total:.1f}%)", col)
            for c, n, col in zip(counts, names, colors) if c > 0]
    sizes, labels, cols = zip(*keep)

    wedges, _ = ax.pie(sizes, colors=cols, startangle=90,
                       explode=[0.03] * len(sizes),
                       wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              ncol=len(sizes), frameon=False)
    ax.set_title(f"{total:,} prompts")

    stamp(fig, src)
    PROV.record("layer1_routing_distribution.png", src,
                "l1_allow / l1_gray_zone / l1_mitigate exactly as recorded")
    savefig(fig, out, "layer1_routing_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2
# ─────────────────────────────────────────────────────────────────────────────

def judge_title(data: dict, cfg: dict) -> str:
    """
    A judge figure title describing the run that produced the data.

    The stored config_snapshot is preferred over the live config, because the
    run may predate a config change; when they disagree both are shown, so a
    figure can never silently claim the current settings for an older run.
    """
    snap = data.get("config_snapshot") or {}
    model = snap.get("model") or cfg_get(cfg, "layer2.judge.model_name", "judge")
    model = str(model).rstrip("/").split("/")[-1]
    total = data.get("total_samples")
    bits = [model]
    if total:
        bits.append(f"n={total}")

    run_mode = snap.get("mode")
    live_mode = cfg_get(cfg, "layer2.consistency.mode")
    mode = run_mode or live_mode
    if mode == "vote" or (mode is None and snap.get("K")):
        k = snap.get("K", cfg_get(cfg, "layer2.consistency.K"))
        c_min = snap.get("c_min", cfg_get(cfg, "layer2.consistency.c_min"))
        bits.append(f"vote K={k}, c_min={c_min}")
    elif mode == "logprob":
        lo = snap.get("allow_below", cfg_get(cfg, "layer2.consistency.allow_below"))
        hi = snap.get("mitigate_above", cfg_get(cfg, "layer2.consistency.mitigate_above"))
        bits.append(f"logprob band {lo}/{hi}")
    if run_mode and live_mode and run_mode != live_mode:
        bits.append(f"config now: {live_mode}")
    return "Layer 2 — LLM-as-Judge  (" + ", ".join(str(b) for b in bits) + ")"


def plot_layer2(data: dict, src: Path, out: Path, cfg: dict):
    print("\n[Layer 2] Generating plots...")

    total      = data.get("total_samples") or 0
    allow_n    = data.get("allow", 0)
    mitigate_n = data.get("mitigate", 0)
    escalated  = data.get("escalated", 0)
    mean_conf  = data.get("mean_confidence")
    mean_sev   = data.get("mean_severity")
    cats       = data.get("top_categories") or {}

    if total == 0:
        PROV.skip("layer2_decisions.png", f"total_samples is 0 or absent in {src}")
        PROV.skip("layer2_categories.png", f"total_samples is 0 or absent in {src}")
        return

    title = judge_title(data, cfg)

    # ── 2a/2b. Decision distribution and recorded metrics ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    pairs = [(allow_n, "Allow", PALETTE["green"]),
             (mitigate_n, "Mitigate", PALETTE["red"]),
             (escalated, "Escalated", PALETTE["amber"])]
    keep = [(c, f"{n}\n{c} ({100 * c / total:.1f}%)", col)
            for c, n, col in pairs if c > 0]
    sizes, labels, cols = zip(*keep)
    wedges, _ = ax.pie(sizes, colors=cols, startangle=90,
                       wedgeprops=dict(edgecolor="white", linewidth=2.5),
                       explode=[0.04] * len(sizes))
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=1, frameon=False)
    ax.set_title("Decision Distribution")

    ax = axes[1]
    recorded = [("Mean Confidence", mean_conf),
                ("Mean Severity", mean_sev),
                ("Allow Rate", allow_n / total),
                ("Mitigate Rate", mitigate_n / total)]
    recorded = [(k, v) for k, v in recorded if v is not None]
    bars = ax.barh([k for k, _ in recorded], [v for _, v in recorded],
                   color=[PALETTE["blue"], PALETTE["red"],
                          PALETTE["green"], PALETTE["red"]][:len(recorded)],
                   edgecolor="white", height=0.5)
    for bar, (_, val) in zip(bars, recorded):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.axvline(1.0, color=PALETTE["slate"], linestyle="--", alpha=0.3)
    ax.set_xlabel("Score / Rate")
    ax.set_title("Recorded Metrics")

    stamp(fig, src)
    PROV.record("layer2_decisions.png", src,
                "decision counts and means as recorded; title from the run's own config_snapshot")
    savefig(fig, out, "layer2_decisions.png")

    # ── 2c. Category distribution ─────────────────────────────────────────────
    if cats:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Layer 2 — Violation Category Distribution",
                     fontsize=14, fontweight="bold")
        names = list(cats.keys())
        values = list(cats.values())
        pcts = [100 * v / total for v in values]
        colors = [PALETTE["slate"] if n == "none" else PALETTE["red"] for n in names]

        bars = ax.bar(names, pcts, color=colors, edgecolor="white", width=0.6)
        for bar, val, pct in zip(bars, values, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Percentage of Samples (%)")
        ax.set_xlabel("Category")
        ax.set_ylim(0, max(pcts) * 1.25)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(handles=[mpatches.Patch(color=PALETTE["red"], label="Harmful category"),
                           mpatches.Patch(color=PALETTE["slate"], label="No violation (allow)")],
                  frameon=False)

        stamp(fig, src)
        PROV.record("layer2_categories.png", src, "top_categories as recorded")
        savefig(fig, out, "layer2_categories.png")
    else:
        PROV.skip("layer2_categories.png", f"no top_categories in {src}")


def plot_layer2_confidence(rows: list, files: list, out: Path, cfg: dict):
    """
    The judge's observed confidence, in the form the data supports.

    Two cases, decided by the data and never assumed:

    vote mode    l2_votes holds one entry per judge call, so the real unanimity
                 distribution is counted and plotted against c_min.
    logprob mode l2_votes holds a single entry, because the decision comes from
                 one forward pass. There is no vote split to plot, so the
                 distribution of l2_confidence is shown instead and labelled as
                 such, against the allow/mitigate band.

    The figure this replaces asserted a 87/10/3 vote split that no run produced,
    and labelled it with a c_min the config did not contain.
    """
    print("\n[Layer 2] Confidence distribution from per-prompt predictions...")

    judged = [r for r in rows if r.get("l2_decision") is not None]
    if not judged:
        PROV.skip("layer2_confidence.png",
                  "no prediction rows reached Layer 2 (l2_decision is null throughout)")
        return

    vote_lengths = {len(r["l2_votes"]) for r in judged
                    if isinstance(r.get("l2_votes"), list) and r["l2_votes"]}
    max_votes = max(vote_lengths) if vote_lengths else 0

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if max_votes > 1:
        # Real K-vote unanimity.
        c_min = cfg_get(cfg, "layer2.consistency.c_min")
        counts: dict = {}
        for r in judged:
            votes = r.get("l2_votes") or []
            if not votes:
                continue
            winner = max(set(votes), key=votes.count)
            counts[f"{votes.count(winner)}/{len(votes)}"] = \
                counts.get(f"{votes.count(winner)}/{len(votes)}", 0) + 1
        order = sorted(counts, key=lambda k: -eval_fraction(k))
        values = [counts[k] for k in order]
        bars = ax.bar(order, values, color=PALETTE["sky"], edgecolor="white", width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Winning votes / K")
        ax.set_ylabel("Number of Prompts")
        fig.suptitle("Layer 2 — Observed Self-Consistency Unanimity",
                     fontsize=14, fontweight="bold")
        ax.set_title(f"{len(judged):,} prompts reached the judge"
                     + (f"; c_min = {c_min}" if c_min is not None else ""))
        note = "counted from l2_votes; vote mode"
    else:
        # One call per prompt: plot the confidence the run recorded.
        conf = [r["l2_confidence"] for r in judged if r.get("l2_confidence") is not None]
        if not conf:
            PROV.skip("layer2_confidence.png",
                      "single-call judge and no l2_confidence recorded")
            plt.close(fig)
            return
        lo = cfg_get(cfg, "layer2.consistency.allow_below")
        hi = cfg_get(cfg, "layer2.consistency.mitigate_above")
        ax.hist(conf, bins=np.linspace(0.5, 1.0, 26), color=PALETTE["blue"],
                alpha=0.8, edgecolor="white")
        ax.set_xlabel("Recorded l2_confidence  (max(p, 1 - p) in log-probability mode)")
        ax.set_ylabel("Number of Prompts")
        fig.suptitle("Layer 2 — Judge Confidence Distribution",
                     fontsize=14, fontweight="bold")
        band = f"band {lo}/{hi}" if lo is not None and hi is not None else "band unrecorded"
        ax.set_title(f"{len(judged):,} prompts reached the judge; one forward pass each, "
                     f"{band}. No vote split exists to plot.")
        note = ("l2_confidence histogram; the judge ran one call per prompt, so there is "
                "no K-vote unanimity to show")

    stamp(fig, *files)
    PROV.record("layer2_confidence.png", files, note)
    savefig(fig, out, "layer2_confidence.png")


def eval_fraction(label: str) -> float:
    """'4/5' -> 0.8, used only to order unanimity bars."""
    num, den = label.split("/")
    return float(num) / float(den)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer3(data: dict, src: Path, jsonl_path, out: Path, cfg: dict):
    print("\n[Layer 3] Generating plots...")

    total       = data.get("total_samples") or 0
    avg_sev     = data.get("avg_severity")
    modes       = data.get("mode_breakdown") or {}
    cats        = data.get("top_categories") or {}

    if total == 0:
        PROV.skip("layer3_overview.png", f"total_samples is 0 or absent in {src}")
        return

    # ── 3a. Mode breakdown and summary ────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5))
    default_mode = cfg_get(cfg, "layer3.mitigation.default_mode", "unrecorded")
    fig.suptitle(f"Layer 3 — Mitigation Results  (n={total}; config default_mode: {default_mode})",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax = fig.add_subplot(gs[0])
    names = [m for m in ("refuse", "redact", "safe_rewrite") if m in modes]
    keep = [(modes[m]["count"],
             f"{m.replace('_', ' ').title()}\n{modes[m]['count']} ({modes[m]['pct']}%)",
             col)
            for m, col in zip(names, [PALETTE["red"], PALETTE["amber"], PALETTE["teal"]])
            if modes[m]["count"] > 0]
    if keep:
        sizes, labels, cols = zip(*keep)
        wedges, _ = ax.pie(sizes, colors=cols, startangle=90,
                           wedgeprops=dict(edgecolor="white", linewidth=2.5),
                           explode=[0.04] * len(sizes))
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                  frameon=False)
    ax.set_title("Mitigation Mode Breakdown (as served)")

    ax2 = fig.add_subplot(gs[1])
    summary = [("Avg Severity", avg_sev)]
    if data.get("constraints_triggered_pct") is not None:
        summary.append(("Constraint\nTrigger Rate", data["constraints_triggered_pct"] / 100))
    if data.get("escalated_pct") is not None:
        summary.append(("Escalation\nRate", data["escalated_pct"] / 100))
    summary = [(k, v) for k, v in summary if v is not None]
    bars = ax2.barh([k for k, _ in summary], [v for _, v in summary],
                    color=[PALETTE["red"], PALETTE["amber"], PALETTE["purple"]][:len(summary)],
                    edgecolor="white", height=0.45)
    for bar, (_, val) in zip(bars, summary):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1.15)
    ax2.axvline(1.0, color=PALETTE["slate"], linestyle="--", alpha=0.3)
    ax2.set_xlabel("Score / Rate")
    ax2.set_title("Summary Metrics")

    stamp(fig, src)
    PROV.record("layer3_overview.png", src,
                "mode_breakdown and summary rates as recorded; default_mode from config")
    savefig(fig, out, "layer3_overview.png")

    # ── 3b. Severity histogram ────────────────────────────────────────────────
    jsonl_path = Path(jsonl_path) if jsonl_path else None
    if jsonl_path and jsonl_path.exists():
        severities = [r["severity"] for r in load_jsonl(jsonl_path)
                      if r.get("severity") is not None]
        if severities:
            refuse_at = cfg_get(cfg, "layer3.mitigation.severity_refuse_threshold")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.suptitle("Layer 3 — Severity Score Distribution",
                         fontsize=14, fontweight="bold")
            ax.hist(severities, bins=15, color=PALETTE["red"],
                    edgecolor="white", alpha=0.85)
            handles = []
            if avg_sev is not None:
                ax.axvline(avg_sev, color=PALETTE["slate"], linestyle="--", linewidth=2)
                handles.append(mpatches.Patch(color=PALETTE["slate"],
                                              label=f"Mean = {avg_sev:.3f}"))
            if refuse_at is not None:
                ax.axvline(refuse_at, color=PALETTE["purple"], linestyle=":", linewidth=2)
                handles.append(mpatches.Patch(
                    color=PALETTE["purple"],
                    label=f"config severity_refuse_threshold = {refuse_at}"))
            ax.set_xlabel("Severity Score")
            ax.set_ylabel("Number of Samples")
            ax.set_title(f"n={len(severities)} scored outputs")
            if handles:
                ax.legend(handles=handles, frameon=False, fontsize=9)

            stamp(fig, jsonl_path)
            PROV.record("layer3_severity_histogram.png", jsonl_path,
                        "per-sample severity; refusal line from config, not a literal")
            savefig(fig, out, "layer3_severity_histogram.png")
        else:
            PROV.skip("layer3_severity_histogram.png",
                      f"no severity values in {jsonl_path}")
    else:
        PROV.skip("layer3_severity_histogram.png",
                  f"per-sample file not found: {jsonl_path}. "
                  "Produce one with run_layer3.py --output PATH")

    # ── 3c. Category distribution ─────────────────────────────────────────────
    if cats:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Layer 3 — Harm Category Distribution",
                     fontsize=14, fontweight="bold")
        names = list(cats.keys())
        values = list(cats.values())
        pcts = [100 * v / total for v in values]
        colors = ([PALETTE["red"], PALETTE["purple"], PALETTE["amber"],
                   PALETTE["slate"], PALETTE["sky"], PALETTE["teal"]]
                  * (1 + len(names) // 6))[:len(names)]
        bars = ax.bar(names, pcts, color=colors, edgecolor="white", width=0.6)
        for bar, val, pct in zip(bars, values, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Percentage of Mitigated Samples (%)")
        ax.set_xlabel("Harm Category")
        ax.set_ylim(0, max(pcts) * 1.3)
        ax.tick_params(axis="x", rotation=20)

        stamp(fig, src)
        PROV.record("layer3_categories.png", src, "top_categories as recorded")
        savefig(fig, out, "layer3_categories.png")
    else:
        PROV.skip("layer3_categories.png", f"no top_categories in {src}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline flow
# ─────────────────────────────────────────────────────────────────────────────

def _flow_box(ax, x, y, w, h, label, sublabel, color, fs=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor="white",
                                   linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.27, sublabel, ha="center", va="center",
            fontsize=fs - 1.5, color="white", alpha=0.95)


def plot_pipeline_flow(stats: dict, n_samples, src: Path, out: Path,
                       redteam: dict = None, rt_src: Path = None):
    """
    The decision flow with the counts a run actually recorded.

    The figure this replaces hard-coded "200 samples", "106 (53%)" and "94 (47%)"
    into the boxes. Those came from one standalone judge run and did not match the
    pipeline evaluation stored beside them, so the diagram described no single run.
    """
    name = "pipeline_flow_with_redteam.png" if redteam else "pipeline_flow.png"
    print(f"\n[Pipeline] {name} from recorded layer_stats...")

    total = n_samples or (stats.get("l1_allow", 0) + stats.get("l1_gray_zone", 0)
                          + stats.get("l1_mitigate", 0))
    if not total:
        PROV.skip(name, f"layer_stats in {src} records no prompts")
        return

    l1_allow = stats.get("l1_allow", 0)
    l1_gray  = stats.get("l1_gray_zone", 0)
    l1_mit   = stats.get("l1_mitigate", 0)
    l2_allow = stats.get("l2_allow", 0)
    l2_mit   = stats.get("l2_mitigate", 0)
    l2_unc   = stats.get("l2_uncertain", 0)
    l3_parts = [(m, stats.get(f"l3_{m}", 0)) for m in ("refuse", "redact", "safe_rewrite")]
    l3_total = sum(v for _, v in l3_parts)

    def pct(n):
        return f"{n:,} ({100 * n / total:.1f}%)"

    fig, ax = plt.subplots(figsize=(14, 6 if redteam else 5.5))
    fig.suptitle("GaurdLLM — Decision Flow as Recorded", fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    base = 0.12 if redteam else 0.06

    _flow_box(ax, 0.01, base + 0.30, 0.13, 0.26, "Layer 1",
              f"{total:,} prompts", PALETTE["blue"])
    _flow_box(ax, 0.22, base + 0.52, 0.13, 0.26, "ALLOW",
              pct(l1_allow), PALETTE["green"])
    _flow_box(ax, 0.22, base + 0.06, 0.13, 0.26, "Layer 2",
              f"{pct(l1_gray)}\ngray zone", PALETTE["sky"])
    _flow_box(ax, 0.46, base + 0.52, 0.13, 0.26, "ALLOW",
              f"L1 {l1_allow:,} + L2 {l2_allow:,}", PALETTE["green"])
    _flow_box(ax, 0.46, base + 0.06, 0.13, 0.26, "MITIGATE",
              f"L1 {l1_mit:,} + L2 {l2_mit:,}"
              + (f"\n+{l2_unc:,} uncertain" if l2_unc else ""), PALETTE["red"])
    served = ", ".join(f"{n:,} {m.replace('_', ' ')}" for m, n in l3_parts if n) or "none served"
    _flow_box(ax, 0.68, base + 0.06, 0.17, 0.26, "Layer 3",
              f"{l3_total:,} handled\n{served}", PALETTE["purple"])

    arrow = dict(arrowstyle="->", color=PALETTE["slate"], lw=1.7)
    ax.annotate("", xy=(0.22, base + 0.66), xytext=(0.14, base + 0.54),
                arrowprops=dict(**arrow, connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", xy=(0.22, base + 0.20), xytext=(0.14, base + 0.36),
                arrowprops=dict(**arrow, connectionstyle="arc3,rad=0.25"))
    ax.annotate("", xy=(0.46, base + 0.65), xytext=(0.35, base + 0.65), arrowprops=dict(**arrow))
    ax.annotate("", xy=(0.46, base + 0.19), xytext=(0.35, base + 0.19), arrowprops=dict(**arrow))
    ax.annotate("", xy=(0.68, base + 0.19), xytext=(0.59, base + 0.19), arrowprops=dict(**arrow))
    ax.text(0.150, base + 0.72, "allow", fontsize=8, color=PALETTE["green"],
            fontstyle="italic", ha="left")
    ax.text(0.150, base + 0.29, "gray_zone", fontsize=8, color=PALETTE["amber"],
            fontstyle="italic", ha="left")

    sources = [src]
    if redteam:
        n_variants = redteam.get("n_variants", 0)
        distinct = redteam.get("n_variants_distinct")
        rate = redteam.get("bypass_rate", 0.0)
        verified = redteam.get("bypass_rate_verified")
        text = (f"Red team: {n_variants:,} variant rows"
                + (f" , {distinct:,} distinct texts" if distinct else "")
                + f"   |   bypass {100 * rate:.1f}% of rows")
        if verified is not None:
            text += (f"   |   {100 * verified:.1f}% over the "
                     f"{redteam.get('n_verified', 0):,} variants that provably keep the request")
        _flow_box(ax, 0.01, 0.01, 0.96, 0.08, text, "", "#1e293b", fs=9)
        sources.append(rt_src)

    stamp(fig, *sources)
    PROV.record(name, sources,
                "every count read from layer_stats and the red-team report; no literals")
    savefig(fig, out, name)


# ─────────────────────────────────────────────────────────────────────────────
# Red team
# ─────────────────────────────────────────────────────────────────────────────

def plot_redteam(data: dict, src: Path, out: Path):
    print("\n[Red-Team] Generating plots...")

    n_variants = data.get("n_variants", 0)
    n_bypassed = data.get("n_bypassed", 0)
    bypass_rate = data.get("bypass_rate", 0.0)
    by_type = data.get("by_attack_type") or {}
    by_type_distinct = data.get("by_attack_type_distinct") or {}
    elapsed = data.get("elapsed_s", 0.0)

    if not by_type:
        PROV.skip("redteam_bypass_rates.png", f"no by_attack_type in {src}")
        PROV.skip("redteam_attack_volume.png", f"no by_attack_type in {src}")
        return

    # Sampling design, in the report's own terms rather than variants / seeds.
    loaded = data.get("n_seeds_loaded") or data.get("n_seeds", 0)
    attacked = data.get("n_seeds_attacked") or data.get("n_seeds", 0)
    distinct = data.get("n_variants_distinct")
    design = f"{loaded:,} seeds loaded, {attacked:,} attacked"
    rounds = data.get("rounds")
    if rounds:
        design += f", {rounds} round(s)"
    design += f"; {n_variants:,} variant rows"
    if distinct:
        design += f", {distinct:,} distinct texts"

    names = list(by_type.keys())
    rates = [by_type[a].get("rate", 0) * 100 for a in names]
    totals = [by_type[a].get("total", 0) for a in names]
    bypassed = [by_type[a].get("bypassed", 0) for a in names]

    # ── RT-1. Bypass rate by attack type ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Red-Team Evaluation — Adversarial Robustness\n{design}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    colors = [PALETTE["red"] if r > 10 else (PALETTE["amber"] if r > 0 else PALETTE["green"])
              for r in rates]
    bars = ax.bar(names, rates, color=colors, edgecolor="white", width=0.55)
    for bar, rate, byp, tot in zip(bars, rates, bypassed, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{byp}/{tot}\n({rate:.1f}%)", ha="center", fontsize=8.5,
                fontweight="bold")
    ax.set_ylabel("Bypass Rate (%)")
    ax.set_xlabel("Attack Type")
    ax.set_ylim(0, max(rates + [5]) * 1.4)
    ax.tick_params(axis="x", rotation=18)
    if by_type_distinct:
        ax.set_title("Bypass Rate per Attack Type (rows; distinct-text rates in the volume chart)")
    else:
        ax.set_title("Bypass Rate per Attack Type")

    # ── RT-2. Overall summary donut ──────────────────────────────────────────
    ax = axes[1]
    blocked = n_variants - n_bypassed
    if n_bypassed > 0:
        sizes = [blocked, n_bypassed]
        labels = [f"Blocked\n{blocked:,} ({100 - bypass_rate * 100:.1f}%)",
                  f"Bypassed\n{n_bypassed:,} ({bypass_rate * 100:.1f}%)"]
        cols = [PALETTE["green"], PALETTE["red"]]
    else:
        sizes, labels, cols = [n_variants], [f"All Blocked\n{n_variants:,} (100%)"], [PALETTE["green"]]
    wedges, _ = ax.pie(sizes, colors=cols, startangle=90,
                       wedgeprops=dict(edgecolor="white", linewidth=2.5, width=0.55),
                       explode=[0.04] * len(sizes))
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    verified = data.get("bypass_rate_verified")
    subtitle = f"Bypass rate over rows: {bypass_rate * 100:.1f}%"
    if verified is not None:
        subtitle += (f"\nOver variants that provably keep the request: "
                     f"{verified * 100:.1f}% of {data.get('n_verified', 0):,}")
    ax.set_title(subtitle)

    stamp(fig, src)
    PROV.record("redteam_bypass_rates.png", src,
                "per-attack-type rates and the report's own sampling-design fields")
    savefig(fig, out, "redteam_bypass_rates.png")

    # ── RT-3. Attack volume ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    fig.suptitle("Red-Team — Attack Volume and Bypasses by Type",
                 fontsize=13, fontweight="bold")
    x = np.arange(len(names))
    width = 0.28 if by_type_distinct else 0.35
    ax.bar(x - width, totals, width, label="Variant rows",
           color=PALETTE["slate"], alpha=0.7, edgecolor="white")
    ax.bar(x, bypassed, width, label="Bypassed (rows)",
           color=PALETTE["red"], alpha=0.9, edgecolor="white")
    if by_type_distinct:
        dist_byp = [by_type_distinct.get(a, {}).get("bypassed", 0) for a in names]
        ax.bar(x + width, dist_byp, width, label="Bypassed (distinct texts)",
               color=PALETTE["purple"], alpha=0.9, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Number of Variants")
    ax.legend(frameon=False)
    ax.set_title(f"Elapsed: {elapsed:.1f}s")

    stamp(fig, src)
    PROV.record("redteam_attack_volume.png", src,
                "row and distinct-text counts per attack type as recorded")
    savefig(fig, out, "redteam_attack_volume.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate GaurdLLM evaluation plots from measurements on disk.")
    p.add_argument("--out", default="artifacts/evaluation_plots",
                   help="Output directory for figures and manifest.json")
    p.add_argument("--config", default="config/config.yaml",
                   help="Config read for thresholds, bands and mode labels")
    p.add_argument("--layer1", default="artifacts/layer1/results.json",
                   help="train_layer1.py results JSON")
    p.add_argument("--layer2", default="artifacts/layer2/results.json",
                   help="run_layer2.py results JSON")
    p.add_argument("--layer3", default="artifacts/layer3/results.json",
                   help="run_layer3.py results JSON")
    p.add_argument("--layer3-jsonl", default=None,
                   help="Per-sample Layer-3 output from run_layer3.py --output "
                        "(needed for the severity histogram)")
    p.add_argument("--predictions", default=None,
                   help="A *_predictions directory or JSONL from run_evaluation.py "
                        "(needed for the score and confidence distributions)")
    p.add_argument("--eval-results", default=None,
                   help="A run_evaluation.py results JSON; its layer_stats drive the "
                        "flow diagrams and the Layer-1 routing pie")
    p.add_argument("--redteam", default=None,
                   help="A run_red_team.py report JSON")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any figure was skipped for want of data")
    return p.parse_args()


def load_config(path) -> dict:
    """Config as a dict, or empty when it cannot be read."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"  [WARNING] could not read {path}: {exc}. "
              "Figures needing config values will be skipped rather than assume one.")
        return {}


def first_benchmark(results: dict) -> tuple:
    """The (layer_stats, n_samples) of the first benchmark in a results JSON."""
    for key, value in results.items():
        if key == "run_info" or not isinstance(value, dict):
            continue
        if "layer_stats" in value:
            return value.get("layer_stats") or {}, value.get("n_samples")
    return {}, None


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")

    cfg = load_config(args.config)
    stacker_label = ("classifier-only"
                     if cfg_get(cfg, "layer1.classifier_only", False) else "three-signal")
    PROV.settings = {
        "config_path": str(args.config),
        "layer1.classifier_only": cfg_get(cfg, "layer1.classifier_only"),
        "layer1.normalise_input": cfg_get(cfg, "layer1.normalise_input"),
        "layer1.payload_views": cfg_get(cfg, "layer1.payload_views"),
        "layer1.artifact_subdir": cfg_get(cfg, "layer1.artifact_subdir"),
        "layer1.thresholds.low": cfg_get(cfg, "layer1.thresholds.low"),
        "layer1.thresholds.high": cfg_get(cfg, "layer1.thresholds.high"),
        "layer2.consistency.mode": cfg_get(cfg, "layer2.consistency.mode"),
        "layer2.consistency.K": cfg_get(cfg, "layer2.consistency.K"),
        "layer2.consistency.c_min": cfg_get(cfg, "layer2.consistency.c_min"),
        "layer2.consistency.allow_below": cfg_get(cfg, "layer2.consistency.allow_below"),
        "layer2.consistency.mitigate_above": cfg_get(cfg, "layer2.consistency.mitigate_above"),
        "layer2.judge.prompt_version": cfg_get(cfg, "layer2.judge.prompt_version"),
        "layer3.mitigation.default_mode": cfg_get(cfg, "layer3.mitigation.default_mode"),
        "layer3.mitigation.severity_refuse_threshold":
            cfg_get(cfg, "layer3.mitigation.severity_refuse_threshold"),
    }

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    l1_path = Path(args.layer1)
    if l1_path.exists():
        plot_layer1(load_json(l1_path), l1_path, out, cfg, stacker_label)
    else:
        PROV.skip("layer1_training_curves.png", f"{l1_path} not found")
        PROV.skip("layer1_test_metrics.png", f"{l1_path} not found")

    # ── Per-prompt predictions ────────────────────────────────────────────────
    rows, pred_files = ([], [])
    if args.predictions:
        rows, pred_files = collect_predictions(args.predictions)
        if not rows:
            print(f"  [WARNING] no prediction rows found under {args.predictions}")
    if rows:
        plot_layer1_scores(rows, pred_files, out, cfg)
        plot_layer2_confidence(rows, pred_files, out, cfg)
    else:
        why = ("--predictions not given" if not args.predictions
               else f"no rows under {args.predictions}")
        PROV.skip("layer1_score_distribution.png", why)
        PROV.skip("layer2_confidence.png", why)

    # ── Recorded routing and flow ─────────────────────────────────────────────
    stats, n_samples = ({}, None)
    eval_path = Path(args.eval_results) if args.eval_results else None
    if eval_path and eval_path.exists():
        stats, n_samples = first_benchmark(load_json(eval_path))
    elif rows:
        # Count routing directly from the prediction rows rather than infer it.
        stats = {
            "l1_allow": sum(1 for r in rows if r.get("l1_decision") == "allow"),
            "l1_gray_zone": sum(1 for r in rows if r.get("l1_decision") == "gray_zone"),
            "l1_mitigate": sum(1 for r in rows if r.get("l1_decision") == "mitigate"),
            "l2_allow": sum(1 for r in rows if r.get("l2_decision") == "allow"),
            "l2_mitigate": sum(1 for r in rows if r.get("l2_decision") == "mitigate"),
            "l2_uncertain": sum(1 for r in rows if r.get("escalated")),
        }
        for mode in ("refuse", "redact", "safe_rewrite"):
            stats[f"l3_{mode}"] = sum(1 for r in rows if r.get("l3_mode") == mode)
        n_samples = len(rows)
        eval_path = pred_files[0] if pred_files else None

    # ── Red team ──────────────────────────────────────────────────────────────
    redteam, rt_path = (None, None)
    if args.redteam:
        rt_path = Path(args.redteam)
        if rt_path.exists():
            warn_if_duplicate_sibling(rt_path)
            redteam = load_json(rt_path)
            plot_redteam(redteam, rt_path, out)
        else:
            PROV.skip("redteam_bypass_rates.png", f"{rt_path} not found")
            PROV.skip("redteam_attack_volume.png", f"{rt_path} not found")
            rt_path = None
    else:
        PROV.skip("redteam_bypass_rates.png", "--redteam not given")
        PROV.skip("redteam_attack_volume.png", "--redteam not given")

    if stats:
        plot_pipeline_flow(stats, n_samples, eval_path, out)
        if redteam:
            plot_pipeline_flow(stats, n_samples, eval_path, out,
                               redteam=redteam, rt_src=rt_path)
        plot_layer1_routing(stats, eval_path, out)
    else:
        why = "neither --eval-results nor --predictions supplied recorded routing counts"
        PROV.skip("pipeline_flow.png", why)
        PROV.skip("pipeline_flow_with_redteam.png", why)
        PROV.skip("layer1_routing_distribution.png", why)

    # ── Layer 2 and Layer 3 summaries ─────────────────────────────────────────
    l2_path = Path(args.layer2)
    if l2_path.exists():
        plot_layer2(load_json(l2_path), l2_path, out, cfg)
    else:
        PROV.skip("layer2_decisions.png", f"{l2_path} not found")
        PROV.skip("layer2_categories.png", f"{l2_path} not found")

    l3_path = Path(args.layer3)
    if l3_path.exists():
        plot_layer3(load_json(l3_path), l3_path, args.layer3_jsonl, out, cfg)
    else:
        PROV.skip("layer3_overview.png", f"{l3_path} not found")
        PROV.skip("layer3_categories.png", f"{l3_path} not found")
        PROV.skip("layer3_severity_histogram.png", f"{l3_path} not found")

    # ── Summary ───────────────────────────────────────────────────────────────
    PROV.save(out)
    print(f"\nFigures drawn : {len(PROV.figures)}")
    print(f"Figures skipped: {len(PROV.skipped)}")
    if PROV.skipped:
        print("\nSkipped for want of data. No figure is drawn from invented numbers,")
        print("so each of these needs the run that produces its input:")
        for entry in PROV.skipped:
            print(f"  - {entry['figure']}: {entry['reason']}")
    print(f"\nAll output in: {out.resolve()}")
    if args.strict and PROV.skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
