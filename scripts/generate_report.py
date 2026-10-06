"""
GaurdLLM — Layer-1 Report Generator
======================================
Generates a Word document (.docx) describing the Layer-1 safety detector:
methodology, training, results, diagnostic plots and provenance.

Every number in the document is read from a file at run time — config/config.yaml,
<artifact-dir>/results.json, the stacker and scaler files, the distillation-pool
metadata, and the pool's own .jsonl line counts. Nothing is hard-coded: when a
source file is missing the report says the value is unavailable and moves on,
rather than falling back to a literal that used to be true.

Steps:
  1. Generate the six diagnostic plots (via scripts.plot_layer1)
  2. Collect the facts the prose needs, each tagged with the file it came from
  3. Build the Word document

Output:
  <artifact-dir>/GaurdLLM_Layer1_Report.docx

Usage:
  pip install python-docx
  python scripts/generate_report.py [--artifact-dir PATH] [--out PATH]
                                    [--stacker {config,full,clf_only}]
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                       # for `from scripts.<module> import ...`
sys.path.insert(0, os.path.join(_ROOT, "src"))  # local checkout without `pip install -e .`

import argparse
import collections
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Matplotlib (non-interactive) ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

from scripts.plot_layer1 import (
    DEFAULT_CONFIG_PATH, DEFAULT_FEATURES_PATH,
    STACKER_CLF_ONLY, STACKER_CONFIG, STACKER_FULL, STACKERS,
    PlotRun, Stacker,
    draw_all, prepare_run, resolve_paths, resolve_stacker,
)
from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger

logger = get_logger("generate_report")

UNAVAILABLE = "unavailable — source file missing"

DEFAULT_REPORT_NAME = "GaurdLLM_Layer1_Report.docx"
DEFAULT_POOL_DIR    = "data/distillation_pool"
POOL_SPLITS         = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
# Reading facts off disk
# ─────────────────────────────────────────────────────────────────────────────

def dig(obj: Any, *keys, default=None):
    """Nested .get that returns `default` the moment anything is missing."""
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def num(value, spec: str = "{:.4f}", missing: str = UNAVAILABLE) -> str:
    """Format a number read from disk, or say plainly that it is not there."""
    if value is None:
        return missing
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return None


def count_lines(path: Path) -> Optional[int]:
    """Non-empty line count, or None when the file is not there."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError as exc:
        logger.warning(f"Could not count {path}: {exc}")
        return None


def pool_category_counts(pool_dir: Path) -> Tuple[Optional[collections.Counter], int]:
    """
    Category labels actually present in the pool, counted across all splits.

    The config lists six aspirational violation_categories; the pool carries
    whatever labels its source datasets used. This reports the latter.
    """
    counts: collections.Counter = collections.Counter()
    seen = 0
    for split in POOL_SPLITS:
        path = Path(pool_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    counts[rec.get("category", "(none recorded)")] += 1
                    seen += 1
        except (OSError, ValueError) as exc:
            logger.warning(f"Could not scan {path}: {exc}")
            return None, 0
    if not seen:
        return None, 0
    return counts, seen


def file_stamp(path: Path) -> str:
    """'present, modified <UTC>' or 'MISSING' — for the provenance table."""
    path = Path(path)
    if not path.exists():
        return "MISSING"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return f"present, modified {mtime.isoformat(timespec='seconds')}"


# ─────────────────────────────────────────────────────────────────────────────
# Facts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Facts:
    """Everything the prose is allowed to state, and where each piece came from."""
    cfg:            dict
    config_path:    Path
    artifact_dir:   Path
    pool_dir:       Path
    stacker:        Stacker
    results:        Optional[dict]              = None
    results_path:   Optional[Path]              = None
    pool_meta:      Optional[dict]              = None
    pool_meta_path: Optional[Path]              = None
    line_counts:    Dict[str, Optional[int]]    = field(default_factory=dict)
    categories:     Optional[collections.Counter] = None
    category_total: int                         = 0
    adapter_cfg:    Optional[dict]              = None
    scaler_ts:      Dict[str, Optional[float]]  = field(default_factory=dict)
    metrics_stacker: Optional[Stacker]          = None
    metrics_stacker_reason: str                 = ""
    judge_labels_path: Optional[Path]           = None
    judge_labels_n:  Optional[int]              = None
    sources:        List[Tuple[str, Path]]      = field(default_factory=list)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def test_metrics(self) -> dict:
        return dig(self.results, "test_metrics", default={}) or {}

    @property
    def training_history(self) -> dict:
        return dig(self.results, "training_history", default={}) or {}

    @property
    def run_info(self) -> dict:
        return dig(self.results, "run_info", default={}) or {}

    def clf_cfg(self, key, default=None):
        return dig(self.cfg, "layer1", "classifier", key, default=default)

    def enabled_benchmarks(self) -> List[dict]:
        benches = dig(self.cfg, "evaluation", "benchmarks", default=[]) or []
        return [b for b in benches if b.get("enabled")]

    def disabled_benchmarks(self) -> List[dict]:
        benches = dig(self.cfg, "evaluation", "benchmarks", default=[]) or []
        return [b for b in benches if not b.get("enabled")]

    def split_sentence(self) -> str:
        """The pool sizes, from line counts, cross-checked against metadata.json."""
        parts = []
        for split in POOL_SPLITS:
            n = self.line_counts.get(split)
            parts.append(f"{split} {n:,}" if n is not None else f"{split} {UNAVAILABLE}")
        text = " / ".join(parts)

        meta_parts = []
        for split in POOL_SPLITS:
            n = dig(self.pool_meta, "splits", split, "n")
            if n is not None:
                meta_parts.append(f"{split} {n:,}")
        if not meta_parts:
            return text
        agree = all(
            dig(self.pool_meta, "splits", s, "n") == self.line_counts.get(s)
            for s in POOL_SPLITS
        )
        if agree:
            return f"{text} (metadata.json agrees)"
        return f"{text}; metadata.json instead records {' / '.join(meta_parts)} — they disagree"


def identify_metrics_stacker(results: Optional[dict],
                             artifact_dir: Path) -> Tuple[Optional[Stacker], str]:
    """
    Work out which stacker the recorded test_metrics belong to, from files only.

    results.json does not name a stacker, but it stores the temperature the run
    calibrated with, and each stacker has its own scaler file. Whichever scaler
    holds that exact T is the one the recorded metrics were produced under.
    """
    recorded = dig(results, "temperature_T")
    if recorded is None:
        return None, "results.json records no temperature_T, so the stacker cannot be identified"

    matches = []
    for stacker in STACKERS.values():
        data = read_json(Path(artifact_dir) / stacker.scaler_file)
        if data is None:
            continue
        if data.get("T") == recorded:
            matches.append(stacker)

    if len(matches) == 1:
        s = matches[0]
        return s, (f"results.json records temperature_T = {recorded!r}, the exact value in "
                   f"{s.scaler_file}, which belongs to the {s.label}")
    if not matches:
        return None, (f"results.json records temperature_T = {recorded!r}, which matches no "
                      f"scaler file in {artifact_dir}, so the stacker cannot be identified")
    names = " and ".join(s.scaler_file for s in matches)
    return None, (f"results.json records temperature_T = {recorded!r}, which appears in both "
                  f"{names}, so the stacker cannot be identified from it")


def collect_facts(cfg: dict, config_path: Path, artifact_dir: Path, pool_dir: Path,
                  stacker: Stacker) -> Facts:
    """Read every source file once. Missing files become None, never a literal."""
    facts = Facts(cfg=cfg, config_path=Path(config_path),
                  artifact_dir=Path(artifact_dir), pool_dir=Path(pool_dir),
                  stacker=stacker)

    facts.results_path = Path(artifact_dir) / "results.json"
    facts.results      = read_json(facts.results_path)

    facts.pool_meta_path = Path(pool_dir) / "metadata.json"
    facts.pool_meta      = read_json(facts.pool_meta_path)

    for split in POOL_SPLITS:
        facts.line_counts[split] = count_lines(Path(pool_dir) / f"{split}.jsonl")

    facts.categories, facts.category_total = pool_category_counts(pool_dir)

    facts.adapter_cfg = read_json(
        Path(artifact_dir) / "classifier" / "checkpoint_best" / "adapter_config.json"
    )

    for s in STACKERS.values():
        data = read_json(Path(artifact_dir) / s.scaler_file)
        facts.scaler_ts[s.key] = (data or {}).get("T")

    facts.metrics_stacker, facts.metrics_stacker_reason = identify_metrics_stacker(
        facts.results, artifact_dir
    )

    pool_path = dig(cfg, "layer2", "distillation", "pool_path", default=str(pool_dir))
    facts.judge_labels_path = Path(pool_path) / "judge_labels.jsonl"
    facts.judge_labels_n    = (count_lines(facts.judge_labels_path)
                               if facts.judge_labels_path.exists() else None)

    facts.sources = [
        ("Config",                    Path(config_path)),
        ("Layer-1 results",           facts.results_path),
        ("Stacker (this report)",     Path(artifact_dir) / stacker.fusion_file),
        ("Temperature scaler",        Path(artifact_dir) / stacker.scaler_file),
        ("LoRA adapter config",       Path(artifact_dir) / "classifier" / "checkpoint_best" / "adapter_config.json"),
        ("Distillation pool metadata", facts.pool_meta_path),
        ("Pool — train split",        Path(pool_dir) / "train.jsonl"),
        ("Pool — val split",          Path(pool_dir) / "val.jsonl"),
        ("Pool — test split",         Path(pool_dir) / "test.jsonl"),
        ("Layer-2 Judge labels",      facts.judge_labels_path),
    ]
    return facts


# ─────────────────────────────────────────────────────────────────────────────
# Word document builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(facts: Facts, run: Optional[PlotRun], plots_dir: Path,
                 out_path: Path) -> None:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    cfg = facts.cfg
    stacker = facts.stacker

    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    # ── Style helpers ─────────────────────────────────────────────────────────
    def set_font(run_, size=11, bold=False, color=None, italic=False):
        run_.font.size   = Pt(size)
        run_.font.bold   = bold
        run_.font.italic = italic
        if color:
            run_.font.color.rgb = RGBColor(*color)

    def heading1(text):
        p = doc.add_heading(text, level=1)
        p.runs[0].font.color.rgb = RGBColor(31, 73, 125)
        return p

    def heading2(text):
        p = doc.add_heading(text, level=2)
        p.runs[0].font.color.rgb = RGBColor(68, 114, 196)
        return p

    def body(text, space_after=6):
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.alignment   = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in p.runs:
            r.font.size = Pt(11)
        return p

    def caution(text):
        """A claim the reader must not skim past."""
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.alignment   = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_font(r, size=11, bold=True, color=(146, 43, 33))
        return p

    def bullet(text):
        p = doc.add_paragraph(text, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        for r in p.runs:
            r.font.size = Pt(11)
        return p

    def add_figure(img_path: Path, caption: str, width=5.8):
        img_path = Path(img_path)
        if img_path.exists():
            doc.add_picture(str(img_path), width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(12)
            for r in cap.runs:
                r.font.size   = Pt(9)
                r.font.italic = True
                r.font.color.rgb = RGBColor(89, 89, 89)
        else:
            body(f"[Figure not generated: {img_path.name}]")

    def table3(rows, headers=("Metric", "Value", "Source / interpretation")):
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Shading Accent 1"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = h
            for r in hdr[i].paragraphs[0].runs:
                r.font.bold = True
                r.font.size = Pt(10)
        for row_data in rows:
            cells = table.add_row().cells
            for i, val in enumerate(row_data):
                cells[i].text = str(val)
                for r in cells[i].paragraphs[0].runs:
                    r.font.size = Pt(9)
        doc.add_paragraph()
        return table

    clf_only_cfg = bool(dig(cfg, "layer1", "classifier_only", default=False))
    shipped      = STACKERS[STACKER_CLF_ONLY if clf_only_cfg else STACKER_FULL]

    # ═════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═════════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_paragraph()
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(title_p.add_run("GaurdLLM"), size=32, bold=True, color=(31, 73, 125))

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(sub_p.add_run("Layer 1 — Safety Detector: Technical Report"),
             size=18, color=(68, 114, 196))

    stack_p = doc.add_paragraph()
    stack_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(stack_p.add_run(f"Stacker described in this report: {stacker.label}"),
             size=13, bold=True, color=(146, 43, 33))

    doc.add_paragraph()
    div_p = doc.add_paragraph()
    div_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(div_p.add_run("─" * 55), color=(128, 128, 128))

    doc.add_paragraph()
    device = dig(facts.results, "config_snapshot", "device")
    fp16   = dig(facts.results, "config_snapshot", "fp16")
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(meta.add_run(
        f"Generated: {date.today().strftime('%B %d, %Y')}\n"
        f"Artifacts: {facts.artifact_dir}\n"
        f"Classifier backbone: {facts.clf_cfg('backbone', UNAVAILABLE)} + LoRA\n"
        f"Training device (results.json): {device if device is not None else UNAVAILABLE}"
        f"  |  fp16: {fp16 if fp16 is not None else UNAVAILABLE}\n"
        f"Every figure in this document names the stacker that produced it."
    ), size=11, italic=True, color=(89, 89, 89))

    doc.add_page_break()

    # ═════════════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    heading1("1. Executive Summary")

    body(
        "GaurdLLM is a multi-layer safety framework that classifies incoming prompts "
        "before they reach a large language model. Layer 1 is the first and cheapest "
        "tier: it scores a prompt and routes it to allow, gray zone, or mitigate."
    )

    body(
        f"Layer 1 ships two stackers in the same artifact directory and "
        f"layer1.classifier_only selects between them. In {facts.config_path} that flag is "
        f"{clf_only_cfg}, so the serving path loads the {shipped.label}."
    )
    if stacker.key == shipped.key:
        body(
            f"This report describes that same stacker: {stacker.fusion_file} + "
            f"{stacker.scaler_file}."
        )
    else:
        caution(
            f"This report was asked for the {stacker.label} ({stacker.fusion_file} + "
            f"{stacker.scaler_file}), which is not the stacker this configuration loads. "
            f"Nothing below describes the serving path."
        )

    body(
        f"Layer 1 was trained on the distillation pool at {facts.pool_dir}. Split sizes, "
        f"counted from the pool files at report time: {facts.split_sentence()}."
    )

    tm = facts.test_metrics
    if tm:
        body(
            f"The evaluation recorded in {facts.results_path} reports "
            f"AUROC = {num(tm.get('auroc'))}, AUPRC = {num(tm.get('auprc'))}, "
            f"accuracy = {num(tm.get('accuracy'))} and FPR@95%TPR = "
            f"{num(tm.get('fpr@95tpr'))} on {num(tm.get('n_samples'), '{:,}')} test prompts."
        )
        if facts.metrics_stacker and facts.metrics_stacker.key != stacker.key:
            caution(
                f"Those metrics describe the {facts.metrics_stacker.label}, not the "
                f"{stacker.label} that this configuration loads. They are not a measurement "
                f"of the shipped model."
            )
        elif facts.metrics_stacker is None:
            caution(
                "Which stacker produced those metrics could not be determined from the "
                f"artifacts: {facts.metrics_stacker_reason}. Read them as unattributed."
            )
    else:
        body(f"No test metrics are available: {facts.results_path} could not be read.")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. SYSTEM OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    heading1("2. System Overview")

    body("GaurdLLM is three cooperative layers, described here as the config configures them:")

    bullet(
        "Layer 1 (this report) — signal fusion over the distilled classifier, with "
        "curvature and log-rank available; routes each prompt to allow, gray zone or mitigate."
    )
    judge_model = dig(cfg, "layer2", "judge", "model_name", default=UNAVAILABLE)
    mode        = dig(cfg, "layer2", "consistency", "mode", default=UNAVAILABLE)
    bullet(
        f"Layer 2 — LLM-as-Judge on gray-zone prompts. Model: {judge_model}. "
        f"Decision mode: {mode} (K = {dig(cfg, 'layer2', 'consistency', 'K', default=UNAVAILABLE)}, "
        f"c_min = {dig(cfg, 'layer2', 'consistency', 'c_min', default=UNAVAILABLE)})."
    )
    l3_mode = dig(cfg, "layer3", "mitigation", "default_mode", default=UNAVAILABLE)
    bullet(
        f"Layer 3 — Guard-LLM with constrained decoding, for prompts Layer 1 or Layer 2 "
        f"mitigate. It is implemented and configured, not planned: "
        f"layer3.mitigation.default_mode is \"{l3_mode}\"."
    )

    heading2("2.1 Layer 1 Architecture")
    body(
        "Layer 1 computes feature signals from the input text, fuses them through a "
        "logistic stacker, applies post-hoc temperature calibration, and routes the "
        "calibrated score through a dual-threshold gate. Which signals are computed "
        "depends on the stacker: the classifier-only stacker declares "
        "needs_lm_features = False, so curvature and log-rank are never computed."
    )

    curv = dig(cfg, "layer1", "curvature", default={}) or {}
    arch_rows = [
        ("(a) CurvatureFeature", "distilgpt2",
         f"M = {curv.get('num_perturbations', UNAVAILABLE)} perturbations, temperature range "
         f"{curv.get('temperature_range', UNAVAILABLE)} (config layer1.curvature). "
         f"{'Not read by this stacker.' if stacker.is_clf_only() else 'Read by this stacker.'}"),
        ("(b) LogRankFeature", "distilgpt2 (shared)",
         f"Average log-percentile token rank; vocab_size "
         f"{dig(cfg, 'layer1', 'log_rank', 'vocab_size', default='null (auto-detected)')}. "
         f"{'Not read by this stacker.' if stacker.is_clf_only() else 'Read by this stacker.'}"),
        ("(c) ClassifierFeature", f"{facts.clf_cfg('backbone', UNAVAILABLE)} + LoRA",
         "Fine-tuned binary classifier; outputs g(x) = P(allow). Read by both stackers."),
        (f"Stacker ({stacker.fusion_file})", "scikit-learn LogisticRegression",
         f"Feature columns: {', '.join(stacker.features)}."),
        (f"TemperatureScaler ({stacker.scaler_file})", "LBFGS optimised",
         f"T = {num(facts.scaler_ts.get(stacker.key), '{:.6f}')} (read from "
         f"{stacker.scaler_file})."),
        ("ThresholdRouter",
         f"τ_low = {dig(cfg, 'layer1', 'thresholds', 'low', default=UNAVAILABLE)}, "
         f"τ_high = {dig(cfg, 'layer1', 'thresholds', 'high', default=UNAVAILABLE)}",
         "From config layer1.thresholds."),
    ]
    table3(arch_rows, headers=("Component", "Model / Method", "Description"))

    # ═════════════════════════════════════════════════════════════════════════
    # 3. DATASET
    # ═════════════════════════════════════════════════════════════════════════
    heading1("3. Dataset")

    heading2("3.1 Distillation Pool")
    body(
        f"The pool at {facts.pool_dir} was built by scripts/build_datasets.py from public "
        f"datasets, not generated programmatically. Sizes below are line counts taken at "
        f"report time, with metadata.json quoted beside them."
    )

    split_rows = []
    for split, role in (
        ("train", "Classifier fine-tuning and stacker fitting"),
        ("val",   "Early stopping and temperature calibration"),
        ("test",  "Final evaluation — held out"),
    ):
        counted = facts.line_counts.get(split)
        meta_n  = dig(facts.pool_meta, "splits", split, "n")
        labels  = dig(facts.pool_meta, "splits", split, "labels", default={}) or {}
        note = role
        if labels:
            note += (f" · allow {num(labels.get('allow'), '{:,}')} / "
                     f"mitigate {num(labels.get('mitigate'), '{:,}')}")
        split_rows.append((
            f"{split} split",
            f"{counted:,} lines" if counted is not None else UNAVAILABLE,
            f"metadata.json: {meta_n:,}" if meta_n is not None else "metadata.json: unavailable",
            note,
        ))
    total_counted = [n for n in facts.line_counts.values() if n is not None]
    if len(total_counted) == len(POOL_SPLITS):
        split_rows.append(("Pool total", f"{sum(total_counted):,} lines", "", "Sum of the three splits"))
    table3(split_rows, headers=("Split", "Counted at report time", "Recorded", "Role / label balance"))

    heading2("3.2 Pool Composition")
    sources = dig(facts.pool_meta, "splits", "train", "sources", default=None)
    if sources:
        body("Source datasets contributing to the train split, from metadata.json:")
        table3(
            [(name, f"{n:,}", "metadata.json → splits.train.sources") for name, n in
             sorted(sources.items(), key=lambda kv: -kv[1])],
            headers=("Source dataset", "Train prompts", "Read from"),
        )
        revs = dig(facts.pool_meta, "source_revisions", default={}) or {}
        if revs:
            body("Each source is pinned to a revision in metadata.json:")
            for name, info in sorted(revs.items()):
                bullet(f"{name}: {info.get('repo', UNAVAILABLE)} @ "
                       f"{info.get('revision', UNAVAILABLE)}")
    else:
        body(f"Pool composition is unavailable: {facts.pool_meta_path} could not be read.")

    heading2("3.3 Category Labels Present in the Pool")
    configured = dig(cfg, "data", "violation_categories", default=[]) or []
    if configured:
        body(
            "The config lists these violation categories under data.violation_categories: "
            + ", ".join(configured) + ". Those are the project's target categories, not a "
            "description of the pool."
        )
    if facts.categories:
        top = facts.categories.most_common(12)
        body(
            f"The category labels the pool records, counted across all three splits "
            f"({facts.category_total:,} prompts, {len(facts.categories)} distinct labels). "
            f"The {len(top)} most frequent:"
        )
        table3(
            [(cat, f"{n:,}", f"{100 * n / facts.category_total:.1f}%") for cat, n in top],
            headers=("Category label in the pool", "Prompts", "Share of pool"),
        )
        body(
            "These labels come from the source datasets, so they do not line up with the "
            "configured category list."
        )
    else:
        body("Pool category labels are unavailable: the pool .jsonl files could not be read.")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. METHODOLOGY
    # ═════════════════════════════════════════════════════════════════════════
    heading1("4. Methodology")

    heading2("4.1 Signal 1 — Curvature Feature (fcurv)")
    body(
        "Following DetectGPT (Mitchell et al., 2023), the curvature feature measures how "
        "stable a text's log-probability is under perturbations of the scoring temperature. "
        f"Config layer1.curvature sets M = {curv.get('num_perturbations', UNAVAILABLE)} "
        f"perturbations over temperature range {curv.get('temperature_range', UNAVAILABLE)} "
        f"and top-p range {curv.get('top_p_range', UNAVAILABLE)}."
    )
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("fcurv(x) = sigmoid( mean_m [ NLL_m(x) − NLL(x) ] )"), size=11, italic=True)
    doc.add_paragraph()
    if stacker.is_clf_only():
        body("This stacker does not read fcurv, and the detector does not compute it.")

    heading2("4.2 Signal 2 — Log-Rank Feature (frank)")
    body(
        "Following DetectLLM (Su et al., 2023) and GLTR (Gehrmann et al., 2019), the "
        "log-rank feature measures the average log-percentile rank of each token under "
        "distilgpt2."
    )
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("frank(x) = (1/T) × Σ_t  log(1 + rank(w_t) / |V|)"), size=11, italic=True)
    doc.add_paragraph()
    if stacker.is_clf_only():
        body("This stacker does not read frank, and the detector does not compute it.")

    heading2("4.3 Signal 3 — Distilled Classifier (g(x))")
    body(
        f"A {facts.clf_cfg('backbone', UNAVAILABLE)} model is fine-tuned for binary safety "
        "classification with Low-Rank Adaptation (Hu et al., 2022). It outputs "
        "g(x) = P(allow), so a high g means the prompt looks safe."
    )
    adapter = facts.adapter_cfg or {}
    clf_rows = [
        ("Base model", facts.clf_cfg("backbone", UNAVAILABLE), "config layer1.classifier.backbone"),
        ("LoRA rank r", adapter.get("r", facts.clf_cfg("lora_r", UNAVAILABLE)),
         "adapter_config.json" if "r" in adapter else "config layer1.classifier.lora_r"),
        ("LoRA alpha", adapter.get("lora_alpha", facts.clf_cfg("lora_alpha", UNAVAILABLE)),
         "adapter_config.json" if "lora_alpha" in adapter else "config layer1.classifier.lora_alpha"),
        ("LoRA dropout", adapter.get("lora_dropout", facts.clf_cfg("lora_dropout", UNAVAILABLE)),
         "adapter_config.json" if "lora_dropout" in adapter else "config layer1.classifier.lora_dropout"),
        ("LoRA target modules", ", ".join(adapter.get("target_modules", [])) or UNAVAILABLE,
         "adapter_config.json"),
        ("Max sequence length", facts.clf_cfg("max_length", UNAVAILABLE), "config layer1.classifier.max_length"),
        ("Batch size", facts.clf_cfg("batch_size", UNAVAILABLE), "config layer1.classifier.batch_size"),
        ("Learning rate", facts.clf_cfg("learning_rate", UNAVAILABLE), "config layer1.classifier.learning_rate"),
        ("Epochs", facts.clf_cfg("epochs", UNAVAILABLE), "config layer1.classifier.epochs"),
        ("Early stopping patience", facts.clf_cfg("early_stopping_patience", UNAVAILABLE),
         "config layer1.classifier.early_stopping_patience"),
    ]
    table3(clf_rows, headers=("Setting", "Value", "Read from"))
    body(
        "The trainable-parameter count is not recorded in any artifact, so it is not "
        "quoted here."
    )

    heading2("4.4 Logistic Fusion")
    body(
        "A logistic-regression stacker combines the signals it is given. The two stackers "
        "in the artifact directory differ only in which columns they read:"
    )
    bullet(
        f"{STACKERS[STACKER_FULL].fusion_file} — three columns "
        f"({', '.join(STACKERS[STACKER_FULL].features)}). "
        "s_raw(x) = σ( w₀ + w₁·fcurv(x) + w₂·frank(x) + w₃·g(x) )."
    )
    bullet(
        f"{STACKERS[STACKER_CLF_ONLY].fusion_file} — one column "
        f"({', '.join(STACKERS[STACKER_CLF_ONLY].features)}). "
        "s_raw(x) = σ( w₀ + w₁·g(x) ). Curvature and log-rank are neither read nor computed."
    )
    if run is not None:
        clf_est   = getattr(run.fusion, "_clf", None)
        coefs     = getattr(clf_est, "coef_", None)
        intercept = getattr(clf_est, "intercept_", None)
        if coefs is not None and intercept is not None:
            pairs = ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(stacker.features, list(coefs[0]))
            )
            body(
                f"Coefficients read from {stacker.fusion_file} at report time (scikit-learn "
                f"fits toward label 1 = allow, so a positive weight pushes toward allow): "
                f"intercept={float(intercept[0]):.4f}, {pairs}."
            )
    body(
        "Both stackers are fitted in the same run: the three-signal one is fitted first "
        "with a g(x) = 0.5 placeholder, then refitted once the classifier is trained."
    )

    heading2("4.5 Temperature Calibration")
    body(
        "Post-hoc temperature scaling (Guo et al., 2017) calibrates the raw fusion score "
        "on the validation split by finding the T that minimises negative log-likelihood. "
        "Each stacker has its own scaler file:"
    )
    table3(
        [(s.scaler_file, num(facts.scaler_ts.get(s.key), "{:.6f}"), s.label)
         for s in STACKERS.values()],
        headers=("Scaler file", "T", "Belongs to"),
    )
    recorded_T = dig(facts.results, "temperature_T")
    if recorded_T is not None:
        body(
            f"results.json records temperature_T = {num(recorded_T, '{:.6f}')} for the run "
            f"that produced the metrics in Section 5.1. The scaler divides the score's logit "
            f"by T before the sigmoid, so a T above 1 moves scores toward 0.5 and a T below 1 "
            f"pushes them toward 0 and 1."
        )
    else:
        body(f"results.json records no temperature_T, so the calibrated temperature of "
             f"the recorded run is {UNAVAILABLE}.")

    heading2("4.6 Threshold Routing")
    tl = dig(cfg, "layer1", "thresholds", "low")
    th = dig(cfg, "layer1", "thresholds", "high")
    body("The calibrated score s(x) is compared against two thresholds from config "
         "layer1.thresholds:")
    if tl is not None and th is not None:
        table3([
            (f"s(x) < {tl}",            "→  ALLOW",     "Pass through to the LLM"),
            (f"{tl} ≤ s(x) ≤ {th}",     "→  GRAY ZONE", "Escalate to the Layer-2 Judge"),
            (f"s(x) > {th}",            "→  MITIGATE",  "Hand to Layer 3"),
        ], headers=("Calibrated score", "Decision", "Effect"))
    else:
        body(f"Routing thresholds are {UNAVAILABLE} in {facts.config_path}.")

    heading2("4.7 Training Pipeline")
    body("scripts/train_layer1.py runs these phases in order:")
    pipeline = [
        f"Load the distillation pool splits from {facts.pool_dir} (sizes in Section 3.1)",
        "Load the distilgpt2 scoring LM (shared by curvature and log-rank)",
        ("Compute fcurv and frank for every record and cache them to "
         "data/processed/layer1_features/ with a fingerprint file"),
        "Fit the three-signal stacker with a g(x) = 0.5 placeholder (fusion_v1)",
        (f"Fine-tune the {facts.clf_cfg('backbone', UNAVAILABLE)} + LoRA classifier "
         f"(batch size {facts.clf_cfg('batch_size', UNAVAILABLE)}, learning rate "
         f"{facts.clf_cfg('learning_rate', UNAVAILABLE)}, up to "
         f"{facts.clf_cfg('epochs', UNAVAILABLE)} epochs, early-stopping patience "
         f"{facts.clf_cfg('early_stopping_patience', UNAVAILABLE)})"),
        "Recompute g(x) with the trained classifier and refit the three-signal stacker (fusion_v2)",
        "Fit the classifier-only stacker on g alone (fusion_clf_only)",
        (f"Calibrate each stacker's TemperatureScaler on the validation split "
         f"(T = {num(facts.scaler_ts.get(STACKER_FULL), '{:.6f}')} for "
         f"{STACKERS[STACKER_FULL].fusion_file}, "
         f"{num(facts.scaler_ts.get(STACKER_CLF_ONLY), '{:.6f}')} for "
         f"{STACKERS[STACKER_CLF_ONLY].fusion_file})"),
        "Evaluate on the test split: AUROC, AUPRC, FPR@95%TPR, accuracy",
        f"Write every artifact to {facts.artifact_dir}",
    ]
    for i, step in enumerate(pipeline, 1):
        bullet(f"Step {i}: {step}")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    heading1("5. Results")

    heading2("5.1 Test Set Performance")
    if tm:
        caution(
            f"Which stacker these metrics describe: {facts.metrics_stacker_reason}."
        )
        if facts.metrics_stacker and facts.metrics_stacker.key != shipped.key:
            caution(
                f"The numbers below therefore belong to the {facts.metrics_stacker.label}. "
                f"The configuration ships the {shipped.label}, and no test-split evaluation "
                f"of that stacker is recorded in {facts.results_path}. Do not read them as "
                f"its performance."
            )
        table3([
            ("AUROC", num(tm.get("auroc")), "results.json → test_metrics.auroc"),
            ("AUPRC", num(tm.get("auprc")), "results.json → test_metrics.auprc"),
            ("FPR @ 95% TPR", num(tm.get("fpr@95tpr")), "results.json → test_metrics['fpr@95tpr']"),
            ("Threshold @ 95% TPR", num(tm.get("threshold_at_95tpr")),
             "results.json → test_metrics.threshold_at_95tpr"),
            ("Accuracy", num(tm.get("accuracy")), "results.json → test_metrics.accuracy"),
            ("Test set size", num(tm.get("n_samples"), "{:,}"), "results.json → test_metrics.n_samples"),
        ])
        n_test = facts.line_counts.get("test")
        n_rec  = tm.get("n_samples")
        if n_test is not None and n_rec is not None and n_test != n_rec:
            caution(
                f"The recorded evaluation covered {n_rec:,} prompts, but the test split on "
                f"disk now holds {n_test:,} lines. The pool has changed since that run."
            )
    else:
        body(f"Test metrics are {UNAVAILABLE}: {facts.results_path} could not be read.")

    heading2("5.2 Training History")
    hist = facts.training_history
    if hist.get("train_loss"):
        tl_, vl_, va_ = hist["train_loss"], hist["val_loss"], hist["val_acc"]
        table3(
            [(f"Epoch {i}", f"{a:.6f}", f"{b:.6f}", f"{c:.4f}")
             for i, (a, b, c) in enumerate(zip(tl_, vl_, va_), 1)],
            headers=("Epoch", "Train Loss", "Val Loss", "Val Accuracy"),
        )
        best_vl = min(range(len(vl_)), key=lambda i: vl_[i]) + 1
        best_va = max(range(len(va_)), key=lambda i: va_[i]) + 1
        body(
            f"Train loss fell monotonically from {tl_[0]:.6f} to {tl_[-1]:.6f} over "
            f"{len(tl_)} epochs. Validation loss was lowest at epoch {best_vl} "
            f"({min(vl_):.6f}) and ended at {vl_[-1]:.6f}, so the last epoch is not the "
            f"best one by validation loss. Validation accuracy peaked at epoch {best_va} "
            f"({max(va_):.4f})."
        )
        patience = facts.clf_cfg("early_stopping_patience")
        if patience is not None:
            body(
                f"Early stopping with patience {patience} was configured; the recorded run "
                f"has {len(tl_)} epochs against a configured maximum of "
                f"{facts.clf_cfg('epochs', UNAVAILABLE)}."
            )
    else:
        body(f"Training history is {UNAVAILABLE}: {facts.results_path} could not be read.")

    heading2("5.3 Reading Accuracy Against AUROC")
    if tm:
        body(
            f"Accuracy ({num(tm.get('accuracy'))}) and AUROC ({num(tm.get('auroc'))}) answer "
            "different questions. AUROC scores the ranking of harmful above safe and ignores "
            "where the thresholds sit; accuracy depends on them. FPR@95%TPR "
            f"({num(tm.get('fpr@95tpr'))}) is the share of safe prompts flagged at the "
            "operating point that catches 95% of harmful ones — it is a measured rate, not zero."
        )
    body(
        "The gray zone between the two thresholds is deliberate: it escalates to the "
        "Layer-2 Judge instead of forcing a decision."
    )
    if run is not None and len(run.cal_scores):
        counts = run.routing_counts()
        total  = sum(counts.values())
        body(
            f"Recomputed at report time over {total:,} cached test-feature records through "
            f"{stacker.fusion_file}: "
            + ", ".join(f"{k} {v:,} ({100 * v / total:.1f}%)" for k, v in counts.items())
            + "."
        )
        if run.caveat:
            caution(
                f"{run.caveat} The routing split just quoted, and Figures 2, 3 and 6, are "
                f"therefore a property of the feature cache, not of the model. "
                f"scripts/train_layer1.py writes g = 0.5 into "
                f"{run.features_path} and never writes the trained classifier score back."
            )

    heading2("5.4 Calibration")
    body(
        f"The scaler this report describes, {stacker.scaler_file}, holds "
        f"T = {num(facts.scaler_ts.get(stacker.key), '{:.6f}')}. Section 4.5 lists the "
        f"temperature of each stacker's scaler and the one recorded in results.json."
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 6. FIGURES
    # ═════════════════════════════════════════════════════════════════════════
    heading1("6. Diagnostic Figures")

    if run is None:
        caution(
            "The figures could not be generated for this report — the stacker, scaler or "
            "cached feature file could not be read. Any images below are left over from an "
            "earlier run and are not described by the text above."
        )
    else:
        body(
            f"All six figures below were produced from the {stacker.label}, calibrated with "
            f"{stacker.scaler_file}, over the cached test features at {run.features_path} "
            f"({len(run.records):,} records). Each filename and each figure title names that "
            f"stacker, so a figure from the other stacker cannot be mistaken for one of these."
        )
        if run.caveat:
            caution(run.caveat)

    def fig_path(name: str) -> Path:
        if run is not None:
            return run.paths[name]
        return Path(plots_dir) / f"{name}_{stacker.slug}.png"

    add_figure(
        fig_path("training_curves"),
        f"Figure 1. Classifier training curves from results.json — train and validation "
        f"loss (left) and validation accuracy (right). These come from the distilled "
        f"classifier's training run and do not depend on the stacker; the file is named "
        f"for the {stacker.label} only so the two runs' outputs stay separate."
    )
    add_figure(
        fig_path("roc_pr_curves"),
        f"Figure 2. ROC (left) and Precision-Recall (right) recomputed at report time for "
        f"the {stacker.label} over the cached test features. The areas printed on the axes "
        f"are what that recomputation produced; they are not the values in Section 5.1."
    )
    add_figure(
        fig_path("score_distribution"),
        f"Figure 3. Calibrated score distribution by true class for the {stacker.label}. "
        f"Dashed lines mark the routing thresholds from config "
        f"(τ_low = {tl if tl is not None else UNAVAILABLE}, "
        f"τ_high = {th if th is not None else UNAVAILABLE}); the amber band is the gray zone."
    )
    add_figure(
        fig_path("feature_scatter"),
        "Figure 4. fcurv against frank for a sample of the cached test features. "
        + ("Neither axis is an input to the classifier-only stacker, which scores on g alone."
           if stacker.is_clf_only() else
           "Both are inputs to the three-signal stacker.")
    )
    add_figure(
        fig_path("feature_weights"),
        f"Figure 5. Logistic coefficients read from {stacker.fusion_file}. "
        f"scikit-learn fits toward label 1 = allow, so the chart flips the signs to make "
        f"up mean 'toward harm'; g is itself P(allow), so its bar pointing down is the "
        f"expected direction."
    )
    add_figure(
        fig_path("decision_routing"),
        f"Figure 6. Routing decisions per true class for the {stacker.label}, produced by "
        f"layer1.fusion.ThresholdRouter at the config thresholds."
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 7. STATUS OF THE REST OF THE PIPELINE
    # ═════════════════════════════════════════════════════════════════════════
    heading1("7. Status of the Rest of the Pipeline")
    body("Described from the configuration and the code as they stand, not as a roadmap.")

    heading2("7.1 Layer 2 and the distillation loop")
    body(
        f"Layer 2 runs {judge_model} as a Judge on gray-zone prompts "
        f"(layer2.consistency.mode = {mode}, K = "
        f"{dig(cfg, 'layer2', 'consistency', 'K', default=UNAVAILABLE)}, c_min = "
        f"{dig(cfg, 'layer2', 'consistency', 'c_min', default=UNAVAILABLE)})."
    )
    caution(
        "The distillation loop does not retrain Layer 1. layer2/distillation.py pools "
        "Judge labels and appends them to judge_labels.jsonl; no code path reads that file "
        "back. scripts/train_layer1.py trains from the pool splits only. Any accuracy "
        "improvement attributed to the loop would be unmeasured."
    )
    retrain_n = dig(cfg, "layer2", "distillation", "retrain_every_n", default=UNAVAILABLE)
    if facts.judge_labels_n is not None:
        body(
            f"config layer2.distillation.retrain_every_n is {retrain_n}. "
            f"{facts.judge_labels_path} currently holds {facts.judge_labels_n:,} buffered "
            f"labels, none of which have been trained on."
        )
    else:
        body(
            f"config layer2.distillation.retrain_every_n is {retrain_n}. "
            f"{facts.judge_labels_path} does not exist, so no Judge labels have been "
            f"buffered yet."
        )

    heading2("7.2 Layer 3")
    l3 = dig(cfg, "layer3", default={}) or {}
    body(
        "Layer 3 is implemented and enabled. Its configured behaviour, read from config "
        "layer3:"
    )
    table3([
        ("Guard-LLM model", dig(l3, "guard_llm", "model_name", default=UNAVAILABLE), "layer3.guard_llm.model_name"),
        ("Default mitigation mode", dig(l3, "mitigation", "default_mode", default=UNAVAILABLE), "layer3.mitigation.default_mode"),
        ("Available modes", ", ".join(dig(l3, "mitigation", "modes", default=[]) or []) or UNAVAILABLE, "layer3.mitigation.modes"),
        ("Severity refuse threshold", dig(l3, "mitigation", "severity_refuse_threshold", default=UNAVAILABLE), "layer3.mitigation.severity_refuse_threshold"),
        ("Refuse when uncategorised", dig(l3, "mitigation", "refuse_when_uncategorised", default=UNAVAILABLE), "layer3.mitigation.refuse_when_uncategorised"),
        ("Candidates K'", dig(l3, "candidates", "K_prime", default=UNAVAILABLE), "layer3.candidates.K_prime"),
        ("Refuse on decline", dig(l3, "candidates", "refuse_on_decline", default=UNAVAILABLE), "layer3.candidates.refuse_on_decline"),
        ("Refusal check mode", dig(l3, "refusal_check", "mode", default=UNAVAILABLE), "layer3.refusal_check.mode"),
        ("Rewrite check mode", dig(l3, "rewrite_check", "mode", default=UNAVAILABLE), "layer3.rewrite_check.mode"),
    ], headers=("Setting", "Value", "Read from"))
    if l3_mode == "refuse":
        body(
            "The default mitigation mode is \"refuse\", not safe_rewrite: a mitigated prompt "
            "receives a refusal rather than a rewritten reply."
        )

    heading2("7.3 Red-teaming")
    rt = dig(cfg, "red_team", default={}) or {}
    body(
        f"config red_team.enabled is {rt.get('enabled', UNAVAILABLE)}, with "
        f"{rt.get('attacks_per_round', UNAVAILABLE)} attacks per round over "
        f"{rt.get('rounds', UNAVAILABLE)} rounds. Attack types: "
        + (", ".join(rt.get("attack_types", []) or []) or UNAVAILABLE) + "."
    )

    heading2("7.4 Benchmark evaluation")
    enabled  = facts.enabled_benchmarks()
    disabled = facts.disabled_benchmarks()
    if enabled:
        body("Benchmarks enabled in config evaluation.benchmarks:")
        table3([(b.get("name", UNAVAILABLE), b.get("path", UNAVAILABLE), "enabled")
                for b in enabled],
               headers=("Benchmark", "Path", "Status"))
    else:
        body(f"No benchmarks are enabled in {facts.config_path}.")
    if disabled:
        body(
            "Also present but disabled, so not part of the default evaluation: "
            + ", ".join(b.get("name", "?") for b in disabled) + "."
        )
    metric_names = dig(cfg, "evaluation", "metrics", default=[]) or []
    if metric_names:
        body("Metrics configured for that evaluation: " + ", ".join(metric_names) + ".")

    heading2("7.5 NIST AI RMF mapping")
    body(
        f"config logging.nist_mapping is {dig(cfg, 'logging', 'nist_mapping', default=UNAVAILABLE)}, "
        f"with the mapping file at "
        f"{dig(cfg, 'logging', 'nist_mapping_path', default=UNAVAILABLE)} and the audit log at "
        f"{dig(cfg, 'logging', 'audit_log_path', default=UNAVAILABLE)}."
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 8. PROVENANCE
    # ═════════════════════════════════════════════════════════════════════════
    heading1("8. Provenance")
    body(
        "So that every number above can be traced to the run that produced it. The fields "
        f"below are read from {facts.results_path} → run_info."
    )

    ri = facts.run_info
    if ri:
        command = ri.get("command")
        if isinstance(command, (list, tuple)):
            command = " ".join(str(c) for c in command)
        dirty = ri.get("git_dirty")
        table3([
            ("Command", command if command else UNAVAILABLE, "run_info.command"),
            ("Git commit", ri.get("git_commit", UNAVAILABLE), "run_info.git_commit"),
            ("Git dirty", str(dirty) if dirty is not None else UNAVAILABLE, "run_info.git_dirty"),
            ("Seed", ri.get("seed", UNAVAILABLE), "run_info.seed"),
            ("Started at", ri.get("started_at", UNAVAILABLE), "run_info.started_at"),
            ("Wall clock (s)", num(ri.get("wall_clock_s"), "{:.1f}"), "run_info.wall_clock_s"),
        ], headers=("Field", "Value", "Read from"))
        if dirty is True:
            caution(
                "run_info.git_dirty is true: the working tree had uncommitted changes when "
                "this run executed, so the commit alone does not reproduce it."
            )
        elif dirty is False:
            body(
                f"run_info.git_dirty is false, so commit {ri.get('git_commit', UNAVAILABLE)} "
                f"is the code that produced these results."
            )
        extra = []
        if ri.get("cwd"):
            extra.append(("Working directory", ri["cwd"], "run_info.cwd"))
        if ri.get("finished_at"):
            extra.append(("Finished at", ri["finished_at"], "run_info.finished_at"))
        if ri.get("python"):
            extra.append(("Python", ri["python"], "run_info.python"))
        gpu = ri.get("gpu") or {}
        if gpu.get("name"):
            extra.append(("GPU", f"{gpu.get('name')} (CUDA {gpu.get('cuda', '?')})", "run_info.gpu"))
        for name, version in (ri.get("packages") or {}).items():
            extra.append((f"Package: {name}", version if version else "not installed",
                          "run_info.packages"))
        if extra:
            body("Environment recorded alongside it:")
            table3(extra, headers=("Field", "Value", "Read from"))
    else:
        body(f"Run provenance is {UNAVAILABLE}: {facts.results_path} has no run_info block.")

    heading2("8.1 Files this report read")
    table3(
        [(label, str(path), file_stamp(path)) for label, path in facts.sources],
        headers=("What", "Path", "State at report time"),
    )
    body(
        f"This document was generated on {date.today().isoformat()}. Every number in it was "
        f"read from one of the files above at that moment; nothing is carried over from an "
        f"earlier run."
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 9. REFERENCES
    # ═════════════════════════════════════════════════════════════════════════
    heading1("9. References")
    for ref in [
        "Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICML 2023.",
        "Su, Y., Zhuo, T. Y., Long, D., Fang, M., Khapra, M., Bing, L., & Poria, S. (2023). DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text. EMNLP 2023.",
        "Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection and Visualization of Generated Text. ACL 2019.",
        "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.",
        "Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML 2017.",
        "Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.",
        "Qwen Team, Alibaba Cloud. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.",
    ]:
        p = doc.add_paragraph(ref, style="List Number")
        for r in p.runs:
            r.font.size = Pt(10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    logger.info(f"Report saved → {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate the Layer-1 technical report (.docx)")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--artifact-dir", default=None,
                   help="override layer1.artifact_subdir (default: artifacts/<layer1.artifact_subdir>)")
    p.add_argument("--out", default=None,
                   help=f"report path (default: <artifact-dir>/{DEFAULT_REPORT_NAME})")
    p.add_argument("--plots-dir", default=None,
                   help="where the figures go (default: <artifact-dir>/plots)")
    p.add_argument("--features-path", default=DEFAULT_FEATURES_PATH)
    p.add_argument("--pool-dir", default=None,
                   help="distillation pool (default: config data.distillation_pool_dir)")
    p.add_argument("--stacker", choices=[STACKER_CONFIG, STACKER_FULL, STACKER_CLF_ONLY],
                   default=STACKER_CONFIG,
                   help="which stacker to report on; 'config' follows layer1.classifier_only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg  = load_config(args.config)

    stacker = resolve_stacker(args.stacker, cfg)
    artifact_dir, plots_dir = resolve_paths(cfg, args.artifact_dir, args.plots_dir)
    report_path = Path(args.out) if args.out else artifact_dir / DEFAULT_REPORT_NAME
    pool_dir = Path(args.pool_dir or dig(cfg, "data", "distillation_pool_dir",
                                         default=DEFAULT_POOL_DIR))

    logger.info("=" * 58)
    logger.info("GaurdLLM — Layer-1 Report Generator")
    logger.info(f"  Stacker   : {stacker.label}")
    logger.info(f"  Artifacts : {artifact_dir}")
    logger.info(f"  Pool      : {pool_dir}")
    logger.info(f"  Plots     : {plots_dir}")
    logger.info(f"  Report    : {report_path}")
    logger.info("=" * 58)

    # ── Step 1: figures ───────────────────────────────────────────────────────
    run: Optional[PlotRun] = None
    try:
        run = prepare_run(cfg, stacker, artifact_dir, plots_dir,
                          Path(args.features_path), Path(args.config))
        draw_all(run)
        logger.info(f"Figures for the {stacker.label} → {plots_dir}")
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        logger.error(
            f"Figures could not be generated ({exc}). The report will be written without "
            "them and will say so."
        )

    # ── Step 2: facts ─────────────────────────────────────────────────────────
    logger.info("Collecting facts from disk...")
    facts = collect_facts(cfg, Path(args.config), artifact_dir, pool_dir, stacker)
    if facts.metrics_stacker_reason:
        logger.info(f"  Recorded metrics: {facts.metrics_stacker_reason}")
    if run is not None and run.caveat:
        logger.warning(f"  {run.caveat}")

    # ── Step 3: document ──────────────────────────────────────────────────────
    logger.info("Building Word document...")
    try:
        build_report(facts, run, plots_dir, report_path)
    except ImportError:
        logger.error(
            "python-docx is not installed.\n"
            "Run:  pip install python-docx\n"
            "Then re-run this script."
        )
        return None

    logger.info("=" * 58)
    logger.info("Done.")
    logger.info(f"  Plots  → {Path(plots_dir).resolve()}")
    logger.info(f"  Report → {Path(report_path).resolve()}")
    logger.info("=" * 58)
    return report_path


if __name__ == "__main__":
    main()
