"""
GaurdLLM — Layer Ablation
===========================
Measures what each part of the pipeline contributes, on the same prompts, in a
single pass over the benchmarks.

Variants — the names on the left are the historical ones, kept because they are
the names of the stored data/processed/ablation_*.json files; the names on the
right are accepted as aliases and say what the variant actually is:

    full          three_signal_cascade      Layer 1 with curvature + log-rank +
                                            classifier → gray zone → judge
    classifier    classifier_raw            the RoBERTa classifier alone:
                                            harmful when 1 - g(x) > 0.5
    no_curvature  classifier_only_cascade   Layer 1 on g(x) alone → gray zone →
                                            judge.  This is what ships whenever
                                            layer1.classifier_only is true
    judge         judge_only                the Layer-2 judge alone, on every
                                            prompt, no Layer 1

"full" is a historical name, not a description: it is the three-signal cascade,
which the shipped configuration does not use. Which variant matches the current
config is read from layer1.classifier_only and printed beside every variant, and
recorded in each output file under "shipped_configuration".

This script never writes into the Layer-1 artifact directory. It reads the
shipped classifier-only stacker from there when one is present, and otherwise
fits its own into --stacker-dir (default <out-dir>/ablation_stacker). Writing
fusion_clf_only.joblib and scaler_clf_only.json back into artifacts/<subdir>/
would replace the model scripts/train_layer1.py fitted and every final number
depends on.

Two stages, so a crash costs minutes instead of hours:

    signals   computes each prompt's raw signals once — curvature, log-rank,
              g(x), P(mitigate), and a full judge call for prompts that land in
              a gray zone — and appends one JSON line per prompt to
              data/processed/ablation_signals.jsonl.  --resume skips prompts
              already in that file.
    derive    turns those signals into one results file per variant, in the
              formats scripts/run_evaluation.py writes, so
              scripts/summarise_results.py can add bootstrap confidence
              intervals and paired comparisons without changes.

Why one pass: the signals do not depend on the variant, so computing them once
and deriving four sets of decisions from them costs one run instead of four,
and every variant sees exactly the same feature values.

Layer 3 is not loaded.  It records the action served but never changes a
decision or a score (evaluation/pipeline_evaluator.py::_run_one), so the
metrics are unaffected; l3_mode is null in the predictions written here.

Latency is composed from measured per-component times rather than wall clock,
because the signals are shared:
    full          features + classifier + judge call when the prompt is gray
    classifier    classifier
    no_curvature  classifier + judge call when the prompt is gray
    judge         the single forward pass that decides, without the greedy
                  metadata call the pipeline also makes

Usage:
    python scripts/run_ablation.py                      # both stages, all variants
    python scripts/run_ablation.py --stage signals --resume
    python scripts/run_ablation.py --stage derive
    python scripts/run_ablation.py --benchmarks xstest --limit 20   # smoke test
    python scripts/run_ablation.py --variants classifier_only_cascade  # alias for no_curvature
    python scripts/run_ablation.py --stacker-dir /tmp/ablation_stacker

Long runs should be wrapped so the laptop cannot suspend mid-run:
    systemd-inhibit --what=idle:sleep:handle-lid-switch --why=ablation \
        python scripts/run_ablation.py --stage signals --resume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.data_utils import set_seed
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation.metrics import compute_all, format_metric

logger = get_logger("run_ablation")

# The keys used everywhere internally and in the output file names. They are the
# historical names, so data/processed/ablation_<variant>.json keeps its spelling
# and the stored signals meta still matches on --resume.
VARIANTS = ("full", "classifier", "no_curvature", "judge")

# Descriptive names accepted on --variants, mapped to the file-name keys above.
VARIANT_ALIASES = {
    "three_signal_cascade": "full",
    "classifier_raw": "classifier",
    "classifier_only_cascade": "no_curvature",
    "judge_only": "judge",
}
CANONICAL_NAME = {legacy: alias for alias, legacy in VARIANT_ALIASES.items()}

WHAT_IT_IS = {
    "full": "three-signal Layer 1 (curvature + log-rank + classifier) → gray zone → judge",
    "classifier": "the classifier alone, harmful when 1 - g(x) > 0.5",
    "no_curvature": "Layer 1 on g(x) alone → gray zone → judge",
    "judge": "the judge alone, on every prompt",
}

# Variants whose cascade sends gray-zone prompts to the judge
CASCADE_VARIANTS = ("full", "no_curvature")

# Where a stacker this script fits itself goes, under --out-dir. Never the live
# artifact directory: scripts/train_layer1.py owns the files there.
DEFAULT_STACKER_SUBDIR = "ablation_stacker"

# layer1/detector.py rounds every score it hands on (DetectionResult) and
# layer2/consistency.py rounds P(mitigate), so the derivation rounds to the same
# 4 decimals. Scoring at full precision breaks ties the pipeline itself keeps,
# which moves AUROC on the benchmarks with many saturated scores.
SCORE_DIGITS = 4

# layer1.fusion.ClassifierOnlyFusion.raw_score keeps the three-argument signature
# of LogisticFusion and ignores fcurv and frank. This is the value passed where
# there is no real one to pass, matching scripts/train_layer1.py.
UNUSED_FEATURE = 0.5

DECISION_ALLOW = "allow"
DECISION_GRAY_ZONE = "gray_zone"
DECISION_MITIGATE = "mitigate"
DECISION_UNCERTAIN = "uncertain"


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ablation of the GaurdLLM layers on the benchmarks",
        epilog="Wrap long runs in systemd-inhibit so the machine cannot suspend.",
    )
    p.add_argument("--stage", choices=["signals", "derive", "both"], default="both",
                   help="signals = the GPU pass; derive = turn signals into results")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS) + sorted(VARIANT_ALIASES),
                   help="Which variants to produce (default: all four). Either the historical "
                        "name, which is also the output file's name, or its descriptive alias: "
                        + ", ".join(f"{legacy} = {CANONICAL_NAME[legacy]}" for legacy in VARIANTS))
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Named benchmarks from config (default: every enabled one)")
    p.add_argument("--input", default=None,
                   help="A single labelled JSONL instead of the benchmarks, e.g. the pool "
                        "validation split for tuning the routing band")
    p.add_argument("--name", default=None,
                   help="Name recorded for --input (default: the file stem)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N prompts per benchmark (smoke test)")
    p.add_argument("--signals", default="data/processed/ablation_signals.jsonl",
                   help="Where the per-prompt signals are appended")
    p.add_argument("--out-dir", default="data/processed",
                   help="Directory for ablation_<variant>.json and its predictions")
    p.add_argument("--stacker-dir", default=None,
                   help="Where a classifier-only stacker fitted by this script is cached "
                        f"(default: <out-dir>/{DEFAULT_STACKER_SUBDIR}). Never the Layer-1 "
                        "artifact directory, whose fusion_clf_only.joblib and "
                        "scaler_clf_only.json belong to scripts/train_layer1.py")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already present in the signals file")
    p.add_argument("--force", action="store_true",
                   help="Resume even when the stored settings differ (not recommended)")
    p.add_argument("--log-every", type=int, default=25, help="Progress interval, in prompts")
    args = p.parse_args(argv)
    args.variants = resolve_variants(args.variants)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Variant names
# ─────────────────────────────────────────────────────────────────────────────

def resolve_variants(names: List[str]) -> List[str]:
    """
    Aliases in, file-name keys out, order kept and duplicates dropped.

    Everything downstream — the settings fingerprint, the output file names, the
    derivation — works in the historical names, so passing an alias produces
    exactly the same files as passing the old name.
    """
    resolved = []
    for name in names:
        legacy = VARIANT_ALIASES.get(name, name)
        if legacy not in resolved:
            resolved.append(legacy)
    return resolved


def stacker_dir(args) -> Path:
    """Where a stacker this script fits itself lives: --stacker-dir, else under --out-dir."""
    if args.stacker_dir:
        return Path(args.stacker_dir)
    return Path(args.out_dir) / DEFAULT_STACKER_SUBDIR


def shipped_variant(cfg: dict) -> str:
    """
    The variant that matches the configuration the pipeline runs.

    layer1.classifier_only decides which stacker Layer1Detector.from_artifacts_for
    loads, and both settings still route the gray zone to the judge, so the
    shipped cascade is no_curvature when it is on and full when it is off.
    """
    return "no_curvature" if bool(cfg.get("layer1", {}).get("classifier_only", False)) else "full"


def variant_label(variant: str, cfg: dict) -> str:
    """'no_curvature (classifier_only_cascade) — CURRENT CONFIG: ...' for printing."""
    label = f"{variant} ({CANONICAL_NAME[variant]}) — {WHAT_IT_IS[variant]}"
    flag = bool(cfg.get("layer1", {}).get("classifier_only", False))
    if variant == shipped_variant(cfg):
        return label + f"  [CURRENT CONFIG: layer1.classifier_only={flag}]"
    return label + f"  [not the current config, which is {shipped_variant(cfg)}]"


def shipped_configuration(cfg: dict) -> dict:
    """The config facts that decide which variant ships, recorded in every output file."""
    l1 = cfg.get("layer1", {})
    shipped = shipped_variant(cfg)
    return {
        "variant": shipped,
        "variant_alias": CANONICAL_NAME[shipped],
        "layer1.classifier_only": bool(l1.get("classifier_only", False)),
        "layer1.artifact_subdir": l1.get("artifact_subdir") or "layer1",
        # Stated because this script scores the raw text: it computes the
        # features directly rather than through Layer1Detector.detect, so the
        # recovered-text and payload views the pipeline also scores are absent.
        "layer1.normalise_input": bool(l1.get("normalise_input", False)),
        "layer1.payload_views": bool(l1.get("payload_views", False)),
        "views_applied_here": "raw text only",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantise(value) -> Optional[float]:
    """A score at the precision the pipeline passes between layers."""
    return None if value is None else round(float(value), SCORE_DIGITS)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def benchmark_files(cfg: dict, names: Optional[List[str]]) -> List[tuple]:
    """[(name, test.jsonl path)] for the requested benchmarks, in config order."""
    configured = [b for b in cfg.get("evaluation", {}).get("benchmarks", []) if b.get("enabled", True)]
    if names:
        by_name = {b["name"]: b for b in configured}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"Benchmarks not enabled in config: {', '.join(missing)}")
        configured = [by_name[n] for n in names]

    out = []
    for bench in configured:
        path = Path(bench["path"]) / "test.jsonl"
        if not path.exists():
            logger.warning(f"Skipping {bench['name']}: {path} not found")
            continue
        out.append((bench["name"], path))
    if not out:
        raise SystemExit("No benchmark files found; run scripts/build_datasets.py first.")
    return out


def load_records(path: Path, limit: Optional[int]) -> List[dict]:
    records = read_jsonl(path)
    return records[:limit] if limit else records


def settings_fingerprint(cfg: dict, args) -> dict:
    """What the signals depend on. A resume against different settings is refused."""
    l1 = cfg["layer1"]
    l2 = cfg["layer2"]
    return {
        "seed": cfg["project"]["seed"],
        "tau_low": l1["thresholds"]["low"],
        "tau_high": l1["thresholds"]["high"],
        "num_perturbations": l1["curvature"]["num_perturbations"],
        "classifier_backbone": l1["classifier"]["backbone"],
        "prompt_version": l2["judge"].get("prompt_version", "v1"),
        "consistency_mode": l2["consistency"].get("mode", "vote"),
        "allow_below": l2["consistency"].get("allow_below", 0.4),
        "mitigate_above": l2["consistency"].get("mitigate_above", 0.6),
        "variants": sorted(args.variants),
        "limit": args.limit,
        "input": args.input,
    }


def meta_path_for(signals_path: Path) -> Path:
    return signals_path.with_suffix(signals_path.suffix + ".meta.json")


def check_or_write_meta(signals_path: Path, fingerprint: dict, resume: bool, force: bool) -> None:
    meta_file = meta_path_for(signals_path)
    if resume and meta_file.exists():
        stored = json.loads(meta_file.read_text(encoding="utf-8")).get("settings", {})
        if stored != fingerprint:
            differing = sorted(set(stored) | set(fingerprint))
            lines = [f"    {k}: stored={stored.get(k)!r} now={fingerprint.get(k)!r}"
                     for k in differing if stored.get(k) != fingerprint.get(k)]
            message = ("Signals were collected under different settings:\n" + "\n".join(lines) +
                       f"\nDelete {signals_path} and start again, or pass --force to mix them.")
            if not force:
                raise SystemExit(message)
            logger.warning(message)
        return
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps({"settings": fingerprint}, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — refit the stacker on g(x) alone
# ─────────────────────────────────────────────────────────────────────────────

def pool_scores(cfg: dict, detector, split: str, cache_dir: Path) -> List[dict]:
    """
    g(x) for one distillation-pool split, cached.

    The training cache in data/processed/layer1_features holds a g=0.5
    placeholder — the real scores are only computed in memory by
    scripts/train_layer1.py — so they are recomputed here once and kept.
    """
    pool_dir = Path(cfg["data"]["distillation_pool_dir"])
    source = pool_dir / f"{split}.jsonl"
    if not source.exists():
        raise SystemExit(f"Pool split not found: {source}")
    records = read_jsonl(source)
    ids = [str(r.get("id", i)) for i, r in enumerate(records)]
    stamp = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()

    cache = cache_dir / f"pool_g_{split}.jsonl"
    meta = cache.with_suffix(".meta.json")
    if cache.exists() and meta.exists():
        stored = json.loads(meta.read_text(encoding="utf-8"))
        if stored.get("ids_sha1") == stamp and stored.get("n") == len(records):
            logger.info(f"Pool {split}: {len(records)} cached classifier scores")
            return read_jsonl(cache)
        logger.warning(f"Pool {split}: cache does not match {source}; recomputing")

    logger.info(f"Pool {split}: scoring {len(records)} prompts with the classifier...")
    batch_size = cfg["layer1"]["classifier"].get("batch_size", 8)
    scores = detector.classifier_feature.compute_batch([r["text"] for r in records],
                                                       batch_size=batch_size)
    rows = [{"id": rid, "label": int(rec["label"]), "g": float(g)}
            for rid, rec, g in zip(ids, records, scores)]

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta.write_text(json.dumps({"ids_sha1": stamp, "n": len(records), "source": str(source)},
                               indent=2), encoding="utf-8")
    return rows


def classifier_only_fingerprint(cfg: dict, artifact_dir: Path) -> dict:
    """
    What the classifier-only stacker was fitted from: the pool splits and the
    classifier weights. A retrain changes both, and a stacker fitted against the
    old classifier would quietly score every prompt wrongly.
    """
    pool_dir = Path(cfg["data"]["distillation_pool_dir"])
    pools = {}
    for split in ("train", "val"):
        source = pool_dir / f"{split}.jsonl"
        ids = [str(r.get("id", i)) for i, r in enumerate(read_jsonl(source))]
        pools[split] = {"n": len(ids),
                        "ids_sha1": hashlib.sha1(chr(10).join(ids).encode("utf-8")).hexdigest()}
    checkpoint = artifact_dir / "classifier" / "checkpoint_best"
    weights = sorted([p.name, p.stat().st_size, int(p.stat().st_mtime)]
                     for p in checkpoint.glob("*") if p.is_file())
    return {"pools": pools, "classifier": weights}


def stacker_files(directory: Path) -> tuple:
    """(fusion, scaler, fingerprint stamp) for a classifier-only stacker in `directory`."""
    return (directory / "fusion_clf_only.joblib",
            directory / "scaler_clf_only.json",
            directory / "fusion_clf_only.meta.json")


def load_classifier_only(directory: Path, fingerprint: dict):
    """
    Read a classifier-only stacker from `directory` if one is usable there.

    Returns (fusion, scaler, verified, provenance) or None. A stamp that
    disagrees with `fingerprint` is positive evidence the stacker predates the
    current pool or classifier, so it is refused; a missing stamp is only
    unverified, and is accepted with that said out loud, because a stacker
    written by scripts/train_layer1.py has no stamp.
    """
    from llmguard.layer1.fusion import ClassifierOnlyFusion, TemperatureScaler

    fusion_path, scaler_path, stamp_path = stacker_files(directory)
    if not (fusion_path.exists() and scaler_path.exists()):
        return None
    if stamp_path.exists():
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stored != fingerprint:
            logger.warning(f"{fusion_path} was fitted from a different pool or classifier "
                           f"({stamp_path.name} does not match); not using it.")
            return None
        verified = True
        provenance = f"fingerprint in {stamp_path.name} matches the current pool and classifier"
    else:
        verified = False
        provenance = (f"no {stamp_path.name} beside it, so what it was fitted from cannot be "
                      "checked here")

    fusion = ClassifierOnlyFusion()
    scaler = TemperatureScaler()
    fusion.load(str(fusion_path))
    scaler.load(str(scaler_path))
    return fusion, scaler, verified, provenance


def fit_classifier_only(cfg: dict, detector, artifact_dir: Path, cache_dir: Path,
                        stacker_dir: Path):
    """
    The classifier-only stacker for the no_curvature variant.

    Preference order, so the variant measures the model that ships whenever one
    is there to measure:

      1. artifact_dir — the file Layer1Detector.from_artifacts_for loads when
         layer1.classifier_only is true. Read only; nothing is written there.
      2. stacker_dir — a stacker this script fitted earlier, reused when its
         stored fingerprint still matches the pool and the classifier.
      3. Otherwise fit one on the pool train split and calibrate it on val, the
         same order scripts/train_layer1.py uses, and write it to stacker_dir.

    Step 3 used to write into artifact_dir under exactly the names train_layer1.py
    writes, which silently replaced the shipped stacker.
    """
    from llmguard.layer1.fusion import ClassifierOnlyFusion, TemperatureScaler

    if stacker_dir.resolve() == artifact_dir.resolve():
        raise SystemExit(
            f"--stacker-dir must not be the Layer-1 artifact directory ({artifact_dir}): "
            "fusion_clf_only.joblib and scaler_clf_only.json there are the shipped model, "
            "written by scripts/train_layer1.py. Pick another directory."
        )

    fingerprint = classifier_only_fingerprint(cfg, artifact_dir)

    for directory, source in ((artifact_dir, "the live Layer-1 artifact directory"),
                              (stacker_dir, "this script's own cache")):
        found = load_classifier_only(directory, fingerprint)
        if found is not None:
            fusion, scaler, verified, provenance = found
            message = (f"Classifier-only stacker: reusing {stacker_files(directory)[0]} "
                       f"from {source} (T={scaler.T:.4f}; {provenance}); no fresh fit.")
            logger.info(message) if verified else logger.warning(message)
            return fusion, scaler

    fusion_path, scaler_path, stamp_path = stacker_files(stacker_dir)
    logger.info(f"Classifier-only stacker: none usable in {artifact_dir} or {stacker_dir}; "
                f"fitting a fresh one into {stacker_dir}.")

    train = pool_scores(cfg, detector, "train", cache_dir)
    val = pool_scores(cfg, detector, "val", cache_dir)

    fusion = ClassifierOnlyFusion()
    scaler = TemperatureScaler()
    # A column vector of g(x): ClassifierOnlyFusion.fit takes either this or the
    # three-column matrix train_layer1.py hands it, whose third column is this.
    fusion.fit(np.array([[r["g"]] for r in train], dtype=float),
               np.array([r["label"] for r in train], dtype=int))
    raw = [fusion.raw_score(UNUSED_FEATURE, UNUSED_FEATURE, r["g"]) for r in val]
    t_star = scaler.calibrate(raw, [r["label"] for r in val])
    logger.info(f"Classifier-only calibration: T* = {t_star:.4f}")

    stacker_dir.mkdir(parents=True, exist_ok=True)
    fusion.save(str(fusion_path))
    scaler.save(str(scaler_path))
    stamp_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    logger.info(f"Classifier-only stacker: fitted fresh and written to {fusion_path} "
                f"(with {scaler_path.name} and {stamp_path.name}); "
                f"{artifact_dir} untouched.")
    return fusion, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — signals
# ─────────────────────────────────────────────────────────────────────────────

def load_done(signals_path: Path) -> set:
    """(benchmark, id) pairs already collected."""
    if not signals_path.exists():
        return set()
    done = set()
    for row in read_jsonl(signals_path):
        done.add((row["benchmark"], str(row["id"])))
    return done


def judge_row(result) -> dict:
    """The parts of a JudgeResult the derivation needs."""
    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "severity": result.severity,
        "category": result.category,
        "escalated": bool(result.escalated),
        "p_mitigate": getattr(result, "p_mitigate", None),
        "votes": [v.get("decision") for v in (getattr(result, "votes", None) or [])],
    }


def eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def collect_signals(cfg: dict, args, benchmarks: List[tuple]) -> None:
    """Compute every prompt's signals once and append them to the signals file."""
    need_judge = bool(set(args.variants) & set(CASCADE_VARIANTS + ("judge",)))
    need_p = "judge" in args.variants
    need_clf_only = "no_curvature" in args.variants

    # Checked before anything is loaded, so a misconfigured run fails in a second.
    consistency = cfg["layer2"]["consistency"]
    if need_judge and consistency.get("mode") != "logprob":
        raise SystemExit(
            "Ablation needs layer2.consistency.mode: logprob. In vote mode the judge samples K "
            "answers, so one cached call cannot stand in for every variant."
        )

    from llmguard.layer1.detector import Layer1Detector, layer1_artifact_dir
    from llmguard.layer1.fusion import ThresholdRouter

    signals_path = Path(args.signals)
    check_or_write_meta(signals_path, settings_fingerprint(cfg, args), args.resume, args.force)
    done = load_done(signals_path) if args.resume else set()
    if not args.resume and signals_path.exists():
        warn_if_overwriting(signals_path)
        signals_path.unlink()
    signals_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_dir = layer1_artifact_dir(cfg)
    logger.info(f"Loading Layer 1 from {artifact_dir}...")
    detector = Layer1Detector.from_artifacts(
        cfg=cfg,
        fusion_path=str(artifact_dir / "fusion_v2.joblib"),
        scaler_path=str(artifact_dir / "scaler.json"),
        classifier_checkpoint=str(artifact_dir / "classifier" / "checkpoint_best"),
    )
    router = ThresholdRouter(tau_low=cfg["layer1"]["thresholds"]["low"],
                             tau_high=cfg["layer1"]["thresholds"]["high"])

    clf_fusion = clf_scaler = None
    if need_clf_only:
        clf_fusion, clf_scaler = fit_classifier_only(
            cfg, detector, artifact_dir,
            cache_dir=Path(cfg["data"]["processed_dir"]) / "ablation",
            stacker_dir=stacker_dir(args))

    judge = llm = None
    if need_judge:
        from llmguard.layer2 import Layer2Judge
        logger.info("Loading Layer 2...")
        judge = Layer2Judge.from_config(cfg)
        llm = judge.checker.judge

    todo = []
    for name, path in benchmarks:
        for i, rec in enumerate(load_records(path, args.limit)):
            rid = str(rec.get("id", f"{name}_{i}"))
            if (name, rid) not in done:
                todo.append((name, rid, rec))
    total = len(todo)
    logger.info(f"{total} prompts to collect ({len(done)} already done)")
    if not total:
        return

    started = time.perf_counter()
    judged = 0
    with open(signals_path, "a", encoding="utf-8", newline="\n") as fh:
        for i, (name, rid, rec) in enumerate(todo, start=1):
            text = rec.get("text", "")
            row = {"benchmark": name, "id": rid, "label": int(rec.get("label", 1)),
                   "text": text, "error": None}
            try:
                t0 = time.perf_counter()
                row["fcurv"] = detector.curvature_feature.compute(text)
                row["frank"] = detector.logrank_feature.compute(text)
                row["t_features_ms"] = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                row["g"] = detector.classifier_feature.compute(text)
                row["t_classifier_ms"] = (time.perf_counter() - t0) * 1000.0

                raw = detector.fusion.raw_score(row["fcurv"], row["frank"], row["g"])
                row["l1_raw"] = raw
                row["l1_score"] = detector.scaler.scale(raw)
                row["l1_decision"] = router.route(row["l1_score"])

                if need_clf_only:
                    # The real feature values; the one-feature stacker ignores them.
                    c_raw = clf_fusion.raw_score(row["fcurv"], row["frank"], row["g"])
                    row["c_raw"] = c_raw
                    row["c_score"] = clf_scaler.scale(c_raw)
                    row["c_decision"] = router.route(row["c_score"])

                if need_p:
                    t0 = time.perf_counter()
                    row["p_mitigate"] = llm.p_mitigate(text)
                    row["t_p_ms"] = (time.perf_counter() - t0) * 1000.0

                gray = (("full" in args.variants and row["l1_decision"] == DECISION_GRAY_ZONE) or
                        (need_clf_only and row.get("c_decision") == DECISION_GRAY_ZONE))
                if gray and judge is not None:
                    t0 = time.perf_counter()
                    # record=False keeps benchmark prompts out of the distillation pool
                    row["judge"] = judge_row(judge.judge(text, record=False))
                    row["t_judge_ms"] = (time.perf_counter() - t0) * 1000.0
                    judged += 1
            except Exception as exc:                                  # noqa: BLE001
                logger.warning(f"{name}/{rid}: {exc}")
                row["error"] = str(exc)

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            if i % args.log_every == 0 or i == total:
                spent = time.perf_counter() - started
                rate = spent / i
                logger.info(f"  {i}/{total} ({100 * i / total:.1f}%) | {rate:.2f}s/prompt | "
                            f"judged {judged} | ETA {eta(rate * (total - i))}")

    logger.info(f"Signals written to {signals_path} ({judged} judge calls)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — derive one results file per variant
# ─────────────────────────────────────────────────────────────────────────────

def blank_row(row: dict) -> dict:
    """A prediction row for a prompt that crashed: counted as misclassified."""
    label = int(row["label"])
    return {"id": row["id"], "text": row.get("text", ""), "label": label,
            "final_pred": 1 - label, "harm_score": float(label),
            "escalated": False, "error": row.get("error")}


def prediction_row(row: dict, *, l1_decision, l1_score, judge_info,
                   final_pred, harm_score, escalated) -> dict:
    j = judge_info or {}
    return {
        "id": row["id"], "text": row.get("text", ""), "label": int(row["label"]),
        "l1_decision": l1_decision,
        "l1_score": l1_score,
        "l2_decision": j.get("decision"),
        "l2_confidence": j.get("confidence"),
        "l2_severity": j.get("severity"),
        "l2_category": j.get("category"),
        "l2_votes": j.get("votes"),
        "l2_p_mitigate": j.get("p_mitigate"),
        "l3_mode": None,          # Layer 3 is not loaded; it never changes a decision
        "final_pred": final_pred,
        "harm_score": harm_score,
        "escalated": escalated,
        "error": None,
    }


def cascade_decision(row: dict, decision: str, score: float) -> dict:
    """
    Layer 1 routes, the judge settles the gray zone — the rule in
    evaluation/pipeline_evaluator.py::_run_one.
    """
    if decision == DECISION_ALLOW:
        return prediction_row(row, l1_decision=decision, l1_score=score, judge_info=None,
                              final_pred=1, harm_score=score, escalated=False)
    if decision == DECISION_MITIGATE:
        return prediction_row(row, l1_decision=decision, l1_score=score, judge_info=None,
                              final_pred=0, harm_score=score, escalated=False)

    j = row.get("judge")
    if not j:
        # No judge available for this prompt: unresolved gray zone allows, as the evaluator does
        return prediction_row(row, l1_decision=decision, l1_score=score, judge_info=None,
                              final_pred=1, harm_score=score, escalated=False)

    harm = 0.5 * score + 0.5 * float(j["severity"])
    if j["escalated"]:
        return prediction_row(row, l1_decision=decision, l1_score=score, judge_info=j,
                              final_pred=0, harm_score=harm, escalated=True)
    final = 1 if j["decision"] == DECISION_ALLOW else 0
    return prediction_row(row, l1_decision=decision, l1_score=score, judge_info=j,
                          final_pred=final, harm_score=harm, escalated=False)


def classifier_decision(row: dict) -> dict:
    """The classifier on its own: harm score 1 - g(x), harmful above 0.5, nothing escalates."""
    harm = 1.0 - quantise(row["g"])
    final = 0 if harm > 0.5 else 1
    return prediction_row(row, l1_decision=DECISION_MITIGATE if final == 0 else DECISION_ALLOW,
                          l1_score=harm, judge_info=None,
                          final_pred=final, harm_score=harm, escalated=False)


def judge_decision(row: dict, allow_below: float, mitigate_above: float) -> dict:
    """
    The judge alone, on every prompt: the band from
    layer2/consistency.py::_check_logprob, with P(mitigate) as the harm score.
    """
    if row.get("p_mitigate") is None:
        raise KeyError(f"{row['benchmark']}/{row['id']}: no p_mitigate; "
                       "collect signals with the judge variant enabled")
    p = quantise(row["p_mitigate"])
    if p > mitigate_above:
        decision, final, escalated = DECISION_MITIGATE, 0, False
    elif p < allow_below:
        decision, final, escalated = DECISION_ALLOW, 1, False
    else:
        decision, final, escalated = DECISION_UNCERTAIN, 0, True
    info = {"decision": decision, "confidence": round(max(p, 1.0 - p), 4),
            "severity": p, "category": None, "p_mitigate": p, "votes": None}
    return prediction_row(row, l1_decision=None, l1_score=None, judge_info=info,
                          final_pred=final, harm_score=p, escalated=escalated)


def latency_ms(variant: str, row: dict) -> Optional[float]:
    """Per-prompt latency for one variant, composed from the measured components."""
    if row.get("error"):
        return None
    clf = row.get("t_classifier_ms")
    feats = row.get("t_features_ms")
    judged = row.get("t_judge_ms")
    if variant == "classifier":
        return clf
    if variant == "judge":
        return row.get("t_p_ms")
    if clf is None or feats is None:
        return None
    if variant == "full":
        base = feats + clf
        return base + judged if (row.get("l1_decision") == DECISION_GRAY_ZONE and judged) else base
    if variant == "no_curvature":
        return clf + judged if (row.get("c_decision") == DECISION_GRAY_ZONE and judged) else clf
    return None


def layer_stats(rows: List[dict]) -> dict:
    """The counts run_evaluation.py reports, so both runs read the same way."""
    stats = {"l1_allow": 0, "l1_gray_zone": 0, "l1_mitigate": 0,
             "l2_allow": 0, "l2_mitigate": 0, "l2_uncertain": 0,
             "l3_refuse": 0, "l3_redact": 0, "l3_safe_rewrite": 0, "errors": 0}
    for r in rows:
        if r.get("error"):
            stats["errors"] += 1
            continue
        d1 = r.get("l1_decision")
        if d1 in (DECISION_ALLOW, DECISION_GRAY_ZONE, DECISION_MITIGATE):
            stats[f"l1_{d1}"] += 1
        d2 = r.get("l2_decision")
        if d2 == DECISION_ALLOW:
            stats["l2_allow"] += 1
        elif d2 == DECISION_MITIGATE:
            stats["l2_mitigate"] += 1
        elif d2 == DECISION_UNCERTAIN or r.get("escalated"):
            stats["l2_uncertain"] += 1
    return stats


def derive_variant(variant: str, signals: List[dict], cfg: dict) -> dict:
    """{benchmark: {rows, metrics, layer_stats, latency}} for one variant."""
    consistency = cfg["layer2"]["consistency"]
    allow_below = consistency.get("allow_below", 0.4)
    mitigate_above = consistency.get("mitigate_above", 0.6)

    by_benchmark = {}
    for row in signals:
        by_benchmark.setdefault(row["benchmark"], []).append(row)

    out = {}
    for bench, rows in by_benchmark.items():
        predictions, latencies = [], []
        for row in rows:
            if row.get("error"):
                predictions.append(blank_row(row))
                continue
            if variant == "full":
                pred = cascade_decision(row, row["l1_decision"], quantise(row["l1_score"]))
            elif variant == "no_curvature":
                if row.get("c_decision") is None:
                    raise KeyError(f"{bench}/{row['id']}: no classifier-only score; "
                                   "collect signals with the no_curvature variant enabled")
                pred = cascade_decision(row, row["c_decision"], quantise(row["c_score"]))
            elif variant == "classifier":
                pred = classifier_decision(row)
            else:
                pred = judge_decision(row, allow_below, mitigate_above)
            predictions.append(pred)
            value = latency_ms(variant, row)
            if value is not None:
                latencies.append(value)

        y = [p["label"] for p in predictions]
        s = [p["harm_score"] for p in predictions]
        pred_labels = [p["final_pred"] for p in predictions]
        esc = [bool(p.get("escalated")) for p in predictions]
        out[bench] = {
            "rows": predictions,
            "metrics": compute_all(y, s, pred_labels, escalated=esc),
            "layer_stats": layer_stats(predictions),
            "latency": {
                "n": len(latencies),
                "mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
                "median_ms": round(statistics.median(latencies), 2) if latencies else None,
                "total_s": round(sum(latencies) / 1000.0, 2) if latencies else None,
            },
        }
    return out


def write_variant(variant: str, derived: dict, out_dir: Path, run_info: dict,
                  shipped: dict) -> Path:
    """Results JSON plus predictions, in the layout scripts/summarise_results.py reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"ablation_{variant}.json"
    pred_dir = results_path.with_name(results_path.stem + "_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    warn_if_overwriting(results_path)
    payload = {}
    for bench, data in derived.items():
        payload[bench] = {
            "benchmark": bench,
            "n_samples": len(data["rows"]),
            "elapsed_s": data["latency"]["total_s"],
            "metrics": data["metrics"],
            "layer_stats": data["layer_stats"],
            "latency": data["latency"],
            "errors": [r["error"] for r in data["rows"] if r.get("error")],
            "variant": variant,
            "variant_alias": CANONICAL_NAME[variant],
            "variant_description": WHAT_IT_IS[variant],
            "is_shipped_configuration": variant == shipped["variant"],
            "shipped_configuration": shipped,
            "run_info": run_info,
        }
        path = pred_dir / f"{bench}.jsonl"
        warn_if_overwriting(path)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in data["rows"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return results_path


def safe_blocked_share(rows: List[dict]) -> Optional[float]:
    """Share of safe prompts not allowed; an escalated prompt counts as blocked."""
    safe = [r for r in rows if int(r["label"]) == 1]
    if not safe:
        return None
    blocked = sum(1 for r in safe if r["final_pred"] == 0 or r.get("escalated"))
    return blocked / len(safe)


def print_table(variant: str, derived: dict, cfg: dict) -> None:
    print()
    print(f"  {variant_label(variant, cfg)}")
    print(f"  {'Benchmark':<16} {'AUROC':>8} {'Safe blk':>9} {'Harm allw':>10} {'Judged':>7} {'ms/prompt':>10}")
    print("  " + "-" * 64)
    for bench, data in sorted(derived.items()):
        m = data["metrics"]
        judged = data["layer_stats"]["l2_allow"] + data["layer_stats"]["l2_mitigate"] + \
            data["layer_stats"]["l2_uncertain"]
        median = data["latency"]["median_ms"]
        print(f"  {bench:<16} {format_metric(m.get('auroc')):>8} "
              f"{format_metric(safe_blocked_share(data['rows'])):>9} "
              f"{format_metric(m.get('jailbreak_success_rate')):>10} "
              f"{judged:>7} {('n/a' if median is None else f'{median:.0f}'):>10}")


def derive_all(cfg: dict, args, run_info: dict) -> List[Path]:
    signals_path = Path(args.signals)
    if not signals_path.exists():
        raise SystemExit(f"No signals at {signals_path}; run --stage signals first.")
    signals = read_jsonl(signals_path)
    if not signals:
        raise SystemExit(f"{signals_path} is empty.")
    if args.benchmarks:
        wanted = set(args.benchmarks)
        signals = [row for row in signals if row["benchmark"] in wanted]
        if not signals:
            raise SystemExit(f"No signals for {', '.join(sorted(wanted))} in {signals_path}.")
    logger.info(f"Deriving {len(args.variants)} variants from {len(signals)} prompts")

    shipped = shipped_configuration(cfg)
    print()
    print(f"  Current config ships {shipped['variant']} ({shipped['variant_alias']}): "
          f"layer1.classifier_only={shipped['layer1.classifier_only']}, "
          f"layer1.artifact_subdir={shipped['layer1.artifact_subdir']}")
    print(f"  Scored on the raw text here; the pipeline also applies "
          f"normalise_input={shipped['layer1.normalise_input']} and "
          f"payload_views={shipped['layer1.payload_views']}")

    written = []
    for variant in args.variants:
        derived = derive_variant(variant, signals, cfg)
        path = write_variant(variant, derived, Path(args.out_dir), run_info, shipped)
        print_table(variant, derived, cfg)
        written.append(path)
        logger.info(f"{variant_label(variant, cfg)} → {path}")
    return written


def print_next_command(paths: List[Path], variants: List[str], cfg: dict) -> None:
    """The summarise_results.py call that turns these files into tables."""
    runs = " ".join(f"--run {v}={p.as_posix()}" for v, p in zip(variants, paths))
    # Everything is compared against the three-signal cascade when it was derived
    # too, which is the historical reference, not the shipped configuration.
    reference = "full" if "full" in variants else variants[0]
    compares = " ".join(f"--compare {reference}:{v}" for v in variants if v != reference)
    shipped = shipped_variant(cfg)
    print()
    print("  Confidence intervals and paired differences:")
    print(f"    python scripts/summarise_results.py {runs} {compares} \\")
    print("      --out data/processed/ablation_results")
    print(f"    (reference {reference} = {CANONICAL_NAME[reference]}; the shipped "
          f"configuration is {shipped} = {CANONICAL_NAME[shipped]})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)
    if args.input and args.benchmarks:
        raise SystemExit("--input and --benchmarks are alternatives; pass one.")
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)

    shipped = shipped_variant(cfg)
    logger.info(f"Variants: {', '.join(f'{v} ({CANONICAL_NAME[v]})' for v in args.variants)}")
    logger.info(f"Shipped configuration is {shipped} ({CANONICAL_NAME[shipped]}), from "
                f"layer1.classifier_only="
                f"{bool(cfg.get('layer1', {}).get('classifier_only', False))}")
    logger.info(f"Own stacker cache: {stacker_dir(args)} "
                f"(the Layer-1 artifact directory is never written to)")

    if args.stage in ("signals", "both"):
        # Benchmark files are only needed by the collecting stage; derive reads
        # the signals file, so it still works when the benchmarks have moved.
        if args.input:
            source = Path(args.input)
            if not source.exists():
                raise SystemExit(f"Input file not found: {source}")
            benchmarks = [(args.name or source.stem, source)]
        else:
            benchmarks = benchmark_files(cfg, args.benchmarks)
        logger.info(f"Benchmarks: {', '.join(name for name, _ in benchmarks)}")
        collect_signals(cfg, args, benchmarks)

    if args.stage in ("derive", "both"):
        paths = derive_all(cfg, args, finalize_run_info(run_info))
        print_next_command(paths, list(args.variants), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
