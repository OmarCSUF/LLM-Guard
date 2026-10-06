"""
Calibration check
=================
Two checks over the cached Layer-1 features. Both run on CPU in seconds and
load no models: only data/processed/layer1_features/*.jsonl and the small
scaler JSON the config points at are read.

  1. Refit path — fits the stacker on the cached training features the same way
     train_layer1.py fits fusion_v1 (placeholder g(x) = 0.5) and refits T on the
     validation split, instead of loading the shipped .joblib, which was pickled
     with a newer scikit-learn. The numbers show calibration direction, not final
     Layer-1 quality.

  2. Shipped path — the temperature the running system actually uses:
     layer1.classifier_only selects scaler_clf_only.json or scaler.json inside
     layer1_artifact_dir(cfg), exactly as Layer1Detector.from_artifacts_for
     does. Check 1 alone exercises a stacker nothing serves, so a bad shipped
     temperature would pass unnoticed.

Each check asserts T ∈ (0, 5] and that harmful validation rows score above safe
ones after calibration. Exits 0 when both pass, 1 otherwise.

Usage:
    python scripts/check_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import json
from pathlib import Path

import numpy as np

from llmguard.utils.config_loader import load_config
# layer1_artifact_dir is the one place that resolves project.artifact_dir /
# layer1.artifact_subdir; importing it keeps this check on the same directory the
# detector loads from.
from llmguard.layer1.detector import layer1_artifact_dir
from llmguard.layer1.fusion import (
    ClassifierOnlyFusion,
    LogisticFusion, TemperatureScaler, ThresholdRouter,
)

FEATURES_DIR = "data/processed/layer1_features"


def load_rows(split: str) -> list:
    with open(f"{FEATURES_DIR}/{split}_features.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def separation(scores: list, labels: list) -> tuple:
    """
    (mean score on harmful rows, mean on safe rows, n_harmful, n_safe).

    label 0 = harmful (mitigate), 1 = safe (allow), as in data.labels.
    """
    harmful = [s for s, label in zip(scores, labels) if label == 0]
    safe = [s for s, label in zip(scores, labels) if label == 1]
    return (
        sum(harmful) / len(harmful),
        sum(safe) / len(safe),
        len(harmful),
        len(safe),
    )


def route_counts(cfg: dict, scores: list) -> dict:
    thresholds = cfg["layer1"]["thresholds"]
    router = ThresholdRouter(tau_low=thresholds["low"], tau_high=thresholds["high"])
    routes = router.route_batch(scores)
    return {d: routes.count(d) for d in ("allow", "gray_zone", "mitigate")}


def shipped_scaler_path(cfg: dict) -> Path:
    """
    The calibrated scaler the config selects.

    Mirrors Layer1Detector.from_artifacts_for: layer1.classifier_only picks the
    file, layer1_artifact_dir picks the directory. from_artifacts_for itself
    cannot be called here because it builds the scoring LM and the classifier.
    """
    classifier_only = bool(cfg.get("layer1", {}).get("classifier_only", False))
    name = "scaler_clf_only.json" if classifier_only else "scaler.json"
    return layer1_artifact_dir(cfg) / name


def refit_check(cfg: dict, train: list, rows: list) -> tuple:
    """
    Check 1: fit the three-signal stacker and the temperature on the cache.

    Returns (ok, raw_scores) — the raw scores are reused by the shipped check so
    the two differ only in which temperature scales them.
    """
    fusion = LogisticFusion()
    fusion.fit(
        np.array([[r["fcurv"], r["frank"], r["g"]] for r in train]),
        np.array([r["label"] for r in train]),
    )
    raw = [fusion.raw_score(r["fcurv"], r["frank"], r["g"]) for r in rows]
    labels = [r["label"] for r in rows]

    scaler = TemperatureScaler()
    T = scaler.calibrate(raw, labels)
    scores = scaler.scale_batch(raw)

    counts = route_counts(cfg, scores)
    mean_harmful, mean_safe, n_harmful, n_safe = separation(scores, labels)

    print("── check 1: refit stacker + refit temperature ───────────────────────")
    print(f"rows        : {len(rows)}  (harmful {n_harmful}, safe {n_safe})")
    print(f"T*          : {T:.4f}")
    print(f"routing     : {counts}")
    print(f"mean score  : harmful {mean_harmful:.3f} | safe {mean_safe:.3f}")

    ok = mean_harmful > mean_safe and 0.0 < T <= 5.0
    print(f"calibration : {'OK' if ok else 'WRONG DIRECTION OR BAD T'}")
    return ok, raw


def shipped_check(cfg: dict, train: list, rows: list, refit_raw: list) -> bool:
    """
    Check 2: the temperature the shipped config serves.

    The scaler file is the one layer1.classifier_only and layer1.artifact_subdir
    select. The raw scores it scales come from the cache: with a classifier-only
    config that means g(x), unless the cached g(x) is still the training
    placeholder (one constant value for every row, which carries no signal at
    all) — then the curvature + log-rank scores of check 1 are scaled instead and
    the basis line below says so. Either way the temperature under test is the
    shipped one, read from the artifact file, never assumed.
    """
    labels = [r["label"] for r in rows]
    path = shipped_scaler_path(cfg)

    print()
    print("── check 2: shipped path (the scaler the config selects) ────────────")
    print(f"artifacts   : {layer1_artifact_dir(cfg)}")
    print(f"stacker     : {'classifier only' if cfg.get('layer1', {}).get('classifier_only', False) else 'three signals'}"
          f"  (layer1.classifier_only)")
    print(f"scaler      : {path}")

    if not path.exists():
        print(f"calibration : MISSING — {path} does not exist")
        return False

    scaler = TemperatureScaler()
    scaler.load(str(path))
    T = scaler.T

    cached_g = {round(r["g"], 6) for r in rows} | {round(r["g"], 6) for r in train}
    if cfg.get("layer1", {}).get("classifier_only", False) and len(cached_g) > 1:
        stacker = ClassifierOnlyFusion()
        stacker.fit(
            np.array([[r["fcurv"], r["frank"], r["g"]] for r in train]),
            np.array([r["label"] for r in train]),
        )
        raw = [stacker.raw_score(r["fcurv"], r["frank"], r["g"]) for r in rows]
        basis = "classifier-only stacker over the cached g(x)"
    else:
        raw = refit_raw
        if len(cached_g) == 1:
            basis = (f"curvature + log-rank stacker — cached g(x) is the constant "
                     f"training placeholder {next(iter(cached_g))}, so it carries "
                     f"no signal")
        else:
            basis = "three-signal stacker over the cached features"

    scores = scaler.scale_batch(raw)
    counts = route_counts(cfg, scores)
    mean_harmful, mean_safe, n_harmful, n_safe = separation(scores, labels)

    print(f"basis       : {basis}")
    print(f"rows        : {len(rows)}  (harmful {n_harmful}, safe {n_safe})")
    print(f"T (shipped) : {T:.4f}")
    print(f"routing     : {counts}")
    print(f"mean score  : harmful {mean_harmful:.3f} | safe {mean_safe:.3f}")

    ok_T = 0.0 < T <= 5.0
    ok_dir = mean_harmful > mean_safe
    if not ok_T:
        print(f"T           : OUT OF RANGE — need 0 < T <= 5, got {T:.4f}")
    if not ok_dir:
        print("direction   : WRONG — harmful rows do not score above safe rows")
    ok = ok_T and ok_dir
    print(f"calibration : {'OK' if ok else 'FAILED'}")
    return ok


def main() -> int:
    import argparse
    global FEATURES_DIR
    ap = argparse.ArgumentParser(description="CPU-only calibration sanity check on the cached Layer-1 features")
    ap.add_argument("--features-dir", default=FEATURES_DIR, help="feature cache written by train_layer1.py (default: %(default)s)")
    FEATURES_DIR = ap.parse_args().features_dir
    cfg = load_config()

    train = load_rows("train")
    rows = load_rows("val")

    refit_ok, refit_raw = refit_check(cfg, train, rows)
    shipped_ok = shipped_check(cfg, train, rows, refit_raw)

    print()
    print(f"refit path  : {'OK' if refit_ok else 'FAILED'}")
    print(f"shipped path: {'OK' if shipped_ok else 'FAILED'}")
    return 0 if (refit_ok and shipped_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
