"""Write (or verify) the freeze record of a final-evaluation study.

The pre-registered study recorded, before any arm was scored, the SHA-256 of the configuration,
of every Python source file, of the Layer-1 artifacts of all four checkpoints, of the three pool
splits and of the benchmark and learner input files, together with the git commit and the GPU
(results/final_eval_20260926/frozen.json). The runner that wrote it was never committed; this
script reproduces the record in the same shape.

Conventions (taken from the stored record and from scripts/run_ablation.py, which uses the same
project convention): file digests are SHA-256 of the bytes; a pool split's ``ids_sha1`` is the
SHA-1 of the newline-joined ``id`` values in file order.

    python scripts/final_eval/freeze.py --study results/<new study>        # write <study>/frozen.json
    python scripts/final_eval/verify_frozen.py --study results/final_eval_20260926   # compare

Layout note: the original record lists source files at the pre-release layout
(``layer1/detector.py`` ...); this script records them at the release layout
(``src/llmguard/layer1/detector.py``). verify_frozen.py maps between the two.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_FILES = ("fusion_clf_only.joblib", "scaler_clf_only.json",
                    "classifier/checkpoint_best/adapter_model.safetensors",
                    "classifier/checkpoint_best/adapter_config.json")
POOL_SPLITS = ("train", "val", "test")
INPUT_FILES = {
    "jailbreakbench": "data/benchmarks/jailbreakbench/test.jsonl",
    "xstest": "data/benchmarks/xstest/test.jsonl",
    "harmbench": "data/benchmarks/harmbench/test.jsonl",
    "toxicchat": "data/benchmarks/toxicchat/test.jsonl",
    "orbench": "data/benchmarks/orbench/test.jsonl",
    "learner_original_wi_a": "data/benchmarks_learner/original/wi_a/test.jsonl",
    "learner_original_wi_b": "data/benchmarks_learner/original/wi_b/test.jsonl",
    "learner_original_wi_c": "data/benchmarks_learner/original/wi_c/test.jsonl",
    "learner_original_locness_n": "data/benchmarks_learner/original/locness_n/test.jsonl",
    "learner_corrected_wi_a": "data/benchmarks_learner/corrected/wi_a/test.jsonl",
    "learner_corrected_wi_b": "data/benchmarks_learner/corrected/wi_b/test.jsonl",
    "learner_corrected_wi_c": "data/benchmarks_learner/corrected/wi_c/test.jsonl",
    "learner_corrected_locness_n": "data/benchmarks_learner/corrected/locness_n/test.jsonl",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pool_entry(path: Path) -> dict:
    ids = [json.loads(line)["id"] for line in open(path, encoding="utf-8") if line.strip()]
    return {"n": len(ids), "ids_sha1": hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest(),
            "sha256": sha256(path)}


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"cuda": True, "name": torch.cuda.get_device_name(0)}
        return {"cuda": False, "name": None}
    except ImportError:
        return {"cuda": None, "name": None}


def source_files() -> list:
    files = [ROOT / "config" / "config.yaml"]
    for base in ("src/llmguard", "scripts"):
        files += sorted(p for p in (ROOT / base).rglob("*.py") if "__pycache__" not in p.parts)
    return files


def build_record(checkpoints: dict, pool_dir: Path) -> dict:
    rec = {"checkpoints": {}, "files": {}, "pool": {}, "inputs": {}, "git": {}, "gpu": {}}
    for name, d in checkpoints.items():
        dpath = ROOT / d
        rec["checkpoints"][name] = {"dir": d, "sha256": {f"{d}/{f}": sha256(dpath / f) for f in CHECKPOINT_FILES if (dpath / f).exists()}}
    for f in source_files():
        rec["files"][f.relative_to(ROOT).as_posix()] = sha256(f)
    for split in POOL_SPLITS:
        p = pool_dir / f"{split}.jsonl"
        if p.exists():
            rec["pool"][split] = pool_entry(p)
    for name, rel in INPUT_FILES.items():
        p = ROOT / rel
        if p.exists():
            rec["inputs"][name] = {"path": rel, "sha256": sha256(p)}
    status = git("status", "--porcelain", "--untracked-files=no")
    rec["git"] = {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}
    rec["gpu"] = gpu()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="study folder; <study>/frozen.json is written")
    ap.add_argument("--checkpoints", default=None,
                    help="JSON mapping name -> checkpoint dir (default: <study>/checkpoints.json, else the four study checkpoints)")
    ap.add_argument("--pool-dir", default="data/distillation_pool")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen.json")
    a = ap.parse_args()
    study = Path(a.study)
    out = study / "frozen.json"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; a freeze record is written once, before any arm is scored (use --force to replace)")
    if a.checkpoints:
        checkpoints = json.load(open(a.checkpoints))
    elif (study / "checkpoints.json").exists():
        checkpoints = json.load(open(study / "checkpoints.json"))
    else:
        checkpoints = {"historical": "artifacts/layer1", "s42": "artifacts/layer1_repeat_study_nciJu1oQ_s42",
                       "s43": "artifacts/layer1_repeat_study_nciJu1oQ_s43", "s44": "artifacts/layer1_repeat_study_nciJu1oQ_s44"}
        study.mkdir(parents=True, exist_ok=True)
        (study / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1))
    rec = build_record(checkpoints, Path(a.pool_dir))
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: {len(rec['files'])} source files, {sum(len(c['sha256']) for c in rec['checkpoints'].values())} checkpoint files, "
          f"{len(rec['pool'])} pool splits, {len(rec['inputs'])} inputs, commit {rec['git']['commit']} dirty={rec['git']['dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
