"""Create the temperature-1 copy of a Layer-1 checkpoint used by the 2026-09-30 post hoc
component ablation (AMENDMENT-2026-09-30.md, arm "T1").

The original study created artifacts/layer1_T1 and artifacts/layer1_repeat_study_nciJu1oQ_s4{2,3,4}_T1
by copying each checkpoint directory and rewriting its two scaler files to T = 1.0; no script for
that step was committed. This script reproduces the copies: the LoRA adapter, tokenizer files,
stackers and results.json are copied unchanged, and scaler.json / scaler_clf_only.json are
replaced with {"T": 1.0, "optimised": false, "ablation": "...; source <subdir>/<file>"}, the exact
content of the original T1 files.

    python scripts/make_t1_variants.py artifacts/layer1
    python scripts/make_t1_variants.py artifacts/layer1_repeat_study_nciJu1oQ_s42 --out artifacts/layer1_repeat_study_nciJu1oQ_s42_T1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

NOTE = "temperature fixed at 1.0 for the 2026-09-30 post hoc ablation; source {subdir}/{name}"


def make_t1(src: Path, out: Path, force: bool = False) -> Path:
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "plots"))
    for name in ("scaler.json", "scaler_clf_only.json"):
        f = out / name
        if f.exists():
            f.write_text(json.dumps({"T": 1.0, "optimised": False,
                                     "ablation": NOTE.format(subdir=src.name, name=name)}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="Layer-1 checkpoint directory, e.g. artifacts/layer1")
    ap.add_argument("--out", default=None, help="output directory (default: <checkpoint>_T1)")
    ap.add_argument("--force", action="store_true", help="replace an existing output directory")
    a = ap.parse_args()
    src = Path(a.checkpoint)
    if not (src / "classifier" / "checkpoint_best").is_dir():
        raise SystemExit(f"{src} does not look like a Layer-1 checkpoint (no classifier/checkpoint_best)")
    out = make_t1(src, Path(a.out) if a.out else src.with_name(src.name + "_T1"), a.force)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
