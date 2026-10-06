"""
Deprecated shim — forwards to scripts/train_layer1.py
=======================================================
This file used to be a 455-line copy of scripts/train_layer1.py that differed
from it in three places: a required --seed, the cfg["project"]["seed"] mutation
that makes the run_info snapshot truthful, and a set_seed() call immediately
before build_lora_classifier (needed because computing the features consumes
random numbers while a cache hit does not, so without it the classifier's
initialisation depended on whether the caches were warm).

All three now live in scripts/train_layer1.py behind its optional --seed flag,
so this file only forwards its arguments there. It is kept rather than deleted
because its path is recorded in manifests of runs that already happened:

    data/processed/repeat_study_nciJu1oQ/protocol.json  ("training_entrypoint")
    data/processed/repeat_study_nciJu1oQ/frozen_files.json
    data/processed/final_eval_20260926/frozen.json      (files[])
    data/processed/final_eval_20260927/frozen.json      (files[])

Keeping the path means the commands behind artifacts/layer1_repeat_study_nciJu1oQ_s42,
_s43 and _s44 still run. New work should call train_layer1.py directly:

    python scripts/train_layer1.py --seed 42 --artifact-dir artifacts/layer1_repeat_s42

--seed stays required here, because that was this entry point's whole purpose;
train_layer1.py itself falls back to project.seed when the flag is absent.
"""

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

NOTICE = (
    "scripts/repeat_study_nciJu1oQ_train.py is deprecated: it now forwards to "
    "scripts/train_layer1.py, which takes --seed itself.\n"
    "  Equivalent command: python scripts/train_layer1.py --seed INT [--artifact-dir DIR]\n"
)


def _load_train_layer1():
    """Import the real entry point from the file beside this one."""
    path = Path(__file__).resolve().parent / "train_layer1.py"
    spec = importlib.util.spec_from_file_location("train_layer1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.stderr.write(NOTICE)
    asking_for_help = any(a in ("-h", "--help") for a in argv)
    if not asking_for_help and not any(a == "--seed" or a.startswith("--seed=") for a in argv):
        raise SystemExit(
            "scripts/repeat_study_nciJu1oQ_train.py: --seed INT is required (this entry "
            "point exists to make the training seed explicit). Either pass it, or call "
            "scripts/train_layer1.py, which defaults to project.seed from the config."
        )
    return _load_train_layer1().main(argv)


if __name__ == "__main__":
    main()
