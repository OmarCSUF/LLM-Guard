"""
Shared test fixtures.

The suite never loads a model and never touches the network: it checks the
pure-Python pieces (normalisation, metrics, interval arithmetic), the stored
study (checkpoint hashes, the final test set, the analysis tables) and that
every module and script still imports.

Run from the repository root:

    PYTHONPATH=src python -m pytest -q tests
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# src/ for `import llmguard`, the root for `from scripts.compare_flips import ...`
# (two scripts import a sibling that way). Both are no-ops when PYTHONPATH already
# provides them.
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_script(rel_path: str):
    """
    Import a script (e.g. "scripts/final_eval/analyse_final.py") as a module
    without running its main(): every script guards main() behind
    `if __name__ == "__main__"`, so exec'ing the module only defines things.
    """
    path = ROOT / rel_path
    name = "script_" + rel_path.replace("/", "_").removesuffix(".py")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def study_dir() -> Path:
    """The pre-registered final study, verbatim."""
    return ROOT / "results" / "final_eval_20260926"


@pytest.fixture(scope="session")
def script_loader():
    return load_script
