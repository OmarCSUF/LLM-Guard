"""
Every llmguard module and every script imports without side effects.

Scripts are loaded by file path (they are not a package); each one keeps its
main() behind `if __name__ == "__main__"`, so importing only defines names.
app.py is deliberately left out: importing it starts a background thread that
loads the models.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

from conftest import ROOT, load_script

import llmguard

LLMGUARD_MODULES = sorted(
    m.name for m in pkgutil.walk_packages(llmguard.__path__, prefix="llmguard.")
)

SCRIPTS = sorted(
    str(p.relative_to(ROOT))
    for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "scripts" / "final_eval").glob("*.py"))
    if p.name != "__init__.py"
)


def test_module_list_is_complete():
    """A silent drop of a subpackage would hide import failures; pin the shape."""
    packages = {"llmguard.layer1", "llmguard.layer2", "llmguard.layer3",
                "llmguard.evaluation", "llmguard.redteam", "llmguard.utils"}
    assert packages <= set(LLMGUARD_MODULES)
    assert len(SCRIPTS) >= 40, SCRIPTS


@pytest.mark.parametrize("name", LLMGUARD_MODULES)
def test_import_llmguard_module(name):
    module = importlib.import_module(name)
    assert module.__name__ == name


@pytest.mark.parametrize("rel_path", SCRIPTS)
def test_import_script(rel_path):
    module = load_script(rel_path)
    # every script defines main(); importing must not have run it
    assert callable(getattr(module, "main", None)), f"{rel_path} has no main()"
