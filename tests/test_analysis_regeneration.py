"""
scripts/final_eval/analyse_final.py regenerates the stored analysis.

The study folder is copied to a temporary directory, the script is run on
the copy as a subprocess (the recorded way), and the new results.json is
compared section by section with results/final_eval_20260926/analysis/
results.json: counts must match exactly, floats to 1e-9 relative. The
bootstrap seed is fixed in the script, so nothing here is stochastic.

Takes about a minute (the 2,000-draw bootstraps), hence the `slow` marker; it
stays in the default run.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys

import pytest

SECTIONS = ("rates", "pooled_auroc", "auroc_diffs", "paired_vs_cascade",
            "seed_range", "ablation", "matched", "redteam", "learner")


def _diffs(expected, actual, path="$"):
    """Recursive comparison; returns a list of human-readable mismatches."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, int):
        # counts: exact (rtol 0)
        return [] if (type(actual) is int and expected == actual) else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return [f"{path}: {expected!r} != {actual!r}"]
        if math.isnan(expected) and math.isnan(actual):
            return []
        return [] if math.isclose(expected, actual, rel_tol=1e-9, abs_tol=0.0) \
            else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a dict"]
        out = []
        if list(expected) != list(actual):
            out.append(f"{path}: keys {list(expected)} != {list(actual)}")
        for k in expected:
            if k in actual:
                out += _diffs(expected[k], actual[k], f"{path}.{k}")
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}: list length {len(expected)} != {len(actual) if isinstance(actual, list) else type(actual).__name__}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += _diffs(e, a, f"{path}[{i}]")
        return out
    return [f"{path}: unhandled type {type(expected).__name__}"]


def test_diff_helper_catches_changes():
    assert _diffs({"a": [1, 2.0]}, {"a": [1, 2.0]}) == []
    assert _diffs({"a": 1}, {"a": 1.0})                       # int vs float is a change
    assert _diffs(0.5, 0.5 * (1 + 1e-8))                      # beyond 1e-9
    assert _diffs(0.5, 0.5 * (1 + 1e-10)) == []
    assert _diffs({"a": 1, "b": 2}, {"b": 2, "a": 1})         # key order


@pytest.mark.slow
def test_analyse_final_regenerates_stored_results(repo_root, study_dir, tmp_path):
    stored = json.loads((study_dir / "analysis" / "results.json").read_text(encoding="utf-8"))
    for section in SECTIONS:
        assert section in stored, f"stored results.json lacks {section!r}"

    study_copy = tmp_path / "study"
    shutil.copytree(study_dir, study_copy)
    shutil.rmtree(study_copy / "analysis")          # regenerate from scratch

    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "final_eval" / "analyse_final.py"),
         "--study", str(study_copy)],
        cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    regenerated = json.loads((study_copy / "analysis" / "results.json").read_text(encoding="utf-8"))
    assert (study_copy / "analysis" / "tables.md").exists()

    assert regenerated.get("skipped") == stored.get("skipped")
    problems = []
    for section in SECTIONS:
        assert section in regenerated, f"regenerated results.json lacks {section!r}"
        problems += _diffs(stored[section], regenerated[section], f"${section}")
    assert not problems, "\n".join(problems[:40])
