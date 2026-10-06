"""
The four pinned Layer-1 checkpoints under artifacts/ are the files the final
study hashed in results/final_eval_20260926/frozen.json.

frozen.json lists each file by its path relative to the repository root
("artifacts/layer1/..."); the release keeps the same relative paths, so the
sha256 of every listed file must match.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "final_eval_20260926" / "frozen.json"

_frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
CASES = [
    pytest.param(tag, rel, sha, id=f"{tag}:{rel}")
    for tag, ck in _frozen["checkpoints"].items()
    for rel, sha in ck["sha256"].items()
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_four_checkpoints_are_listed():
    assert set(_frozen["checkpoints"]) == {"historical", "s42", "s43", "s44"}
    for tag, ck in _frozen["checkpoints"].items():
        assert (ROOT / ck["dir"]).is_dir(), f"{tag}: {ck['dir']} missing"
        assert len(ck["sha256"]) == 4, tag
        for rel in ck["sha256"]:
            assert rel.startswith(ck["dir"] + "/"), (tag, rel)


@pytest.mark.parametrize("tag, rel, sha", CASES)
def test_checkpoint_file_hash(tag, rel, sha):
    path = ROOT / rel
    assert path.is_file(), f"{tag}: {rel} is missing"
    assert sha256_of(path) == sha, f"{tag}: {rel} differs from the frozen hash"


def test_checkpoints_json_agrees_with_frozen(study_dir):
    """checkpoints.json (the study's tag -> directory listing) names the same directories."""
    ck = json.loads((study_dir / "checkpoints.json").read_text(encoding="utf-8"))
    frozen_dirs = {tag: c["dir"] for tag, c in _frozen["checkpoints"].items()}
    assert ck == frozen_dirs
