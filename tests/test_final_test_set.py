"""
data/benchmarks_final/final_test/test.jsonl is the 3,453-prompt final test
set described by data/benchmarks_final/manifest.json: same sha256, same
per-source and per-source:label counts, unique ids.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "data" / "benchmarks_final"
TEST_JSONL = FINAL_DIR / "final_test" / "test.jsonl"
MANIFEST = FINAL_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows() -> list:
    with open(TEST_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_sha256_matches_manifest(manifest):
    assert manifest["output"] == "data/benchmarks_final/final_test/test.jsonl"
    digest = hashlib.sha256(TEST_JSONL.read_bytes()).hexdigest()
    assert digest == manifest["output_sha256"]


def test_row_count(manifest, rows):
    assert manifest["n_total"] == 3453
    assert len(rows) == manifest["n_total"]
    with open(TEST_JSONL, encoding="utf-8") as f:
        assert sum(1 for _ in f) == 3453          # one row per line, no blanks


def test_per_source_counts(manifest, rows):
    by_source = Counter(r["source"] for r in rows)
    assert dict(by_source) == manifest["n_by_source"]
    assert manifest["n_by_source"] == {"sst": 99, "catqa": 543, "dna_core": 573, "dna_other": 364,
                                       "coconot": 378, "phtest": 1000, "no_robots": 496}


def test_per_source_label_counts(manifest, rows):
    by_source_label = Counter(f"{r['source']}:{r['label']}" for r in rows)
    assert dict(by_source_label) == manifest["n_by_source_label"]
    n_safe = sum(1 for r in rows if r["label"] == 1)
    assert n_safe == manifest["n_safe"] == 1874
    harmful = Counter(r["source"] for r in rows if r["label"] == 0)
    assert harmful["sst"] + harmful["catqa"] + harmful["dna_core"] == manifest["n_harmful_primary"] == 1215
    assert harmful["dna_other"] == manifest["n_harmful_secondary"] == 364


def test_rows_are_well_formed(rows):
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for r in rows:
        assert set(r) >= {"id", "text", "label", "source"}, r["id"]
        assert r["label"] in (0, 1), r["id"]
        assert isinstance(r["text"], str) and r["text"].strip(), r["id"]
