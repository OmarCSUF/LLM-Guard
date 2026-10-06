"""
config/config.yaml loads and carries the values the recorded runs used.

The thresholds, band, seed and classifier_only flag are the settings that
every stored result depends on; a change to any of them would make the
results irreproducible, so they are pinned here.
"""
from __future__ import annotations

import pytest

from llmguard.utils.config_loader import load_config

REQUIRED_SECTIONS = ("project", "data", "layer1", "layer2", "layer3",
                     "evaluation", "red_team", "logging")


def _check_values(cfg: dict) -> None:
    for key in REQUIRED_SECTIONS:
        assert key in cfg, f"missing top-level section {key!r}"
    assert cfg["project"]["seed"] == 42
    assert cfg["layer1"]["classifier_only"] is True
    assert cfg["layer1"]["thresholds"]["low"] == pytest.approx(0.3)
    assert cfg["layer1"]["thresholds"]["high"] == pytest.approx(0.7)
    band = cfg["layer2"]["consistency"]
    assert band["allow_below"] == pytest.approx(0.4)
    assert band["mitigate_above"] == pytest.approx(0.6)
    assert band["mode"] == "logprob"
    assert cfg["data"]["labels"] == {"allow": 1, "mitigate": 0}


def test_load_from_repo_root(repo_root, monkeypatch):
    """The recorded runs were launched from the repo root with the default path."""
    monkeypatch.chdir(repo_root)
    cfg = load_config()
    assert isinstance(cfg, dict)
    _check_values(cfg)


def test_load_from_another_cwd(repo_root, tmp_path, monkeypatch):
    """
    From another directory the loader falls back to <repo root>/config/config.yaml
    for the default path; an explicit absolute path always works.
    """
    monkeypatch.chdir(tmp_path)
    explicit = load_config(str(repo_root / "config" / "config.yaml"))
    _check_values(explicit)
    fallback = load_config()
    assert fallback == explicit


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))


def test_layer1_backbone_and_lora(repo_root):
    """The checkpoint shape the pinned adapters were trained with."""
    cfg = load_config(str(repo_root / "config" / "config.yaml"))
    clf = cfg["layer1"]["classifier"]
    assert clf["backbone"] == "roberta-base"
    assert (clf["lora_r"], clf["lora_alpha"], clf["max_length"]) == (8, 16, 512)
