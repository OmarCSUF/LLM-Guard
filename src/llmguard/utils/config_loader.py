"""
Configuration loading
=====================
Reads config/config.yaml and resolves the two things the file names that
depend on where the code runs: repo-relative paths and the judge/guard model.

load_config(path)
    The recorded runs were all launched from the repository root, so every
    path in the config is relative to it. When the default path does not
    exist from the current directory, the copy under the repository root is
    used instead, so `load_config()` also works from another directory. An
    explicit path is used exactly as given.

repo_path(path)
    The same rule for the other repo-relative files the config names
    (safety templates). A path that exists from the cwd, or is absolute, is
    returned unchanged.

judge_model(section)
    (model_name, revision) for layer2.judge or layer3.guard_llm. The
    LLMGUARD_JUDGE_MODEL environment variable replaces model_name with a
    local directory or another hub id; both layers read it through this one
    function, so the Layer-3 weight-sharing check still compares like with
    like.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"

# src/llmguard/utils/config_loader.py -> utils -> llmguard -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

# A local directory or hub id that replaces layer2.judge.model_name and
# layer3.guard_llm.model_name at from_config time.
JUDGE_MODEL_ENV = "LLMGUARD_JUDGE_MODEL"


def repo_path(path) -> Path:
    """
    Resolve a repo-relative path from the config.

    Returns *path* unchanged when it is absolute or exists from the current
    directory (the recorded behaviour: every run started at the repo root).
    Otherwise, when the same relative path exists under the repository root,
    that copy is returned. A path that exists in neither place is returned
    as given, so the caller's own error message names what was configured.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    alt = REPO_ROOT / p
    return alt if alt.exists() else p


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load the YAML config as a dict.

    With the default path, falls back to <repo root>/config/config.yaml when
    the relative path does not exist from the current directory. An explicit
    path is never redirected. Raises FileNotFoundError when nothing is found.
    """
    path = Path(config_path)
    if not path.exists() and str(config_path) == DEFAULT_CONFIG_PATH:
        fallback = REPO_ROOT / DEFAULT_CONFIG_PATH
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def judge_model(section: dict) -> Tuple[str, Optional[str]]:
    """
    (model_name, revision) for a layer2.judge or layer3.guard_llm block.

    LLMGUARD_JUDGE_MODEL, when set and non-empty, replaces model_name. The
    pinned revision belongs to the configured hub id, so it is dropped with
    the override (a local directory has no revision; another hub id has a
    different one). Without the variable, the block's model_name and its
    optional revision are returned; a missing or empty revision is None, and
    from_pretrained is then called exactly as before the key existed.
    """
    override = os.environ.get(JUDGE_MODEL_ENV, "").strip()
    if override:
        return override, None
    revision = section.get("revision") or None
    return section["model_name"], revision


def hub_kwargs(revision: Optional[str]) -> dict:
    """{"revision": revision} when one is set, else {} (the old call)."""
    return {"revision": revision} if revision else {}
