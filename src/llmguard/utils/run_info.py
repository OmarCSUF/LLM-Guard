"""
Run provenance
==============
Records how a result file was produced, so every reported number can be traced
back to the exact command, configuration, code version and environment.

Public API:
    info = collect_run_info(cfg)                   # once, right after seeding
    results["run_info"] = finalize_run_info(info)  # just before saving
    warn_if_overwriting(path)                      # before writing a results file
"""
from __future__ import annotations

import copy
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

from llmguard.utils.logger import get_logger

logger = get_logger("utils.run_info")

_PACKAGES = (
    "torch", "transformers", "bitsandbytes", "peft", "accelerate",
    "scikit-learn", "numpy",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu() -> Optional[dict]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name":             props.name,
        "cuda":             torch.version.cuda,
        "total_memory_gib": round(props.total_memory / 1024 ** 3, 2),
    }


def collect_run_info(cfg: dict) -> dict:
    """
    Snapshot the command, config, seed, code version and environment at run start.

    git_dirty is True when tracked files differ from git_commit, meaning the
    results came from code that is not exactly that commit.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "started_at": _now(),
        "command":    [sys.executable, *sys.argv],
        "cwd":        str(Path.cwd()),
        "seed":       cfg["project"]["seed"],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty":  None if status is None else bool(status),
        "python":     platform.python_version(),
        "packages":   _package_versions(),
        "gpu":        _gpu(),
        "config":     copy.deepcopy(cfg),
        "_t0":        time.perf_counter(),
    }


def finalize_run_info(info: dict) -> dict:
    """Return a JSON-ready copy of `info` with the finish time and wall-clock duration."""
    done = dict(info)
    t0 = done.pop("_t0", None)
    done["finished_at"] = _now()
    if t0 is not None:
        done["wall_clock_s"] = round(time.perf_counter() - t0, 2)
    return done


def warn_if_overwriting(path) -> None:
    """Log a warning when an existing results file is about to be replaced."""
    path = Path(path)
    if path.exists():
        logger.warning(f"Overwriting existing results file: {path.resolve()}")
