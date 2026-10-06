"""
Fetch the final-test sources
============================
Downloads the six sources for the locked final test set through the project's
own verified downloader (scripts/fetch_datasets.py: plain HTTP via HF_ENDPOINT,
every file checked against the checksum the Hub publishes for it), into
data/raw/hf_final/<repo>/<path>, and writes data/raw/hf_final/manifest.json with
the repository revision, size and digest of every file.

Those six sources yield the seven labelled groups the final test set reports on:
scripts/final_eval/build_final_test.py (load_dna) splits Do-Not-Answer by risk
area into dna_core (primary harmful) and dna_other (secondary), so the seven
group names are sst, catqa, dna_core, dna_other, coconot, phtest, no_robots.

The source list is fixed by PREREGISTRATION.md, Section 3.

Run from the repository root with the guard environment active:

    python scripts/final_eval/fetch_final_sources.py            # all sources
    python scripts/final_eval/fetch_final_sources.py --only phtest no_robots

To fetch exactly the files the recorded study used, pin the repository
revisions to the recorded manifest (default: current revisions, as before):

    python scripts/final_eval/fetch_final_sources.py \
        --revision-manifest results/development/raw_final_sources_manifest.json

With a pin, a file whose digest differs from the recorded one is a failure.
See scripts/fetch_datasets.py for --revision REPO=SHA and the details.

Exit status 1 if any source failed; the manifest still records the others.
"""

import sys
import os

REPO = os.environ.get("LLMGUARD_REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))  # local checkout without `pip install -e .`
sys.path.insert(0, os.path.join(REPO, "scripts"))

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import fetch_datasets as fd                      # the project's downloader
from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("fetch_final_sources")

# name -> (dataset repo, files needed). Order matters nowhere here; the build
# script fixes the order of precedence for cross-source duplicates.
FINAL_SOURCES = {
    "sst":       ("Bertievidgen/SimpleSafetyTests",
                  ["sst_test_cases.csv"]),
    "catqa":     ("declare-lab/CategoricalHarmfulQA",
                  ["data/catqa_english.json"]),
    "dna":       ("LibrAI/do-not-answer",
                  ["data_en.csv"]),
    "coconot":   ("allenai/coconot",
                  ["contrast/test-00000-of-00001.parquet"]),
    "phtest":    ("furonghuang-lab/PHTest",
                  ["PHTest.csv"]),
    "no_robots": ("HuggingFaceH4/no_robots",
                  ["data/test-00000-of-00001.parquet"]),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and verify the final-test sources")
    p.add_argument("--only", nargs="+", choices=sorted(FINAL_SOURCES), default=None)
    p.add_argument("--out", default=None, help="Default: <data.raw_dir>/hf_final")
    fd.add_pin_arguments(p)          # --revision-manifest PATH, --revision REPO=SHA
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    run_info = collect_run_info(cfg)
    out_dir = Path(args.out) if args.out else Path(cfg["data"]["raw_dir"]) / "hf_final"
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})

    # fetch_source() reads the module-level SOURCES table; point it at ours.
    fd.SOURCES = FINAL_SOURCES
    token = fd._token()
    logger.info(f"Endpoint: {fd._endpoint()} | token: {'yes' if token else 'no'}")
    pins = fd.load_revision_pins(args.revision_manifest, args.revision, FINAL_SOURCES)
    if pins:
        logger.info(f"Pinned revisions for {len(pins)} repositories"
                    + (f" from {args.revision_manifest}" if args.revision_manifest else ""))

    failed = {}
    for name in args.only or list(FINAL_SOURCES):
        logger.info(f"Fetching {name}...")
        try:
            manifest[name] = fd.fetch_source(name, out_dir, token, pins)
        except HTTPError as exc:
            hint = " (gated: accept the terms on huggingface.co, then `hf auth login`)" \
                if exc.code in (401, 403) else ""
            failed[name] = f"HTTP {exc.code} for {exc.url}{hint}"
        except (URLError, OSError, ValueError, KeyError) as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
        if name in failed:
            logger.error(f"  {name} failed: {failed[name]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(manifest_path)
    manifest_path.write_text(json.dumps(
        {"sources": manifest, "failed": failed, "run_info": finalize_run_info(run_info)},
        indent=2), encoding="utf-8")
    logger.info(f"Manifest -> {manifest_path}")
    if failed:
        logger.error(f"{len(failed)} source(s) failed: {', '.join(failed)}")
        return 1
    logger.info("All final-test sources fetched and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
