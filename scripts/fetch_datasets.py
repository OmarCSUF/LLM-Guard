"""
Dataset Fetch
=============
Downloads the exact source files for the rebuilt Layer-1 training pool and the
evaluation benchmarks over plain HTTP from HF_ENDPOINT (e.g. hf-mirror.com),
and verifies each file against the checksum the Hub publishes for it.

huggingface_hub's downloader rejects some mirror responses ("Distant resource
does not seem to be on huggingface.co"); plain HTTP with redirects does not.

Verification, against the live tree listing for the pinned repo revision:
  Git-LFS files  : sha256 of the content must equal lfs.oid
  regular files  : git blob sha1 of the content must equal oid
A file that fails verification is discarded and reported as failed.

Files land in data/raw/hf/<repo>/<path>, and data/raw/hf/manifest.json records
the revision, size and checksum of each one as data provenance.

Gated datasets (XSTest, HarmBench, WildJailbreak) need a HuggingFace token: accept the terms on
the dataset page, then run `hf auth login`. The token is sent only to
HF_ENDPOINT, never to the storage hosts that downloads redirect to.

Revision pinning (reproducing the recorded study):
  By default every repository is fetched at its CURRENT revision and that
  revision is recorded in the manifest. To fetch exactly the files the study
  used, pin the revisions instead:

    --revision-manifest PATH   pin every repository named in an earlier manifest
                               (data/raw/hf/manifest.json format, or the
                               source_revisions block of a build manifest) to the
                               revision it records, and fail if the digest the
                               Hub lists for a file at that revision, or the
                               content downloaded, differs from the recorded one
    --revision REPO=SHA        pin one repository (repeatable; REPO is the Hub
                               id or the source name); overrides the manifest

  The Hub accepts a commit sha wherever it accepts "main":
  /api/datasets/<repo>/tree/<sha> and /datasets/<repo>/resolve/<sha>/<path>.
  Without either flag the behaviour is unchanged.

Usage:
    python scripts/fetch_datasets.py
    python scripts/fetch_datasets.py --only toxicchat jailbreakbench
    python scripts/fetch_datasets.py --revision-manifest results/development/raw_sources_manifest.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("fetch_datasets")

# name -> (dataset repo, files needed). Evaluation: jailbreakbench, xstest,
# harmbench, toxicchat (test), orbench. Training pool: dolly, beavertails,
# toxicchat (train), wildjailbreak (train.tsv is about 530 MB).
SOURCES = {
    "jailbreakbench": ("JailbreakBench/JBB-Behaviors",
                       ["data/harmful-behaviors.csv", "data/benign-behaviors.csv"]),
    "xstest":         ("walledai/XSTest",
                       ["data/train-00000-of-00001.parquet"]),
    "harmbench":      ("walledai/HarmBench",
                       ["standard/train-00000-of-00001.parquet"]),
    "toxicchat":      ("lmsys/toxic-chat",
                       ["data/0124/toxic-chat_annotation_train.csv",
                        "data/0124/toxic-chat_annotation_test.csv"]),
    "dolly":          ("databricks/databricks-dolly-15k",
                       ["databricks-dolly-15k.jsonl"]),
    "beavertails":    ("PKU-Alignment/BeaverTails",
                       ["round0/30k/train.jsonl.gz"]),
    "wildjailbreak":  ("allenai/wildjailbreak",
                       ["train/train.tsv"]),
    "orbench":        ("bench-llm/or-bench",
                       ["or-bench-hard-1k.csv", "or-bench-toxic.csv"]),
}

_CHUNK = 1 << 20
_REPORT_BYTES = 100 << 20   # progress log interval for large downloads


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and verify LLM-Guard source datasets")
    p.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=None,
                   help="Fetch only these datasets (default: all)")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory (default: <data.raw_dir>/hf)")
    add_pin_arguments(p)
    return p.parse_args()


def add_pin_arguments(p: argparse.ArgumentParser) -> None:
    """Add the revision-pinning options shared with fetch_final_sources.py."""
    p.add_argument("--revision-manifest", type=str, default=None, metavar="PATH",
                   help="Pin every repository to the revision recorded in this manifest "
                        "(a manifest.json written by this script, or any JSON with a "
                        "'sources' or 'source_revisions' block) and fail if a file's digest "
                        "differs from the recorded one. Default: fetch the current revision.")
    p.add_argument("--revision", action="append", default=None, metavar="REPO=SHA",
                   help="Pin one repository (Hub id or source name) to a commit sha; "
                        "repeatable, overrides --revision-manifest for that repository.")


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


class _Redirects(HTTPRedirectHandler):
    """
    Follow 308 (urllib only handles it from Python 3.11; hf-mirror.com uses it
    on API URLs), and keep the token on same-host redirects only, so it never
    reaches the storage hosts that downloads redirect to.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, 307 if code == 308 else code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth and urlsplit(newurl).netloc == urlsplit(req.full_url).netloc:
            new.add_unredirected_header("Authorization", auth)
        return new

    http_error_308 = HTTPRedirectHandler.http_error_302


_OPENER = build_opener(_Redirects)


def _open(url: str, token: Optional[str]):
    req = Request(url, headers={"User-Agent": "llm-guard-fetch/1.0"})
    if token:
        # Unredirected, so _Redirects decides whether a redirect may carry it.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return _OPENER.open(req, timeout=120)


def _get_json(url: str, token: Optional[str]):
    with _open(url, token) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _expected(entry: dict) -> tuple:
    """Return (algorithm, digest, size) the downloaded content must match."""
    lfs = entry.get("lfs")
    if lfs:
        return "sha256", lfs["oid"], int(lfs["size"])
    return "git-sha1", entry["oid"], int(entry["size"])


def _hasher(algorithm: str, size: int):
    if algorithm == "sha256":
        return hashlib.sha256()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)   # git blob object header
    return h


def _file_matches(path: Path, algorithm: str, digest: str, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = _hasher(algorithm, size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest() == digest


def _download(url: str, dest: Path, token: Optional[str],
              algorithm: str, digest: str, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = _hasher(algorithm, size)
    n, next_report = 0, _REPORT_BYTES
    try:
        with _open(url, token) as resp, open(part, "wb") as fh:
            for chunk in iter(lambda: resp.read(_CHUNK), b""):
                fh.write(chunk)
                h.update(chunk)
                n += len(chunk)
                if n >= next_report:
                    logger.info(f"    {dest.name}: {n / 2**20:.0f} of {size / 2**20:.0f} MB")
                    next_report += _REPORT_BYTES
        if n != size:
            raise ValueError(f"size mismatch: got {n} bytes, expected {size}")
        if h.hexdigest() != digest:
            raise ValueError(f"{algorithm} mismatch: got {h.hexdigest()}, expected {digest}")
        part.replace(dest)
    finally:
        if part.exists():
            part.unlink()


def load_revision_pins(manifest_path: Optional[str], overrides: Optional[list],
                       sources: Optional[dict] = None) -> dict:
    """
    Build the pin table {repo: {"revision": sha, "files": {path: {"algorithm", "digest"}}}}.

    `manifest_path` is a JSON file with a "sources" block (data/raw/hf/manifest.json
    as written by this script) or a "source_revisions" block (as written by
    build_datasets.py and build_final_test.py). Each entry needs "repo" and
    "revision"; its optional "files" list gives the digests to enforce.
    `overrides` are "REPO=SHA" strings; REPO may be a Hub id or a source name in
    `sources` (default: SOURCES). An override replaces the manifest revision for
    that repository and drops its recorded digests, since they belong to the
    recorded revision. Returns {} when nothing is pinned.
    """
    sources = SOURCES if sources is None else sources
    pins: dict = {}
    if manifest_path:
        doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        block = doc.get("sources")
        if block is None:
            block = doc.get("source_revisions")
        if not isinstance(block, dict):
            raise ValueError(f"{manifest_path}: no 'sources' or 'source_revisions' block")
        for entry in block.values():
            if not isinstance(entry, dict) or not entry.get("repo") or not entry.get("revision"):
                continue
            files = {}
            for f in entry.get("files") or []:
                if f.get("path") and f.get("digest"):
                    files[f["path"]] = {"algorithm": f.get("algorithm"), "digest": f["digest"]}
            pins[entry["repo"]] = {"revision": entry["revision"], "files": files}
    for item in overrides or []:
        repo, sep, sha = item.partition("=")
        repo, sha = repo.strip(), sha.strip()
        if not sep or not repo or not sha:
            raise ValueError(f"--revision expects REPO=SHA, got {item!r}")
        if repo in sources:                      # a source name, e.g. "xstest"
            repo = sources[repo][0]
        pins[repo] = {"revision": sha, "files": {}}
    return pins


def _check_pinned_digest(repo: str, revision: str, path: str, pin_files: dict,
                         algorithm: str, digest: str) -> None:
    """
    Fail if the Hub lists a different digest for `path` at the pinned revision
    than the manifest recorded. Downloads are verified against the Hub's digest,
    so after this check a verified file is also byte-identical to the recorded one.
    """
    want = pin_files.get(path)
    if not want:
        return
    if want["algorithm"] and want["algorithm"] != algorithm:
        raise ValueError(f"DIGEST MISMATCH {repo}/{path}@{revision[:12]}: manifest records "
                         f"{want['algorithm']} {want['digest']}, Hub lists {algorithm} {digest}")
    if want["digest"] != digest:
        raise ValueError(f"DIGEST MISMATCH {repo}/{path}@{revision[:12]}: manifest records "
                         f"{want['digest']}, Hub lists {digest}")


def fetch_source(name: str, out_dir: Path, token: Optional[str],
                 pins: Optional[dict] = None) -> dict:
    """
    Download and verify every file of one source. Returns its manifest entry.

    `pins` (from load_revision_pins) fixes the revision per repository and the
    digest per file; without it (the default) the current revision is fetched
    and recorded, exactly as before.
    """
    repo, wanted = SOURCES[name]
    ep = _endpoint()
    pin = (pins or {}).get(repo)
    if pin:
        revision = pin["revision"]
        logger.info(f"  {repo}: pinned to revision {revision[:12]}")
    else:
        if pins:
            logger.warning(f"  {repo}: not in the pin table; fetching its current revision")
        revision = _get_json(f"{ep}/api/datasets/{repo}", token)["sha"]
    tree = _get_json(f"{ep}/api/datasets/{repo}/tree/{revision}?recursive=true", token)
    by_path = {e["path"]: e for e in tree if e.get("type") == "file"}

    missing = [p for p in wanted if p not in by_path]
    if missing:
        raise FileNotFoundError(f"{repo}@{revision[:12]} has no {missing}")

    files = []
    for path in wanted:
        algorithm, digest, size = _expected(by_path[path])
        if pin:
            _check_pinned_digest(repo, revision, path, pin["files"], algorithm, digest)
        dest = out_dir / repo / path
        if _file_matches(dest, algorithm, digest, size):
            logger.info(f"  {repo}/{path}: already present and verified")
        else:
            url = f"{ep}/datasets/{repo}/resolve/{revision}/{quote(path)}"
            logger.info(f"  {repo}/{path}: downloading {size / 1024:.0f} KB")
            _download(url, dest, token, algorithm, digest, size)
            logger.info(f"  {repo}/{path}: verified ({algorithm})")
        files.append({"path": path, "local_path": str(dest), "size": size,
                      "algorithm": algorithm, "digest": digest})

    entry = {"repo": repo, "revision": revision, "endpoint": ep, "files": files}
    if pin:
        entry["pinned"] = True       # only written when a pin was applied
    return entry


def main() -> int:
    args = parse_args()
    cfg = load_config()
    run_info = collect_run_info(cfg)

    out_dir = Path(args.out) if args.out else Path(cfg["data"]["raw_dir"]) / "hf"
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})

    token = _token()
    logger.info(f"Endpoint: {_endpoint()} | token: {'yes' if token else 'no'}")
    pins = load_revision_pins(args.revision_manifest, args.revision)
    if pins:
        logger.info(f"Pinned revisions for {len(pins)} repositories"
                    + (f" from {args.revision_manifest}" if args.revision_manifest else ""))

    failed = {}
    for name in args.only or list(SOURCES):
        logger.info(f"Fetching {name}...")
        try:
            manifest[name] = fetch_source(name, out_dir, token, pins)
        except HTTPError as exc:
            hint = ""
            if exc.code in (401, 403):
                hint = " (gated: accept the terms on huggingface.co, then `hf auth login`)"
            failed[name] = f"HTTP {exc.code} for {exc.url}{hint}"
        except (URLError, OSError, ValueError, KeyError) as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
        if name in failed:
            logger.error(f"  {name} failed: {failed[name]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(manifest_path)
    manifest_path.write_text(
        json.dumps({"sources": manifest, "run_info": finalize_run_info(run_info)}, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Manifest → {manifest_path}")

    if failed:
        logger.error(f"{len(failed)} source(s) failed: {', '.join(failed)}")
        return 1
    logger.info("All sources fetched and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
