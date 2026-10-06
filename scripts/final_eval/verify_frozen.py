"""Compare a stored freeze record (frozen.json) with the files present in this checkout.

Reports four groups: the Layer-1 checkpoint artifacts (must all match: the four pinned
checkpoints ship in artifacts/), the configuration (expected to differ in the release, see
CHANGES.md), the Python source files (the stored record used the pre-release
layout; paths are mapped to src/llmguard/...; differences are listed, since the release refactor
changed imports and several scripts were edited after the freeze, see REPRODUCIBILITY.md), and the
pool splits / input files (skipped when the data are not present).

    python scripts/final_eval/verify_frozen.py --study results/final_eval_20260926
Exit status 1 if any checkpoint artifact differs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze import ROOT, sha256, pool_entry  # noqa: E402

PACKAGES = ("layer1", "layer2", "layer3", "evaluation", "redteam", "utils")


def map_path(old: str) -> str:
    """Pre-release layout -> release layout."""
    first = old.split("/", 1)[0]
    if first in PACKAGES:
        return "src/llmguard/" + old
    return old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="results/final_eval_20260926")
    ap.add_argument("--pool-dir", default="data/distillation_pool")
    ap.add_argument("--verbose", action="store_true", help="list every matching file too")
    a = ap.parse_args()
    rec = json.load(open(Path(a.study) / "frozen.json"))
    bad = 0
    print("== checkpoint artifacts ==")
    for name, ck in rec["checkpoints"].items():
        for rel, digest in ck["sha256"].items():
            p = ROOT / rel
            if not p.exists():
                print(f"  MISSING  {rel}"); bad += 1
            elif sha256(p) == digest:
                print(f"  ok       {rel}")
            else:
                print(f"  DIFFERS  {rel}"); bad += 1
    print("== source files (stored at the pre-release layout) ==")
    same = diff = missing = 0
    for rel, digest in rec["files"].items():
        p = ROOT / map_path(rel)
        if not p.exists():
            missing += 1; print(f"  missing  {rel} -> {map_path(rel)}")
        elif sha256(p) == digest:
            same += 1
            if a.verbose: print(f"  ok       {rel}")
        else:
            diff += 1; print(f"  differs  {rel} -> {map_path(rel)}" + ("  (release config: hub ids, pinned revisions, dead keys removed)" if rel == "config/config.yaml" else ""))
    print(f"  {same} identical, {diff} differ, {missing} missing of {len(rec['files'])}")
    print("== pool splits ==")
    for split, e in rec.get("pool", {}).items():
        p = Path(a.pool_dir) / f"{split}.jsonl"
        if not p.exists():
            print(f"  skipped  {split} (not present; rebuild with scripts/build_datasets.py)")
            continue
        now = pool_entry(p)
        print(f"  {'ok     ' if now == e else 'DIFFERS'}  {split}: n={now['n']} ids_sha1={now['ids_sha1'][:12]} sha256={now['sha256'][:12]}")
    print("== input files ==")
    for name, e in rec.get("inputs", {}).items():
        p = ROOT / e["path"]
        if not p.exists():
            print(f"  skipped  {name} ({e['path']} not present)")
        else:
            print(f"  {'ok     ' if sha256(p) == e['sha256'] else 'DIFFERS'}  {name}")
    print(f"== git == stored commit {rec['git'].get('commit')} dirty={rec['git'].get('dirty')}")
    if bad:
        print(f"{bad} checkpoint artifact(s) differ or are missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
