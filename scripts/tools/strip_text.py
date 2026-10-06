"""Remove the ``text`` field from per-prompt prediction files whose prompts may not be
redistributed (the W&I+LOCNESS learner sentences; the training-pool prompts drawn from
gated or non-commercial sources).

Only the ``text`` key is removed; every other field, the row order and the JSON encoding
(json.dumps defaults, one object per line) are preserved, so every analysis script, which
reads ids, labels, decisions and scores, is unaffected. A manifest records each file's row
count and SHA-256 before and after.

    python scripts/tools/strip_text.py --manifest results/final_eval_20260926/TEXT_REMOVED.md \
        results/final_eval_20260926/learner_*_predictions/*.jsonl ...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_file(path: Path, field: str) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    removed = sum(1 for r in rows if field in r)
    for r in rows:
        r.pop(field, None)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(rows), removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="JSONL files to rewrite in place")
    ap.add_argument("--field", default="text", help="field to remove (default: text)")
    ap.add_argument("--manifest", required=True, help="Markdown file to append the record to")
    ap.add_argument("--reason", default="prompt texts come from a corpus that may not be redistributed",
                    help="one-line reason written to the manifest")
    args = ap.parse_args()
    lines = ["| File | Rows | Rows with `%s` removed | SHA-256 before | SHA-256 after |" % args.field,
             "|---|---:|---:|---|---|"]
    for f in args.files:
        p = Path(f)
        before = sha256(p)
        n, removed = strip_file(p, args.field)
        lines.append("| `%s` | %d | %d | `%s` | `%s` |" % (p.as_posix(), n, removed, before, sha256(p)))
        print("stripped %s: %d rows, %d had %s" % (p, n, removed, args.field))
    mp = Path(args.manifest)
    header = "" if mp.exists() else ("# Prompt texts removed from stored prediction files\n\n"
        "Reason: %s. Only the `%s` key was removed from each row (scripts/tools/strip_text.py); "
        "ids, labels, decisions and scores are unchanged, and every analysis script reads only those. "
        "The SHA-256 values before removal match the files in the original workspace.\n\n" % (args.reason, args.field))
    with open(mp, "a", encoding="utf-8") as fh:
        fh.write(header + "\n".join(lines) + "\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
