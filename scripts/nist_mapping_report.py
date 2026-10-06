"""
NIST AI RMF mapping: check and render
=====================================
Reads config/nist_ai_rmf_mapping.yaml, checks it, and renders
docs/nist_ai_rmf_mapping.md.

The checks are what make the mapping worth reading:

  * every subcategory id is one of the 72 in the AI RMF 1.0 Core (NIST AI 100-1);
  * every entry has a verbatim statement, a status and, unless the status is
    not_implemented, at least one piece of evidence;
  * every evidence path exists in the repository;
  * every runtime event the audit logger writes has an entry, and the ids it is
    tagged with all have status implemented or partial, so the audit log never
    cites a control the system does not have;
  * with --nist-text, every statement is found verbatim in the text extracted
    from the NIST PDF, after whitespace, dash and line-break hyphenation are
    normalised.

Usage:
    python scripts/nist_mapping_report.py                    # check and render
    python scripts/nist_mapping_report.py --check            # check only
    python scripts/nist_mapping_report.py --nist-text nist_ai_100_1.txt

Exit status is 1 when any check fails, and the failures are printed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── The AI RMF 1.0 Core, by identifier ────────────────────────────────────────
# GOVERN 19, MAP 18, MEASURE 22, MANAGE 13: 72 subcategories, as in NIST AI 100-1.

_CORE_SHAPE = {
    "GOVERN":  {1: 7, 2: 3, 3: 2, 4: 3, 5: 2, 6: 2},
    "MAP":     {1: 6, 2: 3, 3: 5, 4: 2, 5: 2},
    "MEASURE": {1: 3, 2: 13, 3: 3, 4: 3},
    "MANAGE":  {1: 4, 2: 4, 3: 2, 4: 3},
}
FUNCTIONS = tuple(_CORE_SHAPE)
CORE_IDS = tuple(
    f"{fn} {cat}.{n}"
    for fn, cats in _CORE_SHAPE.items()
    for cat, count in cats.items()
    for n in range(1, count + 1)
)
assert len(CORE_IDS) == 72, len(CORE_IDS)

STATUSES = ("implemented", "partial", "not_implemented")
TAGGABLE = ("implemented", "partial")

DEFAULT_MAPPING = "config/nist_ai_rmf_mapping.yaml"
DEFAULT_OUT = "docs/nist_ai_rmf_mapping.md"


def _runtime_events() -> tuple:
    """The events the audit logger writes, from the logger itself when importable."""
    try:
        from llmguard.utils.logger import EVENT_L1, EVENT_L2, EVENT_L3
        return (EVENT_L1, EVENT_L2, EVENT_L3)
    except Exception:  # pragma: no cover - the logger is in this repository
        return ("l1_decision", "l2_judgement", "l3_mitigation")


def load(path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def normalise_text(s: str) -> str:
    """
    Whitespace, dashes and line-break hyphenation folded away, for comparing a
    quoted statement with text extracted from the PDF. A hyphen at a line break
    cannot be told from a joined word in that text, so all hyphens go.
    """
    s = (s or "").replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = s.replace("-", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.replace("themap", "the map")


def validate(doc: dict, root: Path, nist_text: str | None = None) -> list:
    """Every problem found, as a list of strings. Empty means the mapping passes."""
    problems = []
    entries = doc.get("subcategories") or []
    by_id = {}

    for i, e in enumerate(entries):
        sid = str(e.get("id", "")).strip()
        where = f"entry {i + 1} ({sid or 'no id'})"
        if sid not in CORE_IDS:
            problems.append(f"{where}: id is not in the AI RMF 1.0 Core")
        if sid in by_id:
            problems.append(f"{where}: duplicate id")
        by_id[sid] = e
        if not str(e.get("statement", "")).strip():
            problems.append(f"{where}: empty statement")
        status = e.get("status")
        if status not in STATUSES:
            problems.append(f"{where}: status {status!r} not in {STATUSES}")
        evidence = e.get("evidence") or []
        if status in TAGGABLE and not evidence:
            problems.append(f"{where}: status {status} with no evidence")
        for ev in evidence:
            p = str((ev or {}).get("path", "")).strip()
            if not p:
                problems.append(f"{where}: evidence without a path")
            elif not (root / p).exists():
                problems.append(f"{where}: evidence path does not exist: {p}")
        if status != "implemented" and not str(e.get("gap", "")).strip():
            problems.append(f"{where}: status {status} needs a gap statement")

    events = doc.get("runtime_events") or {}
    for ev in _runtime_events():
        if ev not in events:
            problems.append(f"runtime event {ev} written by the logger has no entry")
    for ev, ids in events.items():
        if ev not in _runtime_events():
            problems.append(f"runtime event {ev} is not one the logger writes")
        if not ids:
            problems.append(f"runtime event {ev} has no ids")
        for sid in ids or []:
            entry = by_id.get(str(sid))
            if entry is None:
                problems.append(f"runtime event {ev}: {sid} has no subcategory entry")
            elif entry.get("status") not in TAGGABLE:
                problems.append(
                    f"runtime event {ev}: {sid} has status {entry.get('status')} "
                    "and may not be written into audit records")

    if nist_text is not None:
        hay = normalise_text(nist_text)
        for e in entries:
            if normalise_text(str(e.get("statement", ""))) not in hay:
                problems.append(f"{e.get('id')}: statement not found verbatim in the NIST text")

    return problems


def render(doc: dict) -> str:
    """The mapping as a Markdown document."""
    entries = doc.get("subcategories") or []
    counts = {s: sum(1 for e in entries if e.get("status") == s) for s in STATUSES}
    events = doc.get("runtime_events") or {}

    out = []
    out.append("# NIST AI RMF 1.0 mapping")
    out.append("")
    out.append(f"Framework: {doc.get('framework', '')}.")
    out.append("")
    out.append(f"Statements: {doc.get('statement_source', '')}.")
    out.append("")
    out.append(
        "This is a self-assessment of a research prototype against the subcategories of the "
        "AI RMF Core that the work bears on, not a conformance claim; the framework is voluntary "
        "and has no conformance procedure. Each row quotes the subcategory verbatim, names the "
        "repository files that bear on it and states the gap. Subcategories not listed are not "
        "claimed. The source of this page is `config/nist_ai_rmf_mapping.yaml`; "
        "`scripts/nist_mapping_report.py` checks every id against the 72 in the Core, every "
        "evidence path against the tree, and renders this file.")
    out.append("")
    out.append("| Status | Meaning | Count |")
    out.append("|---|---|---|")
    out.append(f"| implemented | the statement holds for this system as built | {counts['implemented']} |")
    out.append(f"| partial | something in the repository bears on it, and the gap is named | {counts['partial']} |")
    out.append(f"| not_implemented | nothing in the repository bears on it | {counts['not_implemented']} |")
    out.append(f"| not listed | not claimed | {72 - len(entries)} of 72 |")
    out.append("")

    out.append("## Audit records")
    out.append("")
    out.append(
        "With `logging.nist_mapping: true`, every audit record carries the ids below for its "
        "event, so a line in `logs/audit.jsonl` says which subcategories it is evidence for. "
        "Only ids with status implemented or partial may appear here.")
    out.append("")
    out.append("| Event | Written by | Tagged with |")
    out.append("|---|---|---|")
    writers = {
        "l1_decision":   "Layer 1, `layer1/detector.py`, serving path only",
        "l2_judgement":  "Layer 2, `layer2/layer2_judge.py`, when `record=True`",
        "l3_mitigation": "Layer 3, `layer3/layer3_mitigator.py`, when `record=True`",
    }
    for ev, ids in events.items():
        out.append(f"| `{ev}` | {writers.get(ev, '')} | {', '.join(ids or [])} |")
    out.append("")

    for fn in FUNCTIONS:
        rows = [e for e in entries if str(e.get("id", "")).startswith(fn + " ")]
        if not rows:
            continue
        out.append(f"## {fn}")
        out.append("")
        out.append("| Subcategory | Statement | Status | Evidence | Gap |")
        out.append("|---|---|---|---|---|")
        for e in rows:
            ev_cells = []
            for ev in e.get("evidence") or []:
                note = str(ev.get("note", "")).strip()
                ev_cells.append(f"`{ev.get('path')}`" + (f": {note}" if note else ""))
            evidence = "<br>".join(ev_cells) if ev_cells else "none"
            gap = str(e.get("gap", "")).strip() or "none"
            statement = str(e.get("statement", "")).replace("|", "\\|")
            out.append(f"| {e.get('id')} | {statement} | {e.get('status')} | {evidence} | {gap} |")
        out.append("")

    return "\n".join(out).rstrip("\n") + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", default=DEFAULT_MAPPING, help="mapping file (default: %(default)s)")
    p.add_argument("--out", default=DEFAULT_OUT, help="Markdown to write (default: %(default)s)")
    p.add_argument("--check", action="store_true", help="check only, write nothing")
    p.add_argument("--nist-text", default=None,
                   help="text extracted from NIST AI 100-1; every statement must appear in it verbatim")
    p.add_argument("--root", default=str(ROOT), help="repository root for evidence paths")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    doc = load(root / args.mapping if not Path(args.mapping).is_absolute() else args.mapping)
    nist_text = Path(args.nist_text).read_text(encoding="utf-8") if args.nist_text else None
    problems = validate(doc, root, nist_text)
    n = len(doc.get("subcategories") or [])
    if problems:
        print(f"mapping FAILED {len(problems)} check(s) over {n} entries:")
        for pr in problems:
            print("  -", pr)
        return 1
    verified = " and verbatim against the NIST text" if nist_text is not None else ""
    print(f"mapping ok: {n} entries, ids in the 72-id Core, evidence paths present{verified}")
    if not args.check:
        out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(render(doc).encode("utf-8"))
        print(f"rendered {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
