"""
Score refusal detectors against a saved red-team run
====================================================
Reads a <run>_variants.jsonl, keeps the rows Layer 3 served as rewrites, and scores each
served output with the keyword detector and, when a model is available, with the
model-backed check. It writes one row per output: the id, both verdicts and the model's
P(decline).

No labels are read. The reviewer's labels stay wherever they are and the join happens
there, so the detector is scored against a reading it never saw.

Usage (venv active, from the repository root):
    python scripts/refusal_eval.py --variants data/processed/red_team_pilot4_variants.jsonl \
        --out data/processed/refusal_eval_pilot4.json --ids R01=154,R02=128,...
    python scripts/refusal_eval.py --variants ... --out ... --no-model    # keywords only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.refusal import RefusalCheck, looks_like_refusal
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting

logger = get_logger("refusal_eval")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score refusal detectors on a saved run's served rewrites")
    p.add_argument("--variants", required=True, help="A <run>_variants.jsonl from run_red_team.py")
    p.add_argument("--out", required=True, help="JSON file for the per-output verdicts")
    p.add_argument("--ids", default=None,
                   help="Review ids to row indices, e.g. R01=154,R02=128 (0-based lines of --variants)")
    p.add_argument("--no-model", action="store_true", help="Score the keyword detector only")
    p.add_argument("--threshold", type=float, default=None,
                   help="P(decline) at or above this counts as a refusal (default: config, else 0.5)")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N served outputs")
    return p.parse_args(argv)


def parse_ids(spec: Optional[str]) -> Dict[int, str]:
    """'R01=154,R02=128' -> {154: 'R01', 128: 'R02'}."""
    out: Dict[int, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rid, idx = part.split("=")
        out[int(idx)] = rid.strip()
    return out


def served_rewrites(rows: list) -> list:
    """(row index, row) for every row Layer 3 served as a rewrite with an output."""
    return [(i, r) for i, r in enumerate(rows)
            if r.get("l3_mode") == "safe_rewrite" and (r.get("l3_output") or "").strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    run_info = collect_run_info(cfg)
    threshold = args.threshold
    if threshold is None:
        threshold = float((cfg.get("layer3", {}).get("refusal_check", {}) or {}).get("threshold", 0.5))
    ids = parse_ids(args.ids)

    rows = load_jsonl(args.variants)
    todo = served_rewrites(rows)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        raise SystemExit("No served rewrites in that file (l3_mode == safe_rewrite with an output).")
    logger.info(f"{len(rows)} rows, {len(todo)} served outputs to score")

    check = None
    if not args.no_model:
        from llmguard.layer2 import Layer2Judge
        judge = Layer2Judge.from_config(cfg)
        name, model, tokenizer = judge.shared_hf_model()
        if model is None:
            raise SystemExit("The judge holds no HuggingFace weights; the model check needs logits.")
        check = RefusalCheck(model=model, tokenizer=tokenizer, threshold=threshold)
        logger.info(f"Model check ready on {name}, threshold {threshold:.2f}")

    out_rows = []
    t0 = time.perf_counter()
    for n, (idx, row) in enumerate(todo, 1):
        output = row["l3_output"]
        keyword = bool(looks_like_refusal(output))
        p = check.p_refusal(output) if check else None
        out_rows.append({
            "file_index": idx,
            "id": ids.get(idx),
            "attack_type": row.get("attack_type"),
            "keyword_refusal": keyword,
            "p_decline": None if p is None else round(p, 4),
            "model_refusal": p is not None and p >= threshold,
            "chars": len(output),
        })
        if n % 20 == 0 or n == len(todo):
            logger.info(f"  {n}/{len(todo)} scored ({time.perf_counter() - t0:.0f}s)")

    summary = {
        "n_outputs": len(out_rows), "threshold": threshold,
        "keyword_refusals": sum(r["keyword_refusal"] for r in out_rows),
        "model_refusals": sum(r["model_refusal"] for r in out_rows),
        "model_scored": sum(r["p_decline"] is not None for r in out_rows),
    }
    report = {"variants_file": args.variants, "summary": summary, "rows": out_rows,
              "run_info": finalize_run_info(run_info)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nrefusal detectors on {args.variants}: {summary}")
    print("id      idx  keyword  p_decline  model")
    for r in out_rows:
        p = "   n/a  " if r["p_decline"] is None else f"{r['p_decline']:8.3f}"
        print(f"{(r['id'] or '-'):<7} {r['file_index']:>4}  {'yes' if r['keyword_refusal'] else 'no ':<7}"
              f"{p}  {'yes' if r['model_refusal'] else 'no'}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
