"""
Judge Tuning on the Pool Validation Split
=========================================
Compares Layer-2 judge settings on a sample of the Layer-1 pool's validation
split, never on the evaluation benchmarks, so the settings the thesis reports
are not fitted to its own test numbers.

For every sampled prompt and every prompt version it records:
  p_mitigate : P(mitigate) from one forward pass     (consistency mode "logprob")
  greedy     : one temperature-0 call
  votes      : K calls at the sample temperature     (consistency mode "vote")

and reports, per prompt version and for all prompts / WildJailbreak prompts
(whose benign half is written to resemble harmful requests) / other sources:
  greedy, majority vote and logprob-band decisions: accuracy on decided prompts,
  benign prompts blocked, harmful prompts allowed, share left undecided;
  AUROC of p_mitigate and the error rates at fixed thresholds;
  invalid outputs, vote unanimity and seconds per prompt.

Rows are appended as each prompt finishes; --resume keeps finished rows, so an
interrupted run can continue.

Usage:
    python scripts/tune_judge.py
    python scripts/tune_judge.py --n 20 --k 3          # quick check
    python scripts/tune_judge.py --prompts v2 --resume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from llmguard.utils.config_loader import load_config
from llmguard.utils.data_utils import load_jsonl, set_seed
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation.metrics import auroc, format_metric
from llmguard.layer1.fusion import DECISION_ALLOW, DECISION_MITIGATE
from llmguard.layer2.judge import LLMJudge, PROMPTS, DECISION_UNCERTAIN

logger = get_logger("tune_judge")

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare judge prompt versions and decision modes on the pool validation split")
    p.add_argument("--input", default=None,
                   help="Labelled JSONL (default: <data.distillation_pool_dir>/val.jsonl)")
    p.add_argument("--n", type=int, default=300, help="Prompts to sample, half from each class")
    p.add_argument("--k", type=int, default=None, help="Sampled calls per prompt (default: layer2.consistency.K)")
    p.add_argument("--prompts", nargs="+", default=["v1", "v2"], help="Prompt versions to compare")
    p.add_argument("--output", default="data/processed/judge_tuning.jsonl", help="Per-prompt rows")
    p.add_argument("--summary", default="data/processed/judge_tuning_summary.json")
    p.add_argument("--judge-model", default=None,
                   help="Override layer2.judge.model_name. Used to score an identical "
                        "protocol with a different judge; absent, nothing changes.")
    p.add_argument("--resume", action="store_true", help="Keep rows already in --output and skip those prompts")
    return p.parse_args()


# ── Decisions (mirror ConsistencyChecker) ─────────────────────────────────────

def sample_records(records: list, n: int, rng: random.Random) -> list:
    """n prompts, half per class, drawn from the whole split so sources keep their mix."""
    out = []
    for label in (1, 0):
        recs = [r for r in records if int(r["label"]) == label]
        out += rng.sample(recs, min(n // 2, len(recs)))
    rng.shuffle(out)
    return out


def majority(decisions: list, k: int, c_min: float) -> str:
    """Vote mode: majority of valid votes, confidence over all k calls, ties to mitigate."""
    counts = Counter(decisions)
    allow, mitigate = counts.get(DECISION_ALLOW, 0), counts.get(DECISION_MITIGATE, 0)
    if allow + mitigate == 0:
        return DECISION_UNCERTAIN
    decision, top = (DECISION_ALLOW, allow) if allow > mitigate else (DECISION_MITIGATE, mitigate)
    return decision if top / max(k, 1) >= c_min else DECISION_UNCERTAIN


def band(p: float, allow_below: float, mitigate_above: float) -> str:
    """Logprob mode: allow below allow_below, mitigate above mitigate_above, otherwise uncertain."""
    if p > mitigate_above:
        return DECISION_MITIGATE
    if p < allow_below:
        return DECISION_ALLOW
    return DECISION_UNCERTAIN


# ── Summary ───────────────────────────────────────────────────────────────────

def rates(rows: list, decisions: list) -> dict:
    """Error rates for one decision per row (allow / mitigate / uncertain / invalid)."""
    benign = [d for r, d in zip(rows, decisions) if r["label"] == 1]
    harmful = [d for r, d in zip(rows, decisions) if r["label"] == 0]
    decided = [(r, d) for r, d in zip(rows, decisions) if d in (DECISION_ALLOW, DECISION_MITIGATE)]
    correct = sum((d == DECISION_ALLOW) == (r["label"] == 1) for r, d in decided)

    def share(ds, value):
        return round(sum(d == value for d in ds) / len(ds), 4) if ds else None

    return {
        "n":                len(rows),
        "accuracy_decided": round(correct / len(decided), 4) if decided else None,
        "benign_blocked":   share(benign, DECISION_MITIGATE),
        "harmful_allowed":  share(harmful, DECISION_ALLOW),
        "undecided":        round(1 - len(decided) / len(rows), 4) if rows else None,
    }


def _mean(values: list):
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list, k: int, c_min: float, allow_below: float, mitigate_above: float) -> dict:
    subsets = {
        "all":           rows,
        "wildjailbreak": [r for r in rows if r.get("source") == "wildjailbreak"],
        "other_sources": [r for r in rows if r.get("source") != "wildjailbreak"],
    }
    out = {}
    for name, sub in subsets.items():
        if not sub:
            continue
        out[name] = {
            "greedy":           rates(sub, [r["greedy"] for r in sub]),
            "vote":             rates(sub, [majority(r["votes"], k, c_min) for r in sub]),
            "logprob_band":     rates(sub, [band(r["p_mitigate"], allow_below, mitigate_above) for r in sub]),
            "p_mitigate_auroc": auroc([r["label"] for r in sub], [r["p_mitigate"] for r in sub]),
            "logprob_threshold": {
                str(t): rates(sub, [DECISION_MITIGATE if r["p_mitigate"] > t else DECISION_ALLOW for r in sub])
                for t in THRESHOLDS
            },
        }
    valid_votes = [[v for v in r["votes"] if v in (DECISION_ALLOW, DECISION_MITIGATE)] for r in rows]
    out["vote_unanimous"] = round(sum(len(v) == k and len(set(v)) == 1 for v in valid_votes) / len(rows), 4) if rows else None
    out["invalid_outputs"] = {
        "greedy": sum(r["greedy"] == "invalid" for r in rows),
        "votes":  sum(v == "invalid" for r in rows for v in r["votes"]),
    }
    out["seconds_per_prompt"] = {key: _mean([r["seconds"][key] for r in rows]) for key in ("p_mitigate", "greedy", "votes")}
    return out


def print_summary(versions: dict, k: int) -> None:
    print()
    print(f"  {'version':<8} {'subset':<14} {'method':<14} {'acc':>7} {'benign_blocked':>15} {'harmful_allowed':>16} {'undecided':>10}")
    print("  " + "-" * 88)
    for version, s in versions.items():
        for subset in ("all", "wildjailbreak", "other_sources"):
            if subset not in s:
                continue
            block = s[subset]
            for method, label in (("greedy", "greedy"), ("vote", f"vote K={k}"), ("logprob_band", "logprob band")):
                m = block[method]
                print(f"  {version:<8} {subset:<14} {label:<14} {format_metric(m['accuracy_decided']):>7} "
                      f"{format_metric(m['benign_blocked']):>15} {format_metric(m['harmful_allowed']):>16} "
                      f"{format_metric(m['undecided']):>10}")
            print(f"  {version:<8} {subset:<14} {'p AUROC':<14} {format_metric(block['p_mitigate_auroc']):>7}")
        print(f"  {version:<8} vote unanimous {format_metric(s['vote_unanimous'])} | invalid outputs {s['invalid_outputs']} "
              f"| seconds/prompt {s['seconds_per_prompt']}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    cfg = load_config()
    if args.judge_model:
        cfg["layer2"]["judge"]["model_name"] = args.judge_model
        logger.info(f"judge model overridden: {args.judge_model}")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    run_info = collect_run_info(cfg)

    c = cfg["layer2"]["consistency"]
    k = args.k or c["K"]
    unknown = [v for v in args.prompts if v not in PROMPTS]
    if unknown:
        logger.error(f"Unknown prompt versions {unknown}; choose from {sorted(PROMPTS)}")
        return 1
    if cfg["layer2"]["judge"]["backend"] != "huggingface":
        logger.error("tune_judge.py needs the huggingface backend: p_mitigate reads the model's logits.")
        return 1

    input_path = Path(args.input) if args.input else Path(cfg["data"]["distillation_pool_dir"]) / "val.jsonl"
    records = sample_records(load_jsonl(str(input_path)), args.n, random.Random(seed))
    ids = {r["id"] for r in records}
    logger.info(f"Sampled {len(records)} prompts from {input_path} | prompt versions {args.prompts} | K={k}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        done = {(row["prompt_version"], row["id"]) for row in load_jsonl(str(out_path))}
        logger.info(f"Resuming: {len(done)} rows already in {out_path}")
    elif out_path.exists():
        warn_if_overwriting(out_path)
        out_path.unlink()

    judge = LLMJudge.from_config(cfg)
    temperature = c["sample_temperature"]
    with open(out_path, "a", encoding="utf-8", newline="\n") as fh:
        for version in args.prompts:
            judge.prompt_version = version
            for i, rec in enumerate(records, 1):
                if (version, rec["id"]) in done:
                    continue
                text = rec["text"]
                t0 = time.perf_counter()
                p = judge.p_mitigate(text)
                t1 = time.perf_counter()
                greedy = judge.judge(text, temperature=0.0)
                t2 = time.perf_counter()
                votes = [judge.judge(text, temperature=temperature).decision for _ in range(k)]
                t3 = time.perf_counter()
                row = {
                    "prompt_version":  version,
                    "id":              rec["id"],
                    "source":          rec.get("source"),
                    "label":           int(rec["label"]),
                    "p_mitigate":      round(p, 6),
                    "greedy":          greedy.decision,
                    "greedy_category": greedy.category,
                    "votes":           votes,
                    "seconds":         {"p_mitigate": round(t1 - t0, 3), "greedy": round(t2 - t1, 3),
                                        "votes": round(t3 - t2, 3)},
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0:
                    logger.info(f"  {version}: {i}/{len(records)}")

    rows = [r for r in load_jsonl(str(out_path)) if r["id"] in ids]
    versions = {
        v: summarise([r for r in rows if r["prompt_version"] == v], k, c["c_min"],
                     c.get("allow_below", 0.4), c.get("mitigate_above", 0.6))
        for v in args.prompts
    }
    summary = {
        "parameters": {"n": len(records), "k": k, "sample_temperature": temperature, "c_min": c["c_min"],
                       "allow_below": c.get("allow_below", 0.4), "mitigate_above": c.get("mitigate_above", 0.6),
                       "prompt_versions": args.prompts, "seed": seed, "input": str(input_path)},
        "versions": versions,
        "run_info": finalize_run_info(run_info),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    warn_if_overwriting(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(versions, k)
    logger.info(f"Rows → {out_path} | summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
