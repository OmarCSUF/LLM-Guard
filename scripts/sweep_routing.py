"""
GaurdLLM — Routing Band Sweep
===============================
Simulates the cascade at many routing bands from signals that are already on
disk, so the trade-off between over-refusal and missed harm can be measured
without another GPU run.

The signals written by scripts/run_ablation.py hold, for every prompt, the
Layer-1 score (both the three-signal version and the classifier-only one) and
the judge's P(mitigate).  In logprob mode the judge's decision is a function of
P(mitigate) alone, so every (tau_low, tau_high) band can be replayed exactly.

    allow      score < tau_low
    mitigate   score > tau_high
    otherwise  the judge decides: mitigate above mitigate_above, allow below
               allow_below, uncertain in between (escalated, counted as blocked)

Selection and reporting are kept apart, because tuning a threshold on the
benchmarks would be tuning on the test set:

    --select   signals used to choose a band (the pool validation split)
    --report   signals the chosen band is then measured on (the benchmarks)

With only --select the script sweeps and prints the frontier.  With both it
also applies the chosen band to the report signals, beside the band in config.

Harm scores follow the pipeline: the Layer-1 score when Layer 1 decides, and
0.5 * score + 0.5 * severity when the judge does.  A prompt that was judged
during collection carries its real severity; for the rest severity falls back to
P(mitigate), which is what layer2/consistency.py does when the greedy call
disagrees with the logprob decision.

The ms/prompt column charges every routed prompt what it actually cost during
collection, which makes it honest per row and misleading across rows.  Only the
prompts that reached the judge during collection carry a full generative call;
the rest carry the logprob pass alone, which is several times cheaper.  A narrow
band that happens to contain the generatively judged prompts is therefore priced
against a different cost model from a wide band that does not, and on the pool
validation split the implied per-routed-prompt cost ran from about 700 ms to
about 3,500 ms for that reason alone.  Compare latency between bands only after
re-pricing them on one model.

Usage:
    # collect validation signals once (about 15 minutes on the 4050)
    python scripts/run_ablation.py --stage signals --input data/distillation_pool/val.jsonl \
        --name pool_val --signals data/processed/ablation_val_signals.jsonl

    # sweep, choose, and check the choice on the benchmarks
    python scripts/sweep_routing.py \
        --select data/processed/ablation_val_signals.jsonl \
        --report data/processed/ablation_signals.jsonl \
        --out data/processed/routing_sweep
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from llmguard.utils.config_loader import load_config
from llmguard.utils.logger import get_logger
from llmguard.utils.run_info import collect_run_info, finalize_run_info, warn_if_overwriting
from llmguard.evaluation.metrics import compute_all, format_metric

logger = get_logger("sweep_routing")

# Which stored score each route is built on
SCORES = {
    "l1": ("l1_score", "three-signal Layer 1"),
    "classifier": ("c_score", "Layer 1 refitted on g(x) alone"),
}

SCORE_DIGITS = 4          # the precision the pipeline passes between layers


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sweep the Layer-1 routing band over collected signals",
        epilog="Choose a band on validation signals; report it on the benchmarks.",
    )
    p.add_argument("--select", required=True,
                   help="Signals the band is chosen on (use the pool validation split)")
    p.add_argument("--report", default=None,
                   help="Signals the chosen band is then measured on (the benchmarks)")
    p.add_argument("--score", nargs="+", default=["l1", "classifier"], choices=list(SCORES),
                   help="Which Layer-1 score to route on (default: both)")
    p.add_argument("--step", type=float, default=0.05, help="Grid step for both thresholds")
    p.add_argument("--budget", type=float, default=0.10,
                   help="Largest acceptable share of safe prompts blocked, on the selection set")
    p.add_argument("--max-judge-share", type=float, default=0.20,
                   help="Largest acceptable share of prompts sent to the judge")
    p.add_argument("--objective", choices=["harmful_allowed", "f1", "youden"],
                   default="harmful_allowed",
                   help="What to optimise inside the budget (default: fewest harmful prompts allowed)")
    p.add_argument("--judge-band", nargs=2, type=float, default=None, metavar=("ALLOW_BELOW", "MITIGATE_ABOVE"),
                   help="Override the judge's P(mitigate) band from config")
    p.add_argument("--out", default="data/processed/routing_sweep",
                   help="Output prefix for .csv, .json and .md")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Signals → arrays
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def quantise(value) -> Optional[float]:
    return None if value is None else round(float(value), SCORE_DIGITS)


class Prepared:
    """
    Everything a band replay needs, as parallel arrays.

    y            label, 0 = harmful
    score        the Layer-1 score the band is applied to
    judged_pred  what the judge would decide for this prompt (1 allow, 0 block)
    judged_esc   whether the judge would escalate it
    judged_harm  the blended harm score if the judge sees it
    t_base_ms    Layer-1 cost for this route
    t_judge_ms   what a judge call costs for this prompt
    """

    def __init__(self, rows: List[dict], score_field: str, allow_below: float, mitigate_above: float):
        y, score, jp, je, jh, tb, tj, errors = [], [], [], [], [], [], [], 0
        uses_features = score_field == "l1_score"
        for row in rows:
            if row.get("error"):
                errors += 1
                continue
            if row.get(score_field) is None or row.get("p_mitigate") is None:
                raise SystemExit(
                    f"{row.get('benchmark')}/{row.get('id')}: missing {score_field} or p_mitigate. "
                    "Collect signals with every variant enabled."
                )
            s = quantise(row[score_field])
            p = quantise(row["p_mitigate"])
            judge = row.get("judge") or {}
            severity = judge.get("severity") if judge.get("severity") is not None else p

            y.append(int(row["label"]))
            score.append(s)
            if p > mitigate_above:
                jp.append(0)
                je.append(False)
            elif p < allow_below:
                jp.append(1)
                je.append(False)
            else:
                jp.append(0)
                je.append(True)
            jh.append(0.5 * s + 0.5 * float(severity))
            base = row.get("t_classifier_ms") or 0.0
            if uses_features:
                base += row.get("t_features_ms") or 0.0
            tb.append(base)
            tj.append(row.get("t_judge_ms") or row.get("t_p_ms") or 0.0)

        self.y = np.array(y, dtype=int)
        self.score = np.array(score, dtype=float)
        self.judged_pred = np.array(jp, dtype=int)
        self.judged_esc = np.array(je, dtype=bool)
        self.judged_harm = np.array(jh, dtype=float)
        self.t_base_ms = np.array(tb, dtype=float)
        self.t_judge_ms = np.array(tj, dtype=float)
        self.errors = errors
        if not len(self.y):
            raise SystemExit("No usable rows in the signals file.")


def prepare_by_group(rows: List[dict], score_field: str, allow_below: float,
                     mitigate_above: float) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row.get("benchmark", "all"), []).append(row)
    return {name: Prepared(group, score_field, allow_below, mitigate_above)
            for name, group in groups.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Replaying one band
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_band(d: Prepared, tau_low: float, tau_high: float) -> dict:
    """Metrics, counts and estimated latency for one (tau_low, tau_high)."""
    gray = (d.score >= tau_low) & (d.score <= tau_high)
    pred = np.where(d.score < tau_low, 1, 0)
    pred = np.where(gray, d.judged_pred, pred)
    esc = gray & d.judged_esc
    harm = np.where(gray, d.judged_harm, d.score)

    metrics = compute_all(d.y, harm, pred, escalated=esc)
    safe = d.y == 1
    blocked = (pred == 0) | esc
    latency = d.t_base_ms + np.where(gray, d.t_judge_ms, 0.0)
    return {
        "tau_low": round(tau_low, 4),
        "tau_high": round(tau_high, 4),
        "judge_share": float(gray.mean()),
        "n_judged": int(gray.sum()),
        "safe_blocked": float(blocked[safe].mean()) if safe.any() else None,
        "harmful_allowed": metrics["jailbreak_success_rate"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "escalated": int(esc.sum()),
        "mean_latency_ms": float(latency.mean()),
        "median_latency_ms": float(np.median(latency)),
    }


def sweep(d: Prepared, step: float) -> List[dict]:
    """Every band on the grid, tau_low <= tau_high."""
    grid = [round(x, 4) for x in np.arange(0.0, 1.0 + step / 2, step)]
    rows = []
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            rows.append(evaluate_band(d, lo, hi))
    logger.info(f"Swept {len(rows)} bands")
    return rows


def youden(row: dict) -> Optional[float]:
    """Harmful caught minus safe blocked: one number for the operating point."""
    if row["harmful_allowed"] is None or row["safe_blocked"] is None:
        return None
    return (1.0 - row["harmful_allowed"]) - row["safe_blocked"]


def pareto(rows: List[dict]) -> List[dict]:
    """Bands that no other band beats on both safe_blocked and harmful_allowed."""
    usable = [r for r in rows if r["safe_blocked"] is not None and r["harmful_allowed"] is not None]
    front = []
    for r in usable:
        dominated = any(
            o is not r and o["safe_blocked"] <= r["safe_blocked"] and o["harmful_allowed"] <= r["harmful_allowed"]
            and (o["safe_blocked"] < r["safe_blocked"] or o["harmful_allowed"] < r["harmful_allowed"])
            for o in usable
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r["safe_blocked"], r["harmful_allowed"]))


def choose(rows: List[dict], budget: float, max_share: float, objective: str) -> Optional[dict]:
    """
    The band this run recommends: best objective among bands that stay inside
    both budgets. Ties break toward fewer safe prompts blocked, then less judge
    traffic, then a narrower band.
    """
    feasible = [r for r in rows
                if r["safe_blocked"] is not None and r["safe_blocked"] <= budget
                and r["judge_share"] <= max_share]
    if not feasible:
        return None

    def key(r):
        if objective == "harmful_allowed":
            primary = r["harmful_allowed"] if r["harmful_allowed"] is not None else 1.0
        elif objective == "f1":
            primary = -(r["f1"] if r["f1"] is not None else 0.0)
        else:
            value = youden(r)
            primary = -(value if value is not None else -1.0)
        safe_blocked = r["safe_blocked"] if r["safe_blocked"] is not None else 1.0
        return (primary, safe_blocked, r["judge_share"], -(r["tau_high"] - r["tau_low"]))

    return min(feasible, key=key)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def beats(candidate: dict, reference: dict) -> bool:
    """True when a band is at least as good as another on both operating rates."""
    for key in ("safe_blocked", "harmful_allowed"):
        if candidate[key] is None or reference[key] is None:
            return False
    return (candidate["safe_blocked"] <= reference["safe_blocked"]
            and candidate["harmful_allowed"] <= reference["harmful_allowed"]
            and (candidate["safe_blocked"] < reference["safe_blocked"]
                 or candidate["harmful_allowed"] < reference["harmful_allowed"]))


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def band_table(title: str, rows: List[dict]) -> List[str]:
    out = [f"### {title}", "",
           "| tau_low | tau_high | To judge | Safe blocked | Harmful allowed | AUROC | F1 | ms/prompt |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['tau_low']:.2f} | {r['tau_high']:.2f} | {pct(r['judge_share'])} | "
                   f"{pct(r['safe_blocked'])} | {pct(r['harmful_allowed'])} | "
                   f"{format_metric(r['auroc'])} | {format_metric(r['f1'])} | "
                   f"{r['mean_latency_ms']:.0f} |")
    out.append("")
    return out


def report_table(title: str, per_benchmark: dict) -> List[str]:
    out = [f"### {title}", "",
           "| Benchmark | To judge | Safe blocked | Harmful allowed | AUROC | F1 | ms/prompt |",
           "|---|---|---|---|---|---|---|"]
    for name in sorted(per_benchmark):
        r = per_benchmark[name]
        out.append(f"| {name} | {pct(r['judge_share'])} | {pct(r['safe_blocked'])} | "
                   f"{pct(r['harmful_allowed'])} | {format_metric(r['auroc'])} | "
                   f"{format_metric(r['f1'])} | {r['mean_latency_ms']:.0f} |")
    out.append("")
    return out


def write_outputs(out_prefix: Path, payload: dict, markdown: List[str], csv_rows: List[dict]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".md", ".csv"):
        warn_if_overwriting(out_prefix.with_suffix(suffix))

    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    text = "\n".join(markdown)
    out_prefix.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")

    if csv_rows:
        fields = ["route"] + [k for k in csv_rows[0] if k != "route"]
        with open(out_prefix.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    run_info = collect_run_info(cfg)

    consistency = cfg["layer2"]["consistency"]
    allow_below, mitigate_above = (args.judge_band if args.judge_band else
                                   (consistency.get("allow_below", 0.4),
                                    consistency.get("mitigate_above", 0.6)))
    current = (cfg["layer1"]["thresholds"]["low"], cfg["layer1"]["thresholds"]["high"])
    logger.info(f"Judge band: allow below {allow_below}, mitigate above {mitigate_above}")
    logger.info(f"Band in config: {current[0]} / {current[1]}")

    select_rows = read_jsonl(Path(args.select))
    report_rows = read_jsonl(Path(args.report)) if args.report else []
    logger.info(f"Selection signals: {len(select_rows)} prompts from {args.select}")

    markdown = ["# Routing band sweep", "",
                f"Judge band: allow below {allow_below}, mitigate above {mitigate_above}. "
                f"Config band: {current[0]} / {current[1]}.",
                f"Selection set: `{args.select}` ({len(select_rows)} prompts). "
                "Bands are chosen here and only then measured on the benchmarks.", ""]
    payload = {"run_info": finalize_run_info(run_info), "routes": {},
               "judge_band": [allow_below, mitigate_above], "config_band": list(current),
               "objective": args.objective, "budget": args.budget,
               "max_judge_share": args.max_judge_share}
    csv_rows = []

    for route in args.score:
        field, description = SCORES[route]
        logger.info(f"Route {route}: {description}")
        prepared = Prepared(select_rows, field, allow_below, mitigate_above)
        grid = sweep(prepared, args.step)
        for row in grid:
            csv_rows.append({"route": route, **row})

        front = pareto(grid)
        picked = choose(grid, args.budget, args.max_judge_share, args.objective)
        at_current = evaluate_band(prepared, *current)

        markdown += [f"## Route: {route} ({description})", ""]
        markdown += band_table("Frontier on the selection set", front)
        markdown += band_table("Config band, on the selection set", [at_current])
        if picked:
            markdown += band_table(
                f"Recommended band (best {args.objective} with safe blocked <= {pct(args.budget)} "
                f"and judge share <= {pct(args.max_judge_share)})", [picked])
            # The recommended band optimises one objective, so it failing to beat the
            # configured band says nothing about the rest of the feasible set. Count them,
            # or a reader draws a universal conclusion from a single comparison.
            dominating = [r for r in grid
                          if r["safe_blocked"] is not None
                          and r["safe_blocked"] <= args.budget
                          and r["judge_share"] <= args.max_judge_share
                          and beats(r, at_current)]
            if dominating:
                closest = min(dominating, key=lambda r: (r["safe_blocked"], r["harmful_allowed"]))
                lo, hi = closest["tau_low"], closest["tau_high"]
                markdown += band_table(
                    f"Bands inside the budgets that beat the configured one on both rates "
                    f"({len(dominating)} of {len(grid)} swept)", dominating[:10])
                markdown += [f"The closest is {lo:.2f} / {hi:.2f}. Check the margin against the "
                             "sampling error before moving: on a balanced split of this size a "
                             "tenth of a point is about one prompt per class, and the binomial "
                             "standard error is several times that.", ""]
            elif not beats(picked, at_current):
                markdown += [f"None of the {len(grid)} swept bands inside those budgets beats the "
                             "configured one on both rates. Raise --budget or --max-judge-share, "
                             "or read the frontier and pick the trade-off you want.", ""]
        else:
            markdown += [f"No band meets safe blocked <= {pct(args.budget)} with judge share "
                         f"<= {pct(args.max_judge_share)} on the selection set.", ""]

        route_payload = {"description": description, "grid": grid, "pareto": front,
                         "config_band": at_current, "recommended": picked,
                         "beats_config_band": bool(picked and beats(picked, at_current))}

        if report_rows:
            by_bench = prepare_by_group(report_rows, field, allow_below, mitigate_above)
            baseline = {name: evaluate_band(d, *current) for name, d in by_bench.items()}
            markdown += report_table(f"Benchmarks at the config band {current[0]} / {current[1]}", baseline)
            route_payload["report_config_band"] = baseline
            # A band that loses to the configured one on the selection set is not
            # worth measuring on the benchmarks; the frontier already shows the cost.
            if picked and route_payload["beats_config_band"]:
                chosen = {name: evaluate_band(d, picked["tau_low"], picked["tau_high"])
                          for name, d in by_bench.items()}
                markdown += report_table(
                    f"Benchmarks at the recommended band {picked['tau_low']:.2f} / {picked['tau_high']:.2f}",
                    chosen)
                route_payload["report_recommended_band"] = chosen

        payload["routes"][route] = route_payload

    markdown += ["Notes:",
                 "- Bands are replayed from stored signals: in logprob mode the judge's decision is a",
                 "  function of P(mitigate), so no model is called here.",
                 "- Safe blocked counts escalated prompts as blocked, as in the evaluation tables.",
                 "- Latency is the measured Layer-1 cost plus a measured judge call for routed prompts.",
                 "- A prompt judged during collection keeps its real severity; the rest fall back to",
                 "  P(mitigate), which is what the judge does when its greedy call disagrees.", ""]

    write_outputs(Path(args.out), payload, markdown, csv_rows)
    logger.info(f"Tables → {Path(args.out).with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
