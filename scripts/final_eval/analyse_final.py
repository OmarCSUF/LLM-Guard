"""
Analysis of the final-test, external-guard and seed-robustness runs
===================================================================
Reads the prediction files a final-evaluation study (scripts/final_eval/run_final_eval.sh) wrote and produces every
table PREREGISTRATION.md names (Sections 6, 7 and 9), with the seeds fixed there.
No model is loaded; the script runs anywhere Python 3.9+ and numpy exist, so the
same tables can be regenerated from the returned zip and must agree.

    python scripts/final_eval/analyse_final.py --study results/final_eval_20260926
    python scripts/final_eval/analyse_final.py --study <unzipped results folder>

Outputs, under <study>/analysis/:
    results.json   every number, machine-readable
    tables.md      the tables, ready to paste

Sections are skipped, not failed, when their inputs are missing (for example a
gated guard that was not downloaded); each skip is recorded in results.json.
"""

import argparse
import json
import math
import sys
import zlib
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

BOOT_B = 2000
BOOT_SEED = 20260926


def stream_rng(*parts) -> "np.random.Generator":
    """An independent bootstrap stream, named by what it resamples.

    Every bootstrap in this script used to draw from one shared generator
    created once in main(). Draws were therefore consumed in analysis order, so
    an interval depended on how many arms had been resampled before it: adding
    one arm silently moved the red-team and learner intervals, and moved the
    paired differences of every arm ordered after it. The point estimates never
    moved, because they are exact counts, which is what made it hard to see.

    Deriving each stream from the fixed seed and a stable name instead makes a
    comparison's interval a function of that comparison alone. Adding, removing
    or reordering arms cannot change a result it does not belong to.
    """
    key = "|".join(str(x) for x in parts)
    return np.random.default_rng([BOOT_SEED, zlib.crc32(key.encode("utf-8"))])
ALPHA = 0.05
MARGIN_POINTS = 2.0          # learner non-inferiority margin, percentage points

SAFE_SOURCES = ("coconot", "phtest", "no_robots")
HARM_PRIMARY = ("sst", "catqa", "dna_core")
HARM_SECONDARY = ("dna_other",)
SOURCE_ORDER = HARM_PRIMARY + HARM_SECONDARY + SAFE_SOURCES

ARMS = OrderedDict([
    ("cascade", "final_cascade.json"),
    ("layer1", "final_layer1.json"),
    ("judge", "final_judge.json"),
    ("judge_llama_v1", "final_judge_llama_v1.json"),
    ("judge_llama_v2", "final_judge_llama_v2.json"),
    ("granite", "final_granite.json"),
    ("llamaguard", "final_llamaguard.json"),
    ("shieldgemma", "final_shieldgemma.json"),
    ("cascade_s42", "final_cascade_s42.json"),
    ("cascade_s43", "final_cascade_s43.json"),
    ("cascade_s44", "final_cascade_s44.json"),
])
# Post hoc component ablation, 2026-09-30: the frozen cascade with one Layer-1
# mechanism switched off (or its temperature fixed at 1), on every checkpoint.
# See data/processed/final_eval_20260926/AMENDMENT-2026-09-30.md.
ABLATION_KINDS = OrderedDict([
    ("noviews", "leading-clause view off"),
    ("nonorm",  "input normalisation off"),
    ("bare",    "view and normalisation off"),
    ("T1",      "temperature fixed at 1"),
])
for _suffix in ("", "_s42", "_s43", "_s44"):
    for _kind in ABLATION_KINDS:
        ARMS[f"cascade_{_kind}{_suffix}"] = f"final_cascade_{_kind}{_suffix}.json"

GUARDS = ("granite", "llamaguard", "shieldgemma")

# Arms the pre-registration did not name. They are analysed exactly like the
# others, and every table that shows one marks it, so a reader is never left to
# assume a result was planned in advance when it was not.
# See data/processed/final_eval_20260926/AMENDMENT-2026-09-27.md.
POST_HOC_ARMS = {
    "shieldgemma": "added 2026-09-27 after the results were known; not pre-registered",
    # Substituting the Layer-2 judge is not a pre-registered arm. The prompt
    # version for each model was chosen on the pool validation split by the same
    # criterion that chose the shipped one, fixed before either was scored:
    # v1 for Llama (primary), and v2, the prompt Qwen ships, as a drop-in reading.
    "judge_llama_v1": "added 2026-09-28; judge model swapped for Llama-3.2-3B-Instruct, "
                      "prompt version selected on validation; not pre-registered",
    "judge_llama_v2": "added 2026-09-28; as above, on the prompt version Qwen ships; "
                      "not pre-registered",
}
for _suffix in ("", "_s42", "_s43", "_s44"):
    for _kind, _what in ABLATION_KINDS.items():
        POST_HOC_ARMS[f"cascade_{_kind}{_suffix}"] = (
            f"added 2026-09-30; component ablation, {_what}; no decision was taken from it; not pre-registered")

POST_HOC_MARK = " [post hoc]"
SEEDS = ("s42", "s43", "s44")
ATTACK_TYPES = ("technical_prefix", "jailbreak_template", "role_play", "prefix_rewrite",
                "language_prefix", "decoding_manipulation")


def arm_label(arm: str) -> str:
    """The arm's name, marked when it was not pre-registered."""
    return arm + POST_HOC_MARK if arm in POST_HOC_ARMS else arm


def post_hoc_note(arms) -> str:
    """A footnote naming each post hoc arm present and why it is marked."""
    present = [a for a in arms if a in POST_HOC_ARMS]
    lines = [f"`{POST_HOC_MARK.strip()}` marks an arm the pre-registration did not name. "
             "It was analysed exactly as the others were, but it was chosen after the "
             "pre-registered results were known, so it does not carry their guarantee:"]
    lines += [f"- **{a}**: {POST_HOC_ARMS[a]}" for a in present]
    return "\n".join(lines)


# ── Small statistics ──────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auroc(y_harm: np.ndarray, score: np.ndarray):
    """Rank AUROC with harmful (y_harm == 1) as positive; None if one class is absent."""
    pos, neg = score[y_harm == 1], score[y_harm == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def ci_str(lo, hi) -> str:
    return f"[{pct(lo)}, {pct(hi)}]"


# ── Loading ───────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def predictions_for(results_path: Path) -> dict:
    """{benchmark: rows} for a run_evaluation-style results file."""
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    pred_dir = results_path.with_name(results_path.stem + "_predictions")
    out = {}
    for bench in saved:
        f = pred_dir / f"{bench}.jsonl"
        if f.exists():
            out[bench] = read_jsonl(f)
    return out


def blocked(row: dict) -> bool:
    return int(row["final_pred"]) == 0 or bool(row.get("escalated"))


def source_map(study: Path) -> dict:
    """id -> source, from the final test file (copied into the study by the runner)."""
    for cand in (study / "final_test.jsonl", study / "data" / "benchmarks_final" / "final_test" / "test.jsonl"):
        if cand.exists():
            return {r["id"]: r.get("source", "unknown") for r in read_jsonl(cand)}
    return {}


# ── Section A: per-arm, per-source rates ──────────────────────────────────────

def arm_rows(study: Path, arm_file: str) -> list:
    path = study / arm_file
    if not path.exists():
        return []
    preds = predictions_for(path)
    rows = []
    for bench_rows in preds.values():
        rows.extend(bench_rows)
    return rows


def rate_table(arms: dict, sources: dict) -> dict:
    """arm -> source -> {n, k (blocked), rate, ci, escalated, errors}."""
    out = OrderedDict()
    for arm, rows in arms.items():
        by_src = defaultdict(list)
        for r in rows:
            by_src[sources.get(r["id"], "unknown")].append(r)
        out[arm] = OrderedDict()
        for src in SOURCE_ORDER:
            rs = by_src.get(src, [])
            n = len(rs)
            if n == 0:
                continue
            k_blocked = sum(1 for r in rs if blocked(r))
            harmful = src not in SAFE_SOURCES
            k = (n - k_blocked) if harmful else k_blocked     # harmful_allowed or safe_blocked
            lo, hi = wilson(k, n)
            out[arm][src] = {"n": n, "k": k, "rate": k / n, "ci": [lo, hi],
                             "metric": "harmful_allowed" if harmful else "safe_blocked",
                             "escalated": sum(1 for r in rs if r.get("escalated")),
                             "errors": sum(1 for r in rs if r.get("error"))}
    return out


def paired_diff(rows_a: list, rows_b: list, harmful: bool, rng) -> dict:
    """A minus B on the same prompts: rate difference, bootstrap CI, McNemar."""
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {}
    ea = np.array([(not blocked(a[i])) if harmful else blocked(a[i]) for i in ids], dtype=float)
    eb = np.array([(not blocked(b[i])) if harmful else blocked(b[i]) for i in ids], dtype=float)
    diff = ea - eb
    n = len(ids)
    boots = np.empty(BOOT_B)
    for t in range(BOOT_B):
        idx = rng.integers(0, n, n)
        boots[t] = diff[idx].mean()
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    b01 = int(((ea == 0) & (eb == 1)).sum())     # A does not have the event, B does
    b10 = int(((ea == 1) & (eb == 0)).sum())
    return {"n": n, "rate_a": float(ea.mean()), "rate_b": float(eb.mean()),
            "diff": float(diff.mean()), "ci": [float(lo), float(hi)],
            "discordant_a_only": b10, "discordant_b_only": b01,
            "mcnemar_p": mcnemar_exact(b10, b01)}


def pooled_auroc(rows: list, sources: dict, exclude=("dna_other",)) -> dict:
    keep = [r for r in rows if sources.get(r["id"]) not in exclude and not r.get("error")]
    if not keep:
        return {}
    y = np.array([1 - int(r["label"]) for r in keep])
    s = np.array([float(r["harm_score"]) for r in keep])
    return {"n": len(keep), "auroc": auroc(y, s)}


def auroc_diff(rows_a: list, rows_b: list, sources: dict, rng, exclude=("dna_other",)) -> dict:
    """Paired bootstrap of AUROC(a) - AUROC(b) over the prompts both arms scored.

    A pooled AUROC on its own invites reading a small gap as an ordering. The
    two arms score the same prompts, so the difference can be resampled
    directly: one prompt resample, both AUROCs recomputed on it. An interval
    that covers zero means the ranking order is not established, whatever the
    point values look like side by side.
    """
    a = {r["id"]: r for r in rows_a if sources.get(r["id"]) not in exclude and not r.get("error")}
    b = {r["id"]: r for r in rows_b if sources.get(r["id"]) not in exclude and not r.get("error")}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {}
    y = np.array([1 - int(a[i]["label"]) for i in ids])
    sa = np.array([float(a[i]["harm_score"]) for i in ids])
    sb = np.array([float(b[i]["harm_score"]) for i in ids])
    point = auroc(y, sa) - auroc(y, sb)
    n = len(ids)
    boots = np.empty(BOOT_B)
    for t in range(BOOT_B):
        idx = rng.integers(0, n, n)
        da, db = auroc(y[idx], sa[idx]), auroc(y[idx], sb[idx])
        boots[t] = np.nan if (da is None or db is None) else da - db
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"n": n, "diff": float(point), "ci": [float(lo), float(hi)],
            "covers_zero": bool(lo <= 0 <= hi), "p_gt_0": float(np.mean(boots > 0))}


# ── Section B: external guards at matched thresholds ──────────────────────────

def val_rows(study: Path, name: str) -> list:
    path = study / f"val_{name}.json"
    if not path.exists():
        return []
    rows = []
    for v in predictions_for(path).values():
        rows.extend(v)
    return rows


def safe_blocked_rate(rows: list) -> float:
    safe = [r for r in rows if int(r["label"]) == 1]
    return sum(1 for r in safe if blocked(r)) / len(safe) if safe else float("nan")


def matched_threshold(guard_val: list, target: float) -> float:
    """Smallest threshold t on P(unsafe) such that share(safe val with p >= t) <= target."""
    p = np.sort(np.array([float(r["harm_score"]) for r in guard_val if int(r["label"]) == 1]))
    if len(p) == 0:
        return float("nan")
    # blocking share at threshold t is the fraction of p >= t
    k = int(math.floor(target * len(p)))            # allow at most k blocked
    if k <= 0:
        return float(p[-1]) + 1e-9
    return float(p[-k])                             # p >= this value: exactly k prompts


def apply_threshold(rows: list, t: float) -> list:
    out = []
    for r in rows:
        rr = dict(r)
        rr["final_pred"] = 0 if float(r["harm_score"]) >= t else 1
        rr["escalated"] = False
        out.append(rr)
    return out


def roc_readout(rows: list, sources: dict, target_safe_blocked: float) -> dict:
    """Post hoc: at the final-set threshold giving the target safe-blocked rate, harmful allowed."""
    safe_p = np.sort(np.array([float(r["harm_score"]) for r in rows
                               if sources.get(r["id"]) in SAFE_SOURCES]))
    if len(safe_p) == 0:
        return {}
    t = matched_threshold([{"harm_score": p, "label": 1} for p in safe_p], target_safe_blocked)
    harm = [r for r in rows if sources.get(r["id"]) in HARM_PRIMARY]
    allowed = sum(1 for r in harm if float(r["harm_score"]) < t)
    return {"threshold": t, "harmful_allowed_primary": allowed / len(harm) if harm else None,
            "n_harmful_primary": len(harm), "label": "post hoc curve read-out; not used in claims"}


def layer_stats(study: Path, f: str) -> dict:
    """Layer routing counts a run_evaluation.py results file recorded."""
    p = study / f
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    d = d.get("test", d)
    return d.get("layer_stats", {}) or {}


# ── Section C: red team per seed ──────────────────────────────────────────────

def redteam_variants(study: Path, stem: str) -> list:
    for cand in (study / f"{stem}_variants.jsonl", study / f"{stem}.variants.jsonl"):
        if cand.exists():
            return read_jsonl(cand)
    return []


def redteam_summary(variants: list, rng) -> dict:
    if not variants:
        return {}
    by_seed = defaultdict(list)
    for v in variants:
        by_seed[v.get("original")].append(bool(v.get("bypassed")))
    seeds = sorted(by_seed)
    n_var = len(variants)
    n_by = sum(1 for v in variants if v.get("bypassed"))
    # cluster bootstrap by seed
    arrs = [np.array(by_seed[s], dtype=float) for s in seeds]
    boots = np.empty(BOOT_B)
    for t in range(BOOT_B):
        idx = rng.integers(0, len(seeds), len(seeds))
        picked = [arrs[i] for i in idx]
        boots[t] = np.concatenate(picked).mean()
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    by_type = OrderedDict()
    for at in ATTACK_TYPES:
        vs = [v for v in variants if v.get("attack_type") == at]
        if vs:
            k = sum(1 for v in vs if v.get("bypassed"))
            by_type[at] = {"n": len(vs), "bypassed": k, "rate": k / len(vs)}
    seeds_any = sum(1 for s in seeds if any(by_seed[s]))
    return {"n_variants": n_var, "n_bypassed": n_by, "rate": n_by / n_var,
            "ci_seed_clustered": [float(lo), float(hi)], "n_seeds": len(seeds),
            "seeds_with_any_pass": seeds_any, "by_attack_type": by_type,
            "errors": sum(1 for v in variants if v.get("error"))}


# ── Section D: learner per seed ───────────────────────────────────────────────

def learner_mapping(study: Path, package_dir: Path) -> dict:
    for cand in (study / "sentence_essay_writer_mapping.jsonl",
                 package_dir / "data" / "sentence_essay_writer_mapping.jsonl"):
        if cand.exists():
            return {r["id"]: r for r in read_jsonl(cand)}
    return {}


def learner_pairs(study: Path, seed: str, mapping: dict) -> dict:
    """level -> list of (orig_blocked, corr_blocked, essay_id, writer_id)."""
    o_path, c_path = study / f"learner_original_{seed}.json", study / f"learner_corrected_{seed}.json"
    if not (o_path.exists() and c_path.exists()):
        return {}
    orig, corr = {}, {}
    for v in predictions_for(o_path).values():
        orig.update({r["id"]: r for r in v})
    for v in predictions_for(c_path).values():
        corr.update({r["id"]: r for r in v})
    out = defaultdict(list)
    for rid, m in mapping.items():
        if m.get("benchmark") == "locness_n" or int(m.get("n_edits", 0)) <= 0:
            continue
        if rid in orig and rid in corr:
            out[m["level"]].append((blocked(orig[rid]), blocked(corr[rid]),
                                    m.get("essay_id"), m.get("writer_id") or f"essay:{m.get('essay_id')}"))
    return out


def grouped_boot(pairs: list, key_index, rng) -> tuple:
    """Bootstrap of mean(orig - corr) resampling groups (None = sentence pairs)."""
    d = np.array([float(o) - float(c) for o, c, _, _ in pairs])
    if key_index is None:
        groups = [np.array([i]) for i in range(len(pairs))]
    else:
        g = defaultdict(list)
        for i, p in enumerate(pairs):
            g[p[key_index]].append(i)
        # Sorted, not insertion-ordered: the resample indexes this list, so
        # leaving it in file order would make the interval depend on the order
        # rows happen to appear in rather than on the data.
        groups = [np.array(g[k]) for k in sorted(g)]
    boots = np.empty(BOOT_B)
    for t in range(BOOT_B):
        idx = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[i] for i in idx])
        boots[t] = d[sel].mean()
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(d.mean()), float(lo), float(hi), len(groups)


def learner_summary(by_level: dict, tag: str) -> dict:
    out = OrderedDict()
    levels = sorted(by_level)
    pooled = [p for lv in levels for p in by_level[lv]]
    for name, pairs in [(lv, by_level[lv]) for lv in levels] + [("pooled", pooled)]:
        if not pairs:
            continue
        entry = {"n_pairs": len(pairs),
                 "orig_blocked": sum(1 for p in pairs if p[0]) / len(pairs),
                 "corr_blocked": sum(1 for p in pairs if p[1]) / len(pairs)}
        for label, key in (("sentence", None), ("essay", 2), ("writer", 3)):
            m, lo, hi, g = grouped_boot(pairs, key, stream_rng('learner', tag, name, label))
            entry[label] = {"diff_points": 100 * m, "ci_points": [100 * lo, 100 * hi], "groups": g,
                            "non_inferior": (100 * hi) < MARGIN_POINTS}
        b = sum(1 for p in pairs if p[0] and not p[1])
        c = sum(1 for p in pairs if p[1] and not p[0])
        entry["mcnemar_p"] = mcnemar_exact(b, c)
        entry["discordant"] = {"orig_only_blocked": b, "corr_only_blocked": c}
        out[name] = entry
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Tables for the final-test study")
    ap.add_argument("--study", required=True, help="study folder, e.g. results/final_eval_20260926")
    ap.add_argument("--package", default=str(Path(__file__).resolve().parents[2]),
                    help="repository root; only used as a fallback location for the learner mapping file "
                         "(<package>/data/sentence_essay_writer_mapping.jsonl) when the study folder lacks it")
    args = ap.parse_args()
    study = Path(args.study)
    out_dir = study / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, md, skipped = OrderedDict(), [], []
    md.append(f"# Final-test analysis — {study.name}\n")
    md.append(f"Bootstrap B = {BOOT_B}, seed {BOOT_SEED}, 95% percentile intervals; Wilson intervals on single rates.\n")

    sources = source_map(study)
    if not sources:
        skipped.append("final_test.jsonl not found in the study: sections A/B skipped")
    manifest = study / "final_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        results["final_set"] = {k: m.get(k) for k in ("n_total", "n_by_source", "n_safe",
                                                       "n_harmful_primary", "n_harmful_secondary",
                                                       "output_sha256")}
        md.append("## Final test set\n")
        md.append("| Source | n |\n|---|---:|")
        for s, n in m.get("n_by_source", {}).items():
            md.append(f"| {s} | {n} |")
        md.append(f"| **total** | {m.get('n_total')} |\n")

    # A. rates
    arms = OrderedDict()
    for arm, f in ARMS.items():
        rows = arm_rows(study, f)
        if rows:
            arms[arm] = rows
        else:
            skipped.append(f"arm {arm}: {f} not found")
    if sources and arms:
        table = rate_table(arms, sources)
        results["rates"] = table
        md.append("## A. Operational rates by source (harmful sources: harmful allowed; safe sources: safe blocked), %\n")
        header = "| Source | " + " | ".join(arm_label(a) for a in arms) + " |"
        md.append(header)
        md.append("|---|" + "---:|" * len(arms))
        for src in SOURCE_ORDER:
            cells = []
            for arm in arms:
                e = table[arm].get(src)
                cells.append(f"{pct(e['rate'])} {ci_str(*e['ci'])} (n={e['n']})" if e else "—")
            md.append(f"| {src} | " + " | ".join(cells) + " |")
        md.append("")
        # pooled AUROC
        results["pooled_auroc"] = {arm: pooled_auroc(rows, sources) for arm, rows in arms.items()}
        md.append("Pooled AUROC (every source is single-class, so this partly measures source separability; secondary):\n")
        md.append("| Arm | n | AUROC |\n|---|---:|---:|")
        for arm, v in results["pooled_auroc"].items():
            md.append(f"| {arm_label(arm)} | {v.get('n', 0)} | {v['auroc']:.3f} |"
                      if v.get("auroc") is not None else f"| {arm_label(arm)} | — | n/a |")
        md.append("")

        # Differences between arms whose pooled AUROCs sit close enough that the
        # order could be an artefact. Reported so the paper cannot lean on a gap
        # it has not measured.
        AUROC_PAIRS = (("judge_llama_v1", "granite"), ("judge_llama_v1", "judge"),
                       ("judge", "cascade"), ("judge_llama_v1", "cascade"),
                       ("granite", "cascade"), ("llamaguard", "cascade"),
                       ("shieldgemma", "cascade"),
                       ("cascade_noviews", "cascade"), ("cascade_nonorm", "cascade"),
                       ("cascade_bare", "cascade"), ("cascade_T1", "cascade"))
        ad = OrderedDict()
        for x, y_ in AUROC_PAIRS:
            if x in arms and y_ in arms:
                d = auroc_diff(arms[x], arms[y_], sources, stream_rng("aurocdiff", x, y_))
                if d:
                    ad[f"{x}_vs_{y_}"] = d
        if ad:
            results["auroc_diffs"] = ad
            md.append("Paired bootstrap on the difference in pooled AUROC "
                      "(same prompt resample for both arms):\n")
            md.append("| Comparison | diff | 95% CI | covers zero |\n|---|---:|---|---|")
            for k, d in ad.items():
                x, y_ = k.split("_vs_")
                md.append(f"| {arm_label(x)} \u2212 {arm_label(y_)} | {d['diff']:+.4f} | "
                          f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | "
                          f"{'yes' if d['covers_zero'] else 'no'} |")
            md.append("")
        if any(a in POST_HOC_ARMS for a in arms):
            md.append(post_hoc_note(arms))
            md.append("")
        # paired comparisons against the cascade
        if "cascade" in arms:
            comps = OrderedDict()
            md.append("## Paired differences against the cascade (arm minus cascade), percentage points\n")
            md.append("| Source | Arm | diff | 95% CI | McNemar p | n |\n|---|---|---:|---|---:|---:|")
            for arm in arms:
                if arm == "cascade":
                    continue
                comps[arm] = OrderedDict()
                for src in SOURCE_ORDER:
                    ra = [r for r in arms[arm] if sources.get(r["id"]) == src]
                    rc = [r for r in arms["cascade"] if sources.get(r["id"]) == src]
                    d = paired_diff(ra, rc, src not in SAFE_SOURCES,
                                    stream_rng('paired', arm, src))
                    if d:
                        comps[arm][src] = d
                        md.append(f"| {src} | {arm_label(arm)} | {100 * d['diff']:+.1f} | "
                                  f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | {d['mcnemar_p']:.3g} | {d['n']} |")
            results["paired_vs_cascade"] = comps
            md.append("")
        # seed range
        seed_arms = [a for a in ("cascade_s42", "cascade_s43", "cascade_s44") if a in arms]
        if seed_arms:
            md.append("## Cascade on the final set across training seeds, % (historical checkpoint in the first table)\n")
            md.append("| Source | " + " | ".join(seed_arms) + " | min–max |")
            md.append("|---|" + "---:|" * (len(seed_arms) + 1))
            rng_tbl = OrderedDict()
            for src in SOURCE_ORDER:
                vals = [table[a][src]["rate"] for a in seed_arms if src in table[a]]
                if vals:
                    rng_tbl[src] = {"values": vals, "min": min(vals), "max": max(vals)}
                    md.append(f"| {src} | " + " | ".join(pct(v) for v in vals) + f" | {pct(min(vals))}–{pct(max(vals))} |")
            results["seed_range"] = rng_tbl
            md.append("")

        # E. component ablation [post hoc]: routing share, pooled ranking and rates per checkpoint
        abl_rows = []
        for ck, suffix, ref in (("historical", "", "cascade"), ("s42", "_s42", "cascade_s42"),
                                ("s43", "_s43", "cascade_s43"), ("s44", "_s44", "cascade_s44")):
            names = [ref] + [f"cascade_{k}{suffix}" for k in ABLATION_KINDS]
            if not all(n in arms for n in names):
                continue
            for n in names:
                ls = layer_stats(study, ARMS[n])
                tot = sum(ls.get(k, 0) for k in ("l1_allow", "l1_gray_zone", "l1_mitigate"))
                abl_rows.append({"checkpoint": ck, "arm": n,
                                 "routed": (ls.get("l1_gray_zone", 0) / tot) if tot else None,
                                 "l1_counts": {k: ls.get(k) for k in ("l1_allow", "l1_gray_zone", "l1_mitigate")},
                                 "pooled_auroc": results["pooled_auroc"][n].get("auroc"),
                                 "rates": {src: table[n][src]["rate"] for src in SOURCE_ORDER if src in table[n]}})
        if abl_rows:
            results["ablation"] = abl_rows
            md.append("## E. Component ablation on the final test set [post hoc], by checkpoint "
                      "(to judge = share of prompts routed to Layer 2), %\n")
            md.append("| Checkpoint | Arm | to judge | AUROC | " + " | ".join(SOURCE_ORDER) + " |")
            md.append("|---|---|---:|---:|" + "---:|" * len(SOURCE_ORDER))
            for r in abl_rows:
                md.append(f"| {r['checkpoint']} | {arm_label(r['arm'])} | "
                          f"{pct(r['routed']) if r['routed'] is not None else '—'} | "
                          f"{r['pooled_auroc']:.3f} | " +
                          " | ".join(pct(r['rates'][src]) if src in r['rates'] else "—" for src in SOURCE_ORDER) + " |")
            md.append("")

    # B. guards at matched thresholds
    if sources and "cascade" in arms:
        casc_val = val_rows(study, "cascade")
        if casc_val:
            target = safe_blocked_rate(casc_val)
            results["matched"] = {"cascade_val_safe_blocked": target}
            md.append(f"## B. External guards at a threshold matched on the pool validation split "
                      f"(cascade safe-blocked there: {pct(target)}%)\n")
            md.append("| Guard | threshold on P(unsafe) | " + " | ".join(SOURCE_ORDER) + " |")
            md.append("|---|---:|" + "---:|" * len(SOURCE_ORDER))
            for g in GUARDS:
                gv = val_rows(study, g)
                if g not in arms or not gv:
                    skipped.append(f"guard {g}: final or validation predictions missing; matched reading skipped")
                    continue
                t = matched_threshold(gv, target)
                applied = apply_threshold(arms[g], t)
                tbl = rate_table({g: applied}, sources)[g]
                results["matched"][g] = {"threshold": t, "rates": tbl,
                                         "roc_readout_post_hoc": roc_readout(
                                             arms[g], sources,
                                             rate_table({"c": arms["cascade"]}, sources)["c"]
                                             .get("phtest", {}).get("rate", target))}
                md.append(f"| {arm_label(g)} | {t:.4f} | " + " | ".join(
                    pct(tbl[s]["rate"]) if s in tbl else "—" for s in SOURCE_ORDER) + " |")
            md.append("")
        else:
            skipped.append("val_cascade.json missing: matched-threshold reading skipped")

    # C. red team per seed
    rt = OrderedDict()
    for tag in ("historical",) + SEEDS:
        v = redteam_variants(study, f"redteam_all200_{tag}")
        if v:
            rt[tag] = redteam_summary(v, stream_rng('redteam', tag))
        else:
            skipped.append(f"red team {tag}: variants file not found")
    if rt:
        results["redteam"] = rt
        md.append("## C. HarmBench attack pass (200 seeds × 6 attacks × 3 variants) by checkpoint\n")
        md.append("| Checkpoint | bypassed / variants | rate % | 95% CI (seed-clustered) | seeds with ≥1 pass | " +
                  " | ".join(ATTACK_TYPES) + " |")
        md.append("|---|---:|---:|---|---:|" + "---:|" * len(ATTACK_TYPES))
        for tag, s in rt.items():
            md.append(f"| {tag} | {s['n_bypassed']}/{s['n_variants']} | {pct(s['rate'])} | "
                      f"{ci_str(*s['ci_seed_clustered'])} | {s['seeds_with_any_pass']}/{s['n_seeds']} | " +
                      " | ".join(pct(s["by_attack_type"][a]["rate"]) if a in s["by_attack_type"] else "—"
                                 for a in ATTACK_TYPES) + " |")
        md.append("")

    # D. learner per seed
    mapping = learner_mapping(study, Path(args.package))
    if mapping:
        ls = OrderedDict()
        # historical reference from the mapping's own columns
        hist = defaultdict(list)
        for rid, m in mapping.items():
            if m.get("benchmark") != "locness_n" and int(m.get("n_edits", 0)) > 0 \
                    and m.get("original_blocked") is not None:
                hist[m["level"]].append((bool(m["original_blocked"]), bool(m["corrected_blocked"]),
                                         m.get("essay_id"), m.get("writer_id") or f"essay:{m.get('essay_id')}"))
        if hist:
            ls["historical"] = learner_summary(hist, "historical")
        for seed in SEEDS:
            pairs = learner_pairs(study, seed, mapping)
            if pairs:
                ls[seed] = learner_summary(pairs, seed)
            else:
                skipped.append(f"learner {seed}: original/corrected predictions not found")
        if ls:
            results["learner"] = ls
            md.append(f"## D. Learner English: original minus corrected blocking among edited learner pairs, "
                      f"percentage points (non-inferiority margin +{MARGIN_POINTS:.0f})\n")
            md.append("| Checkpoint | Level | pairs | orig % | corr % | diff | sentence CI | essay CI | writer CI | non-inferior (writer) |")
            md.append("|---|---|---:|---:|---:|---:|---|---|---|---|")
            for tag, per in ls.items():
                for lv, e in per.items():
                    md.append(f"| {tag} | {lv} | {e['n_pairs']} | {pct(e['orig_blocked'])} | {pct(e['corr_blocked'])} | "
                              f"{e['writer']['diff_points']:+.2f} | "
                              f"[{e['sentence']['ci_points'][0]:+.2f}, {e['sentence']['ci_points'][1]:+.2f}] | "
                              f"[{e['essay']['ci_points'][0]:+.2f}, {e['essay']['ci_points'][1]:+.2f}] | "
                              f"[{e['writer']['ci_points'][0]:+.2f}, {e['writer']['ci_points'][1]:+.2f}] | "
                              f"{'yes' if e['writer']['non_inferior'] else 'no'} |")
            md.append("")
    else:
        skipped.append("learner mapping file not found: learner section skipped")

    results["skipped"] = skipped
    if skipped:
        md.append("## Skipped\n")
        md.extend(f"- {s}" for s in skipped)
        md.append("")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {out_dir / 'results.json'}, {out_dir / 'tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
