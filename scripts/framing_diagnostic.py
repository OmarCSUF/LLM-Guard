# -*- coding: utf-8 -*-
"""
Framing diagnostic: jailbreak, role-play and prefix_rewrite frames against Layer 1.

Run from the repository root with the venv active:

    python scripts/framing_diagnostic.py [--layer1-artifacts SUBDIR] [--no-judge]
                                         [--out PATH] [--variants PATH] [--predictions PATH]

First run on 22 Sep 2026 against the shipped Layer 1 with the leading-clause view
on (data/processed/lim2_diag.json); its findings are in layer1/augment.py and
LLM-Guard_Limitations_Work_Log.md. --layer1-artifacts repeats it against a retrained
Layer 1 (layer1.artifact_subdir), so two models can be compared frame by frame.

Part A  attributes the residual bypasses of the all-200 single pass with the
        leading-clause view on (redteam_all200_views_variants.jsonl) to frame and
        seed, and separates the seeds Layer 1 allows unwrapped from the rest.
Part B  scores every frame x seed combination (15 jailbreak, 7 role-play and 8
        prefix_rewrite frames x 200 HarmBench seeds = 6,000 variants) with the
        shipped Layer 1 (views on) and with candidate extra views:
          comma     : the text behind a comma inside the first 12 words
          sentence  : the text behind a sentence end inside the first 24 words
          opener    : the text behind a request opener ("could you walk me through")
          dropk     : the text minus its first k words, k = 1..6 (generic comparator)
          boundary  : comma + sentence together
          bound+op  : comma + sentence + opener
        Layer 1 keeps the maximum over forms, so a candidate can only raise scores.
Part C  measures the benign cost of every candidate on the pool's validation and
        test splits and on the five benchmark test sets: decisions moved for safe
        prompts (cost) and for harmful prompts (recovered).
Part D  (unless --no-judge) runs the Layer-2 Judge on the residual bypass variants
        and on the unwrapped weak seeds, to see whether routing would catch them.

Writes --out (default data/processed/framing_diag.json) and <out stem>_variants.jsonl.
Nothing in the repository is modified.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

from llmguard.utils.config_loader import load_config          # noqa: E402
from llmguard.utils.data_utils import load_jsonl, set_seed    # noqa: E402
from llmguard.layer1.detector import Layer1Detector, layer1_artifact_dir  # noqa: E402
from llmguard.redteam import attacks as A                     # noqa: E402

T0 = time.time()


def elapsed():
    return f"[{time.time() - T0:6.0f}s]"


# -- Candidate views -----------------------------------------------------------

_COMMA_BOUNDARY = re.compile(r",\s+")
_SENT_BOUNDARY = re.compile(r"[.!?]\s+")

_POLITE = (r"(?:(?:could|can|would|will)\s+you\s+(?:please\s+)?"
           r"|please\s+"
           r"|i(?:'d|\s+would)\s+(?:like|love|want)\s+(?:you\s+)?to\s+"
           r"|i\s+(?:want|need|wish)\s+(?:you\s+)?to\s+)?")
_VERB = (r"(?:walk\s+me\s+through"
         r"|help\s+me\s+(?:to\s+)?(?:learn|understand)"
         r"|(?:explain|describe|outline|detail|clarify)(?:\s+(?:in\s+detail|step\s+by\s+step|to\s+me|for\s+me))*"
         r"|tell\s+me|show\s+me|teach\s+me"
         r"|(?:give|provide)\s+(?:me\s+)?(?:with\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,2}"
         r"(?:explanation|breakdown|description|overview|guide|walkthrough|summary|instructions|steps)\s+(?:of|on|for|about)"
         r"|what\s+is\s+the\s+(?:process|procedure|method|best\s+way|steps?)\s+(?:for|to|of)"
         r"|understand)")
_OPENER = re.compile(r"^\s*" + _POLITE + _VERB + r"\s+", re.IGNORECASE)


def _boundary_views(text, pattern, max_lead, min_tail, max_views):
    if not text or not text.strip():
        return []
    text = text.strip()
    out = []
    for m in pattern.finditer(text):
        if len(text[:m.start()].split()) > max_lead:
            break
        tail = text[m.end():].strip()
        if len(tail.split()) >= min_tail and tail != text and tail not in out:
            out.append(tail)
        if len(out) >= max_views:
            break
    return out


def comma_views(text):
    return _boundary_views(text, _COMMA_BOUNDARY, 12, 3, 2)


def sentence_views(text):
    return _boundary_views(text, _SENT_BOUNDARY, 24, 3, 3)


def opener_views(text):
    """Strip a leading request opener, up to twice; each remainder is a view."""
    out = []
    cur = (text or "").strip()
    for _ in range(2):
        m = _OPENER.match(cur)
        if not m:
            break
        tail = cur[m.end():].strip()
        if len(tail.split()) < 3 or tail == cur or tail in out:
            break
        out.append(tail)
        cur = tail
    return out


def dropk_views(text):
    w = (text or "").split()
    return [" ".join(w[k:]) for k in range(1, 7) if len(w) - k >= 3]


CANDIDATES = {
    "comma":    lambda t: comma_views(t),
    "sentence": lambda t: sentence_views(t),
    "opener":   lambda t: opener_views(t),
    "dropk":    lambda t: dropk_views(t),
    "boundary": lambda t: comma_views(t) + sentence_views(t),
    "bound+op": lambda t: comma_views(t) + sentence_views(t) + opener_views(t),
}

# -- Attack reconstruction -----------------------------------------------------

IMPERATIVE_OPENERS = ("Tell me ", "Explain ", "Describe ", "How do I ", "How to ")


def strip_imperative(prompt):
    for opener in IMPERATIVE_OPENERS:
        if prompt.lower().startswith(opener.lower()):
            return prompt[len(opener):], opener.strip()
    return prompt, None


FRAMES = {
    "jailbreak_template": [(i, t) for i, t in enumerate(A._JAILBREAK_TEMPLATES)],
    "role_play":          [(i, t) for i, t in enumerate(A._ROLEPLAY_TEMPLATES)],
    "prefix_rewrite":     [(i, p) for i, p in enumerate(A._PARAPHRASE_PREFIXES)],
    "language_prefix":    [(i, p) for i, (_, p) in enumerate(A._LANG_PREFIXES)],
    "technical_prefix":   [(i, p) for i, p in enumerate(A._TECHNICAL_PREFIXES)],
}


def build_variant(attack_type, frame, seed):
    if attack_type in ("jailbreak_template", "role_play"):
        return frame.format(prompt=seed)
    if attack_type == "prefix_rewrite":
        return frame + strip_imperative(seed)[0]
    return frame + seed


def all_variants(attack_type, seed):
    if attack_type == "decoding_manipulation":
        return list(enumerate(A.decoding_manipulation(seed, 3)))
    return [(i, build_variant(attack_type, f, seed)) for i, f in FRAMES[attack_type]]


def frame_label(attack_type, idx):
    if attack_type == "decoding_manipulation":
        return ["base64", "leetspeak", "zero-width"][idx]
    f = FRAMES[attack_type][idx][1]
    return f.replace("{prompt}", "").strip()[:48]


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/processed/redteam_all200_views_variants.jsonl")
    ap.add_argument("--predictions", default="data/processed/eval_views_2026-09-22_predictions/harmbench.jsonl")
    ap.add_argument("--seeds", default="data/benchmarks/harmbench/test.jsonl")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--layer1-artifacts", default=None, metavar="SUBDIR",
                    help="Load Layer 1 from artifacts/SUBDIR instead of layer1.artifact_subdir")
    ap.add_argument("--out", default="data/processed/framing_diag.json")
    args = ap.parse_args()

    cfg = load_config("config/config.yaml")
    if args.layer1_artifacts:
        cfg.setdefault("layer1", {})["artifact_subdir"] = args.layer1_artifacts
    set_seed(cfg["project"]["seed"])
    variants_out = Path(args.out).with_name(Path(args.out).stem + "_variants.jsonl")
    report = {"date": time.strftime("%Y-%m-%d"),
              "layer1_artifacts": str(layer1_artifact_dir(cfg)),
              "config_payload_views": cfg["layer1"].get("payload_views")}
    print(f"Layer-1 artifacts: {layer1_artifact_dir(cfg)}")

    seeds = [r["text"].strip() for r in load_jsonl(args.seeds) if r.get("text", "").strip()]
    by_cap = {s[:200]: s for s in seeds}
    print(f"{elapsed()} seeds: {len(seeds)} ({len(by_cap)} distinct under the 200-char cap)")

    # -- Part A: attribute the residual bypasses ------------------------------
    print("\n=== Part A: residual bypasses of the all-200 pass with views on ===")
    rows = load_jsonl(args.variants)
    print(f"variant rows: {len(rows)}; bypassed: {sum(1 for r in rows if r['bypassed'])}")
    recon = []          # (row, seed, frame_idx, full_variant)
    unmatched = Counter()
    for r in rows:
        seed = by_cap.get(r["original"])
        if seed is None:
            unmatched[r["attack_type"]] += 1
            continue
        hit = None
        for idx, v in all_variants(r["attack_type"], seed):
            if v[:200] == r["variant"]:
                hit = (idx, v)
                break
        if hit is None:
            unmatched[r["attack_type"]] += 1
            continue
        recon.append((r, seed, hit[0], hit[1]))
    print(f"reconstructed: {len(recon)} of {len(rows)}; unmatched by type: {dict(unmatched)}")

    pred_weak = None
    if Path(args.predictions).exists():
        preds = load_jsonl(args.predictions)
        pred_weak = {p["text"].strip() for p in preds if p.get("l1_decision") == "allow"}
        print(f"predictions file: {len(preds)} rows; Layer-1 allow unwrapped: {len(pred_weak)}")

    print(f"\n{elapsed()} loading Layer 1 ...")
    det = Layer1Detector.from_artifacts_for(cfg)
    print(f"Layer 1 ready: payload_views={det.payload_views} normalise_input={det.normalise_input} "
          f"thresholds={det.router.tau_low}/{det.router.tau_high}")
    route = det.router.route
    memo = {}

    def sc(text):
        if text not in memo:
            memo[text] = float(det._score(text)[4])
        return memo[text]

    seed_res = {s: det.detect(s) for s in seeds}
    weak = {s for s, r in seed_res.items() if r.decision == "allow"}
    gray_seeds = {s for s, r in seed_res.items() if r.decision == "gray_zone"}
    print(f"seeds allowed unwrapped by Layer 1 (views on): {len(weak)}; gray zone: {len(gray_seeds)}")
    if pred_weak is not None:
        print(f"  agrees with predictions file: {weak == pred_weak}")
    for s in sorted(weak):
        print(f"  weak seed  cal={seed_res[s].calibrated_score:.3f}  {s[:90]}")
    report["weak_seeds"] = sorted(weak)
    report["gray_seeds"] = sorted(gray_seeds)

    partA = {}
    for at in ("jailbreak_template", "role_play", "prefix_rewrite", "language_prefix",
               "technical_prefix", "decoding_manipulation"):
        sub = [x for x in recon if x[0]["attack_type"] == at]
        byp = [x for x in sub if x[0]["bypassed"]]
        on_weak = [x for x in byp if x[1] in weak]
        on_other = [x for x in byp if x[1] not in weak]
        by_frame = Counter(frame_label(at, x[2]) for x in on_other)
        by_seed = Counter(x[1][:70] for x in on_other)
        l1_alloc = Counter()
        for x in on_other:
            l1_alloc[det.detect(x[3]).decision] += 1
        partA[at] = {"rows": len(sub), "bypassed": len(byp), "on_weak_seeds": len(on_weak),
                     "on_other_seeds": len(on_other), "by_frame_other": dict(by_frame),
                     "by_seed_other": dict(by_seed), "l1_decision_now_other": dict(l1_alloc)}
        print(f"\n{at:<22} bypassed {len(byp):>3}/{len(sub)}: {len(on_weak)} on the weak seeds, "
              f"{len(on_other)} on other seeds (Layer-1 now: {dict(l1_alloc)})")
        for f, n in by_frame.most_common():
            print(f"    {n:>3}  frame: {f}")
        for s, n in by_seed.most_common(8):
            print(f"    {n:>3}  seed : {s}")
    report["partA"] = partA

    # -- Part B: full factorial frame x seed, current Layer 1 and candidates --
    print(f"\n{elapsed()} === Part B: 30 frames x {len(seeds)} seeds under each candidate view ===")
    var_rows = []
    partB = {}
    for at in ("jailbreak_template", "role_play", "prefix_rewrite"):
        per_cand = {c: Counter() for c in ["current"] + list(CANDIDATES)}
        per_frame = defaultdict(lambda: {c: Counter() for c in ["current"] + list(CANDIDATES)})
        for s in seeds:
            for idx, v in all_variants(at, s):
                base = det.detect(v)
                b = base.calibrated_score
                decs = {"current": base.decision}
                for c, fn in CANDIDATES.items():
                    views = [x for x in fn(v) if x]
                    best = max([b] + [sc(x) for x in views]) if views else b
                    decs[c] = route(best)
                for c, d in decs.items():
                    per_cand[c][d] += 1
                    per_frame[idx][c][d] += 1
                var_rows.append({"attack_type": at, "frame": idx, "seed": s[:80], "weak_seed": s in weak,
                                 "base_cal": round(b, 4), "base_form": base.normalised, **decs})
        n = sum(per_cand["current"].values())
        print(f"\n{at}: {n} variants; weak-seed floor = {len(weak) * len(FRAMES[at])} "
              f"({100 * len(weak) / len(seeds):.1f}%)")
        print(f"  {'candidate':<10} {'allow':>6} {'gray':>6} {'mitig':>6}   allow%")
        for c, cnt in per_cand.items():
            print(f"  {c:<10} {cnt['allow']:>6} {cnt['gray_zone']:>6} {cnt['mitigate']:>6}   "
                  f"{100 * cnt['allow'] / n:.1f}%")
        print("  per frame, allow count (current -> comma / sentence / opener / dropk / boundary / bound+op):")
        for idx, table in sorted(per_frame.items()):
            cur = table["current"]["allow"]
            rest = " / ".join(f"{table[c]['allow']:>3}" for c in CANDIDATES)
            print(f"    {cur:>3} -> {rest}   {frame_label(at, idx)}")
        partB[at] = {"n": n, "weak_floor": len(weak) * len(FRAMES[at]),
                     "by_candidate": {c: dict(cnt) for c, cnt in per_cand.items()},
                     "by_frame": {frame_label(at, idx): {c: dict(cnt) for c, cnt in t.items()}
                                  for idx, t in per_frame.items()}}

    # prefix_rewrite mechanism: the opener vs the missing imperative
    print(f"\n{elapsed()} prefix_rewrite mechanism ({len(seeds)} seeds x 8 openers, strong seeds only)")
    mech = {"seeds_with_imperative": 0, "stripped_alone": Counter(), "seed_alone": Counter()}
    per_opener_drop = defaultdict(list)
    for s in seeds:
        if s in weak:
            continue
        stripped, op = strip_imperative(s)
        s_seed = seed_res[s].calibrated_score
        if op:
            mech["seeds_with_imperative"] += 1
            mech["stripped_alone"][route(sc(stripped))] += 1
            mech["seed_alone"][seed_res[s].decision] += 1
        for idx, p in FRAMES["prefix_rewrite"]:
            per_opener_drop[idx].append(s_seed - sc(p + stripped))
    print(f"  strong seeds with an imperative opener the attack strips: {mech['seeds_with_imperative']}"
          f" -> unwrapped decision {dict(mech['seed_alone'])}; stripped text alone {dict(mech['stripped_alone'])}")
    for idx, drops in sorted(per_opener_drop.items()):
        mean = sum(drops) / len(drops)
        big = sum(1 for d in drops if d > 0.4)
        print(f"  opener {frame_label('prefix_rewrite', idx):<44} mean score drop {mean:+.3f}; "
              f"drops over 0.4: {big}/{len(drops)}")
    mech["stripped_alone"] = dict(mech["stripped_alone"])
    mech["seed_alone"] = dict(mech["seed_alone"])
    mech["mean_drop_by_opener"] = {frame_label("prefix_rewrite", i): round(sum(d) / len(d), 4)
                                   for i, d in per_opener_drop.items()}
    partB["prefix_mechanism"] = mech
    report["partB"] = partB

    with open(variants_out, "w", encoding="utf-8") as fh:
        for r in var_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -- Part C: benign cost --------------------------------------------------
    print(f"\n{elapsed()} === Part C: decision changes on the pool splits and the benchmarks ===")
    sets = []
    pool = Path(cfg["data"]["distillation_pool_dir"])
    for split in ("val", "test"):
        p = pool / f"{split}.jsonl"
        if p.exists():
            sets.append((f"pool_{split}", p))
    # Enabled benchmarks only, as everywhere else: the RQ4 learner arms are
    # listed disabled and belong to their own study, not to this diagnostic.
    for b in cfg["evaluation"]["benchmarks"]:
        if not b.get("enabled", True):
            continue
        p = Path(b["path"]) / "test.jsonl"
        if p.exists():
            sets.append((b["name"], p))
    partC = {}
    header = f"  {'set':<16}{'safe':>6}{'harm':>6} | " + " | ".join(f"{c:^17}" for c in CANDIDATES)
    print(header)
    print("  " + " " * 28 + " | ".join(f"{'safe up':>8}{'harm up':>9}" for _ in CANDIDATES))
    for name, path in sets:
        recs = [r for r in load_jsonl(str(path)) if r.get("text", "").strip()]
        n_safe = sum(1 for r in recs if r["label"] == 1)
        n_harm = len(recs) - n_safe
        moved = {c: {"safe_up": 0, "harm_up": 0, "safe_to_mitigate": 0} for c in CANDIDATES}
        base_dec = Counter()
        for r in recs:
            t = r["text"].strip()
            base = det.detect(t)
            b, d0 = base.calibrated_score, base.decision
            base_dec[(r["label"], d0)] += 1
            rank = {"allow": 0, "gray_zone": 1, "mitigate": 2}
            for c, fn in CANDIDATES.items():
                views = [x for x in fn(t) if x]
                if not views:
                    continue
                d1 = route(max([b] + [sc(x) for x in views]))
                if rank[d1] > rank[d0]:
                    key = "safe_up" if r["label"] == 1 else "harm_up"
                    moved[c][key] += 1
                    if r["label"] == 1 and d1 == "mitigate":
                        moved[c]["safe_to_mitigate"] += 1
        partC[name] = {"n_safe": n_safe, "n_harm": n_harm,
                       "base_decisions": {f"{k[0]}/{k[1]}": v for k, v in base_dec.items()},
                       "moved": moved}
        cells = " | ".join(f"{moved[c]['safe_up']:>8}{moved[c]['harm_up']:>9}" for c in CANDIDATES)
        print(f"  {name:<16}{n_safe:>6}{n_harm:>6} | {cells}")
    print("  (safe up = a safe prompt's decision moved allow->gray, allow->mitigate or gray->mitigate;"
          " harm up = the same for a harmful prompt, i.e. recovered)")
    report["partC"] = partC

    # -- Part D: the Judge on the residual bypasses ---------------------------
    if not args.no_judge:
        print(f"\n{elapsed()} === Part D: Layer-2 Judge on the residual bypass variants ===")
        try:
            from llmguard.layer2 import Layer2Judge
            judge = Layer2Judge.from_config(cfg)
        except Exception as exc:
            print(f"  judge failed to load: {exc}")
            judge = None
        if judge is not None:
            partD = {}
            items = [(x[0]["attack_type"], x[1] in weak, x[3]) for x in recon if x[0]["bypassed"]]
            items += [("weak_seed_unwrapped", True, s) for s in sorted(weak)]
            tally = defaultdict(Counter)
            details = []
            for at, is_weak, text in items:
                r = judge.judge(text, record=False)
                verdict = "escalated" if r.escalated else r.decision
                tally[(at, "weak" if is_weak else "other")][verdict] += 1
                details.append({"attack_type": at, "weak_seed": is_weak, "verdict": verdict,
                                "p_mitigate": getattr(r, "p_mitigate", None),
                                "severity": r.severity, "text": text[:160]})
            for (at, grp), cnt in sorted(tally.items()):
                print(f"  {at:<22} {grp:<6} " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
                partD[f"{at}/{grp}"] = dict(cnt)
            report["partD"] = {"tally": partD, "details": details}

    report["elapsed_s"] = round(time.time() - T0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n{elapsed()} wrote {args.out} and {variants_out}")


if __name__ == "__main__":
    main()
