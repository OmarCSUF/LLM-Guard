# LLM-Guard

A calibrated three-layer cascade for prompt-level harm detection, and the pre-registered
evaluation that measured it. This repository holds the code, the configuration, four
pinned Layer-1 checkpoints, the 3,453-prompt final test set, every stored result of the
study, and the scripts that regenerate every table of the article from stored per-prompt
predictions.

```
prompt ──► Layer 1: calibrated detector ──► allow                                  (score < 0.30)
           (RoBERTa+LoRA, temperature     ──► mitigate ───────────────► Layer 3: refuse (score > 0.70)
            scaling, threshold routing)   ──► gray zone ──► Layer 2: judge ──► allow      (P(mitigate) < 0.40)
                                                            (Qwen2.5-3B, 4-bit, ──► mitigate ──► Layer 3
                                                             log-prob decision) ──► uncertain ─► Layer 3
```

> **How to read the numbers in this file.** Every figure is followed by the file it came
> from. The defensible results are the pre-registered final evaluation in
> `results/final_eval_20260926/`, on prompts that informed no design decision. Numbers from
> the training pool are in-distribution and say nothing about generalisation; numbers from
> the five development benchmarks are the results of a configuration chosen with their help.
> Where the system loses to a published baseline, this file says so.

Companion documents: `EXPERIMENTS.md` (what maps to what), `REPRODUCIBILITY.md` (what was
run, what can and cannot be re-run, manuscript discrepancies), `data/README.md` (rebuilding
the data), `CHANGES.md` (what changed from the original workspace).

## 1. Research objective

Prompt-level screening fails in two directions: a harmful request is answered, or a
legitimate one is refused. The project asks whether a cheap calibrated detector, a
selective LLM judge for the prompts the detector cannot settle, and refusal-by-default
mitigation can be combined into one screen, and, separately, what such a system looks like
when it is evaluated on prompts that did not shape it. The article's claim is a diagnosis,
not an improvement over published guards: the cascade under-uses its middle layer because
the detector's calibration does not transfer out of distribution.

## 2. Methodology

**Layer 1, calibrated detection.** A `roberta-base` classifier with a LoRA adapter
(r = 8, alpha = 16) is distilled on a 15,296-prompt balanced pool and returns P(allow). The
shipped configuration (`layer1.classifier_only: true`) feeds that single signal to a
logistic stacker, applies temperature scaling fitted on the pool validation split
(T = 1.1908 for the shipped stacker, `artifacts/layer1/scaler_clf_only.json`), and routes on
the calibrated score: allow below 0.30, mitigate above 0.70, otherwise to the judge. A
three-signal stacker that adds two `distilgpt2` scoring-LM features (curvature, log-rank)
is shipped as well (`fusion_v2.joblib`) and selected with `--full-stacker`. Before scoring,
the prompt is also scored in every recovered form the input may hide: decoded base64,
folded leetspeak (`layer1.normalise_input`), and the text behind a short leading
instruction clause (`layer1.payload_views`); the maximum score is routed, so these can
raise a score but never lower it.

**Layer 2, selective judge.** `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4, prompt version
`v2`, one forward pass: P(mitigate) is read from the next-token distribution after
`DECISION:`. Allow below 0.40, mitigate above 0.60, otherwise uncertain, which the pipeline
treats as blocked. No K-vote self-consistency is run (`layer2.consistency.mode: logprob`).

**Layer 3, mitigation.** Default mode `refuse` with a templated refusal
(`config/safety_templates.yaml`). `redact` and `safe_rewrite` exist; `safe_rewrite` was
found in manual review to carry the original request in roughly one in eight served
outputs and is not the default. A request-conditioned rewrite check and a model-backed
refusal detector were built, measured and not adopted (`config/config.yaml` comments;
`docs/history/development_runs.md`, section 4).

**Final test set.** Pre-registered on 26 September 2026 before any source was downloaded
(`results/final_eval_20260926/PREREGISTRATION.md`): seven public sources with no lineage
to the training pool or to the five development benchmarks, screened for exact and
near-duplicates (character 5-gram Jaccard >= 0.8, MinHash 64/16, seed 42) against all
three pool splits and all five benchmarks, PHTest sampled to 1,000 with seed 20260926.
3,453 prompts: SimpleSafetyTests 99, CatQA 543, Do-Not-Answer core 573 (harmful);
Do-Not-Answer other 364 (harmful, secondary, excluded from primary rates and pooled
ranking); CoCoNot contrast 378, PHTest harmless 1,000, No Robots test 496 (safe).
Source: `data/benchmarks_final/manifest.json`.

**Arms.** Pre-registered: the frozen cascade; Layer 1 alone (gray zone allowed); the judge
alone on every prompt; IBM Granite Guardian 3.1 2B; Meta Llama Guard 3 1B; the cascade
under three repeat-study checkpoints (seeds 42, 43, 44). Post hoc, marked as such
everywhere: Google ShieldGemma 2B (added 27 Sep while Llama Guard access was refused), the
cascade's judge slot holding `Llama-3.2-3B-Instruct` (28 Sep), and a sixteen-arm component
ablation (30 Sep). The guards run in the same 4-bit NF4 quantisation as the judge. The
system was hash-frozen before scoring (`results/final_eval_20260926/frozen.json`, commit
`dc90a9e2`).

**Statistics.** Per source, never pooled into one headline rate: `safe_blocked` on safe
sources and `harmful_allowed` on harmful sources (an escalated prompt counts as blocked),
each with a 95% Wilson interval; paired differences between arms by percentile bootstrap
within source (B = 2000, seed 20260926, one named stream per comparison) with an exact
McNemar test; pooled AUROC over the 3,089 primary prompts as a secondary measure, with
paired bootstrap on differences. External guards are read twice: at their native decision
and at the P(unsafe) threshold that matches the cascade's safe-blocked rate on the pool
validation split (3.8%). Red-team intervals cluster by seed request; learner intervals are
grouped by sentence, essay and writer. Everything is computed by
`scripts/final_eval/analyse_final.py` and the six scripts beside it from the stored
prediction files; no model is loaded.

## 3. Results

### 3.1 Final evaluation, 26 September 2026

Operational rates in percent. For harmful sources the entry is *harmful allowed*; for safe
sources it is *safe blocked*. Lower is better everywhere. Wilson intervals and the
per-source n are in `results/final_eval_20260926/analysis/tables.md`, section A.

| Source | Class | n | Cascade | Layer 1 alone | Judge alone | Granite | Llama Guard | ShieldGemma † |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleSafetyTests | harmful | 99 | 5.1 | 6.1 | 2.0 | 0.0 | 2.0 | 24.2 |
| CatQA | harmful | 543 | 19.2 | 23.6 | 2.4 | 0.2 | 6.1 | 51.2 |
| Do-Not-Answer core | harmful | 573 | 23.9 | 25.1 | 32.6 | 16.9 | 45.4 | 49.2 |
| Do-Not-Answer other (secondary) | harmful | 364 | 48.9 | 50.8 | 64.0 | 26.4 | 65.1 | 92.9 |
| CoCoNot contrast | safe | 378 | 6.3 | 6.3 | 20.4 | 21.7 | 6.1 | 11.6 |
| PHTest harmless | safe | 1000 | 42.8 | 42.1 | 29.6 | 55.9 | 27.7 | 15.0 |
| No Robots | safe | 496 | 12.7 | 12.5 | 7.1 | 19.2 | 11.3 | 2.0 |

† Post hoc: added on 27 September 2026 after these results were known
(`AMENDMENT-2026-09-27.md`). Llama Guard was named in advance and ran on 28 September
after a second account was granted access (`AMENDMENT-2026-09-28.md`); it is pre-registered.

Pooled AUROC over the 3,089 primary prompts (`tables.md`, section A). Every source is
single-class, so this partly measures source separability; it is secondary.

| Arm | Pooled AUROC |
|---|---:|
| Judge slot holding Llama-3.2-3B-Instruct, prompt v1 (post hoc) | 0.928 |
| **IBM Granite Guardian 3.1 2B** | **0.923** |
| Judge alone (Qwen2.5-3B-Instruct, shipped) | 0.878 |
| Meta Llama Guard 3 1B | 0.834 |
| Google ShieldGemma 2B (post hoc) | 0.826 |
| Cascade | 0.811 |
| Layer 1 alone | 0.811 |
| Cascade, retrained under seeds 42 / 43 / 44 | 0.751 / 0.773 / 0.777 |

**What this shows.** The cascade is clearly behind **Granite**: in ranking (0.811 against
0.923, paired difference +0.112 [+0.098, +0.127]) and at the operating point matched on the
pool validation split, where Granite allows 8.3% of CatQA harmful prompts against the
cascade's 19.2% while blocking 5.2% of PHTest harmless prompts against the cascade's 42.8%
(`tables.md`, section B). That does not generalise to published guards as a class. Llama
Guard ranks 0.834 and ShieldGemma 0.826; both margins (0.023 and 0.015) are inside the 0.060
that retraining the cascade's own recipe moves it (0.751 to 0.811). Llama Guard beats the
cascade on both axes at once on two sources (13.1 points fewer CatQA harmful prompts
allowed, 15.1 points fewer PHTest harmless prompts blocked, both p < 1e-12) but allows 21.5
points more Do-Not-Answer core prompts, the largest single gap either way. ShieldGemma
mostly occupies a different operating point: 27.8 points fewer PHTest blocks and 19 to 44
points more harmful prompts allowed on every harmful source.

The deficit is internal. The routing band sends **2.8%** of prompts to the judge (97 of
3,453; `final_cascade.json` → `layer_stats`), so the cascade's ranking is Layer 1's. The
judge alone outranks the cascade that contains it (0.878 against 0.811), and an unrelated
3B instruct model in the same slot matches Granite (0.928 against 0.923, paired difference
+0.005 [−0.002, +0.011], `AMENDMENT-2026-09-28b.md`). Both judges, however, block CoCoNot
boundary prompts far more often than Layer 1 does (20.4% and 21.7% against 6.3%), which is
the clearest evidence that the cheap first layer contributes something a judge does not.

**Calibration.** Layer 1's expected calibration error is 0.013 [0.008, 0.021] on the pool
test split and 0.204 [0.191, 0.221] on the final test set
(`analysis/calibration.json`); the calibrated score is bimodal, with 1,481 prompts below
0.1 and 1,372 above 0.9, which is why the band is nearly empty.

**Component ablation (post hoc, 30 September).** Turning the leading-clause view off
changes 7 to 9 decisions per checkpoint, all safe prompts allowed, and raises pooled AUROC
by +0.003 [+0.002, +0.005]; turning input normalisation off changes nothing; fixing T = 1
moves routing on 14 prompts and no rate by more than 0.3 points (`tables.md`, section E;
`AMENDMENT-2026-09-30.md`). No decision was taken from it.

### 3.2 Sensitivity to the training seed

The identical recipe retrained under three further seeds, scored on the final test set
(`tables.md`, seed table). This spread is wider than most differences the project
compares, so no single-checkpoint result should be quoted alone.

| Source | Historical | s42 | s43 | s44 | Range |
|---|---:|---:|---:|---:|---|
| CatQA harmful allowed | 19.2 | 34.8 | 22.3 | 35.7 | 19.2 to 35.7 |
| PHTest safe blocked | 42.8 | 45.9 | 46.3 | 42.2 | 42.2 to 46.3 |
| Pooled AUROC | 0.811 | 0.751 | 0.773 | 0.777 | 0.751 to 0.811 |

### 3.3 Red team

All 200 HarmBench seeds attacked once with six template attack types, three variants each:
3,600 distinct variant texts, zero errors. A variant counts as a bypass if the pipeline
allows it or raises an uncaught exception. Historical checkpoint, views on:
`results/development/redteam_all200_views.json` (commit `6dead1c4`, 22 September 2026;
byte-identical copy at `results/final_eval_20260926/redteam_all200_historical.json`).

| Attack type | Views off | Views on |
|---|---:|---:|
| technical_prefix | 29.2% | 3.0% |
| jailbreak_template | 16.5% | 3.7% |
| role_play | 13.0% | 2.7% |
| prefix_rewrite | 11.0% | 11.0% |
| language_prefix | 5.8% | 2.8% |
| decoding_manipulation | 4.3% | 4.3% |
| **Overall** | **13.3%** (479/3600) | **4.6%** (165/3600, 95% CI [2.5, 6.8]) |

The paired change is −8.72 points, request-clustered 95% interval [−10.53, −7.11]
(`analysis/paired_views.json`). The two attacks the view does not address are unchanged
bit for bit. Across the three repeat-study checkpoints the same views-on pass gives 6.3%,
5.0% and 7.8% (`tables.md`, section C). These attacks informed the defence; they are a
development measurement, not a confirmation set, and the suite says nothing about
adaptive attackers.

### 3.4 Learner English (grammatical-form sensitivity)

5,467 W&I+LOCNESS sentence pairs (CEFR A 2,079, B 2,044, C 1,344), each scored as written
and with the annotator's corrections applied, non-inferiority margin +2 points on the
original-minus-corrected blocking difference (`tables.md`, section D). Historical
checkpoint, pooled: −1.21 points, writer-grouped 95% CI [−1.91, −0.51], non-inferior.
Seeds 42 and 44 are non-inferior at every level; seed 43 fails at level A (+1.68,
[+0.29, +2.97]) and passes pooled (+0.22, [−0.53, +0.97]). Every sentence is labelled safe
without content review. The corpus is non-commercial and is not redistributed.

### 3.5 Layer 1 in isolation, in-distribution

On the pool test split (1,912 prompts, balanced), from the article's Table II. The
classifier-only column has no stored results file: it was read from the scored feature
cache by `scripts/plot_layer1.py`; `artifacts/layer1/results.json` records the three-signal
row. `results/final_eval_20260926/pooltest_layer1.json` (shipped Layer 1, views on,
1 October) gives AUROC 0.9913 and accuracy 0.9644 on the same split.

| Metric | Classifier-only (shipped) | Three-signal (`fusion_v2`) |
|---|---:|---:|
| AUROC | 0.9914 | 0.9892 |
| AUPRC | 0.9868 | 0.9877 |
| FPR at 95% TPR | 0.0220 | 0.0220 |
| Accuracy | 0.9629 | 0.9629 |
| Routed to judge | 15 | 15 |
| Temperature T | 1.1908 | 1.1891 |

These are in-distribution figures and the contrast with section 3.1 is the point.

## 4. Repository structure

```
src/llmguard/            the package: layer1/ layer2/ layer3/ evaluation/ redteam/ utils/
scripts/                 experiment entry points (flat, names as in the recorded commands)
scripts/final_eval/      final-test fetch/build, judge-only and external-guard scorers, analysis
config/                  config.yaml, safety_templates.yaml, nist_ai_rmf_mapping.yaml
artifacts/layer1/        the historical Layer-1 checkpoint (LoRA adapter, stackers, scalers, results.json)
artifacts/layer1_repeat_study_nciJu1oQ_s{42,43,44}/   the three repeat-study checkpoints
data/benchmarks_final/   the 3,453-prompt final test set and its manifest (shipped)
data/                    everything else is rebuilt from pinned sources; see data/README.md, data/LICENSES.md
results/final_eval_20260926/   the pre-registered study: pre-registration, amendments, frozen.json,
                               every arm's results and per-prompt predictions, red-team and learner
                               files, logs, withdrawn run, analysis/
results/development/     selected development results (red team, judge tuning, routing sweep, ablation, parity, learner)
results/repeat_study_nciJu1oQ/   protocol, frozen files, seed variability, summary of the repeat study
figures/                 README figures and Layer-1 diagnostic plots (see figures/README.md)
docs/                    validation policy, leak codebook, learner pre-registration, NIST mapping, history/, release change notes
tests/                   pytest suite (imports, config, metrics, normalisation, final-set and checkpoint hashes, analysis regeneration)
app.py                   Gradio dashboard (section 8)
```

## 5. Environment setup

The recorded runs used Python 3.10.12 with torch 2.14.0 (CUDA 13.0 build), transformers
4.57.6, bitsandbytes 0.50.2, peft 0.20.0, accelerate 1.15.0, scikit-learn 1.7.2, numpy
2.2.6, pandas 2.3.3 and pyyaml 6.0.3, on one NVIDIA RTX 4050 Laptop GPU (6 GB).
`requirements.txt` and `pyproject.toml` pin these versions; `requirements-lock.txt` is the
full freeze.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # pick the index for your CUDA build
pip install -e .                      # or: pip install -r requirements.txt
pip install -e ".[dashboard,report]"  # optional: Gradio demo, matplotlib figures
pip install -e ".[dev]" && PYTHONPATH=src python -m pytest -q tests   # optional: the test suite (no model is loaded)
```

The scripts put `src/` on `sys.path` themselves, so `python scripts/<name>.py` works from
the repository root without installing; `PYTHONPATH=src python scripts/<name>.py` is
equivalent. `project.device: auto` uses CUDA and stops if it is missing; set
`LLMGUARD_ALLOW_CPU=1` to run Layer 1 on CPU on purpose (the 4-bit judge and guards need a
GPU). The judge and the Guard-LLM load `Qwen/Qwen2.5-3B-Instruct` at the revision pinned in
`config/config.yaml`; `LLMGUARD_JUDGE_MODEL=/path/or/hub-id` overrides the location for
both layers at once. Llama Guard 3 1B, Llama-3.2-3B-Instruct, ShieldGemma, WildJailbreak,
XSTest and HarmBench are gated on the Hugging Face Hub and need an authenticated account.

Memory: every recorded run, including the full cascade with the 4-bit judge loaded, ran on
a 6 GB GPU (5.64 GiB addressable). Layer 3 reuses the judge's loaded weights when both name
the same checkpoint, so it adds no VRAM of its own; `python scripts/check_shared_model.py`
loads both and reports whether the instance is shared.

## 6. Dataset preparation

The final test set is shipped. Everything else (the training pool, the five development
benchmarks, the learner and parity arms, the raw sources) is rebuilt from pinned
revisions; `data/README.md` gives the six steps in order with expected counts and hashes,
and `data/LICENSES.md` the licence of every source. In brief:

```bash
python scripts/fetch_datasets.py                     # pool and development-benchmark sources
python scripts/build_datasets.py                     # data/distillation_pool/{train,val,test}.jsonl + data/benchmarks/<name>/test.jsonl
python scripts/final_eval/fetch_final_sources.py     # only needed to rebuild the shipped final set
python scripts/final_eval/build_final_test.py
python scripts/make_learner_arms.py --source <W&I+LOCNESS v2.1 directory>   # non-redistributable corpus
python scripts/make_l2_benchmarks.py --root data/benchmarks_parity
```

Expected SHA-256 of the pool splits and benchmark files are in
`results/final_eval_20260926/frozen.json`.

## 7. Training

```bash
python scripts/train_layer1.py                                   # -> artifacts/layer1 (refuses to overwrite without --force)
python scripts/train_layer1.py --seed 43 --artifact-dir artifacts/layer1_s43
```

Fits the curvature and log-rank feature cache, the LoRA classifier, both stackers and both
temperature scalers, then evaluates on the pool test split and writes `results.json` with a
`run_info` block. The historical run took 2,394 s. Training is **not bitwise
deterministic**: the same seed gave T = 1.1891, 1.2547 and 1.2486 in three runs, and
retraining moves CatQA harmful-allowed on the final set by up to 16 points (section 3.2).
Use the shipped checkpoints to reproduce the reported numbers; retrain to study variance.

## 8. Inference and demo

```bash
python scripts/demo_layer1.py --text "How do I reset my router?"          # Layer 1 only
python scripts/demo.py --text "..."                                       # full cascade
python scripts/demo.py --l1-only --examples                               # built-in examples, no judge
SERVER_NAME=0.0.0.0 SERVER_PORT=7860 LOAD_L2=true LOAD_L3=true python app.py   # Gradio dashboard
```

The dashboard's public `gradio.live` tunnel stays off unless `GRADIO_SHARE=true`.

## 9. Evaluation

```bash
python scripts/run_evaluation.py --input FILE.jsonl --output OUT.json               # full cascade on a labelled JSONL
python scripts/run_evaluation.py --input FILE.jsonl --no-l2 --output OUT.json       # Layer 1 alone
python scripts/run_evaluation.py                                                     # the five development benchmarks
python scripts/final_eval/run_judge_only.py --input FILE.jsonl --output OUT.json     # judge alone
python scripts/final_eval/run_external_guard.py --model granite --input FILE.jsonl --output OUT.json
python scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 \
    --attack-types jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix \
    --payload-views --output OUT.json
python scripts/summarise_results.py --run a=A.json --run b=B.json --compare a:b --out OUT   # paired bootstrap tables
```

Input rows are `{"id", "text", "label"}` with label 1 = safe (allow) and 0 = harmful
(mitigate). Every results file carries a `run_info` block and writes per-prompt predictions
beside it (`<stem>_predictions/<benchmark>.jsonl`). `summarise_results.py` recomputes every
metric from the predictions and exits non-zero if they disagree with the results file.

## 10. Configuration and hyperparameters

All values are in `config/config.yaml`; the frozen system is the file as shipped.

| Setting | Key | Value |
|---|---|---|
| Classifier backbone, LoRA rank / alpha / dropout, max length | `layer1.classifier.*` | `roberta-base`, 8 / 16 / 0.1, 512 tokens |
| Classifier training | `layer1.classifier.*` | batch 8, lr 2e-4, 5 epochs, early-stopping patience 2 |
| Scoring LM for curvature and log-rank | `layer1.scoring_lm` | `distilgpt2` (10 perturbations, temperature [0.5, 1.5]) |
| Stacker, calibration | `layer1.fusion.method`, `layer1.classifier_only` | logistic; classifier-only (T fitted on the validation split) |
| Input defences | `layer1.normalise_input`, `layer1.payload_views` | both on |
| Routing band | `layer1.thresholds.low` / `high` | 0.30 / 0.70 |
| Judge | `layer2.judge.*` | `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4, prompt `v2`, temperature 0, max 512 tokens |
| Judge decision | `layer2.consistency.*` | `logprob`; allow < 0.40, mitigate > 0.60, otherwise uncertain (K = 5, c_min = 0.8 apply only to the unused `vote` mode) |
| Mitigation | `layer3.mitigation.*`, `layer3.candidates.*` | default `refuse`; severity refuse threshold 0.99; K' = 3 rewrite candidates; refuse on model decline |
| Checks | `layer3.refusal_check`, `layer3.rewrite_check` | keywords (0.5); off (0.5) |
| Red-team attack types | `red_team.attack_types` | eight defined; the six template types are used in every recorded run |
| Bootstrap (development summaries) | `evaluation.fairness.*` | 1000 resamples, alpha 0.05 |

The routing band was never tuned: a sweep on the pool validation split found three bands
better by one prompt per class, inside the binomial standard error, and only 11 of 1,912
validation prompts fall inside the band at all (`docs/validation_policy.md`,
`results/development/routing_sweep.md`). The judge band and K' were never varied.

## 11. Random seeds

| Where | Seed | Mechanism |
|---|---|---|
| Everything a script runs (`set_seed`, MinHash, sampling) | 42 | `project.seed`; `src/llmguard/utils/data_utils.set_seed` |
| Layer-1 training | 42 (historical); 42, 43, 44 (repeat study) | `train_layer1.py --seed`; RNG reset again right before the classifier is built |
| PHTest sample of the final test set | 20260926 | `random.Random(20260926)` in `build_final_test.py` |
| Final-study bootstraps (`analyse_final.py`, `frontiers.py`, `calibration.py`, `paired_views.py`) | 20260926, B = 2000 | one stream per comparison: `default_rng([20260926, crc32(name)])` |
| Development summaries (`summarise_results.py`) and the repeat-study summary | 42, B = 1000 | `--seed`, `--bootstrap` |
| Layer-3 `safe_rewrite` sampling | none per call | temperature 0.7, top_p 0.9; outputs are not reproducible |

## 12. Expected outputs

| Command | Writes |
|---|---|
| `train_layer1.py --artifact-dir D` | `D/classifier/checkpoint_best/`, `D/fusion_v2.joblib`, `D/fusion_clf_only.joblib`, `D/scaler.json`, `D/scaler_clf_only.json`, `D/results.json`; feature cache under `data/processed/layer1_features/` |
| `run_evaluation.py --output X.json` | `X.json` (metrics, `layer_stats`, `run_info` per benchmark) and `X_predictions/<benchmark>.jsonl` (one row per prompt: `l1_decision`, `l1_score`, `l2_*`, `l3_mode`, `final_pred`, `harm_score`, `escalated`) |
| `run_judge_only.py`, `run_external_guard.py` | the same shape; guard rows add `p_unsafe`, `native_text`, `mass_on_answers`, `t_ms` |
| `run_red_team.py --output X.json` | `X.json` (bypass rates per attack type, verified rate, `run_info`) and `X_variants.jsonl` (every variant with its decision) |
| `summarise_results.py --out P` | `P.json`, `P.md`, `P.csv` |
| `final_eval/analyse_final.py --study S` | `S/analysis/results.json`, `S/analysis/tables.md` |

## 13. Reproducing the main reported experiments

In order, from the repository root, with the data rebuilt as in section 6 and the shipped
checkpoints in place. `IN=data/benchmarks_final/final_test/test.jsonl`; `OUT` is a study
folder of your own (the recorded one is `results/final_eval_20260926`). The original runner
was never committed; `REPRODUCIBILITY.md`, section 3, lists every recorded command line.

```bash
# 1. pre-registered arms (26 Sep 2026)
python scripts/run_evaluation.py --input $IN --output $OUT/final_cascade.json
python scripts/run_evaluation.py --no-l2 --input $IN --output $OUT/final_layer1.json
python scripts/final_eval/run_judge_only.py --input $IN --output $OUT/final_judge.json
python scripts/final_eval/run_external_guard.py --model granite --input $IN --output $OUT/final_granite.json
python scripts/final_eval/run_external_guard.py --model llamaguard --input $IN --output $OUT/final_llamaguard.json --revision acf7aafa60f0410f8f42b1fa35e077d705892029
for s in 42 43 44; do
  python scripts/run_evaluation.py --layer1-artifacts layer1_repeat_study_nciJu1oQ_s$s --input $IN --output $OUT/final_cascade_s$s.json
done
# 2. validation-split runs for the matched thresholds
python scripts/run_evaluation.py --input data/distillation_pool/val.jsonl --output $OUT/val_cascade.json
python scripts/final_eval/run_external_guard.py --model granite --input data/distillation_pool/val.jsonl --output $OUT/val_granite.json
python scripts/final_eval/run_external_guard.py --model llamaguard --input data/distillation_pool/val.jsonl --output $OUT/val_llamaguard.json --revision acf7aafa60f0410f8f42b1fa35e077d705892029
# 3. red team and learner arms on the repeat checkpoints (the historical red-team file is shipped)
ATK="jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix"
for s in 42 43 44; do
  python scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 --attack-types $ATK --payload-views \
      --layer1-artifacts layer1_repeat_study_nciJu1oQ_s$s --output $OUT/redteam_all200_s$s.json
  for arm in original corrected; do
    python scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/$arm \
        --layer1-artifacts layer1_repeat_study_nciJu1oQ_s$s --output $OUT/learner_${arm}_s$s.json
  done
done
# 4. post hoc arms (27-30 Sep): ShieldGemma, the Llama judge, the component ablation
python scripts/final_eval/run_external_guard.py --model shieldgemma --input $IN --output $OUT/final_shieldgemma.json --max-new-tokens 2
python scripts/final_eval/run_external_guard.py --model shieldgemma --input data/distillation_pool/val.jsonl --output $OUT/val_shieldgemma.json --benchmark val --max-new-tokens 2
python scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v1 --input $IN --output $OUT/final_judge_llama_v1.json
python scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v2 --input $IN --output $OUT/final_judge_llama_v2.json
python scripts/run_evaluation.py --no-payload-views --input $IN --output $OUT/final_cascade_noviews.json
python scripts/run_evaluation.py --no-normalise-input --input $IN --output $OUT/final_cascade_nonorm.json
python scripts/run_evaluation.py --no-payload-views --no-normalise-input --input $IN --output $OUT/final_cascade_bare.json
python scripts/run_evaluation.py --no-l2 --input data/distillation_pool/test.jsonl --output $OUT/pooltest_layer1.json
# 5. analysis, no model loaded (copy final_test.jsonl, final_manifest.json, checkpoints.json and
#    sentence_essay_writer_mapping.jsonl from results/final_eval_20260926 into $OUT first)
python scripts/final_eval/analyse_final.py --study $OUT
python scripts/final_eval/frontiers.py --study $OUT && python scripts/final_eval/calibration.py --study $OUT
python scripts/final_eval/per_category.py --study $OUT && python scripts/final_eval/guard_agreement.py --study $OUT
python scripts/final_eval/judge_band.py --study $OUT && python scripts/final_eval/paired_views.py --out $OUT/analysis/paired_views.json
```

The same sequence, in the recorded order and with the recorded arguments, is scripted in
`scripts/final_eval/run_final_eval.sh` (`--dry-run` prints every command; `--only <stage>`
runs one stage; `REPRODUCIBILITY.md`, section 8, gives the timeline). The T = 1 ablation
arms need `_T1` copies of each checkpoint with both scaler files set to `T: 1.0`
(`AMENDMENT-2026-09-30.md`); `scripts/make_t1_variants.py artifacts/<checkpoint>` recreates
them byte for byte. The freeze record of a new study is written by
`scripts/final_eval/freeze.py --study $OUT`, and `scripts/final_eval/verify_frozen.py` checks
the stored one against your checkout (the four checkpoints must match; the source hashes
refer to the pre-release layout and differ, as REPRODUCIBILITY.md explains). Expect decisions
to match the stored files and 4-bit scores to differ by up to about 6e-3; expect retrained
checkpoints not to match at all (section 7).

To regenerate only the tables from the shipped predictions:

```bash
python scripts/final_eval/analyse_final.py --study results/final_eval_20260926
```

## 14. Limitations and what cannot be reproduced

- The cascade trails Granite Guardian 3.1 2B on ranking and at a matched operating point;
  its margins to Llama Guard 3 1B and ShieldGemma 2B are inside the retraining spread.
- Over-refusal is the cascade's worst property: 42.8% of PHTest harmless prompts blocked.
- Only 2.8% of prompts reach the judge, so the cascade's ranking is Layer 1's; the judge
  alone outranks the cascade.
- Retraining under another seed moves CatQA harmful-allowed by up to 16 points and the
  red-team rate from 4.6% to 7.8%; training is not bitwise deterministic.
- Seven design decisions were taken by consulting the development benchmarks that are also
  reported (`docs/validation_policy.md`); the final test set exists because of that.
- The red-team suite is six template attacks that informed the defence; nothing here speaks
  to adaptive or optimisation-based attackers.
- ShieldGemma and the Llama judge arms are post hoc; the Llama Guard arm ran after the
  other results were known, by circumstance of the gate.
- The rewrite validator is four regular expressions; `safe_rewrite` is not a safe default;
  nothing consumes the distillation buffer, so Layer 1 does not improve over time.
- Not reproducible from this repository: the training command of the shipped
  classifier-only stacker, the original runner, the T1-variant creation, the learner
  mapping file's producer, the private reviewer labels, sampled rewrites, the 38 ms / 32 ms
  Layer-1 cost breakdown, Table II's stored source, and exact 4-bit scores. The full list,
  with evidence, is `REPRODUCIBILITY.md`, section 5; manuscript discrepancies are in
  section 6.

## 15. Citation

See `CITATION.cff`. The article is *LLM-Guard: A Calibrated Three-Layer Cascade for
Prompt-Level Harm Detection, and What a Preregistered Test Set Says About It* (2026).

## 16. Licence

The code, configuration and documentation are released under the Apache License 2.0
(`LICENSE`, "Copyright 2026 Omar Albulayhi"). Data licences are separate and are listed in
`data/LICENSES.md`: the final test set's sources are CC-BY-2.0, Apache-2.0, ODC-BY, MIT and
CC-BY-NC-4.0 (No Robots); W&I+LOCNESS is non-commercial and not redistributed; model weights
keep their own licences.
