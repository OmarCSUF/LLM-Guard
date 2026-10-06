# Reproducibility

What was run, under what, and how far it can be run again. Every claim here was checked
against the files under `results/`, `artifacts/` and `config/`; where a number is quoted,
the file it came from is named. The stored results are the source of truth: nothing in this
release re-derives them, and the analysis scripts regenerate the tables from the stored
per-prompt predictions without loading a model.

Contents: 1 environment; 2 provenance mechanism; 3 recorded commands by stage; 4 what is
included and what must be rebuilt; 5 what cannot be reproduced; 6 discrepancies between the
manuscript and the stored files; 7 time zones.

## 1. Environment

Every recorded run used the same interpreter and machine:

| Item | Value | Where recorded |
|---|---|---|
| Interpreter | `/home/OmarLaptop/venvs/guard/bin/python`, Python 3.10.12 | `run_info.command`, `run_info.python` in every results file |
| Packages | torch 2.14.0 (CUDA 13.0 build, `+cu130`), transformers 4.57.6, bitsandbytes 0.50.2, peft 0.20.0, accelerate 1.15.0, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, pyyaml 6.0.3 | `run_info.packages`; `requirements.txt` pins these |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB (5.64 GiB addressable), CUDA 13.0 | `run_info.gpu` |
| Working directory | the repository root (`/home/OmarLaptop/Omar/GaurdLLm-vibe-main` at the time) | `run_info.cwd` |
| OS | Linux (Ubuntu); kernel not recorded in `run_info` | — |

Install with `requirements.txt` (pinned to the recorded versions) after installing the
torch wheel for your CUDA build; see `README.md`, section "Environment setup". Three
earlier install instructions disagreed (the old README said torch 2.5.1 / cu124, the old
`requirements.txt` said cu118 or cu121 with `torch>=2.5.1`); this release pins what the
runs actually used.

Layer 1 runs unquantised in float16 on the GPU; the judge and all three external guards
run in 4-bit NF4 through bitsandbytes. `project.device: auto` refuses to fall back to CPU
silently; set `LLMGUARD_ALLOW_CPU=1` to run Layer 1 on CPU on purpose.

## 2. Provenance mechanism

### 2.1 `run_info` blocks

Every results file written by a script (`src/llmguard/utils/run_info.py`,
`collect_run_info` / `finalize_run_info`) carries a `run_info` block with: `started_at`
(UTC), `command` (argv with the interpreter path), `cwd`, `seed`, `git_commit`,
`git_dirty`, `python`, `packages`, `gpu` and a full snapshot of `config/config.yaml`. In
`run_evaluation.py` outputs the block is nested under each benchmark key (`{"test":
{"run_info": ...}}`); in red-team reports, manifests and summaries it is at the top level.
Check any file with:

```bash
python3 -c "import json;d=json.load(open('results/final_eval_20260926/final_cascade.json'))['test'];print(d['run_info']['git_commit'], d['run_info']['git_dirty'], d['run_info']['started_at'])"
python3 -c "import json;d=json.load(open('artifacts/layer1/results.json'));print(d['run_info']['git_commit'], d['run_info']['started_at'])"
```

Files written before 15 September 2026 predate the mechanism and have none; the release
ships no result without one except where section 5 says so.

### 2.2 `frozen.json`

`results/final_eval_20260926/frozen.json` was written at the start of the final study by
the (never committed) runner `run_final_eval.sh`. It records SHA-256 of `config/config.yaml`
and 62 source files under the old flat layout (`layer1/ layer2/ layer3/ evaluation/
redteam/ utils/ scripts/`, 63 files in all), the Layer-1 artifacts of the four checkpoints
(adapter, adapter config, classifier-only stacker and scaler), the three pool splits (with
row counts and an id hash), the thirteen input files (five development benchmarks, eight
learner arm files), the git commit `dc90a9e2124dac1ee81588059bde5239855c9fdf` with
`dirty: false`, and the GPU. `checkpoints.json` beside it maps checkpoint names to
directories. The source files have since moved to `src/llmguard/`, so their paths in
`frozen.json` no longer resolve; the hashes still identify the versions that ran.

### 2.3 Named bootstrap streams

`scripts/final_eval/analyse_final.py` draws every bootstrap from its own generator,
`np.random.default_rng([20260926, crc32(name)])`, where `name` identifies the comparison
(for example `aurocdiff|granite|cascade`). An interval therefore depends only on the
comparison it belongs to, not on how many arms the study contains. This replaced a single
shared generator on 28 September 2026; point estimates never moved (they are exact counts),
intervals moved at the first decimal and by up to one point on SimpleSafetyTests (n = 99).
`frontiers.py`, `calibration.py` and `paired_views.py` use the same scheme
(`BOOT_SEED = 20260926`, B = 2000).

### 2.4 Pinned models

| Model | Repository | Revision | Where pinned |
|---|---|---|---|
| Judge and Guard-LLM | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | The runs loaded a local copy (`/home/OmarLaptop/hf_models/Qwen2.5-3B-Instruct`) whose safetensors match this revision by SHA-256 |
| Layer-1 backbone | `roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | HF cache snapshot used during the runs |
| Scoring LM (three-signal stacker only) | `distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` | HF cache snapshot |
| Granite Guardian 3.1 2B | `ibm-granite/granite-guardian-3.1-2b` | `81145486e85c6c82c01e759c0356d9d6da4d21a5` | `final_granite.json` → `test.model.revision` |
| Llama Guard 3 1B | `meta-llama/Llama-Guard-3-1B` | `acf7aafa60f0410f8f42b1fa35e077d705892029` | `--revision` on the command line; `final_llamaguard.json` |
| ShieldGemma 2B | `google/shieldgemma-2b` | `d1dffc9c8c9237a90aab09c61383791e718ef9e8` | `final_shieldgemma.json` |
| Substitute judge | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | `AMENDMENT-2026-09-28b.md` |

Dataset revisions are in `results/development/raw_sources_manifest.json` (pool and
development benchmark sources), `results/development/raw_final_sources_manifest.json` and
`data/benchmarks_final/manifest.json` → `source_revisions` (final-test sources).

## 3. Recorded commands, by stage

All commands were run from the repository root with the interpreter above. Output paths
were `data/processed/...`; the release keeps those files under `results/` as shown. In
this release the scripts put `src/` on `sys.path` themselves, so `python scripts/<x>.py`
works from the root; `PYTHONPATH=src python scripts/<x>.py` is equivalent.

### 3.1 Data (15 September 2026, commit `c8f6c0da`)

```bash
python scripts/fetch_datasets.py                 # recorded as: --only wildjailbreak orbench (the rest were present)
python scripts/build_datasets.py                 # pool train/val/test + the five development benchmarks
```

Outputs kept: `results/development/raw_sources_manifest.json`,
`results/development/pool_metadata.json`, `results/development/pool_build_2026-09-15.log`.
Pool: 15,296 / 1,912 / 1,912 rows, balanced; SHA-256 of the three split files in
`frozen.json` → `pool`.

### 3.2 Layer-1 training

```bash
python scripts/train_layer1.py                                               # 2026-09-15T06:30Z, seed 42, 2,394 s -> artifacts/layer1
python scripts/repeat_study_nciJu1oQ_train.py --seed 42 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s42   # 2026-09-23, dc90a9e2
python scripts/repeat_study_nciJu1oQ_train.py --seed 43 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s43
python scripts/repeat_study_nciJu1oQ_train.py --seed 44 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s44
```

`repeat_study_nciJu1oQ_train.py` now forwards to `train_layer1.py --seed`; the equivalent
command is `python scripts/train_layer1.py --seed 42 --artifact-dir <dir>`.

### 3.3 Final test set (26 September 2026, commit `dc90a9e2`, clean tree)

```bash
python scripts/final_eval/fetch_final_sources.py       # 06:35Z -> results/development/raw_final_sources_manifest.json
python scripts/final_eval/build_final_test.py          # 06:35Z -> data/benchmarks_final/final_test/test.jsonl + manifest.json
```

`data/benchmarks_final/final_test/test.jsonl` has SHA-256
`39425a51fc671079dc341f57ec6a4821b1fb2732b50ae23cc95ccda3135e11a0` (3,453 rows), as the
manifest records. `results/final_eval_20260926/final_test.jsonl` is the copy the study
scored and is byte-identical to it.

### 3.4 Pre-registered arms (26 September 2026, commit `dc90a9e2`, clean tree)

`IN=data/benchmarks_final/final_test/test.jsonl`, `OUT=data/processed/final_eval_20260926`
(now `results/final_eval_20260926`).

```bash
python scripts/run_evaluation.py --input $IN --output $OUT/final_cascade.json
python scripts/run_evaluation.py --no-l2 --input $IN --output $OUT/final_layer1.json
python scripts/final_eval/run_judge_only.py --input $IN --output $OUT/final_judge.json
python scripts/final_eval/run_external_guard.py --model granite --input $IN --output $OUT/final_granite.json
python scripts/final_eval/run_external_guard.py --model granite --input data/distillation_pool/val.jsonl --output $OUT/val_granite.json
python scripts/run_evaluation.py --input data/distillation_pool/val.jsonl --output $OUT/val_cascade.json
python scripts/run_evaluation.py --layer1-artifacts layer1_repeat_study_nciJu1oQ_s42 --input $IN --output $OUT/final_cascade_s42.json   # and s43, s44
python scripts/final_eval/run_external_guard.py --model llamaguard --input $IN --output $OUT/final_llamaguard.json   # skipped on 26 Sep: access refused; see final_llamaguard.skipped.json
```

Seed-robustness arms (Q3), same day:

```bash
python scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 \
    --attack-types jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix \
    --payload-views --layer1-artifacts layer1_repeat_study_nciJu1oQ_s42 --output $OUT/redteam_all200_s42.json   # and s43, s44
python scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/original \
    --layer1-artifacts layer1_repeat_study_nciJu1oQ_s42 --output $OUT/learner_original_s42.json                    # and corrected; s43, s44
```

The historical red-team file `redteam_all200_historical.json` is a byte-identical copy of
`results/development/redteam_all200_views.json`, run on 22 September 2026 at commit
`6dead1c4` with the same command minus `--layer1-artifacts`. The views-off counterpart
(`results/development/redteam_all200.json`, commit `6efe54d5`) omits `--payload-views`.

### 3.5 Post hoc arms (27 September to 1 October 2026, commit `dc90a9e2`, `git_dirty: true`)

```bash
# 27 Sep: ShieldGemma (AMENDMENT-2026-09-27.md)
python scripts/final_eval/run_external_guard.py --model shieldgemma --input $IN --output $OUT/final_shieldgemma.json --max-new-tokens 2
python scripts/final_eval/run_external_guard.py --model shieldgemma --input data/distillation_pool/val.jsonl --output $OUT/val_shieldgemma.json --benchmark val --max-new-tokens 2
# 28 Sep: Llama Guard, second account (AMENDMENT-2026-09-28.md)
python scripts/final_eval/run_external_guard.py --model llamaguard --input $IN --output $OUT/final_llamaguard.json --revision acf7aafa60f0410f8f42b1fa35e077d705892029
python scripts/final_eval/run_external_guard.py --model llamaguard --input data/distillation_pool/val.jsonl --output $OUT/val_llamaguard.json --revision acf7aafa60f0410f8f42b1fa35e077d705892029
# 28 Sep: judge substitution (AMENDMENT-2026-09-28b.md)
python scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct --output data/processed/judge_tuning_llama.jsonl --summary data/processed/judge_tuning_llama_summary.json
python scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v1 --input $IN --output $OUT/final_judge_llama_v1.json
python scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v2 --input $IN --output $OUT/final_judge_llama_v2.json
# 30 Sep: component ablation, 16 arms (AMENDMENT-2026-09-30.md)
python scripts/run_evaluation.py --no-payload-views --input $IN --output $OUT/final_cascade_noviews.json
python scripts/run_evaluation.py --no-normalise-input --input $IN --output $OUT/final_cascade_nonorm.json
python scripts/run_evaluation.py --no-payload-views --no-normalise-input --input $IN --output $OUT/final_cascade_bare.json
python scripts/run_evaluation.py --layer1-artifacts layer1_T1 --input $IN --output $OUT/final_cascade_T1.json
#   and each of the four with --layer1-artifacts layer1_repeat_study_nciJu1oQ_s4X[_T1] -> final_cascade_<kind>_s4X.json
# 1 Oct: Layer 1 on the pool test split, for the calibration analysis (AMENDMENT-2026-10-01.md)
python scripts/run_evaluation.py --no-l2 --input data/distillation_pool/test.jsonl --output $OUT/pooltest_layer1.json
```

### 3.6 Analysis (no model loaded)

```bash
python scripts/final_eval/analyse_final.py --study results/final_eval_20260926      # analysis/results.json, analysis/tables.md
python scripts/final_eval/frontiers.py       --study results/final_eval_20260926   # analysis/frontiers.json
python scripts/final_eval/calibration.py     --study results/final_eval_20260926 --dev results/development   # analysis/calibration.json
python scripts/final_eval/per_category.py    --study results/final_eval_20260926   # analysis/per_category.json, .md
python scripts/final_eval/guard_agreement.py --study results/final_eval_20260926   # analysis/guard_agreement.json
python scripts/final_eval/judge_band.py      --study results/final_eval_20260926   # analysis/judge_band.json
python scripts/final_eval/paired_views.py    --out results/final_eval_20260926/analysis/paired_views.json   # reads results/development/redteam_all200*_variants.jsonl
python scripts/summarise_results.py --run cascade=results/final_eval_20260926/final_cascade.json --run layer1=results/final_eval_20260926/final_layer1.json \
    --run judge=results/final_eval_20260926/final_judge.json --run granite=results/final_eval_20260926/final_granite.json \
    --compare cascade:layer1 --compare cascade:judge --compare cascade:granite --out results/final_eval_20260926/final_summary
```

`analyse_final.py` also needs `sentence_essay_writer_mapping.jsonl` in the study folder for
section D (learner intervals grouped by essay and writer). At the time of the runs the six
single-purpose scripts read the study folder from a path constant
(`data/processed/final_eval_20260926`); the release versions take `--study` / `--out`
(and `--dev`, `--views-off`, `--views-on` where they read development files), with defaults
that point at the shipped locations, so the bare commands above regenerate the stored
`analysis/` files byte for byte (`make analysis`). The whole study, in the recorded order,
is scripted in `scripts/final_eval/run_final_eval.sh` (see section 8).

Development-phase commands (ablation, routing sweep, judge tuning, parity, learner arms,
the `lim*` drivers) are listed in `docs/history/development_runs.md`.

## 4. What is included, what must be rebuilt

### Included

| Item | Location |
|---|---|
| Source code (`llmguard` package), scripts, configuration | `src/llmguard/`, `scripts/`, `config/` |
| Four Layer-1 checkpoints: LoRA adapter, tokenizer files, both stackers, both scalers, `results.json` | `artifacts/layer1`, `artifacts/layer1_repeat_study_nciJu1oQ_s{42,43,44}` |
| The final test set (3,453 prompts with id, source, category, label, text) and its manifest | `data/benchmarks_final/` |
| The pre-registered study: pre-registration, five amendments, `frozen.json`, `checkpoints.json`, every arm's results JSON and per-prompt predictions (27 arms on the final set, 4 on the pool validation split, Layer 1 on the pool test split), red-team reports and per-variant rows for four checkpoints, learner results for three checkpoints, logs, the withdrawn Llama Guard attempt, and `analysis/` | `results/final_eval_20260926/` |
| Selected development results (red team views on/off, judge tuning, routing sweep, ablation summary, parity and learner summaries, review scorers' outputs, manifests, pool metadata and build log) | `results/development/` |
| Repeat-study protocol, frozen-file hashes, seed variability and summary | `results/repeat_study_nciJu1oQ/` |
| Figures of the old README and the Layer-1 diagnostic plots | `figures/` (see `figures/README.md`) |

### Must be rebuilt from pinned sources (not redistributed)

| Item | How | Notes |
|---|---|---|
| Pool splits (`data/distillation_pool/{train,val,test}.jsonl`) and the five development benchmarks (`data/benchmarks/<name>/test.jsonl`) | `scripts/fetch_datasets.py`, `scripts/build_datasets.py` | Source revisions in `results/development/raw_sources_manifest.json`; expected SHA-256 of the splits and benchmark files in `frozen.json`. WildJailbreak, XSTest and HarmBench are gated on the Hub. |
| Raw final-test sources | `scripts/final_eval/fetch_final_sources.py` | Only needed to re-run `build_final_test.py`; the built set is shipped |
| Learner arms (`data/benchmarks_learner/{original,corrected}/`) | `scripts/make_learner_arms.py --source <W&I+LOCNESS v2.1 directory>` | W&I+LOCNESS is non-commercial and is not redistributed; download it from the BEA-2019 shared task page. Expected SHA-256 of the eight arm files in `frozen.json` → `inputs` |
| Parity arms (`data/benchmarks_parity/`) | `scripts/make_l2_benchmarks.py --root data/benchmarks_parity` | Needs the development benchmarks |
| Layer-1 feature cache (`data/processed/layer1_features/`) | written by `scripts/train_layer1.py`; scored by `scripts/score_feature_cache.py` | Needed only by `plot_layer1.py` and `check_calibration.py` |
| `_T1` artifact directories (temperature fixed at 1) | copies of a checkpoint with both scaler files rewritten to `T = 1.0`; see `AMENDMENT-2026-09-30.md` | The creation was not committed; a reconstruction script is described in section 5 |
| Model weights | downloaded from the Hub at the revisions in section 2.4 | Llama Guard 3 1B, Llama-3.2-3B-Instruct and ShieldGemma are gated |

See `data/README.md` for the rebuild procedure and `data/LICENSES.md` for the licences.

## 5. What cannot be reproduced, or not exactly

Each item below was verified against the files; none is a guess.

1. **The shipped classifier-only stacker has no training command.**
   `artifacts/layer1/fusion_clf_only.joblib`, `scaler_clf_only.json` (T = 1.1908) and
   `fusion_clf_only.meta.json` were written by an older version of `scripts/run_ablation.py`
   on 22 September 2026 (local time 23 Sep 00:31 +08), after the training run recorded in
   `artifacts/layer1/results.json` (15 September). `train_layer1.py` now writes both stackers,
   but the shipped file is not its output. The final study pins the file by SHA-256
   (`frozen.json` → `checkpoints.historical`), so the study is internally consistent, but the
   stacker cannot be traced to a command.
2. **`artifacts/layer1` is an unrecorded copy.** Its adapter and `results.json` are
   byte-identical to the 15 September training run (which also survives as
   `artifacts/layer1_v2` in the original workspace), but the directory's files are dated
   21 September 2026 and no command records the copy.
3. **Layer-1 training is not bitwise deterministic.** The same recipe with seed 42 gave
   fitted temperatures T = 1.1891 (15 Sep), 1.2547 and 1.2486 (`artifacts/layer1_repeat_study_nciJu1oQ_s42`)
   in three runs. On the final test set, retraining under seeds 42/43/44 moves CatQA
   harmful-allowed from 19.2% to 34.8 / 22.3 / 35.7% (`analysis/tables.md`, seed table).
   Retraining will not reproduce the historical checkpoint; use the shipped one.
4. **4-bit judge and guard passes drift slightly between runs.** Re-scoring 60 prompts with
   Granite moved P(unsafe) by at most 6.03e-3 and changed no decision
   (`AMENDMENT-2026-09-28.md`, section 3). Expect the same for the judge arms: identical
   decisions in the overwhelming majority of prompts, not identical scores.
5. **Every run from 27 September on was made with uncommitted edits** (`git_dirty: true` at
   `dc90a9e2`), and 16 scripts differ from the hashes in `frozen.json`. Only two of those edits
   are documented in amendments: `run_evaluation.py` gained `--normalise-input` /
   `--no-normalise-input` (30 Sep), and `tune_judge.py` / `run_judge_only.py` gained
   `--judge-model` (28 Sep). Absent the new flags both behave as before. The other
   differences are undocumented; the release code is the final state of that tree, moved
   under `src/llmguard/`.
6. **The runner and the T1 variants were never committed.** `run_final_eval.sh`, which wrote
   `frozen.json` and ran the 26 September arms in order, and the step that created the four
   `layer1*_T1` directories, do not exist in the git history. The release reconstructs them
   from the `run_info` blocks as `scripts/final_eval/run_final_eval.sh` (with
   `scripts/final_eval/freeze.py` for the freeze record) and `scripts/make_t1_variants.py`,
   which reproduces the original T1 scaler, stacker and adapter files byte for byte. The
   reconstruction reproduces the recorded commands, not the original file.
7. **`sentence_essay_writer_mapping.jsonl` ("the 24 September mapping") has no producer
   script.** It maps every learner sentence to its essay and writer id and is read by
   `analyse_final.py` for the essay- and writer-grouped intervals. It is shipped as data.
8. **The Layer-3 leak review's reviewer label files are private** and absent; the leak rate
   in `docs/leak_codebook.md` and the inputs of `scripts/leak_reliability.py` cannot be
   regenerated. `rewrite_check_pilot4.json` and `refusal_eval_pilot4.json` are the scorers'
   outputs over the 90 reviewed items.
9. **Layer-3 `safe_rewrite` outputs are sampled** (temperature 0.7, top_p 0.9) with no per-call
   seed, so served rewrites differ between runs. The shipped default is `refuse`, which is
   deterministic.
10. **The pool and development benchmarks are not redistributed** (section 4). A rebuild from
    the pinned revisions should reproduce the SHA-256 in `frozen.json`; the near-duplicate
    screen is seeded (MinHash seed 42), so the result is deterministic given the same sources.
11. **W&I+LOCNESS is non-redistributable.** The learner arms must be rebuilt; the learner
    prediction files under `results/` are intended to carry decisions and scores only, with
    sentence texts stripped (check `learner_*_predictions/*.jsonl` in your checkout).
12. **Gated models and data need Hugging Face access**: Llama Guard 3 1B (manual review by
    Meta; one account was refused, another granted), Llama-3.2-3B-Instruct, ShieldGemma
    (terms acceptance), WildJailbreak, XSTest and HarmBench.
13. **The historical red-team run predates the freeze.** `redteam_all200_views.json` was run
    on 22 September 2026 at commit `6dead1c4`, four days before `frozen.json`; the three
    repeat-checkpoint passes were run inside the study at `dc90a9e2`.
14. **Latencies are specific to the RTX 4050 Laptop GPU** and to machine load at the time
    (`latency_ms` in the judge and guard arm files; `elapsed_s` in every arm). They should not
    be compared across days.
15. **Table II of the paper (pool-test stackers) has no stored result file.** The
    classifier-only column (AUROC 0.9914, AUPRC 0.9868, FPR@95%TPR 0.0220, accuracy 0.9629,
    15 gray-zone prompts, T = 1.1908) was read from the scored feature cache by
    `scripts/plot_layer1.py`; `artifacts/layer1/results.json` stores only the three-signal
    row (AUROC 0.9892, T = 1.1891). `results/final_eval_20260926/pooltest_layer1.json`
    (Layer 1 with views on, 1 Oct) gives AUROC 0.9913 and accuracy 0.9644 on the same split.
    Regenerate the table with `scripts/plot_layer1.py` after rebuilding the feature cache.
16. **The 38 ms / 32 ms Layer-1 cost breakdown** (three-signal stacker per prompt; the two
    scoring-LM signals' share) quoted in the old README and `config/config.yaml` comments
    was never stored in a results file.
17. **Guard parameter counts** in the paper (about 2.5 B, 1.5 B, 2.6 B) come from the model
    cards, not from a measurement in this repository.

## 6. Discrepancies between the manuscript and the stored files

An independent check found that every manuscript table matches the stored analysis files
except the following:

| # | Manuscript | Stored value | Reading |
|---|---|---|---|
| (a) | Guard-agreement sentence: aggregated unsafe and safe spellings "differ by less than 3e-5" | `analysis/guard_agreement.json` → Granite `disagree_scores.max` = 0.500021 on the final set (0.500029 on validation) | Better stated as \|p_unsafe − 0.5\| ≤ 2.1e-5 for the six final-set ties; the 3e-5 bound holds for both sets |
| (b) | Frontier read-outs at the cascade's macro safe-blocked rate (Layer 1 allows 25.7% on s42; cascade macro harmful-allowed 16.0) | `analysis/frontiers.json` | The paper reads the step of the frontier at or below the cascade's safe-blocked rate; linear interpolation would give 25.6 and 15.9 |
| (c) | `AMENDMENT-2026-10-01.md` prints the Layer-1 pool-test ECE interval as [0.008, 0.022] | `analysis/calibration.json` → `layer1_pool_test.ci` = [0.00752, 0.02149] | 0.02149 rounds to 0.021, as the paper's Table VII has it; the amendment's 0.022 is a rounding slip |
| (d) | Number of attack families | `src/llmguard/redteam/attacks.py` defines eight (`jailbreak_template`, `role_play`, `prefix_rewrite`, `language_prefix`, `decoding_manipulation`, `technical_prefix`, `model_paraphrase`, `model_translation`); every reported red-team run used six; `config/nist_ai_rmf_mapping.yaml` says "seven attack types" in two places; the old README said eight | Eight defined, six used; the NIST YAML's "seven" is stale |
| (e) | Install instructions | Old README: torch 2.5.1 / cu124; old `requirements.txt`: cu118 or cu121, `torch>=2.5.1`; `run_info`: torch 2.14.0+cu130 | The release pins the recorded versions |

## 7. Time zones

`run_info.started_at` is UTC (`+00:00`). The log files under
`results/final_eval_20260926/logs/` and `results/development/pool_build_2026-09-15.log`
are stamped in the machine's local time, UTC+8: for example `final_cascade.json` records
`2026-09-26T06:39:16+00:00` and the first line of `logs/final_cascade.log` reads
`[2026-09-26 14:39:17]`. File modification times quoted in this document and in
`figures/README.md` are local (+08). Dates in the amendments and in this file are the
local calendar date unless marked `Z`.

## 8. Final-evaluation runbook

How the pre-registered study in `results/final_eval_20260926/` was produced, reconstructed from the
`run_info` block of every stored result file (the original orchestrating script was never committed;
`scripts/final_eval/run_final_eval.sh` replays the sequence below). Times are UTC as recorded in
`run_info.started_at`; the amendments and logs use local time (UTC+8).

### Before any arm ran (26 September 2026)

| Step | Command | Notes |
|---|---|---|
| Pre-registration written | `PREREGISTRATION.md` | before any source was downloaded |
| 06:35:29 fetch sources | `python scripts/final_eval/fetch_final_sources.py` | six open repositories; revisions recorded in `results/development/raw_final_sources_manifest.json` |
| 06:35:45 build the set | `python scripts/final_eval/build_final_test.py` | 3,453 prompts; counts at each screening step in `final_manifest.json`; SHA-256 `39425a51…` |
| freeze | (runner) wrote `frozen.json` and `checkpoints.json` | SHA-256 of `config/config.yaml`, 62 source files, the four checkpoints' adapters and stackers, the three pool splits, 13 input files; git commit `dc90a9e2`, clean. Reproduced by `scripts/final_eval/freeze.py`; checked by `verify_frozen.py` |

### Pre-registered arms (26 September, commit `dc90a9e2`, clean tree)

| Started | Arm | Command | Elapsed |
|---|---|---|---|
| 06:39:16 | cascade | `scripts/run_evaluation.py --input data/benchmarks_final/final_test/test.jsonl --output <study>/final_cascade.json` | 188 s |
| 06:42:39 | Layer 1 alone | `… --no-l2 …final_layer1.json` | 19 s |
| 06:43:08 | judge alone | `scripts/final_eval/run_judge_only.py --input … --output <study>/final_judge.json` | 2,014 s |
| 07:16:50 | cascade on pool validation | `scripts/run_evaluation.py --input data/distillation_pool/val.jsonl --output <study>/val_cascade.json` | the matched-threshold target (3.8%) |
| 07:17:34 / 07:28:09 | Granite Guardian 3.1 2B, final / validation | `scripts/final_eval/run_external_guard.py --model granite --input … --output …` | NF4 |
| 07:32:34, 07:50:37, 08:09:57 | cascade with checkpoints s42, s43, s44 | `… --layer1-artifacts layer1_repeat_study_nciJu1oQ_s4X …` | |
| 07:35:19, 07:53:27, 08:12:53 | attack pass per checkpoint | `scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 --attack-types jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix --payload-views --layer1-artifacts … --output …` | 3,600 variants each; the configured checkpoint's pass is the 22 Sep run copied in as `redteam_all200_historical.*` |
| 07:36–08:22 | learner arms per checkpoint | `scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/{original,corrected} --layer1-artifacts … --output …` | needs the W&I+LOCNESS corpus |
| 08:29:56 | summary (superseded) | `scripts/summarise_results.py …` | replaced by `analyse_final.py`; kept in `superseded/` |
| 14:11 | Llama Guard, skipped | `run_external_guard.py --model llamaguard …` → `*.skipped.json` | gated weights not available (AMENDMENT-2026-09-27) |

### Post hoc arms (uncommitted edits present, `git_dirty: true`)

| Date | Arm | Command | Amendment |
|---|---|---|---|
| 27 Sep 11:48 / 12:14 | ShieldGemma 2B, final / validation | `run_external_guard.py --model shieldgemma … --max-new-tokens 2` (`--benchmark val` for the validation file) | 2026-09-27 |
| 28 Sep 04:13 / 04:19 | Llama Guard 3 1B, final / validation | `run_external_guard.py --model llamaguard … --revision acf7aafa60f0410f8f42b1fa35e077d705892029` | 2026-09-28 (a 04:01 attempt was withdrawn: `withdrawn_badstep_20260928/`) |
| 28 Sep 07:52 | Llama judge prompt choice | `scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct --output … --summary …` | 2026-09-28b |
| 28 Sep 09:31 / 09:46 | judge slot with Llama-3.2-3B, prompt v1 / v2 | `run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v1|v2 …` | 2026-09-28b |
| 30 Sep 05:50–06:33 | 16 component-ablation arms | `run_evaluation.py {--no-payload-views | --no-normalise-input | both | --layer1-artifacts <ckpt>_T1} [--layer1-artifacts layer1_repeat_study_nciJu1oQ_s4X]` | 2026-09-30; T1 checkpoints made by `scripts/make_t1_variants.py` (reconstruction of a manual copy) |
| 1 Oct 15:16 | Layer 1 on the pool test split | `run_evaluation.py --no-l2 --input data/distillation_pool/test.jsonl --output <study>/pooltest_layer1.json` | 2026-10-01 (calibration reference) |

### Analysis (no model loaded)

```
python scripts/final_eval/analyse_final.py --study results/final_eval_20260926
python scripts/final_eval/frontiers.py        --study results/final_eval_20260926
python scripts/final_eval/calibration.py      --study results/final_eval_20260926 --dev results/development
python scripts/final_eval/per_category.py     --study results/final_eval_20260926
python scripts/final_eval/guard_agreement.py  --study results/final_eval_20260926
python scripts/final_eval/judge_band.py       --study results/final_eval_20260926
python scripts/final_eval/paired_views.py     --out results/final_eval_20260926/analysis/paired_views.json
```

`analyse_final.py` was re-run after each amendment (last on 30 Sep); the six post hoc scripts ran on 1 Oct. Re-running them over the stored predictions reproduces `analysis/` byte for byte (`make analysis`, `tests/test_analysis_regeneration.py`).

### What each stage needs

- Models at the pinned revisions in `config/config.yaml` and the commands above; gated: `meta-llama/Llama-Guard-3-1B` (manual approval by Meta), `meta-llama/Llama-3.2-3B-Instruct`, `google/shieldgemma-2b` (Gemma terms). The Qwen judge and the Layer-1 backbones are open.
- Data: the final test set is in the repository; the pool validation/test splits and the HarmBench seed file must be rebuilt (`data/README.md`); the learner arms need your own copy of W&I+LOCNESS.
- Hardware: every arm ran on one RTX 4050 Laptop GPU (5.64 GiB) with 4-bit NF4 for the judge and guards; latencies are specific to it. Layer 1 alone is deterministic; the 4-bit arms drift by about 6e-3 in probability between runs (no decision changed in a 60-prompt check, AMENDMENT-2026-09-28).
- The red-team and learner arms use the configured seeds (42) and the per-comparison bootstrap streams; see `results/final_eval_20260926/analysis/README.md`.
