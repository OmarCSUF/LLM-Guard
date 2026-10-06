# Development runs: a plain record

This file records the development-phase runs that produced the files under
`results/development/`, and the root-level driver scripts of the original workspace that
were not carried into this release. It is a record, not a recipe: the development
benchmarks these runs score were also used for selection (see `docs/validation_policy.md`),
and nothing here is a headline result. The pre-registered study is in
`results/final_eval_20260926/`.

Conventions. Every command was run from the repository root with the project virtual
environment (`/home/OmarLaptop/venvs/guard/bin/python`, Python 3.10.12). Output paths of
the form `data/processed/<file>` are the original locations; where a file was kept, its
release location is given. Commit hashes are the `run_info.git_commit` recorded in the
output file. Dates are UTC, from `run_info.started_at`.

## 1. Pool, benchmarks and training (15 September 2026)

| Step | Command | Commit | Kept as |
|---|---|---|---|
| Fetch pool and benchmark sources | `python scripts/fetch_datasets.py --only wildjailbreak orbench` (the recorded manifest run; the other sources were already present) | `c8f6c0da` | `results/development/raw_sources_manifest.json` |
| Build the pool and the five benchmarks | `python scripts/build_datasets.py` | `c8f6c0da` | `results/development/pool_metadata.json`, `results/development/pool_build_2026-09-15.log` |
| Train Layer 1 (historical checkpoint) | `python scripts/train_layer1.py` (seed 42 from `project.seed`, 2,394 s) | `c8f6c0da` | `artifacts/layer1/results.json` |
| Judge prompt v1 vs v2 on 300 validation prompts | `python scripts/tune_judge.py --resume` | `b08c108d` | `results/development/judge_tuning.jsonl`, `judge_tuning_summary.json` |
| Thesis tables, Layer 1 alone vs pipeline | `python scripts/summarise_results.py --run layer1=data/processed/eval_layer1_only_v2.json --run pipeline=data/processed/eval_pipeline_v2.json --compare pipeline:layer1 --out data/processed/thesis_results` | `aab18ac1` | `results/development/thesis_results.{json,md}` (the two input runs are not shipped) |

## 2. Component ablation on the benchmarks (18 September 2026)

`scripts/run_ablation.py` collects per-prompt signals in one GPU pass (`--stage signals`)
and derives the four historical variants from them (`--stage derive`): `full`
(three-signal cascade), `classifier` (raw classifier), `no_curvature` (classifier-only
cascade) and `judge` (judge only). The derived results were then compared:

```bash
python scripts/summarise_results.py \
    --run full=data/processed/ablation_full.json \
    --run classifier=data/processed/ablation_classifier.json \
    --run no_curvature=data/processed/ablation_no_curvature.json \
    --run judge=data/processed/ablation_judge.json \
    --compare full:no_curvature --compare full:classifier --compare full:judge \
    --compare classifier:no_curvature \
    --out data/processed/ablation_results
```

Commit `c069337c`. Kept as `results/development/ablation_results.{json,md}`; the four
`ablation_*.json` inputs and the signals file are not shipped. This is the ablation that
made the classifier-only stacker the default (decision register, patches 28 and 29). The
shipped `artifacts/layer1/fusion_clf_only.joblib` carries a fingerprint written by an older
version of this script, not by `train_layer1.py`; see `REPRODUCIBILITY.md`.

## 3. Red team, framing diagnostic and the payload view (22 September 2026)

| Run | Command | Commit | Kept as |
|---|---|---|---|
| Views off, all 200 HarmBench seeds | `python scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 --attack-types jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix --output data/processed/redteam_all200.json` | `6efe54d5` | `results/development/redteam_all200.json`, `redteam_all200_variants.jsonl` |
| Views on, same seeds | the same command with `--payload-views` and `--output data/processed/redteam_all200_views.json` | `6dead1c4` | `results/development/redteam_all200_views.json`, `redteam_all200_views_variants.jsonl` (also `results/final_eval_20260926/redteam_all200_historical.json`) |
| Benchmarks, views vs shipped default | `python scripts/summarise_results.py --run base=data/processed/eval_shipped_default_2026-09-22.json --run views=data/processed/eval_views_2026-09-22.json --compare views:base --out data/processed/views_vs_base` | `6dead1c4` | `results/development/views_vs_base.{json,md}` |
| Seed coverage across four benchmarks | `python scripts/run_red_team.py --seed-benchmarks harmbench jailbreakbench orbench toxicchat --per-category 12 --all-seeds --n-per-type 3 --attack-types <six> --output data/processed/redteam_coverage.json` | `f9e14bf8` | `results/development/redteam_coverage.json` |
| HarmBench alone through the same harness | `python scripts/run_red_team.py --seed-benchmarks harmbench --all-seeds --n-per-type 3 --attack-types <six> --output data/processed/redteam_harmbench_only.json` | `f9e14bf8` | `results/development/redteam_harmbench_only.json` |
| Routing-band sweep on the validation split | `python scripts/sweep_routing.py --select data/processed/ablation_val_signals.jsonl --out data/processed/routing_sweep` | `ab60defe` | `results/development/routing_sweep.{csv,json,md}` (the signals file is not shipped) |

`<six>` stands for `jailbreak_template role_play prefix_rewrite language_prefix
decoding_manipulation technical_prefix`.

### 3a. `lim2_runs.sh` (opener augmentation, limitation 2)

Root-level driver of the original workspace, run on 22 September 2026. It retrained
Layer 1 twice and compared the three models. Transcribed without the `tee` redirections:

```bash
RT="--seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 --attack-types jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix"
# 1/8 retrain with opener augmentation
python scripts/train_layer1.py --artifact-dir artifacts/layer1_openers --augment-openers
# 2/8 benchmarks, augmented model
python scripts/run_evaluation.py --layer1-artifacts layer1_openers --output data/processed/eval_openers_$D.json
# 3/8 red team, augmented model
python scripts/run_red_team.py $RT --layer1-artifacts layer1_openers --output data/processed/redteam_all200_openers.json
# 4/8 framing diagnostic, augmented model
python scripts/framing_diagnostic.py --layer1-artifacts layer1_openers --no-judge --out data/processed/framing_diag_openers.json
# 5/8 control retrain, same recipe, no augmentation
python scripts/train_layer1.py --artifact-dir artifacts/layer1_control
# 6/8 benchmarks, red team, diagnostic, control model
python scripts/run_evaluation.py --layer1-artifacts layer1_control --output data/processed/eval_control_$D.json
python scripts/run_red_team.py $RT --layer1-artifacts layer1_control --output data/processed/redteam_all200_control.json
python scripts/framing_diagnostic.py --layer1-artifacts layer1_control --no-judge --out data/processed/framing_diag_control.json
# 7/8 pool test split, Layer 1 alone, three models
python scripts/run_evaluation.py --input data/distillation_pool/test.jsonl --no-l2 --output data/processed/pooltest_shipped.json
python scripts/run_evaluation.py --input data/distillation_pool/test.jsonl --no-l2 --layer1-artifacts layer1_openers --output data/processed/pooltest_openers.json
python scripts/run_evaluation.py --input data/distillation_pool/test.jsonl --no-l2 --layer1-artifacts layer1_control --output data/processed/pooltest_control.json
# 8/8 paired bootstrap against the views baseline
python scripts/summarise_results.py --run base=data/processed/eval_views_2026-09-22.json --run openers=data/processed/eval_openers_$D.json --run control=data/processed/eval_control_$D.json --compare openers:base --compare control:base --compare openers:control --out data/processed/lim2_openers_vs_base
python scripts/summarise_results.py --run base=data/processed/pooltest_shipped.json --run openers=data/processed/pooltest_openers.json --run control=data/processed/pooltest_control.json --compare openers:base --compare control:base --out data/processed/lim2_pooltest
```

`$D` was the run date, `2026-09-22`. Kept: `results/development/lim2_openers_vs_base.{json,md}`
(commit `b3518409`). Not kept: the two retrained checkpoints (`layer1_openers`,
`layer1_control`), their evaluation, red-team and framing-diagnostic outputs, and the
pool-test comparison. The decision this run informed (opener augmentation rejected, patch 35)
is in the decision register. The control retrain is the first evidence that the same recipe
retrained gives a different model; the three-seed repeat study of 23 September followed from it.

### 3b. `lim2_diag.py` (framing diagnostic prototype)

A root-level script, superseded by `scripts/framing_diagnostic.py`. Usage as written:

```
python lim2_diag.py [--no-judge] [--variants PATH] [--predictions PATH]
```

Defaults: `--variants data/processed/redteam_all200_views_variants.jsonl`,
`--predictions data/processed/eval_views_2026-09-22_predictions/harmbench.jsonl`,
`--seeds data/benchmarks/harmbench/test.jsonl`, `--out data/processed/lim2_diag.json`.
Four parts: (A) attribute the residual bypasses of the views-on pass to frame and seed;
(B) score every frame x seed combination (15 jailbreak, 7 role-play and 8 prefix_rewrite
frames x 200 seeds) under the shipped Layer 1 and six candidate extra views (`comma`,
`sentence`, `opener`, `dropk`, `boundary`, `bound+op`); (C) the benign cost of each
candidate on the pool splits and the five benchmarks; (D) the judge on the residual
bypasses. It imported `layer1.detector`, `redteam.attacks` and the config loader from the
old flat package layout. Its outputs (`lim2_diag.json`, `lim2_diag_variants.jsonl`) are not
shipped. The candidate views it tested were not adopted; the shipped view is the
leading-clause view in `src/llmguard/layer1/normalise.py`.

### 3c. `lim4_coverage.sh` and `lim4_cats.py` (seed coverage, limitation 4)

`lim4_coverage.sh` applied patch 38 and ran the two coverage commands listed in the table
above (`redteam_coverage.json`, `redteam_harmbench_only.json`). `lim4_cats.py` was a
twelve-line reader that aggregated `data/processed/redteam_coverage_variants.jsonl` by
`seed_source/seed_category` over variants with intent `verbatim` or `mechanical` and printed
the bypass rate per stratum. Its printed table (`lim4_categories.txt`) and the variants file
are not shipped; the per-source and per-category rates are inside
`results/development/redteam_coverage.json`.

## 4. Layer-3 review scoring (22 September 2026)

Both drivers below scored the 90 served rewrites of red-team pilot 4 against the reviewed
labels, using a fixed mapping from review id to 0-based line in
`red_team_pilot4_variants.jsonl`. The mapping, identical in both:

```
R01=154,R02=128,R03=171,R04=167,R05=250,R06=227,R07=31,R08=164,R09=85,R10=266,
R11=11,R12=35,R13=273,R14=114,R15=91,R16=186,R17=213,R18=80,R19=83,R20=33,
R21=21,R22=147,R23=6,R24=73,R25=103,R26=201,R27=271,R28=96,R29=217,R30=125,
R31=45,R32=196,R33=143,R34=28,R35=95,R36=105,R37=7,R38=238,R39=109,R40=112,
R41=94,R42=99,R43=259,R44=24,R45=3,R46=170,R47=34,R48=39,R49=157,R50=32,
R51=203,R52=87,R53=209,R54=207,R55=156,R56=225,R57=245,R58=150,R59=237,R60=49,
R61=102,R62=74,R63=17,R64=101,R65=90,R66=98,R67=44,R68=142,R69=75,R70=269,
R71=249,R72=202,R73=38,R74=76,R75=264,R76=18,R77=151,R78=166,R79=235,R80=19,
R81=10,R82=200,R83=206,R84=20,R85=199,R86=2,R87=197,R88=30,R89=117,R90=268
```

(passed to `--ids` as one comma-separated string without spaces or line breaks).

**`lim3_stage1.sh`** (patch 36, request-conditioned rewrite check):

```bash
V=$(find . -name "red_team_pilot4_variants.jsonl" | head -1)
sha256sum "$V"
python scripts/rewrite_check_eval.py --variants "$V" --out data/processed/rewrite_check_pilot4.json --ids "$IDS"
```

Commit `db4515ba`, 2026-09-22T14:37Z. Kept: `results/development/rewrite_check_pilot4.json`
and the input `results/development/red_team_pilot4_variants.jsonl`. Result: recall 6 of 7
leaks, AUROC 0.874, but 7 false alarms, so the check stays `off` (decision register,
patches 36 and 37; `config/config.yaml`, `layer3.rewrite_check`).

**`lim6_run.sh`** (patches 43 and 44, model-backed refusal detector):

```bash
V=$(find . -name "red_team_pilot4_variants.jsonl" | head -1)
python scripts/refusal_eval.py --variants "$V" --out data/processed/refusal_eval_pilot4.json --ids "$IDS"
```

Commit `548fb334`, 2026-09-22T16:13Z. Kept: `results/development/refusal_eval_pilot4.json`.
Result: recall 0.95 against the keyword list's 0.34, precision 0.56 against a bar of 0.90, so
the keyword detector stays the default (`layer3.refusal_check.mode: keywords`).

The reviewer label files these two evaluations were read against are private and are not in
the repository; `docs/leak_codebook.md` describes the protocol.

### 4a. `apply41.py` (patch 41 as a direct editor)

A root-level Python script that applied patch 41 by exact string replacement rather than
`git apply`: it taught `scripts/leak_reliability.py` to read a reviewer's `.xlsx` workbook
(sheet `Review`, else the first sheet) as well as CSV, added `openpyxl>=3.1.0` as an optional
requirement, and changed one sentence of `docs/leak_codebook.md`. Usage was
`python apply41.py [REPO_ROOT]`; it exited early if `_rows_from_xlsx` was already present.
The resulting code is what `scripts/leak_reliability.py` contains now; the editor itself is
not shipped.

## 5. Parity and learner-English arms (22 to 23 September 2026)

| Run | Command | Commit | Kept as |
|---|---|---|---|
| Build the three parity arms | `python scripts/make_l2_benchmarks.py --root data/benchmarks_parity` | `b6567449` | `results/development/parity_arms_manifest.json` |
| Score each arm | `python scripts/run_evaluation.py --benchmark-root data/benchmarks_parity/<arm> --output data/processed/parity_<arm>.json` for `original`, `general`, `arabic` | `b6567449` | not shipped (inputs to the summary) |
| Paired summary | `python scripts/summarise_results.py --run original=data/processed/parity_original.json --run general=data/processed/parity_general.json --run arabic=data/processed/parity_arabic.json --compare general:original --compare arabic:original --out data/processed/parity_summary` | `b6567449` | `results/development/parity_summary.{json,md}` |
| Flips and attribution | `scripts/compare_flips.py --original ... --variant ...` and `scripts/parity_attribution.py` | — | `results/development/parity_flips_{general,arabic}.json`, `parity_attr_{general,arabic}.json` (no `run_info` in these files) |
| Build the learner arms | `python scripts/make_learner_arms.py --source data/raw/wi_locness` | `dc90a9e2` | `results/development/learner_arms_manifest.json` |
| Score the learner arms, historical checkpoint | `python scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/original --output data/processed/learner_original.json` and the same with `corrected` | `dc90a9e2` | `results/development/learner_{original,corrected}.json`, `learner_attr.json` |

The W&I+LOCNESS corpus is non-commercial and not redistributed; the learner arms must be
rebuilt from the BEA-2019 release to re-run these commands.

## 6. Three-seed repeat study (23 September 2026)

```bash
python scripts/repeat_study_nciJu1oQ_train.py --seed 42 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s42
python scripts/repeat_study_nciJu1oQ_train.py --seed 43 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s43
python scripts/repeat_study_nciJu1oQ_train.py --seed 44 --artifact-dir artifacts/layer1_repeat_study_nciJu1oQ_s44
python scripts/summarise_results.py --run s42=data/processed/repeat_study_nciJu1oQ/eval_s42.json --run s43=data/processed/repeat_study_nciJu1oQ/eval_s43.json --run s44=data/processed/repeat_study_nciJu1oQ/eval_s44.json --bootstrap 1000 --seed 42 --out data/processed/repeat_study_nciJu1oQ/summary
```

Commit `dc90a9e2`. The three checkpoints are shipped under `artifacts/`; the protocol,
frozen-file hashes, per-seed variability and summary are in `results/repeat_study_nciJu1oQ/`.
`scripts/repeat_study_nciJu1oQ_train.py` is now a shim that forwards to
`scripts/train_layer1.py --seed`; it is kept because these commands are recorded in
manifests.

## 7. Judge substitution tuning (28 September 2026)

```bash
python scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct \
    --output data/processed/judge_tuning_llama.jsonl \
    --summary data/processed/judge_tuning_llama_summary.json
```

Commit `dc90a9e2`, `git_dirty: true`. Kept as `results/development/judge_tuning_llama.jsonl`
and `judge_tuning_llama_summary.json`. This is the validation-split selection of the prompt
version for the post hoc Llama judge arm; see `results/final_eval_20260926/AMENDMENT-2026-09-28b.md`.

## 8. What the development record does not contain

- The commands behind `results/development/learner_attr.json`, `parity_attr_*.json` and
  `parity_flips_*.json` were not recorded in those files (no `run_info`).
- The retrained development checkpoints `layer1_openers`, `layer1_control`, `layer1_v3`,
  `layer1_cap` and `layer1_lowwjb`, and their evaluations, are not shipped.
- The six red-team pilot runs (`red_team_pilot*.json`) that informed the Layer-3 default
  (refuse) are not shipped except `red_team_pilot4_variants.jsonl`, which the two review
  scorers read.
- The evaluation runs on the five development benchmarks (`eval_*.json`) are not shipped;
  their paired summaries are (`thesis_results`, `views_vs_base`, `lim2_openers_vs_base`,
  `ablation_results`, `parity_summary`).
