# Experiments: what maps to what

Tables that connect each experiment, dataset, model component, configuration key,
metric and paper table or figure to the script that produced it, the command, the result
file and the function that computes it. Paths are release paths; the recorded commands
wrote to `data/processed/...`, which is now `results/...` (see `REPRODUCIBILITY.md`,
section 3, for the full recorded command lines). `$IN` is
`data/benchmarks_final/final_test/test.jsonl` and `$OUT` is `results/final_eval_20260926`.

## 1. Experiments

### 1.1 Pre-registered study (26 September 2026)

| Experiment | Script and command | Result file(s) | Analysed in |
|---|---|---|---|
| Final test set construction | `scripts/final_eval/fetch_final_sources.py`; `scripts/final_eval/build_final_test.py` | `data/benchmarks_final/final_test/test.jsonl`, `data/benchmarks_final/manifest.json` (= `$OUT/final_manifest.json`) | `tables.md` "Final test set" |
| Cascade (frozen system) | `scripts/run_evaluation.py --input $IN --output $OUT/final_cascade.json` | `final_cascade.json`, `final_cascade_predictions/test.jsonl` | `tables.md` A; `results.json` → `rates.cascade` |
| Layer 1 alone (gray zone allowed) | `scripts/run_evaluation.py --no-l2 --input $IN --output $OUT/final_layer1.json` | `final_layer1.json` + predictions | A |
| Judge alone, every prompt | `scripts/final_eval/run_judge_only.py --input $IN --output $OUT/final_judge.json` | `final_judge.json` + predictions | A |
| Granite Guardian 3.1 2B | `scripts/final_eval/run_external_guard.py --model granite --input $IN --output $OUT/final_granite.json` | `final_granite.json` + predictions | A, B |
| Llama Guard 3 1B (pre-registered; ran 28 Sep) | `... --model llamaguard ... --revision acf7aafa60f0410f8f42b1fa35e077d705892029` | `final_llamaguard.json` + predictions; `final_llamaguard.skipped.json` (26 Sep skip record); `withdrawn_badstep_20260928/` | A, B |
| Guards on the pool validation split (for matched thresholds) | `run_external_guard.py --model <g> --input data/distillation_pool/val.jsonl --output $OUT/val_<g>.json`; `run_evaluation.py --input data/distillation_pool/val.jsonl --output $OUT/val_cascade.json` | `val_{granite,llamaguard,shieldgemma,cascade}.json` + predictions | B (`results.json` → `matched`) |
| Cascade under the three repeat checkpoints | `run_evaluation.py --layer1-artifacts layer1_repeat_study_nciJu1oQ_s4X --input $IN --output $OUT/final_cascade_s4X.json` | `final_cascade_s{42,43,44}.json` + predictions | A, seed table |
| HarmBench attack pass, four checkpoints | `scripts/run_red_team.py --seeds data/benchmarks/harmbench/test.jsonl --all-seeds --n-per-type 3 --attack-types <six> --payload-views [--layer1-artifacts ...] --output ...` | `redteam_all200_historical.json` (= `results/development/redteam_all200_views.json`), `redteam_all200_s{42,43,44}.json`, each with `_variants.jsonl` | C (`results.json` → `redteam`) |
| Learner English, three checkpoints | `run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/{original,corrected} --layer1-artifacts ... --output ...` | `learner_{original,corrected}_s{42,43,44}.json` + predictions; `sentence_essay_writer_mapping.jsonl` | D (`results.json` → `learner`) |
| Summary with `summarise_results.py` | `scripts/summarise_results.py --run cascade=... --run layer1=... --run judge=... --run granite=... --compare ... --out $OUT/final_summary` | `final_summary.{json,md,csv}` | — (pooled, not per source; superseded by `analyse_final.py`) |

`<six>` = `jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix`.

### 1.2 Post hoc arms and analyses

| Experiment | Script and command | Result file(s) | Amendment |
|---|---|---|---|
| ShieldGemma 2B (final + validation) | `run_external_guard.py --model shieldgemma ... --max-new-tokens 2` | `final_shieldgemma.json`, `val_shieldgemma.json` + predictions | 2026-09-27 |
| Judge model substitution (Llama-3.2-3B-Instruct, v1 primary, v2 secondary) | `scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct --output ... --summary ...`; `run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v1\|v2 --input $IN --output ...` | `results/development/judge_tuning_llama.jsonl`, `judge_tuning_llama_summary.json`; `final_judge_llama_v{1,2}.json` + predictions | 2026-09-28b |
| Component ablation, 16 arms | `run_evaluation.py {--no-payload-views \| --no-normalise-input \| both \| --layer1-artifacts layer1_T1} [--layer1-artifacts layer1_repeat_study_nciJu1oQ_s4X[_T1]] --input $IN --output ...` | `final_cascade_{noviews,nonorm,bare,T1}[_s42\|_s43\|_s44].json` + predictions | 2026-09-30 |
| Layer 1 on the pool test split | `run_evaluation.py --no-l2 --input data/distillation_pool/test.jsonl --output $OUT/pooltest_layer1.json` | `pooltest_layer1.json` + predictions | 2026-10-01 |
| Main analysis | `scripts/final_eval/analyse_final.py --study results/final_eval_20260926` | `analysis/results.json`, `analysis/tables.md` | — |
| Whole study in recorded order (reconstructed runner) | `scripts/final_eval/run_final_eval.sh [--dry-run] [--only STAGE]`; freeze record `scripts/final_eval/freeze.py --study S`; check `scripts/final_eval/verify_frozen.py`; T = 1 checkpoints `scripts/make_t1_variants.py artifacts/<ckpt>` | `frozen.json`, `checkpoints.json`, every arm file | `REPRODUCIBILITY.md` §8 |
| Operating-point frontiers and band sweep | `scripts/final_eval/frontiers.py --study results/final_eval_20260926` | `analysis/frontiers.json` | 2026-10-01 §1 |
| Calibration (ECE, reliability bins) | `scripts/final_eval/calibration.py --study results/final_eval_20260926 --dev results/development` | `analysis/calibration.json` | 2026-10-01 §2 |
| Rates by category | `scripts/final_eval/per_category.py --study results/final_eval_20260926` | `analysis/per_category.{json,md}` | 2026-10-01 §3 |
| Native verdict vs harm score, per guard | `scripts/final_eval/guard_agreement.py --study results/final_eval_20260926` | `analysis/guard_agreement.json` | 2026-10-01 §4 |
| Judge band and uncertain policy | `scripts/final_eval/judge_band.py --study results/final_eval_20260926` | `analysis/judge_band.json` | 2026-10-01 §5 |
| Paired effect of the payload view | `scripts/final_eval/paired_views.py --out results/final_eval_20260926/analysis/paired_views.json` (reads `results/development/redteam_all200_variants.jsonl` and `redteam_all200_views_variants.jsonl`) | `analysis/paired_views.json` | — |

### 1.3 Development experiments

Commands in `docs/history/development_runs.md`.

| Experiment | Script | Result file(s) in `results/development/` |
|---|---|---|
| Pool and benchmark build | `scripts/fetch_datasets.py`, `scripts/build_datasets.py` | `raw_sources_manifest.json`, `pool_metadata.json`, `pool_build_2026-09-15.log` |
| Judge prompt v1 vs v2, 300 validation prompts | `scripts/tune_judge.py --resume` | `judge_tuning.jsonl`, `judge_tuning_summary.json` |
| Layer-1 component ablation on the benchmarks | `scripts/run_ablation.py` (`--stage signals`, `--stage derive`) then `scripts/summarise_results.py` | `ablation_results.{json,md}` |
| Routing-band sweep on the validation split | `scripts/sweep_routing.py --select data/processed/ablation_val_signals.jsonl --out data/processed/routing_sweep` | `routing_sweep.{csv,json,md}` |
| Red team views off / on | `scripts/run_red_team.py` (see 1.1) | `redteam_all200.json`, `redteam_all200_views.json` + `_variants.jsonl` |
| Benchmarks, views vs default | `scripts/summarise_results.py --compare views:base` | `views_vs_base.{json,md}` |
| Red-team seed coverage | `run_red_team.py --seed-benchmarks harmbench jailbreakbench orbench toxicchat --per-category 12 ...` and `--seed-benchmarks harmbench` | `redteam_coverage.json`, `redteam_harmbench_only.json` |
| Opener augmentation (rejected) | `lim2_runs.sh` → `train_layer1.py --augment-openers`, `run_evaluation.py`, `run_red_team.py`, `summarise_results.py` | `lim2_openers_vs_base.{json,md}` |
| Rewrite check (not adopted) | `scripts/rewrite_check_eval.py --variants ... --ids R01=154,...` | `rewrite_check_pilot4.json`, input `red_team_pilot4_variants.jsonl` |
| Model-backed refusal detector (not adopted) | `scripts/refusal_eval.py --variants ... --ids R01=154,...` | `refusal_eval_pilot4.json` |
| Grammatical-profile parity arms | `scripts/make_l2_benchmarks.py --root data/benchmarks_parity`; `run_evaluation.py --benchmark-root ...`; `summarise_results.py`; `compare_flips.py`; `parity_attribution.py` | `parity_arms_manifest.json`, `parity_summary.{json,md}`, `parity_flips_*.json`, `parity_attr_*.json` |
| Learner arms, historical checkpoint | `scripts/make_learner_arms.py --source data/raw/wi_locness`; `run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root ...` | `learner_arms_manifest.json`, `learner_{original,corrected}.json`, `learner_attr.json` |
| Thesis tables (Layer 1 vs pipeline on the five benchmarks) | `scripts/summarise_results.py --compare pipeline:layer1` | `thesis_results.{json,md}` |
| Three-seed repeat study | `scripts/repeat_study_nciJu1oQ_train.py --seed {42,43,44} --artifact-dir ...`; `summarise_results.py --bootstrap 1000 --seed 42` | `results/repeat_study_nciJu1oQ/{protocol,frozen_files,seed_variability,summary}.json`, `summary.md`; checkpoints under `artifacts/` |

## 2. Datasets

| Dataset | Role | Fetch / build script | Manifest | Shipped? |
|---|---|---|---|---|
| Pool (WildJailbreak vanilla, BeaverTails, Dolly, ToxicChat train): 15,296 / 1,912 / 1,912 | Layer-1 training, validation (selection set), test (in-distribution) | `scripts/fetch_datasets.py`; `scripts/build_datasets.py` | `results/development/raw_sources_manifest.json`, `results/development/pool_metadata.json`; SHA-256 in `results/final_eval_20260926/frozen.json` → `pool` | No (rebuild) |
| Development benchmarks: JailbreakBench 200, XSTest 450, HarmBench 200, ToxicChat test 4,951, OR-Bench 1,974 | Reporting set, also used for selection (`docs/validation_policy.md`) | `scripts/build_datasets.py` | `raw_sources_manifest.json`; SHA-256 in `frozen.json` → `inputs` | No (rebuild) |
| Final test set: SST 99, CatQA 543, DNA core 573, DNA other 364, CoCoNot contrast 378, PHTest harmless 1,000, No Robots test 496 = 3,453 | Pre-registered held-out evaluation | `scripts/final_eval/fetch_final_sources.py`; `scripts/final_eval/build_final_test.py` | `data/benchmarks_final/manifest.json` (stage counts, rules, source revisions, SHA-256) | Yes |
| HarmBench seeds for the red team (200 harmful) | Attack seeds | `data/benchmarks/harmbench/test.jsonl` from `build_datasets.py` | `frozen.json` → `inputs.harmbench` | No (rebuild) |
| Learner arms: W&I+LOCNESS v2.1, `original` and `corrected`, levels A/B/C + LOCNESS N | Grammatical-form sensitivity (RQ4, second study) | `scripts/make_learner_arms.py --source <corpus dir>` | `results/development/learner_arms_manifest.json`; `frozen.json` → `inputs.learner_*` | No (non-commercial licence) |
| Parity arms: safe prompts of the development benchmarks rewritten under two grammatical profiles | RQ4, first study | `scripts/make_l2_benchmarks.py --root data/benchmarks_parity` | `results/development/parity_arms_manifest.json` | No (rebuild) |

## 3. Model components

| Component | Implementation | Artifacts / weights |
|---|---|---|
| Layer-1 detector (orchestrates signals, normalisation, views, routing) | `src/llmguard/layer1/detector.py` → `Layer1Detector`, `DetectionResult`, `layer1_artifact_dir` | `artifacts/<layer1.artifact_subdir>/` |
| Distilled classifier `g(x)` = P(allow) | `src/llmguard/layer1/classifier.py` → `build_lora_classifier`, `ClassifierTrainer`, `PromptDataset`; `features.py` → `ClassifierFeature` | `artifacts/layer1/classifier/checkpoint_best/` (LoRA adapter on `roberta-base`) |
| Curvature and log-rank signals (three-signal stacker only) | `src/llmguard/layer1/features.py` → `CurvatureFeature`, `LogRankFeature` (`distilgpt2`) | — |
| Logistic stackers | `src/llmguard/layer1/fusion.py` → `LogisticFusion` (three signals), `ClassifierOnlyFusion` | `fusion_v2.joblib`, `fusion_clf_only.joblib` |
| Temperature scaling | `fusion.py` → `TemperatureScaler` | `scaler.json`, `scaler_clf_only.json` |
| Threshold router (allow / gray_zone / mitigate) | `fusion.py` → `ThresholdRouter` | `layer1.thresholds` |
| Input normalisation (base64, leetspeak) and leading-clause payload views | `src/llmguard/layer1/normalise.py` → `normalise`, `payloads`, `leading_clause_views` | — |
| Opener augmentation (rejected, kept for the record) | `src/llmguard/layer1/augment.py` | — |
| Layer-2 judge | `src/llmguard/layer2/layer2_judge.py` → `Layer2Judge`; `judge.py` → `LLMJudge` (prompt versions `v1`, `v2`, logprob decision); `consistency.py` → `ConsistencyChecker`, `JudgeResult` (vote mode, not default); `distillation.py` → `DistillationBuffer` (nothing consumes it) | `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 |
| Layer-3 mitigator | `src/llmguard/layer3/layer3_mitigator.py` → `Layer3Mitigator`; `mitigator.py` → `ModeSelector`, `RefuseHandler`, `RedactHandler`; `guard_llm.py` → `GuardLLM` (safe_rewrite); `constraints.py` → `ConstraintEngine` (four regexes from `config/safety_templates.yaml`); `fulfilment.py` → `FulfilmentCheck` (rewrite check, off) | shares the judge's weights when both name the same checkpoint |
| Refusal detection | `src/llmguard/utils/refusal.py` → `looks_like_refusal` (keywords, default), `RefusalCheck` (model-backed, not adopted) | — |
| Pipeline evaluator | `src/llmguard/evaluation/pipeline_evaluator.py` → `PipelineEvaluator`, `EvalResult` | — |
| External guards | `scripts/final_eval/run_external_guard.py` → `score_one`, `answer_step`, `parse_native`, `build_input_ids` | Granite, Llama Guard, ShieldGemma at the pinned revisions |
| Judge-only arm | `scripts/final_eval/run_judge_only.py` → `decide` | — |
| Attacks | `src/llmguard/redteam/attacks.py` → `jailbreak_template`, `role_play`, `prefix_rewrite`, `language_prefix`, `decoding_manipulation`, `technical_prefix` (template, used), `model_paraphrase`, `model_translation` (model-written, not used in reported runs), `generate_attacks`, `resolve_attack_type`; `red_teamer.py` → `RedTeamer`, `RedTeamReport`, `RedTeamResult` | — |
| Grammatical-profile rewrites (parity arms) | `src/llmguard/evaluation/l2_variants.py` | — |
| Provenance | `src/llmguard/utils/run_info.py` → `collect_run_info`, `finalize_run_info`, `warn_if_overwriting` | — |
| Audit log and NIST tags | `src/llmguard/utils/logger.py` → `AuditLogger`, `text_sha256`; `config/nist_ai_rmf_mapping.yaml` | — |

## 4. Configuration keys (`config/config.yaml`)

| Setting | Key | Value in the frozen system |
|---|---|---|
| Project seed | `project.seed` | 42 |
| Device policy | `project.device`, `project.fp16` | `auto` (CUDA or stop; `LLMGUARD_ALLOW_CPU=1` allows CPU), fp16 on |
| Classifier-only stacker | `layer1.classifier_only` | `true` |
| Input normalisation | `layer1.normalise_input` | `true` |
| Leading-clause payload view | `layer1.payload_views` | `true` |
| Layer-1 artifact directory | `layer1.artifact_subdir` | `layer1` |
| Classifier backbone and LoRA | `layer1.classifier.{backbone,lora_r,lora_alpha,lora_dropout,max_length}` | `roberta-base`, 8, 16, 0.1, 512 |
| Classifier training | `layer1.classifier.{batch_size,learning_rate,epochs,early_stopping_patience}` | 8, 2e-4, 5, 2 |
| Curvature signal | `layer1.curvature.{num_perturbations,temperature_range,top_p_range}` | 10, [0.5, 1.5], [0.8, 1.0] |
| Fusion | `layer1.fusion.method` | `logistic` |
| Routing band | `layer1.thresholds.{low,high}` | 0.30, 0.70 |
| Judge model and loading | `layer2.judge.{model_name,load_in_4bit,max_tokens,temperature,prompt_version}` | Qwen2.5-3B-Instruct, true, 512, 0.0, `v2` |
| Judge decision | `layer2.consistency.{mode,allow_below,mitigate_above}` | `logprob`, 0.40, 0.60 |
| Vote mode (not used) | `layer2.consistency.{K,sample_temperature,c_min}` | 5, 0.7, 0.8 |
| Distillation buffer (unused output) | `layer2.distillation.{retrain_every_n,pool_path}` | 1000, `data/distillation_pool/` |
| Layer-3 default mode | `layer3.mitigation.default_mode` | `refuse` |
| Severity refuse threshold | `layer3.mitigation.severity_refuse_threshold` | 0.99 |
| Rewrite candidates, decline veto | `layer3.candidates.{K_prime,refuse_on_decline}` | 3, true |
| Refusal detector | `layer3.refusal_check.{mode,threshold}` | `keywords`, 0.5 |
| Rewrite check | `layer3.rewrite_check.{mode,threshold}` | `off`, 0.5 |
| Constraint templates | `layer3.constraints.templates_path` | `config/safety_templates.yaml` |
| Development benchmarks | `evaluation.benchmarks[]` | five enabled; four learner arms disabled (run by name) |
| Bootstrap for `summarise_results.py` | `evaluation.fairness.{bootstrap_samples,alpha}` | 1000, 0.05 |
| Red-team attack types | `red_team.attack_types` | eight listed; the six template types are passed explicitly in every recorded run |
| Audit log and NIST tagging | `logging.{audit_log_path,nist_mapping,nist_mapping_path}` | `logs/audit.jsonl`, true, `config/nist_ai_rmf_mapping.yaml` |

Command-line flags that override these (`run_evaluation.py`, `run_red_team.py`):
`--classifier-only` / `--full-stacker`, `--payload-views` / `--no-payload-views`,
`--normalise-input` / `--no-normalise-input`, `--layer1-artifacts SUBDIR`, `--no-l2`,
`--no-l3`, `--guard-llm`, `--l3-mode`, `--rewrite-check`. Absent a flag, the config value
applies.

## 5. Evaluation metrics

| Metric | Definition | Function |
|---|---|---|
| safe_blocked | share of safe prompts (label 1) not allowed; an escalated prompt counts as blocked | `scripts/summarise_results.py` → `metric_values` (`blocked = (p == 0) \| esc`); `scripts/final_eval/analyse_final.py` → `blocked`, `rate_table` |
| harmful_allowed (= jailbreak success rate) | share of harmful prompts (label 0) allowed | `src/llmguard/evaluation/metrics.py` → `jailbreak_success_rate`; `analyse_final.py` → `rate_table` |
| Wilson 95% interval on a single rate | score interval, z = 1.959964 | `analyse_final.py` → `wilson`; `per_category.py` |
| Paired difference between arms | percentile bootstrap of the rate difference, prompts resampled within source, B = 2000, seed 20260926, named stream | `analyse_final.py` → `paired_diff`, `stream_rng` |
| McNemar test | exact two-sided binomial test on discordant decisions | `analyse_final.py` → `mcnemar_exact` |
| Pooled AUROC and paired AUROC difference | harm score vs harmful label over the 3,089 primary prompts | `analyse_final.py` → `auroc`, `pooled_auroc`, `auroc_diff`; `metrics.py` → `auroc` |
| AUPRC, FPR at 95% TPR, accuracy, precision/recall/F1 | standard, on harm score / decisions | `metrics.py` → `auprc`, `fpr_at_tpr`, `accuracy`, `precision_recall_f1`, `compute_all` |
| ECE | 10 equal-width bins, \|mean score − fraction harmful\| weighted by bin share | `metrics.py` → `ece`; `scripts/final_eval/calibration.py` → `ece`, `reliability`, `boot_ece` |
| Matched guard threshold | smallest P(unsafe) threshold whose validation safe-blocked rate does not exceed the cascade's (3.8%) | `analyse_final.py` → `matched_threshold`, `apply_threshold`, `safe_blocked_rate` |
| ROC read-out at the cascade's safe-blocked rate (descriptive) | | `analyse_final.py` → `roc_readout` |
| Red-team bypass rate and seed-clustered interval | bypass = allowed or uncaught error; cluster bootstrap over the 200 seeds | `analyse_final.py` → `redteam_summary`; `paired_views.py` |
| Learner non-inferiority | original − corrected blocking among edited pairs; sentence-, essay- and writer-grouped bootstraps; margin +2 points | `analyse_final.py` → `learner_pairs`, `grouped_boot`, `learner_summary` (`MARGIN_POINTS = 2.0`) |
| Benchmark-level bootstrap intervals (development) | 1000 stratified resamples, seed 42 | `summarise_results.py` → `summarise_benchmark`, `stratified_indices`, `interval` |
| Per-prompt checks against the stored results | recomputed metrics must agree with the results file | `summarise_results.py` → `check_saved` |

## 6. Paper tables and figures

Every table and figure of the article is drawn from the files below; none is drawn from a
model run at writing time. `tables.md` is `results/final_eval_20260926/analysis/tables.md`.

| Paper item | Content | Generating script | Result file | `tables.md` section |
|---|---|---|---|---|
| Table I | Composition of the final test set | `scripts/final_eval/build_final_test.py` | `data/benchmarks_final/manifest.json` → `n_by_source` | "Final test set" |
| Table II | Layer-1 stackers on the pool test split | `scripts/plot_layer1.py` (classifier-only column from the scored feature cache); `artifacts/layer1/results.json` (three-signal column) | no stored file for the classifier-only column; `pooltest_layer1.json` is the nearest stored run | — |
| Table III | Operational rates by source, six arms | `analyse_final.py` | `analysis/results.json` → `rates` | A |
| Table IV | External guards at the matched threshold | `analyse_final.py` | `results.json` → `matched` | B |
| Table V | Pooled AUROC and paired differences | `analyse_final.py` | `results.json` → `pooled_auroc`, `auroc_diffs` | A (AUROC tables) |
| Table VI | Judge slot under two models | `analyse_final.py` | `results.json` → `rates.judge_llama_v1`, `rates.judge_llama_v2`, `paired_vs_cascade` | A, "Paired differences" |
| Table VII | ECE in and out of distribution | `scripts/final_eval/calibration.py --study results/final_eval_20260926 --dev results/development` | `analysis/calibration.json` → `ece` | — |
| Table VIII | Rates by category | `scripts/final_eval/per_category.py --study results/final_eval_20260926` | `analysis/per_category.{json,md}` | — |
| Table IX | Judge band and uncertain policy | `scripts/final_eval/judge_band.py --study results/final_eval_20260926` | `analysis/judge_band.json` | — |
| Table X | Seed sensitivity on the final set | `analyse_final.py` | `results.json` → `seed_range` | "Cascade on the final set across training seeds" |
| Table XI | HarmBench attack pass by checkpoint | `analyse_final.py` | `results.json` → `redteam` | C |
| Table XII | Component ablation | `analyse_final.py` | `results.json` → `ablation` | E |
| Table XIII | Routing-band alternatives on the validation split | `scripts/sweep_routing.py` | `results/development/routing_sweep.md` | — (`docs/validation_policy.md`) |
| Fig. operating points (PHTest vs CatQA) | native points of every arm | drawn from Table III values | `results.json` → `rates` | A |
| Fig. frontiers and band sweep | threshold sweeps, macro rates; band half-width sweep | `scripts/final_eval/frontiers.py --study results/final_eval_20260926` | `analysis/frontiers.json` → `frontiers`, `native`, `band_sweep` | — |
| Fig. Layer-1 score distribution | `l1_score` by label | from predictions | `final_cascade_predictions/test.jsonl` | — |
| Fig. reliability diagrams | ten-bin reliability of Layer 1 and the judge | `calibration.py` | `analysis/calibration.json` → `reliability` | — |
| Fig. payload view, paired | bypass per attack type, views off vs on | `scripts/final_eval/paired_views.py` | `analysis/paired_views.json`; `results/development/redteam_all200{,_views}.json` | C |
| Learner table (Section on grammatical-form sensitivity) | original − corrected blocking | `analyse_final.py` | `results.json` → `learner` | D |
| Guard-agreement paragraph | native vs score agreement, answer mass | `scripts/final_eval/guard_agreement.py --study results/final_eval_20260926` | `analysis/guard_agreement.json` | — |

## 7. README figures (`figures/`)

Documented in `figures/README.md`; the source of each PNG is listed in
`figures/manifest.json` with its SHA-256. In brief: the `layer1_*`, `pipeline_flow*` and
`redteam_*` PNGs were drawn by `scripts/plot_evaluation.py` from `artifacts/layer1/results.json`,
`results/final_eval_20260926/final_cascade.json` and its predictions, and
`results/development/redteam_all200_views.json`; the `layer2_*` and `layer3_*` PNGs come from
legacy May-2026 standalone runs that the paper does not use; two images
(`layer2_self_consistency.png`, `layer3_severity_histogram.png`) are not in the manifest and
must not be used. `figures/layer1/*.png` come from `scripts/plot_layer1.py` on the historical
checkpoint.
