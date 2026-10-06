#!/usr/bin/env bash
# Reconstructed runner for the pre-registered final evaluation (results/final_eval_20260926).
#
# The original runner was never committed; this file replays, in the recorded order and with the
# recorded arguments, every command found in the run_info blocks of the stored result files
# (started_at in UTC is given per stage; see docs/final_evaluation_runbook.md). Stages after
# "26 Sep" are the post hoc arms, each recorded in a dated AMENDMENT file.
#
# Usage
#   PYTHON=python STUDY=results/my_rerun bash scripts/final_eval/run_final_eval.sh --dry-run
#   bash scripts/final_eval/run_final_eval.sh --only analysis        # one stage
#   FORCE=1 bash scripts/final_eval/run_final_eval.sh                 # overwrite an existing STUDY
# Stages: fetch build freeze arms val judge guards seeds redteam learner summarise shieldgemma
#         llamaguard llamajudge ablation pooltest analysis
#
# Requirements per stage are stated inline. Scoring arms need the GPU (RTX 4050 6 GB in the
# study; 4-bit NF4 for the judge and guards), the Hugging Face models at the pinned revisions in
# config/config.yaml and the recorded commands, the gated repositories where marked, the rebuilt
# pool and benchmarks (data/README.md) and, for the learner arms, data/benchmarks_learner built
# from your own copy of W&I+LOCNESS. Each arm took minutes to ~35 min on the recorded GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
STUDY="${STUDY:-results/final_eval_20260926}"
FINAL="data/benchmarks_final/final_test/test.jsonl"
VAL="data/distillation_pool/val.jsonl"
POOLTEST="data/distillation_pool/test.jsonl"
HB="data/benchmarks/harmbench/test.jsonl"
ATK="jailbreak_template role_play prefix_rewrite language_prefix decoding_manipulation technical_prefix"
DRY=0; ONLY=""
for arg in "$@"; do case "$arg" in
  --dry-run) DRY=1 ;;
  --only=*) ONLY="${arg#--only=}" ;;
  --only) shift_next=1 ;;
  *) if [ "${shift_next:-0}" = 1 ]; then ONLY="$arg"; shift_next=0; else echo "unknown argument $arg" >&2; exit 2; fi ;;
esac; done
run() { if [ "$DRY" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
stage() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
if [ -e "$STUDY/final_cascade.json" ] && [ "$DRY" = 0 ] && [ "${FORCE:-0}" != 1 ] && [ -z "$ONLY" ]; then
  echo "$STUDY already holds results; set FORCE=1 to overwrite or use --only <stage>" >&2; exit 1
fi
mkdir -p "$STUDY/logs"

# ---- 26 Sep 2026, 06:35 UTC: sources and the final test set (pre-registered) --------------------
if stage fetch; then   # network; the six source repos are not gated; pins the recorded revisions
  run $PY scripts/final_eval/fetch_final_sources.py --revision-manifest results/development/raw_final_sources_manifest.json
fi
if stage build; then   # needs data/distillation_pool and data/benchmarks (rebuilt locally) for the near-duplicate screen
  run $PY scripts/final_eval/build_final_test.py --require-tokenizer
  run cp data/benchmarks_final/final_test/test.jsonl "$STUDY/final_test.jsonl"
  run cp data/benchmarks_final/manifest.json "$STUDY/final_manifest.json"
fi
if stage freeze; then  # written once, before any arm is scored
  run $PY scripts/final_eval/freeze.py --study "$STUDY"
fi

# ---- 26 Sep 2026, 06:39-07:28 UTC: pre-registered arms -----------------------------------------
if stage arms; then
  run $PY scripts/run_evaluation.py --input "$FINAL" --output "$STUDY/final_cascade.json"          # 06:39
  run $PY scripts/run_evaluation.py --no-l2 --input "$FINAL" --output "$STUDY/final_layer1.json"   # 06:42
fi
if stage judge; then
  run $PY scripts/final_eval/run_judge_only.py --input "$FINAL" --output "$STUDY/final_judge.json"  # 06:43, ~34 min
fi
if stage val; then
  run $PY scripts/run_evaluation.py --input "$VAL" --output "$STUDY/val_cascade.json"               # 07:16
fi
if stage guards; then  # ibm-granite/granite-guardian-3.1-2b (open); Llama Guard is in its own stage below
  run $PY scripts/final_eval/run_external_guard.py --model granite --input "$FINAL" --output "$STUDY/final_granite.json"  # 07:17
  run $PY scripts/final_eval/run_external_guard.py --model granite --input "$VAL" --output "$STUDY/val_granite.json"      # 07:28
fi

# ---- 26 Sep 2026, 07:32-08:22 UTC: the three repeat-study checkpoints (Q3) ----------------------
for S in 42 43 44; do
  CK="layer1_repeat_study_nciJu1oQ_s$S"
  if stage seeds; then
    run $PY scripts/run_evaluation.py --layer1-artifacts "$CK" --input "$FINAL" --output "$STUDY/final_cascade_s$S.json"
  fi
  if stage redteam; then   # HarmBench seeds (gated dataset); 200 x 6 x 3 = 3,600 variants; ~1.5 min each on the study GPU
    run $PY scripts/run_red_team.py --seeds "$HB" --all-seeds --n-per-type 3 --attack-types $ATK --payload-views --layer1-artifacts "$CK" --output "$STUDY/redteam_all200_s$S.json"
  fi
  if stage learner; then   # needs data/benchmarks_learner (W&I+LOCNESS, non-redistributable)
    run $PY scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/original  --layer1-artifacts "$CK" --output "$STUDY/learner_original_s$S.json"
    run $PY scripts/run_evaluation.py --benchmarks wi_a wi_b wi_c locness_n --benchmark-root data/benchmarks_learner/corrected --layer1-artifacts "$CK" --output "$STUDY/learner_corrected_s$S.json"
  fi
done
# The historical (configured-checkpoint) attack pass was run on 22 Sep 2026 at commit 6dead1c4
# and copied into the study as redteam_all200_historical.json/_variants.jsonl:
if stage redteam; then
  run cp results/development/redteam_all200_views.json          "$STUDY/redteam_all200_historical.json"
  run cp results/development/redteam_all200_views_variants.jsonl "$STUDY/redteam_all200_historical_variants.jsonl"
fi
if stage summarise; then  # 08:29; superseded by analyse_final.py (kept in <study>/superseded/)
  run $PY scripts/summarise_results.py --run cascade="$STUDY/final_cascade.json" --run layer1="$STUDY/final_layer1.json" --run judge="$STUDY/final_judge.json" --compare cascade:layer1 --compare cascade:judge --run granite="$STUDY/final_granite.json" --compare cascade:granite --out "$STUDY/final_summary"
fi

# ---- post hoc arms (each recorded in an AMENDMENT file) ----------------------------------------
if stage shieldgemma; then   # 27 Sep, google/shieldgemma-2b (gated, Gemma terms); AMENDMENT-2026-09-27
  run $PY scripts/final_eval/run_external_guard.py --model shieldgemma --input "$FINAL" --output "$STUDY/final_shieldgemma.json" --max-new-tokens 2
  run $PY scripts/final_eval/run_external_guard.py --model shieldgemma --input "$VAL" --output "$STUDY/val_shieldgemma.json" --benchmark val --max-new-tokens 2
fi
if stage llamaguard; then    # 28 Sep, meta-llama/Llama-Guard-3-1B (gated, manual approval); AMENDMENT-2026-09-28
  run $PY scripts/final_eval/run_external_guard.py --model llamaguard --input "$FINAL" --output "$STUDY/final_llamaguard.json" --revision acf7aafa60f0410f8f42b1fa35e077d705892029
  run $PY scripts/final_eval/run_external_guard.py --model llamaguard --input "$VAL" --output "$STUDY/val_llamaguard.json" --revision acf7aafa60f0410f8f42b1fa35e077d705892029
fi
if stage llamajudge; then    # 28 Sep, meta-llama/Llama-3.2-3B-Instruct (gated); AMENDMENT-2026-09-28b
  run $PY scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct --output results/development/judge_tuning_llama.jsonl --summary results/development/judge_tuning_llama_summary.json
  run $PY scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v1 --input "$FINAL" --output "$STUDY/final_judge_llama_v1.json"
  run $PY scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct --prompt-version v2 --input "$FINAL" --output "$STUDY/final_judge_llama_v2.json"
fi
if stage ablation; then      # 30 Sep, 16 arms; AMENDMENT-2026-09-30. T1 checkpoints are copies with T=1.0 scalers.
  for CK in layer1 layer1_repeat_study_nciJu1oQ_s42 layer1_repeat_study_nciJu1oQ_s43 layer1_repeat_study_nciJu1oQ_s44; do
    [ -d "artifacts/${CK}_T1" ] || run $PY scripts/make_t1_variants.py "artifacts/$CK"
  done
  for S in "" _s42 _s43 _s44; do
    CKARG=(); [ -n "$S" ] && CKARG=(--layer1-artifacts "layer1_repeat_study_nciJu1oQ${S#_}")
    T1="layer1${S:+_repeat_study_nciJu1oQ$S}_T1"
    run $PY scripts/run_evaluation.py --no-payload-views "${CKARG[@]}" --input "$FINAL" --output "$STUDY/final_cascade_noviews$S.json"
    run $PY scripts/run_evaluation.py --no-normalise-input "${CKARG[@]}" --input "$FINAL" --output "$STUDY/final_cascade_nonorm$S.json"
    run $PY scripts/run_evaluation.py --no-payload-views --no-normalise-input "${CKARG[@]}" --input "$FINAL" --output "$STUDY/final_cascade_bare$S.json"
    run $PY scripts/run_evaluation.py --layer1-artifacts "$T1" --input "$FINAL" --output "$STUDY/final_cascade_T1$S.json"
  done
fi
if stage pooltest; then      # 1 Oct, Layer 1 on the pool test split for the in-distribution calibration reference; AMENDMENT-2026-10-01
  run $PY scripts/run_evaluation.py --no-l2 --input "$POOLTEST" --output "$STUDY/pooltest_layer1.json"
fi

# ---- analysis: no model is loaded; everything is recomputed from the stored predictions ---------
if stage analysis; then
  run $PY scripts/final_eval/analyse_final.py --study "$STUDY"
  run $PY scripts/final_eval/frontiers.py --study "$STUDY"
  run $PY scripts/final_eval/calibration.py --study "$STUDY" --dev results/development
  run $PY scripts/final_eval/per_category.py --study "$STUDY"
  run $PY scripts/final_eval/guard_agreement.py --study "$STUDY"
  run $PY scripts/final_eval/judge_band.py --study "$STUDY"
  run $PY scripts/final_eval/paired_views.py --out "$STUDY/analysis/paired_views.json"
fi
echo "done: $STUDY"
