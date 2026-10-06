# Figures

Two sets of PNGs are kept here. Neither is a figure of the paper: the paper's figures are
drawn in TikZ from the numbers in `results/final_eval_20260926/analysis/` (see
`EXPERIMENTS.md`, section "Paper tables and figures"). These images are the README and
report figures of the development phase, kept because the old README and the thesis refer
to them.

## `figures/*.png`: evaluation plots

Copied byte-for-byte from `artifacts/evaluation_plots/` in the original workspace.
`figures/manifest.json` is that directory's manifest, written by `scripts/plot_evaluation.py`
on 27 September 2026: for every figure it lists the source file(s) and their SHA-256, and it
states the rule that every plotted value is read from a results file, a predictions file or
`config/config.yaml`. The source paths in the manifest are the original workspace paths;
their release locations are given below.

| Figure | Source (per `manifest.json`) | Release location of the source | What it shows |
|---|---|---|---|
| `layer1_training_curves.png` | `artifacts/layer1/results.json` | same path | Epoch train/validation loss and validation accuracy of the historical training run (15 Sep 2026) |
| `layer1_test_metrics.png` | `artifacts/layer1/results.json` | same path | Pool-test metrics and temperature as recorded; note the recorded row is the three-signal stacker (AUROC 0.9892, T = 1.1891), not the shipped classifier-only one |
| `layer1_score_distribution.png` | `data/processed/final_eval_20260926/final_cascade_predictions/test.jsonl` | `results/final_eval_20260926/final_cascade_predictions/test.jsonl` | Calibrated Layer-1 score by true label on the final test set, with the 0.30/0.70 band |
| `layer1_routing_distribution.png` | `data/processed/final_eval_20260926/final_cascade.json` | `results/final_eval_20260926/final_cascade.json` | Allow / gray-zone / mitigate counts (1,733 / 97 / 1,623) from `layer_stats` |
| `layer2_confidence.png` | `final_cascade_predictions/test.jsonl` (as above) | as above | `l2_confidence` histogram over the 97 prompts that reached the judge; one call per prompt, so there is no vote unanimity to show |
| `pipeline_flow.png` | `final_cascade.json` | as above | Flow diagram with every count read from `layer_stats` |
| `pipeline_flow_with_redteam.png` | `final_cascade.json` + `data/processed/redteam_all200_views.json` | as above + `results/development/redteam_all200_views.json` | The same with the red-team pass attached |
| `redteam_bypass_rates.png` | `data/processed/redteam_all200_views.json` | `results/development/redteam_all200_views.json` | Per-attack bypass rates of the historical views-on pass (4.6% overall) |
| `redteam_attack_volume.png` | same | same | Row and distinct-text counts per attack type |
| `layer2_decisions.png` | `artifacts/layer2/results.json` | **not shipped** | Decision counts of a legacy standalone Layer-2 run |
| `layer2_categories.png` | `artifacts/layer2/results.json` | **not shipped** | Top categories of the same legacy run |
| `layer3_overview.png` | `artifacts/layer3/results.json` | **not shipped** | Mode breakdown of a legacy standalone Layer-3 run |
| `layer3_categories.png` | `artifacts/layer3/results.json` | **not shipped** | Top categories of the same legacy run |

### The `layer2_*` and `layer3_*` figures are legacy

The four figures drawn from `artifacts/layer2/results.json` and `artifacts/layer3/results.json`
describe standalone runs from **May 2026** (the two results files are dated 10 May 2026 and
carry no `run_info`): a Layer-2 pass over 200 gray-zone prompts from an early Layer 1
(`data/processed/layer1_gray_zone.jsonl`, K = 5 vote mode, `c_min` 0.6, a configuration the
shipped system does not use) and a Layer-3 pass over the 94 prompts that run flagged. They
predate the rebuilt pool, the classifier-only stacker, the logprob judge and the refusal
default, were not used in the paper, and are not comparable to anything in
`results/final_eval_20260926/`. They are kept because the thesis reproduced them.

### Two withdrawn images, not included

The original workspace also held `layer2_self_consistency.png` and `layer3_severity_histogram.png`
under `artifacts/evaluation_plots/withdrawn_fabricated/`, whose README records that their
contents were not measured (a vote-unanimity split that was a literal in an earlier version of
`scripts/plot_evaluation.py`, and an "auto-refuse threshold = 0.85" annotation that does not
exist in the configuration). They are deliberately not part of this release and are cited
nowhere in the repository or the paper.

## `figures/layer1/*.png`: Layer-1 diagnostic plots

Produced by `scripts/plot_layer1.py`, which names the stacker in every filename and title.

| Suffix | Stacker | Drawn from (original workspace) | Date |
|---|---|---|---|
| `*_clf_only.png` (6 files) | classifier-only, the shipped configuration | `artifacts/layer1/plots/` | 27 Sep 2026 |
| no suffix (6 files) | three-signal `fusion_v2` | `artifacts/layer1_v2/plots/` | 18 Sep 2026 |

`artifacts/layer1_v2` in the original workspace holds the same checkpoint as the shipped
`artifacts/layer1` (identical `results.json`, LoRA adapter and `fusion_v2.joblib`; verified by
SHA-256), so both sets describe the historical training run of 15 September 2026 on the pool
test split (1,912 prompts). The six plots per set are `training_curves`, `roc_pr_curves`,
`score_distribution`, `feature_scatter`, `feature_weights` and `decision_routing`.

These plots read the Layer-1 feature cache (`data/processed/layer1_features/test_scored.jsonl`,
or the placeholder `test_features.jsonl`), which is not shipped; to regenerate them, rebuild the
pool, retrain or reuse the shipped checkpoint, run `scripts/score_feature_cache.py --splits test`
and then:

```bash
python scripts/plot_layer1.py --stacker clf_only --out-dir figures/layer1
python scripts/plot_layer1.py --stacker full --out-dir figures/layer1
```

The classifier-only column of the paper's Table II (AUROC 0.9914, AUPRC 0.9868, T = 1.1908)
was read from these plots' inputs; no results file stores it. See `REPRODUCIBILITY.md`.

## Regenerating the evaluation plots

```bash
python scripts/plot_evaluation.py \
    --layer1 artifacts/layer1/results.json \
    --predictions results/final_eval_20260926/final_cascade_predictions \
    --eval-results results/final_eval_20260926/final_cascade.json \
    --redteam results/development/redteam_all200_views.json \
    --out figures
```

Without `--layer2` and `--layer3` the four legacy figures are skipped and named in the new
`manifest.json`; pass `--strict` to make a skip a non-zero exit.
