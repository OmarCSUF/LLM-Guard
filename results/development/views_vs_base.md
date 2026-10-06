# Results with 95% bootstrap confidence intervals

## Run `base`

Results: `data/processed/eval_shipped_default_2026-09-22.json` (commit `b656744908b484f67167afe902ec1be78b526fad`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.803 [0.744, 0.860] | 0.736 [0.676, 0.798] | 0.530 [0.330, 0.730] | 0.670 [0.580, 0.750] | 0.020 [0.000, 0.050] | 0.740 [0.714, 0.767] | 0.318 [0.278, 0.365] |
| xstest | 200 / 250 | 0.882 [0.852, 0.909] | 0.844 [0.800, 0.885] | 0.412 [0.344, 0.604] | 0.176 [0.132, 0.228] | 0.230 [0.180, 0.290] | 0.774 [0.728, 0.814] | 0.156 [0.125, 0.198] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.030 [0.010, 0.055] | 0.985 [0.972, 0.995] | 0.053 [0.033, 0.077] |
| toxicchat | 354 / 4597 | 0.943 [0.931, 0.953] | 0.551 [0.507, 0.596] | 0.249 [0.150, 0.336] | 0.087 [0.079, 0.094] | 0.141 [0.110, 0.181] | 0.576 [0.550, 0.602] | 0.098 [0.091, 0.105] |
| orbench | 655 / 1319 | 0.648 [0.624, 0.674] | 0.439 [0.416, 0.464] | 0.952 [0.941, 0.964] | 0.415 [0.388, 0.442] | 0.344 [0.308, 0.380] | 0.527 [0.503, 0.551] | 0.343 [0.321, 0.364] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 34 | 3 | 163 | 1.5% | 1 | 2 | 0 |
| xstest | 248 | 10 | 192 | 2.2% | 4 | 6 | 0 |
| harmbench | 6 | 1 | 193 | 0.5% | 0 | 1 | 0 |
| toxicchat | 4192 | 81 | 678 | 1.6% | 56 | 24 | 1 |
| orbench | 975 | 72 | 927 | 3.6% | 22 | 49 | 1 |

## Run `views`

Results: `data/processed/eval_views_2026-09-22.json` (commit `6dead1c4d49b8360e9defe6422714939a922508e`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.803 [0.744, 0.860] | 0.736 [0.676, 0.798] | 0.530 [0.330, 0.730] | 0.670 [0.580, 0.750] | 0.020 [0.000, 0.050] | 0.740 [0.714, 0.767] | 0.318 [0.278, 0.365] |
| xstest | 200 / 250 | 0.882 [0.852, 0.909] | 0.844 [0.800, 0.885] | 0.412 [0.344, 0.604] | 0.176 [0.132, 0.228] | 0.230 [0.180, 0.290] | 0.774 [0.728, 0.814] | 0.156 [0.125, 0.198] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.030 [0.010, 0.055] | 0.985 [0.972, 0.995] | 0.053 [0.033, 0.077] |
| toxicchat | 354 / 4597 | 0.942 [0.930, 0.952] | 0.547 [0.504, 0.591] | 0.245 [0.156, 0.346] | 0.090 [0.081, 0.098] | 0.133 [0.102, 0.169] | 0.572 [0.546, 0.599] | 0.102 [0.094, 0.109] |
| orbench | 655 / 1319 | 0.648 [0.623, 0.674] | 0.439 [0.415, 0.463] | 0.952 [0.941, 0.964] | 0.415 [0.390, 0.443] | 0.344 [0.308, 0.380] | 0.527 [0.503, 0.551] | 0.344 [0.322, 0.365] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 34 | 3 | 163 | 1.5% | 1 | 2 | 0 |
| xstest | 248 | 10 | 192 | 2.2% | 4 | 6 | 0 |
| harmbench | 6 | 1 | 193 | 0.5% | 0 | 1 | 0 |
| toxicchat | 4170 | 86 | 695 | 1.7% | 61 | 24 | 1 |
| orbench | 973 | 73 | 928 | 3.7% | 23 | 49 | 1 |

## Paired difference `views - base`

| Benchmark | AUROC diff | Safe blocked diff | Harmful allowed diff | F1 diff |
|---|---|---|---|---|
| jailbreakbench | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| xstest | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| harmbench | n/a | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| toxicchat | -0.001 [-0.002, -0.000] * | 0.003 [0.002, 0.005] * | -0.008 [-0.017, 0.000] | -0.004 [-0.009, 0.002] |
| orbench | -0.000 [-0.001, 0.000] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | -0.000 [-0.001, 0.000] |

Notes:
- Intervals: 95% percentile bootstrap, 1000 resamples drawn with replacement within each class (seed 42). `*` marks a paired difference whose interval excludes 0.
- Safe blocked: share of safe prompts not allowed (escalated prompts count as blocked). Harmful allowed: share of harmful prompts allowed (jailbreak success rate).
- AUROC, AUPRC, FPR@95%TPR and ECE use the harm score; n/a means undefined for the benchmark (e.g. AUROC on a single-class set).
