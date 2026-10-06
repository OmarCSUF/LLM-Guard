# Results with 95% bootstrap confidence intervals

## Run `layer1`

Results: `data/processed/eval_layer1_only_v2.json` (commit `82a73b6f948564d35ba41e3e5dc6a7f44d850320`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.776 [0.706, 0.836] | 0.729 [0.654, 0.804] | 0.540 [0.370, 0.740] | 0.670 [0.580, 0.750] | 0.040 [0.010, 0.080] | 0.730 [0.701, 0.762] | 0.322 [0.280, 0.368] |
| xstest | 200 / 250 | 0.879 [0.847, 0.907] | 0.838 [0.788, 0.885] | 0.468 [0.352, 0.628] | 0.164 [0.120, 0.212] | 0.245 [0.190, 0.305] | 0.770 [0.725, 0.812] | 0.158 [0.127, 0.199] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.035 [0.010, 0.060] | 0.982 [0.969, 0.995] | 0.055 [0.034, 0.078] |
| toxicchat | 354 / 4597 | 0.935 [0.920, 0.948] | 0.553 [0.505, 0.606] | 0.261 [0.148, 0.542] | 0.083 [0.075, 0.090] | 0.155 [0.121, 0.192] | 0.578 [0.552, 0.605] | 0.099 [0.092, 0.106] |
| orbench | 655 / 1319 | 0.632 [0.605, 0.658] | 0.433 [0.406, 0.463] | 0.933 [0.885, 0.962] | 0.395 [0.368, 0.422] | 0.379 [0.342, 0.415] | 0.514 [0.488, 0.538] | 0.345 [0.324, 0.367] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 34 | 3 | 163 | 1.5% | 0 | 0 | 0 |
| xstest | 248 | 10 | 192 | 2.2% | 0 | 0 | 0 |
| harmbench | 6 | 1 | 193 | 0.5% | 0 | 0 | 0 |
| toxicchat | 4193 | 78 | 680 | 1.6% | 0 | 0 | 0 |
| orbench | 974 | 72 | 928 | 3.6% | 0 | 0 | 0 |

## Run `pipeline`

Results: `data/processed/eval_pipeline_v2.json` (commit `a4fd84708b8625ea37e25bef4c26072eb1e02959`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.776 [0.706, 0.836] | 0.729 [0.654, 0.804] | 0.540 [0.370, 0.730] | 0.670 [0.580, 0.750] | 0.020 [0.000, 0.050] | 0.740 [0.714, 0.767] | 0.318 [0.278, 0.365] |
| xstest | 200 / 250 | 0.879 [0.847, 0.907] | 0.838 [0.788, 0.885] | 0.468 [0.352, 0.628] | 0.176 [0.132, 0.228] | 0.230 [0.180, 0.290] | 0.774 [0.728, 0.814] | 0.156 [0.125, 0.198] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.030 [0.010, 0.055] | 0.985 [0.972, 0.995] | 0.054 [0.034, 0.077] |
| toxicchat | 354 / 4597 | 0.935 [0.920, 0.948] | 0.553 [0.505, 0.607] | 0.261 [0.148, 0.542] | 0.087 [0.079, 0.095] | 0.141 [0.110, 0.181] | 0.575 [0.550, 0.602] | 0.098 [0.091, 0.105] |
| orbench | 655 / 1319 | 0.632 [0.606, 0.659] | 0.433 [0.407, 0.463] | 0.933 [0.885, 0.962] | 0.415 [0.390, 0.443] | 0.344 [0.308, 0.380] | 0.527 [0.503, 0.551] | 0.344 [0.322, 0.366] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 34 | 3 | 163 | 1.5% | 1 | 2 | 0 |
| xstest | 248 | 10 | 192 | 2.2% | 4 | 6 | 0 |
| harmbench | 6 | 1 | 193 | 0.5% | 0 | 1 | 0 |
| toxicchat | 4193 | 78 | 680 | 1.6% | 54 | 23 | 1 |
| orbench | 974 | 72 | 928 | 3.6% | 22 | 49 | 1 |

## Paired difference `pipeline - layer1`

| Benchmark | AUROC diff | Safe blocked diff | Harmful allowed diff | F1 diff |
|---|---|---|---|---|
| jailbreakbench | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.000] | -0.020 [-0.050, 0.000] | 0.010 [0.000, 0.025] |
| xstest | -0.000 [-0.000, 0.000] | 0.012 [0.000, 0.028] | -0.015 [-0.035, 0.000] | 0.003 [-0.009, 0.016] |
| harmbench | n/a | n/a | -0.005 [-0.015, 0.000] | 0.003 [0.000, 0.008] |
| toxicchat | 0.000 [0.000, 0.000] * | 0.004 [0.002, 0.006] * | -0.014 [-0.028, -0.003] * | -0.003 [-0.010, 0.005] |
| orbench | 0.000 [0.000, 0.001] * | 0.020 [0.013, 0.030] * | -0.035 [-0.049, -0.021] * | 0.013 [0.004, 0.021] * |

Notes:
- Intervals: 95% percentile bootstrap, 1000 resamples drawn with replacement within each class (seed 42). `*` marks a paired difference whose interval excludes 0.
- Safe blocked: share of safe prompts not allowed (escalated prompts count as blocked). Harmful allowed: share of harmful prompts allowed (jailbreak success rate).
- AUROC, AUPRC, FPR@95%TPR and ECE use the harm score; n/a means undefined for the benchmark (e.g. AUROC on a single-class set).
