# Results with 95% bootstrap confidence intervals

## Run `original`

Results: `data/processed/parity_original.json` (commit `b656744908b484f67167afe902ec1be78b526fad`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 0 / 100 | n/a | n/a | n/a | 0.670 [0.580, 0.750] | n/a | n/a | 0.664 [0.583, 0.741] |
| xstest | 0 / 250 | n/a | n/a | n/a | 0.176 [0.132, 0.228] | n/a | n/a | 0.203 [0.163, 0.251] |
| toxicchat | 0 / 4597 | n/a | n/a | n/a | 0.087 [0.079, 0.095] | n/a | n/a | 0.118 [0.111, 0.126] |
| orbench | 0 / 1319 | n/a | n/a | n/a | 0.415 [0.389, 0.441] | n/a | n/a | 0.426 [0.403, 0.450] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 32 | 1 | 67 | 1.0% | 1 | 0 | 0 |
| xstest | 203 | 6 | 41 | 2.4% | 3 | 3 | 0 |
| toxicchat | 4144 | 74 | 379 | 1.6% | 54 | 19 | 1 |
| orbench | 752 | 46 | 521 | 3.5% | 20 | 25 | 1 |

## Run `general`

Results: `data/processed/parity_general.json` (commit `b656744908b484f67167afe902ec1be78b526fad`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 0 / 100 | n/a | n/a | n/a | 0.530 [0.430, 0.630] | n/a | n/a | 0.549 [0.462, 0.634] |
| xstest | 0 / 250 | n/a | n/a | n/a | 0.172 [0.128, 0.224] | n/a | n/a | 0.195 [0.157, 0.239] |
| toxicchat | 0 / 4597 | n/a | n/a | n/a | 0.080 [0.072, 0.088] | n/a | n/a | 0.112 [0.105, 0.119] |
| orbench | 0 / 1319 | n/a | n/a | n/a | 0.404 [0.378, 0.431] | n/a | n/a | 0.415 [0.392, 0.440] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 43 | 4 | 53 | 4.0% | 4 | 0 | 0 |
| xstest | 204 | 6 | 40 | 2.4% | 3 | 3 | 0 |
| toxicchat | 4184 | 64 | 349 | 1.4% | 45 | 17 | 2 |
| orbench | 768 | 45 | 506 | 3.4% | 18 | 25 | 2 |

## Run `arabic`

Results: `data/processed/parity_arabic.json` (commit `b656744908b484f67167afe902ec1be78b526fad`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 0 / 100 | n/a | n/a | n/a | 0.530 [0.430, 0.630] | n/a | n/a | 0.551 [0.464, 0.634] |
| xstest | 0 / 250 | n/a | n/a | n/a | 0.176 [0.132, 0.228] | n/a | n/a | 0.199 [0.159, 0.243] |
| toxicchat | 0 / 4597 | n/a | n/a | n/a | 0.082 [0.075, 0.090] | n/a | n/a | 0.114 [0.107, 0.121] |
| orbench | 0 / 1319 | n/a | n/a | n/a | 0.428 [0.402, 0.455] | n/a | n/a | 0.434 [0.410, 0.459] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 43 | 4 | 53 | 4.0% | 4 | 0 | 0 |
| xstest | 205 | 3 | 42 | 1.2% | 1 | 2 | 0 |
| toxicchat | 4172 | 69 | 356 | 1.5% | 47 | 18 | 4 |
| orbench | 741 | 41 | 537 | 3.1% | 14 | 26 | 1 |

## Paired difference `arabic - original`

| Benchmark | AUROC diff | Safe blocked diff | Harmful allowed diff | F1 diff |
|---|---|---|---|---|
| jailbreakbench | n/a | -0.140 [-0.210, -0.080] * | n/a | n/a |
| xstest | n/a | 0.000 [-0.020, 0.020] | n/a | n/a |
| toxicchat | n/a | -0.005 [-0.009, 0.000] | n/a | n/a |
| orbench | n/a | 0.013 [-0.002, 0.028] | n/a | n/a |

## Paired difference `general - original`

| Benchmark | AUROC diff | Safe blocked diff | Harmful allowed diff | F1 diff |
|---|---|---|---|---|
| jailbreakbench | n/a | -0.140 [-0.210, -0.070] * | n/a | n/a |
| xstest | n/a | -0.004 [-0.028, 0.016] | n/a | n/a |
| toxicchat | n/a | -0.007 [-0.011, -0.002] * | n/a | n/a |
| orbench | n/a | -0.011 [-0.027, 0.005] | n/a | n/a |

Notes:
- Intervals: 95% percentile bootstrap, 1000 resamples drawn with replacement within each class (seed 42). `*` marks a paired difference whose interval excludes 0.
- Safe blocked: share of safe prompts not allowed (escalated prompts count as blocked). Harmful allowed: share of harmful prompts allowed (jailbreak success rate).
- AUROC, AUPRC, FPR@95%TPR and ECE use the harm score; n/a means undefined for the benchmark (e.g. AUROC on a single-class set).
