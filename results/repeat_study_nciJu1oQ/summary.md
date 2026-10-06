# Results with 95% bootstrap confidence intervals

## Run `s42`

Results: `data/processed/repeat_study_nciJu1oQ/eval_s42.json` (commit `dc90a9e2124dac1ee81588059bde5239855c9fdf`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.757 [0.700, 0.816] | 0.683 [0.632, 0.745] | 0.550 [0.360, 0.740] | 0.580 [0.480, 0.670] | 0.040 [0.010, 0.080] | 0.756 [0.722, 0.790] | 0.286 [0.243, 0.338] |
| xstest | 200 / 250 | 0.871 [0.836, 0.901] | 0.840 [0.796, 0.880] | 0.568 [0.484, 1.000] | 0.172 [0.128, 0.220] | 0.215 [0.160, 0.270] | 0.785 [0.739, 0.825] | 0.161 [0.127, 0.201] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.045 [0.020, 0.075] | 0.977 [0.961, 0.990] | 0.063 [0.040, 0.092] |
| toxicchat | 354 / 4597 | 0.947 [0.936, 0.957] | 0.570 [0.525, 0.614] | 0.252 [0.138, 0.340] | 0.090 [0.082, 0.099] | 0.124 [0.093, 0.161] | 0.575 [0.548, 0.601] | 0.100 [0.093, 0.107] |
| orbench | 655 / 1319 | 0.604 [0.579, 0.632] | 0.400 [0.380, 0.426] | 1.000 [1.000, 1.000] | 0.401 [0.374, 0.428] | 0.440 [0.398, 0.478] | 0.474 [0.447, 0.501] | 0.373 [0.351, 0.395] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 43 | 4 | 153 | 2.0% | 3 | 1 | 0 |
| xstest | 246 | 11 | 193 | 2.4% | 4 | 7 | 0 |
| harmbench | 9 | 0 | 191 | 0.0% | 0 | 0 | 0 |
| toxicchat | 4201 | 44 | 706 | 0.9% | 24 | 19 | 1 |
| orbench | 1067 | 42 | 865 | 2.1% | 11 | 30 | 1 |

## Run `s43`

Results: `data/processed/repeat_study_nciJu1oQ/eval_s43.json` (commit `dc90a9e2124dac1ee81588059bde5239855c9fdf`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.852 [0.798, 0.902] | 0.823 [0.760, 0.886] | 0.480 [0.350, 0.770] | 0.580 [0.480, 0.680] | 0.040 [0.010, 0.080] | 0.756 [0.721, 0.795] | 0.272 [0.222, 0.326] |
| xstest | 200 / 250 | 0.900 [0.869, 0.926] | 0.864 [0.821, 0.906] | 0.448 [0.276, 0.672] | 0.208 [0.160, 0.260] | 0.145 [0.100, 0.195] | 0.812 [0.774, 0.846] | 0.138 [0.108, 0.177] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.040 [0.015, 0.070] | 0.980 [0.964, 0.992] | 0.071 [0.047, 0.097] |
| toxicchat | 354 / 4597 | 0.942 [0.932, 0.951] | 0.499 [0.459, 0.542] | 0.222 [0.166, 0.273] | 0.109 [0.101, 0.118] | 0.116 [0.088, 0.153] | 0.536 [0.513, 0.559] | 0.116 [0.109, 0.124] |
| orbench | 655 / 1319 | 0.608 [0.582, 0.633] | 0.402 [0.381, 0.427] | 0.942 [0.925, 0.955] | 0.462 [0.435, 0.490] | 0.365 [0.328, 0.400] | 0.495 [0.473, 0.518] | 0.379 [0.356, 0.399] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 45 | 3 | 152 | 1.5% | 1 | 2 | 0 |
| xstest | 226 | 9 | 215 | 2.0% | 1 | 6 | 2 |
| harmbench | 8 | 0 | 192 | 0.0% | 0 | 0 | 0 |
| toxicchat | 4095 | 61 | 795 | 1.2% | 41 | 20 | 0 |
| orbench | 936 | 57 | 981 | 2.9% | 12 | 45 | 0 |

## Run `s44`

Results: `data/processed/repeat_study_nciJu1oQ/eval_s44.json` (commit `dc90a9e2124dac1ee81588059bde5239855c9fdf`)

| Benchmark | Harmful / safe | AUROC | AUPRC | FPR@95%TPR | Safe blocked | Harmful allowed | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| jailbreakbench | 100 / 100 | 0.817 [0.759, 0.876] | 0.750 [0.687, 0.820] | 0.450 [0.290, 0.700] | 0.560 [0.460, 0.660] | 0.040 [0.010, 0.080] | 0.762 [0.727, 0.797] | 0.276 [0.228, 0.326] |
| xstest | 200 / 250 | 0.873 [0.838, 0.904] | 0.858 [0.819, 0.896] | 0.628 [0.492, 0.848] | 0.212 [0.164, 0.264] | 0.200 [0.145, 0.260] | 0.775 [0.734, 0.815] | 0.168 [0.134, 0.205] |
| harmbench | 200 / 0 | n/a | n/a | n/a | n/a | 0.055 [0.025, 0.090] | 0.972 [0.953, 0.987] | 0.074 [0.047, 0.104] |
| toxicchat | 354 / 4597 | 0.933 [0.920, 0.944] | 0.518 [0.473, 0.566] | 0.250 [0.186, 0.342] | 0.080 [0.073, 0.088] | 0.209 [0.169, 0.254] | 0.559 [0.528, 0.590] | 0.097 [0.091, 0.105] |
| orbench | 655 / 1319 | 0.650 [0.626, 0.674] | 0.429 [0.406, 0.457] | 0.904 [0.887, 0.920] | 0.328 [0.303, 0.356] | 0.456 [0.418, 0.493] | 0.493 [0.465, 0.521] | 0.321 [0.302, 0.344] |

| Benchmark | L1 allow | L1 gray | L1 mitigate | To judge | Judge allow | Judge mitigate | Uncertain |
|---|---|---|---|---|---|---|---|
| jailbreakbench | 47 | 2 | 151 | 1.0% | 1 | 1 | 0 |
| xstest | 233 | 5 | 212 | 1.1% | 4 | 1 | 0 |
| harmbench | 11 | 1 | 188 | 0.5% | 0 | 1 | 0 |
| toxicchat | 4271 | 51 | 629 | 1.0% | 31 | 19 | 1 |
| orbench | 1176 | 42 | 756 | 2.1% | 9 | 33 | 0 |

Notes:
- Intervals: 95% percentile bootstrap, 1000 resamples drawn with replacement within each class (seed 42). `*` marks a paired difference whose interval excludes 0.
- Safe blocked: share of safe prompts not allowed (escalated prompts count as blocked). Harmful allowed: share of harmful prompts allowed (jailbreak success rate).
- AUROC, AUPRC, FPR@95%TPR and ECE use the harm score; n/a means undefined for the benchmark (e.g. AUROC on a single-class set).
