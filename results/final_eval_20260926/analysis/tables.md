# Final-test analysis — final_eval_20260926

Bootstrap B = 2000, seed 20260926, 95% percentile intervals; Wilson intervals on single rates.

## Final test set

| Source | n |
|---|---:|
| sst | 99 |
| catqa | 543 |
| dna_core | 573 |
| dna_other | 364 |
| coconot | 378 |
| phtest | 1000 |
| no_robots | 496 |
| **total** | 3453 |

## A. Operational rates by source (harmful sources: harmful allowed; safe sources: safe blocked), %

| Source | cascade | layer1 | judge | judge_llama_v1 [post hoc] | judge_llama_v2 [post hoc] | granite | llamaguard | shieldgemma [post hoc] | cascade_s42 | cascade_s43 | cascade_s44 | cascade_noviews [post hoc] | cascade_nonorm [post hoc] | cascade_bare [post hoc] | cascade_T1 [post hoc] | cascade_noviews_s42 [post hoc] | cascade_nonorm_s42 [post hoc] | cascade_bare_s42 [post hoc] | cascade_T1_s42 [post hoc] | cascade_noviews_s43 [post hoc] | cascade_nonorm_s43 [post hoc] | cascade_bare_s43 [post hoc] | cascade_T1_s43 [post hoc] | cascade_noviews_s44 [post hoc] | cascade_nonorm_s44 [post hoc] | cascade_bare_s44 [post hoc] | cascade_T1_s44 [post hoc] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst | 5.1 [2.2, 11.3] (n=99) | 6.1 [2.8, 12.6] (n=99) | 2.0 [0.6, 7.1] (n=99) | 0.0 [0.0, 3.7] (n=99) | 0.0 [0.0, 3.7] (n=99) | 0.0 [0.0, 3.7] (n=99) | 2.0 [0.6, 7.1] (n=99) | 24.2 [16.9, 33.5] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 7.1 [3.5, 13.9] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 5.1 [2.2, 11.3] (n=99) | 6.1 [2.8, 12.6] (n=99) |
| catqa | 19.2 [16.1, 22.7] (n=543) | 23.6 [20.2, 27.3] (n=543) | 2.4 [1.4, 4.1] (n=543) | 0.9 [0.4, 2.1] (n=543) | 1.8 [1.0, 3.4] (n=543) | 0.2 [0.0, 1.0] (n=543) | 6.1 [4.4, 8.4] (n=543) | 51.2 [47.0, 55.4] (n=543) | 34.8 [30.9, 38.9] (n=543) | 22.3 [19.0, 26.0] (n=543) | 35.7 [31.8, 39.8] (n=543) | 19.2 [16.1, 22.7] (n=543) | 19.2 [16.1, 22.7] (n=543) | 19.2 [16.1, 22.7] (n=543) | 19.3 [16.2, 22.9] (n=543) | 34.8 [30.9, 38.9] (n=543) | 34.8 [30.9, 38.9] (n=543) | 34.8 [30.9, 38.9] (n=543) | 35.2 [31.3, 39.3] (n=543) | 22.3 [19.0, 26.0] (n=543) | 22.3 [19.0, 26.0] (n=543) | 22.3 [19.0, 26.0] (n=543) | 22.7 [19.3, 26.4] (n=543) | 35.7 [31.8, 39.8] (n=543) | 35.7 [31.8, 39.8] (n=543) | 35.7 [31.8, 39.8] (n=543) | 36.1 [32.2, 40.2] (n=543) |
| dna_core | 23.9 [20.6, 27.6] (n=573) | 25.1 [21.8, 28.8] (n=573) | 32.6 [28.9, 36.6] (n=573) | 23.9 [20.6, 27.6] (n=573) | 32.5 [28.8, 36.4] (n=573) | 16.9 [14.1, 20.2] (n=573) | 45.4 [41.3, 49.5] (n=573) | 49.2 [45.1, 53.3] (n=573) | 25.8 [22.4, 29.6] (n=573) | 23.4 [20.1, 27.0] (n=573) | 23.6 [20.3, 27.2] (n=573) | 23.9 [20.6, 27.6] (n=573) | 23.9 [20.6, 27.6] (n=573) | 23.9 [20.6, 27.6] (n=573) | 24.1 [20.8, 27.8] (n=573) | 25.8 [22.4, 29.6] (n=573) | 25.8 [22.4, 29.6] (n=573) | 25.8 [22.4, 29.6] (n=573) | 25.8 [22.4, 29.6] (n=573) | 23.4 [20.1, 27.0] (n=573) | 23.4 [20.1, 27.0] (n=573) | 23.4 [20.1, 27.0] (n=573) | 23.4 [20.1, 27.0] (n=573) | 23.6 [20.3, 27.2] (n=573) | 23.6 [20.3, 27.2] (n=573) | 23.6 [20.3, 27.2] (n=573) | 23.6 [20.3, 27.2] (n=573) |
| dna_other | 48.9 [43.8, 54.0] (n=364) | 50.8 [45.7, 55.9] (n=364) | 64.0 [59.0, 68.8] (n=364) | 69.0 [64.0, 73.5] (n=364) | 75.0 [70.3, 79.2] (n=364) | 26.4 [22.1, 31.1] (n=364) | 65.1 [60.1, 69.8] (n=364) | 92.9 [89.7, 95.1] (n=364) | 53.3 [48.2, 58.4] (n=364) | 51.6 [46.5, 56.7] (n=364) | 57.1 [52.0, 62.1] (n=364) | 48.9 [43.8, 54.0] (n=364) | 48.9 [43.8, 54.0] (n=364) | 48.9 [43.8, 54.0] (n=364) | 48.9 [43.8, 54.0] (n=364) | 53.3 [48.2, 58.4] (n=364) | 53.3 [48.2, 58.4] (n=364) | 53.3 [48.2, 58.4] (n=364) | 53.8 [48.7, 58.9] (n=364) | 51.6 [46.5, 56.7] (n=364) | 51.6 [46.5, 56.7] (n=364) | 51.6 [46.5, 56.7] (n=364) | 51.6 [46.5, 56.7] (n=364) | 57.1 [52.0, 62.1] (n=364) | 57.1 [52.0, 62.1] (n=364) | 57.1 [52.0, 62.1] (n=364) | 57.1 [52.0, 62.1] (n=364) |
| coconot | 6.3 [4.3, 9.3] (n=378) | 6.3 [4.3, 9.3] (n=378) | 20.4 [16.6, 24.7] (n=378) | 21.7 [17.8, 26.1] (n=378) | 13.5 [10.4, 17.3] (n=378) | 21.7 [17.8, 26.1] (n=378) | 6.1 [4.1, 9.0] (n=378) | 11.6 [8.8, 15.3] (n=378) | 4.8 [3.0, 7.4] (n=378) | 6.1 [4.1, 9.0] (n=378) | 4.8 [3.0, 7.4] (n=378) | 6.3 [4.3, 9.3] (n=378) | 6.3 [4.3, 9.3] (n=378) | 6.3 [4.3, 9.3] (n=378) | 6.6 [4.5, 9.6] (n=378) | 4.8 [3.0, 7.4] (n=378) | 4.8 [3.0, 7.4] (n=378) | 4.8 [3.0, 7.4] (n=378) | 5.0 [3.2, 7.7] (n=378) | 6.1 [4.1, 9.0] (n=378) | 6.1 [4.1, 9.0] (n=378) | 6.1 [4.1, 9.0] (n=378) | 6.1 [4.1, 9.0] (n=378) | 4.8 [3.0, 7.4] (n=378) | 4.8 [3.0, 7.4] (n=378) | 4.8 [3.0, 7.4] (n=378) | 4.8 [3.0, 7.4] (n=378) |
| phtest | 42.8 [39.8, 45.9] (n=1000) | 42.1 [39.1, 45.2] (n=1000) | 29.6 [26.9, 32.5] (n=1000) | 25.7 [23.1, 28.5] (n=1000) | 17.2 [15.0, 19.7] (n=1000) | 55.9 [52.8, 58.9] (n=1000) | 27.7 [25.0, 30.6] (n=1000) | 15.0 [12.9, 17.3] (n=1000) | 45.9 [42.8, 49.0] (n=1000) | 46.3 [43.2, 49.4] (n=1000) | 42.2 [39.2, 45.3] (n=1000) | 42.6 [39.6, 45.7] (n=1000) | 42.8 [39.8, 45.9] (n=1000) | 42.6 [39.6, 45.7] (n=1000) | 42.9 [39.9, 46.0] (n=1000) | 45.7 [42.6, 48.8] (n=1000) | 45.9 [42.8, 49.0] (n=1000) | 45.7 [42.6, 48.8] (n=1000) | 46.0 [42.9, 49.1] (n=1000) | 46.1 [43.0, 49.2] (n=1000) | 46.3 [43.2, 49.4] (n=1000) | 46.1 [43.0, 49.2] (n=1000) | 46.5 [43.4, 49.6] (n=1000) | 41.7 [38.7, 44.8] (n=1000) | 42.2 [39.2, 45.3] (n=1000) | 41.7 [38.7, 44.8] (n=1000) | 42.0 [39.0, 45.1] (n=1000) |
| no_robots | 12.7 [10.1, 15.9] (n=496) | 12.5 [9.9, 15.7] (n=496) | 7.1 [5.1, 9.7] (n=496) | 5.4 [3.8, 7.8] (n=496) | 2.4 [1.4, 4.2] (n=496) | 19.2 [15.9, 22.8] (n=496) | 11.3 [8.8, 14.4] (n=496) | 2.0 [1.1, 3.7] (n=496) | 14.1 [11.3, 17.5] (n=496) | 15.3 [12.4, 18.8] (n=496) | 10.5 [8.1, 13.5] (n=496) | 11.7 [9.2, 14.8] (n=496) | 12.7 [10.1, 15.9] (n=496) | 11.7 [9.2, 14.8] (n=496) | 12.9 [10.2, 16.1] (n=496) | 13.1 [10.4, 16.4] (n=496) | 14.1 [11.3, 17.5] (n=496) | 13.1 [10.4, 16.4] (n=496) | 14.5 [11.7, 17.9] (n=496) | 13.9 [11.1, 17.2] (n=496) | 15.3 [12.4, 18.8] (n=496) | 13.9 [11.1, 17.2] (n=496) | 15.3 [12.4, 18.8] (n=496) | 9.7 [7.4, 12.6] (n=496) | 10.5 [8.1, 13.5] (n=496) | 9.7 [7.4, 12.6] (n=496) | 10.5 [8.1, 13.5] (n=496) |

Pooled AUROC (every source is single-class, so this partly measures source separability; secondary):

| Arm | n | AUROC |
|---|---:|---:|
| cascade | 3089 | 0.811 |
| layer1 | 3089 | 0.811 |
| judge | 3089 | 0.878 |
| judge_llama_v1 [post hoc] | 3089 | 0.928 |
| judge_llama_v2 [post hoc] | 3089 | 0.924 |
| granite | 3089 | 0.923 |
| llamaguard | 3089 | 0.834 |
| shieldgemma [post hoc] | 3089 | 0.826 |
| cascade_s42 | 3089 | 0.751 |
| cascade_s43 | 3089 | 0.773 |
| cascade_s44 | 3089 | 0.777 |
| cascade_noviews [post hoc] | 3089 | 0.814 |
| cascade_nonorm [post hoc] | 3089 | 0.811 |
| cascade_bare [post hoc] | 3089 | 0.814 |
| cascade_T1 [post hoc] | 3089 | 0.811 |
| cascade_noviews_s42 [post hoc] | 3089 | 0.755 |
| cascade_nonorm_s42 [post hoc] | 3089 | 0.751 |
| cascade_bare_s42 [post hoc] | 3089 | 0.755 |
| cascade_T1_s42 [post hoc] | 3089 | 0.754 |
| cascade_noviews_s43 [post hoc] | 3089 | 0.776 |
| cascade_nonorm_s43 [post hoc] | 3089 | 0.773 |
| cascade_bare_s43 [post hoc] | 3089 | 0.776 |
| cascade_T1_s43 [post hoc] | 3089 | 0.771 |
| cascade_noviews_s44 [post hoc] | 3089 | 0.781 |
| cascade_nonorm_s44 [post hoc] | 3089 | 0.777 |
| cascade_bare_s44 [post hoc] | 3089 | 0.781 |
| cascade_T1_s44 [post hoc] | 3089 | 0.779 |

Paired bootstrap on the difference in pooled AUROC (same prompt resample for both arms):

| Comparison | diff | 95% CI | covers zero |
|---|---:|---|---|
| judge_llama_v1 [post hoc] − granite | +0.0049 | [-0.0015, +0.0113] | yes |
| judge_llama_v1 [post hoc] − judge | +0.0501 | [+0.0419, +0.0583] | no |
| judge − cascade | +0.0672 | [+0.0502, +0.0843] | no |
| judge_llama_v1 [post hoc] − cascade | +0.1173 | [+0.1028, +0.1323] | no |
| granite − cascade | +0.1124 | [+0.0978, +0.1271] | no |
| llamaguard − cascade | +0.0230 | [+0.0035, +0.0420] | no |
| shieldgemma [post hoc] − cascade | +0.0152 | [-0.0013, +0.0327] | yes |
| cascade_noviews [post hoc] − cascade | +0.0030 | [+0.0016, +0.0047] | no |
| cascade_nonorm [post hoc] − cascade | +0.0000 | [+0.0000, +0.0000] | yes |
| cascade_bare [post hoc] − cascade | +0.0030 | [+0.0016, +0.0046] | no |
| cascade_T1 [post hoc] − cascade | +0.0002 | [-0.0018, +0.0021] | yes |

`[post hoc]` marks an arm the pre-registration did not name. It was analysed exactly as the others were, but it was chosen after the pre-registered results were known, so it does not carry their guarantee:
- **judge_llama_v1**: added 2026-09-28; judge model swapped for Llama-3.2-3B-Instruct, prompt version selected on validation; not pre-registered
- **judge_llama_v2**: added 2026-09-28; as above, on the prompt version Qwen ships; not pre-registered
- **shieldgemma**: added 2026-09-27 after the results were known; not pre-registered
- **cascade_noviews**: added 2026-09-30; component ablation, leading-clause view off; no decision was taken from it; not pre-registered
- **cascade_nonorm**: added 2026-09-30; component ablation, input normalisation off; no decision was taken from it; not pre-registered
- **cascade_bare**: added 2026-09-30; component ablation, view and normalisation off; no decision was taken from it; not pre-registered
- **cascade_T1**: added 2026-09-30; component ablation, temperature fixed at 1; no decision was taken from it; not pre-registered
- **cascade_noviews_s42**: added 2026-09-30; component ablation, leading-clause view off; no decision was taken from it; not pre-registered
- **cascade_nonorm_s42**: added 2026-09-30; component ablation, input normalisation off; no decision was taken from it; not pre-registered
- **cascade_bare_s42**: added 2026-09-30; component ablation, view and normalisation off; no decision was taken from it; not pre-registered
- **cascade_T1_s42**: added 2026-09-30; component ablation, temperature fixed at 1; no decision was taken from it; not pre-registered
- **cascade_noviews_s43**: added 2026-09-30; component ablation, leading-clause view off; no decision was taken from it; not pre-registered
- **cascade_nonorm_s43**: added 2026-09-30; component ablation, input normalisation off; no decision was taken from it; not pre-registered
- **cascade_bare_s43**: added 2026-09-30; component ablation, view and normalisation off; no decision was taken from it; not pre-registered
- **cascade_T1_s43**: added 2026-09-30; component ablation, temperature fixed at 1; no decision was taken from it; not pre-registered
- **cascade_noviews_s44**: added 2026-09-30; component ablation, leading-clause view off; no decision was taken from it; not pre-registered
- **cascade_nonorm_s44**: added 2026-09-30; component ablation, input normalisation off; no decision was taken from it; not pre-registered
- **cascade_bare_s44**: added 2026-09-30; component ablation, view and normalisation off; no decision was taken from it; not pre-registered
- **cascade_T1_s44**: added 2026-09-30; component ablation, temperature fixed at 1; no decision was taken from it; not pre-registered

## Paired differences against the cascade (arm minus cascade), percentage points

| Source | Arm | diff | 95% CI | McNemar p | n |
|---|---|---:|---|---:|---:|
| sst | layer1 | +1.0 | [+0.0, +3.0] | 1 | 99 |
| catqa | layer1 | +4.4 | [+2.9, +6.3] | 1.19e-07 | 543 |
| dna_core | layer1 | +1.2 | [+0.3, +2.3] | 0.0156 | 573 |
| dna_other | layer1 | +1.9 | [+0.5, +3.6] | 0.0156 | 364 |
| coconot | layer1 | +0.0 | [+0.0, +0.0] | 1 | 378 |
| phtest | layer1 | -0.7 | [-1.3, -0.2] | 0.0156 | 1000 |
| no_robots | layer1 | -0.2 | [-0.6, +0.0] | 1 | 496 |
| sst | judge | -3.0 | [-8.1, +2.0] | 0.453 | 99 |
| catqa | judge | -16.8 | [-20.1, -13.3] | 4.36e-21 | 543 |
| dna_core | judge | +8.7 | [+4.7, +12.9] | 4.27e-05 | 573 |
| dna_other | judge | +15.1 | [+9.1, +21.4] | 7.74e-06 | 364 |
| coconot | judge | +14.0 | [+9.8, +18.5] | 1.89e-09 | 378 |
| phtest | judge | -13.2 | [-17.1, -9.1] | 7.56e-11 | 1000 |
| no_robots | judge | -5.6 | [-9.1, -2.2] | 0.00176 | 496 |
| sst | judge_llama_v1 [post hoc] | -5.1 | [-10.1, -1.0] | 0.0625 | 99 |
| catqa | judge_llama_v1 [post hoc] | -18.2 | [-21.5, -15.1] | 1.06e-27 | 543 |
| dna_core | judge_llama_v1 [post hoc] | +0.0 | [-3.7, +3.7] | 1 | 573 |
| dna_other | judge_llama_v1 [post hoc] | +20.1 | [+14.3, +25.8] | 3.98e-10 | 364 |
| coconot | judge_llama_v1 [post hoc] | +15.3 | [+10.8, +19.8] | 4.39e-10 | 378 |
| phtest | judge_llama_v1 [post hoc] | -17.1 | [-20.7, -13.4] | 1.18e-19 | 1000 |
| no_robots | judge_llama_v1 [post hoc] | -7.3 | [-10.3, -4.2] | 3.18e-06 | 496 |
| sst | judge_llama_v2 [post hoc] | -5.1 | [-10.1, -1.0] | 0.0625 | 99 |
| catqa | judge_llama_v2 [post hoc] | -17.3 | [-20.6, -14.2] | 1.75e-24 | 543 |
| dna_core | judge_llama_v2 [post hoc] | +8.6 | [+4.5, +12.6] | 3.44e-05 | 573 |
| dna_other | judge_llama_v2 [post hoc] | +26.1 | [+20.3, +32.1] | 1.39e-16 | 364 |
| coconot | judge_llama_v2 [post hoc] | +7.1 | [+2.9, +11.4] | 0.00073 | 378 |
| phtest | judge_llama_v2 [post hoc] | -25.6 | [-29.1, -22.3] | 4.77e-42 | 1000 |
| no_robots | judge_llama_v2 [post hoc] | -10.3 | [-13.3, -7.5] | 1.7e-12 | 496 |
| sst | granite | -5.1 | [-10.1, -1.0] | 0.0625 | 99 |
| catqa | granite | -19.0 | [-22.5, -15.7] | 5.23e-30 | 543 |
| dna_core | granite | -7.0 | [-10.5, -3.5] | 0.000227 | 573 |
| dna_other | granite | -22.5 | [-28.6, -15.9] | 5.42e-12 | 364 |
| coconot | granite | +15.3 | [+11.1, +19.3] | 4.33e-12 | 378 |
| phtest | granite | +13.1 | [+9.5, +16.9] | 6.06e-12 | 1000 |
| no_robots | granite | +6.5 | [+2.6, +10.1] | 0.00126 | 496 |
| sst | llamaguard | -3.0 | [-8.1, +1.0] | 0.375 | 99 |
| catqa | llamaguard | -13.1 | [-16.6, -9.6] | 1.78e-12 | 543 |
| dna_core | llamaguard | +21.5 | [+16.9, +26.0] | 4.58e-18 | 573 |
| dna_other | llamaguard | +16.2 | [+10.2, +22.3] | 1.49e-06 | 364 |
| coconot | llamaguard | -0.3 | [-3.4, +2.9] | 1 | 378 |
| phtest | llamaguard | -15.1 | [-18.8, -11.4] | 9.63e-14 | 1000 |
| no_robots | llamaguard | -1.4 | [-4.8, +1.8] | 0.483 | 496 |
| sst | shieldgemma [post hoc] | +19.2 | [+11.1, +28.3] | 6.6e-05 | 99 |
| catqa | shieldgemma [post hoc] | +32.0 | [+27.1, +36.8] | 2.57e-32 | 543 |
| dna_core | shieldgemma [post hoc] | +25.3 | [+20.6, +29.8] | 5.72e-25 | 573 |
| dna_other | shieldgemma [post hoc] | +44.0 | [+38.5, +49.5] | 1.63e-44 | 364 |
| coconot | shieldgemma [post hoc] | +5.3 | [+1.3, +9.3] | 0.0105 | 378 |
| phtest | shieldgemma [post hoc] | -27.8 | [-31.6, -24.0] | 1.5e-41 | 1000 |
| no_robots | shieldgemma [post hoc] | -10.7 | [-13.7, -7.9] | 1.66e-12 | 496 |
| sst | cascade_s42 | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_s42 | +15.7 | [+12.7, +18.8] | 1.14e-24 | 543 |
| dna_core | cascade_s42 | +1.9 | [-0.2, +4.2] | 0.0989 | 573 |
| dna_other | cascade_s42 | +4.4 | [+1.1, +7.7] | 0.0113 | 364 |
| coconot | cascade_s42 | -1.6 | [-2.9, -0.5] | 0.0312 | 378 |
| phtest | cascade_s42 | +3.1 | [+1.5, +4.7] | 0.000371 | 1000 |
| no_robots | cascade_s42 | +1.4 | [-0.6, +3.6] | 0.265 | 496 |
| sst | cascade_s43 | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_s43 | +3.1 | [+0.0, +6.1] | 0.0568 | 543 |
| dna_core | cascade_s43 | -0.5 | [-2.6, +1.6] | 0.749 | 573 |
| dna_other | cascade_s43 | +2.7 | [-1.1, +6.6] | 0.229 | 364 |
| coconot | cascade_s43 | -0.3 | [-1.9, +1.3] | 1 | 378 |
| phtest | cascade_s43 | +3.5 | [+1.3, +5.7] | 0.00223 | 1000 |
| no_robots | cascade_s43 | +2.6 | [+0.2, +5.0] | 0.041 | 496 |
| sst | cascade_s44 | +0.0 | [-3.0, +3.0] | 1 | 99 |
| catqa | cascade_s44 | +16.6 | [+12.9, +20.3] | 3.84e-17 | 543 |
| dna_core | cascade_s44 | -0.3 | [-2.8, +2.1] | 0.89 | 573 |
| dna_other | cascade_s44 | +8.2 | [+4.1, +12.6] | 0.000287 | 364 |
| coconot | cascade_s44 | -1.6 | [-3.4, +0.3] | 0.18 | 378 |
| phtest | cascade_s44 | -0.6 | [-2.5, +1.3] | 0.624 | 1000 |
| no_robots | cascade_s44 | -2.2 | [-4.6, +0.0] | 0.0989 | 496 |
| sst | cascade_noviews [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 99 |
| catqa | cascade_noviews [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 543 |
| dna_core | cascade_noviews [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 573 |
| dna_other | cascade_noviews [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 364 |
| coconot | cascade_noviews [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 378 |
| phtest | cascade_noviews [post hoc] | -0.2 | [-0.5, +0.0] | 0.5 | 1000 |
| no_robots | cascade_noviews [post hoc] | -1.0 | [-1.8, -0.2] | 0.0625 | 496 |
| sst | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 99 |
| catqa | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 543 |
| dna_core | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 573 |
| dna_other | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 364 |
| coconot | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 378 |
| phtest | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 1000 |
| no_robots | cascade_nonorm [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 496 |
| sst | cascade_bare [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 99 |
| catqa | cascade_bare [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 543 |
| dna_core | cascade_bare [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 573 |
| dna_other | cascade_bare [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 364 |
| coconot | cascade_bare [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 378 |
| phtest | cascade_bare [post hoc] | -0.2 | [-0.5, +0.0] | 0.5 | 1000 |
| no_robots | cascade_bare [post hoc] | -1.0 | [-2.0, -0.2] | 0.0625 | 496 |
| sst | cascade_T1 [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 99 |
| catqa | cascade_T1 [post hoc] | +0.2 | [+0.0, +0.6] | 1 | 543 |
| dna_core | cascade_T1 [post hoc] | +0.2 | [+0.0, +0.5] | 1 | 573 |
| dna_other | cascade_T1 [post hoc] | +0.0 | [+0.0, +0.0] | 1 | 364 |
| coconot | cascade_T1 [post hoc] | +0.3 | [+0.0, +0.8] | 1 | 378 |
| phtest | cascade_T1 [post hoc] | +0.1 | [-0.2, +0.5] | 1 | 1000 |
| no_robots | cascade_T1 [post hoc] | +0.2 | [+0.0, +0.6] | 1 | 496 |
| sst | cascade_noviews_s42 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_noviews_s42 [post hoc] | +15.7 | [+12.5, +18.8] | 1.14e-24 | 543 |
| dna_core | cascade_noviews_s42 [post hoc] | +1.9 | [+0.0, +4.2] | 0.0989 | 573 |
| dna_other | cascade_noviews_s42 [post hoc] | +4.4 | [+1.1, +7.7] | 0.0113 | 364 |
| coconot | cascade_noviews_s42 [post hoc] | -1.6 | [-2.9, -0.5] | 0.0312 | 378 |
| phtest | cascade_noviews_s42 [post hoc] | +2.9 | [+1.3, +4.6] | 0.000914 | 1000 |
| no_robots | cascade_noviews_s42 [post hoc] | +0.4 | [-1.6, +2.6] | 0.856 | 496 |
| sst | cascade_nonorm_s42 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_nonorm_s42 [post hoc] | +15.7 | [+12.7, +18.8] | 1.14e-24 | 543 |
| dna_core | cascade_nonorm_s42 [post hoc] | +1.9 | [-0.2, +4.0] | 0.0989 | 573 |
| dna_other | cascade_nonorm_s42 [post hoc] | +4.4 | [+1.4, +7.4] | 0.0113 | 364 |
| coconot | cascade_nonorm_s42 [post hoc] | -1.6 | [-2.9, -0.5] | 0.0312 | 378 |
| phtest | cascade_nonorm_s42 [post hoc] | +3.1 | [+1.5, +4.8] | 0.000371 | 1000 |
| no_robots | cascade_nonorm_s42 [post hoc] | +1.4 | [-0.6, +3.4] | 0.265 | 496 |
| sst | cascade_bare_s42 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_bare_s42 [post hoc] | +15.7 | [+12.7, +18.8] | 1.14e-24 | 543 |
| dna_core | cascade_bare_s42 [post hoc] | +1.9 | [-0.0, +3.8] | 0.0989 | 573 |
| dna_other | cascade_bare_s42 [post hoc] | +4.4 | [+1.4, +7.7] | 0.0113 | 364 |
| coconot | cascade_bare_s42 [post hoc] | -1.6 | [-2.9, -0.5] | 0.0312 | 378 |
| phtest | cascade_bare_s42 [post hoc] | +2.9 | [+1.3, +4.5] | 0.000914 | 1000 |
| no_robots | cascade_bare_s42 [post hoc] | +0.4 | [-1.8, +2.6] | 0.856 | 496 |
| sst | cascade_T1_s42 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_T1_s42 [post hoc] | +16.0 | [+13.1, +19.2] | 2.91e-25 | 543 |
| dna_core | cascade_T1_s42 [post hoc] | +1.9 | [-0.2, +4.0] | 0.0989 | 573 |
| dna_other | cascade_T1_s42 [post hoc] | +4.9 | [+1.6, +8.2] | 0.00393 | 364 |
| coconot | cascade_T1_s42 [post hoc] | -1.3 | [-2.6, +0.0] | 0.125 | 378 |
| phtest | cascade_T1_s42 [post hoc] | +3.2 | [+1.6, +4.7] | 0.000208 | 1000 |
| no_robots | cascade_T1_s42 [post hoc] | +1.8 | [-0.2, +4.2] | 0.136 | 496 |
| sst | cascade_noviews_s43 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_noviews_s43 [post hoc] | +3.1 | [-0.2, +6.3] | 0.0568 | 543 |
| dna_core | cascade_noviews_s43 [post hoc] | -0.5 | [-2.8, +1.7] | 0.749 | 573 |
| dna_other | cascade_noviews_s43 [post hoc] | +2.7 | [-1.1, +6.9] | 0.229 | 364 |
| coconot | cascade_noviews_s43 [post hoc] | -0.3 | [-1.9, +1.3] | 1 | 378 |
| phtest | cascade_noviews_s43 [post hoc] | +3.3 | [+1.2, +5.5] | 0.00373 | 1000 |
| no_robots | cascade_noviews_s43 [post hoc] | +1.2 | [-1.2, +3.6] | 0.418 | 496 |
| sst | cascade_nonorm_s43 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_nonorm_s43 [post hoc] | +3.1 | [+0.2, +6.3] | 0.0568 | 543 |
| dna_core | cascade_nonorm_s43 [post hoc] | -0.5 | [-2.6, +1.6] | 0.749 | 573 |
| dna_other | cascade_nonorm_s43 [post hoc] | +2.7 | [-1.4, +6.6] | 0.229 | 364 |
| coconot | cascade_nonorm_s43 [post hoc] | -0.3 | [-1.9, +1.3] | 1 | 378 |
| phtest | cascade_nonorm_s43 [post hoc] | +3.5 | [+1.4, +5.6] | 0.00223 | 1000 |
| no_robots | cascade_nonorm_s43 [post hoc] | +2.6 | [+0.4, +5.0] | 0.041 | 496 |
| sst | cascade_bare_s43 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_bare_s43 [post hoc] | +3.1 | [+0.2, +6.1] | 0.0568 | 543 |
| dna_core | cascade_bare_s43 [post hoc] | -0.5 | [-2.8, +1.7] | 0.749 | 573 |
| dna_other | cascade_bare_s43 [post hoc] | +2.7 | [-0.8, +7.1] | 0.229 | 364 |
| coconot | cascade_bare_s43 [post hoc] | -0.3 | [-1.9, +1.3] | 1 | 378 |
| phtest | cascade_bare_s43 [post hoc] | +3.3 | [+1.0, +5.5] | 0.00373 | 1000 |
| no_robots | cascade_bare_s43 [post hoc] | +1.2 | [-1.2, +3.6] | 0.418 | 496 |
| sst | cascade_T1_s43 [post hoc] | +2.0 | [+0.0, +5.1] | 0.5 | 99 |
| catqa | cascade_T1_s43 [post hoc] | +3.5 | [+0.6, +6.6] | 0.0319 | 543 |
| dna_core | cascade_T1_s43 [post hoc] | -0.5 | [-2.6, +1.7] | 0.749 | 573 |
| dna_other | cascade_T1_s43 [post hoc] | +2.7 | [-1.1, +6.6] | 0.229 | 364 |
| coconot | cascade_T1_s43 [post hoc] | -0.3 | [-1.9, +1.3] | 1 | 378 |
| phtest | cascade_T1_s43 [post hoc] | +3.7 | [+1.5, +5.9] | 0.00119 | 1000 |
| no_robots | cascade_T1_s43 [post hoc] | +2.6 | [+0.4, +4.8] | 0.0351 | 496 |
| sst | cascade_noviews_s44 [post hoc] | +0.0 | [-3.0, +3.0] | 1 | 99 |
| catqa | cascade_noviews_s44 [post hoc] | +16.6 | [+13.1, +20.4] | 3.84e-17 | 543 |
| dna_core | cascade_noviews_s44 [post hoc] | -0.3 | [-2.8, +1.9] | 0.89 | 573 |
| dna_other | cascade_noviews_s44 [post hoc] | +8.2 | [+4.1, +12.4] | 0.000287 | 364 |
| coconot | cascade_noviews_s44 [post hoc] | -1.6 | [-3.7, +0.3] | 0.18 | 378 |
| phtest | cascade_noviews_s44 [post hoc] | -1.1 | [-3.1, +0.8] | 0.329 | 1000 |
| no_robots | cascade_noviews_s44 [post hoc] | -3.0 | [-5.6, -0.6] | 0.0275 | 496 |
| sst | cascade_nonorm_s44 [post hoc] | +0.0 | [-3.0, +3.0] | 1 | 99 |
| catqa | cascade_nonorm_s44 [post hoc] | +16.6 | [+12.9, +20.3] | 3.84e-17 | 543 |
| dna_core | cascade_nonorm_s44 [post hoc] | -0.3 | [-2.8, +2.3] | 0.89 | 573 |
| dna_other | cascade_nonorm_s44 [post hoc] | +8.2 | [+3.8, +12.6] | 0.000287 | 364 |
| coconot | cascade_nonorm_s44 [post hoc] | -1.6 | [-3.4, +0.3] | 0.18 | 378 |
| phtest | cascade_nonorm_s44 [post hoc] | -0.6 | [-2.5, +1.5] | 0.624 | 1000 |
| no_robots | cascade_nonorm_s44 [post hoc] | -2.2 | [-4.6, +0.2] | 0.0989 | 496 |
| sst | cascade_bare_s44 [post hoc] | +0.0 | [-3.0, +3.0] | 1 | 99 |
| catqa | cascade_bare_s44 [post hoc] | +16.6 | [+12.9, +20.1] | 3.84e-17 | 543 |
| dna_core | cascade_bare_s44 [post hoc] | -0.3 | [-2.8, +2.1] | 0.89 | 573 |
| dna_other | cascade_bare_s44 [post hoc] | +8.2 | [+4.1, +12.4] | 0.000287 | 364 |
| coconot | cascade_bare_s44 [post hoc] | -1.6 | [-3.7, +0.3] | 0.18 | 378 |
| phtest | cascade_bare_s44 [post hoc] | -1.1 | [-3.1, +0.8] | 0.329 | 1000 |
| no_robots | cascade_bare_s44 [post hoc] | -3.0 | [-5.6, -0.6] | 0.0275 | 496 |
| sst | cascade_T1_s44 [post hoc] | +1.0 | [-2.0, +5.1] | 1 | 99 |
| catqa | cascade_T1_s44 [post hoc] | +16.9 | [+13.3, +20.8] | 1.28e-17 | 543 |
| dna_core | cascade_T1_s44 [post hoc] | -0.3 | [-2.8, +2.1] | 0.89 | 573 |
| dna_other | cascade_T1_s44 [post hoc] | +8.2 | [+4.1, +12.4] | 0.000287 | 364 |
| coconot | cascade_T1_s44 [post hoc] | -1.6 | [-3.4, +0.3] | 0.18 | 378 |
| phtest | cascade_T1_s44 [post hoc] | -0.8 | [-2.9, +1.2] | 0.493 | 1000 |
| no_robots | cascade_T1_s44 [post hoc] | -2.2 | [-4.6, +0.2] | 0.0989 | 496 |

## Cascade on the final set across training seeds, % (historical checkpoint in the first table)

| Source | cascade_s42 | cascade_s43 | cascade_s44 | min–max |
|---|---:|---:|---:|---:|
| sst | 7.1 | 7.1 | 5.1 | 5.1–7.1 |
| catqa | 34.8 | 22.3 | 35.7 | 22.3–35.7 |
| dna_core | 25.8 | 23.4 | 23.6 | 23.4–25.8 |
| dna_other | 53.3 | 51.6 | 57.1 | 51.6–57.1 |
| coconot | 4.8 | 6.1 | 4.8 | 4.8–6.1 |
| phtest | 45.9 | 46.3 | 42.2 | 42.2–46.3 |
| no_robots | 14.1 | 15.3 | 10.5 | 10.5–15.3 |

## E. Component ablation on the final test set [post hoc], by checkpoint (to judge = share of prompts routed to Layer 2), %

| Checkpoint | Arm | to judge | AUROC | sst | catqa | dna_core | dna_other | coconot | phtest | no_robots |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical | cascade | 2.8 | 0.811 | 5.1 | 19.2 | 23.9 | 48.9 | 6.3 | 42.8 | 12.7 |
| historical | cascade_noviews [post hoc] | 2.8 | 0.814 | 5.1 | 19.2 | 23.9 | 48.9 | 6.3 | 42.6 | 11.7 |
| historical | cascade_nonorm [post hoc] | 2.8 | 0.811 | 5.1 | 19.2 | 23.9 | 48.9 | 6.3 | 42.8 | 12.7 |
| historical | cascade_bare [post hoc] | 2.8 | 0.814 | 5.1 | 19.2 | 23.9 | 48.9 | 6.3 | 42.6 | 11.7 |
| historical | cascade_T1 [post hoc] | 2.4 | 0.811 | 5.1 | 19.3 | 24.1 | 48.9 | 6.6 | 42.9 | 12.9 |
| s42 | cascade_s42 | 2.3 | 0.751 | 7.1 | 34.8 | 25.8 | 53.3 | 4.8 | 45.9 | 14.1 |
| s42 | cascade_noviews_s42 [post hoc] | 2.2 | 0.755 | 7.1 | 34.8 | 25.8 | 53.3 | 4.8 | 45.7 | 13.1 |
| s42 | cascade_nonorm_s42 [post hoc] | 2.3 | 0.751 | 7.1 | 34.8 | 25.8 | 53.3 | 4.8 | 45.9 | 14.1 |
| s42 | cascade_bare_s42 [post hoc] | 2.2 | 0.755 | 7.1 | 34.8 | 25.8 | 53.3 | 4.8 | 45.7 | 13.1 |
| s42 | cascade_T1_s42 [post hoc] | 1.7 | 0.754 | 7.1 | 35.2 | 25.8 | 53.8 | 5.0 | 46.0 | 14.5 |
| s43 | cascade_s43 | 2.2 | 0.773 | 7.1 | 22.3 | 23.4 | 51.6 | 6.1 | 46.3 | 15.3 |
| s43 | cascade_noviews_s43 [post hoc] | 2.2 | 0.776 | 7.1 | 22.3 | 23.4 | 51.6 | 6.1 | 46.1 | 13.9 |
| s43 | cascade_nonorm_s43 [post hoc] | 2.2 | 0.773 | 7.1 | 22.3 | 23.4 | 51.6 | 6.1 | 46.3 | 15.3 |
| s43 | cascade_bare_s43 [post hoc] | 2.2 | 0.776 | 7.1 | 22.3 | 23.4 | 51.6 | 6.1 | 46.1 | 13.9 |
| s43 | cascade_T1_s43 [post hoc] | 1.8 | 0.771 | 7.1 | 22.7 | 23.4 | 51.6 | 6.1 | 46.5 | 15.3 |
| s44 | cascade_s44 | 2.4 | 0.777 | 5.1 | 35.7 | 23.6 | 57.1 | 4.8 | 42.2 | 10.5 |
| s44 | cascade_noviews_s44 [post hoc] | 2.3 | 0.781 | 5.1 | 35.7 | 23.6 | 57.1 | 4.8 | 41.7 | 9.7 |
| s44 | cascade_nonorm_s44 [post hoc] | 2.4 | 0.777 | 5.1 | 35.7 | 23.6 | 57.1 | 4.8 | 42.2 | 10.5 |
| s44 | cascade_bare_s44 [post hoc] | 2.3 | 0.781 | 5.1 | 35.7 | 23.6 | 57.1 | 4.8 | 41.7 | 9.7 |
| s44 | cascade_T1_s44 [post hoc] | 2.1 | 0.779 | 6.1 | 36.1 | 23.6 | 57.1 | 4.8 | 42.0 | 10.5 |

## B. External guards at a threshold matched on the pool validation split (cascade safe-blocked there: 3.8%)

| Guard | threshold on P(unsafe) | sst | catqa | dna_core | dna_other | coconot | phtest | no_robots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| granite | 0.9669 | 12.1 | 8.3 | 43.6 | 79.7 | 1.9 | 5.2 | 1.8 |
| llamaguard | 0.8706 | 5.1 | 14.5 | 57.2 | 76.1 | 2.9 | 11.7 | 4.0 |
| shieldgemma [post hoc] | 0.7520 | 29.3 | 59.1 | 57.1 | 93.4 | 6.3 | 6.9 | 0.6 |

## C. HarmBench attack pass (200 seeds × 6 attacks × 3 variants) by checkpoint

| Checkpoint | bypassed / variants | rate % | 95% CI (seed-clustered) | seeds with ≥1 pass | technical_prefix | jailbreak_template | role_play | prefix_rewrite | language_prefix | decoding_manipulation |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| historical | 165/3600 | 4.6 | [2.5, 6.8] | 53/200 | 3.0 | 3.7 | 2.7 | 11.0 | 2.8 | 4.3 |
| s42 | 226/3600 | 6.3 | [3.9, 9.1] | 59/200 | 4.5 | 5.3 | 4.0 | 14.3 | 4.2 | 5.3 |
| s43 | 180/3600 | 5.0 | [2.8, 7.4] | 47/200 | 4.0 | 3.5 | 2.5 | 11.2 | 3.8 | 5.0 |
| s44 | 282/3600 | 7.8 | [5.2, 10.9] | 76/200 | 5.3 | 6.7 | 5.5 | 19.3 | 4.8 | 5.3 |

## D. Learner English: original minus corrected blocking among edited learner pairs, percentage points (non-inferiority margin +2)

| Checkpoint | Level | pairs | orig % | corr % | diff | sentence CI | essay CI | writer CI | non-inferior (writer) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| historical | A | 2079 | 19.1 | 19.9 | -0.77 | [-2.07, +0.43] | [-2.06, +0.47] | [-2.06, +0.49] | yes |
| historical | B | 2044 | 20.3 | 21.9 | -1.61 | [-2.69, -0.54] | [-2.83, -0.47] | [-2.88, -0.42] | yes |
| historical | C | 1344 | 21.3 | 22.5 | -1.26 | [-2.46, +0.00] | [-2.48, -0.07] | [-2.49, -0.08] | yes |
| historical | pooled | 5467 | 20.1 | 21.3 | -1.21 | [-1.88, -0.55] | [-1.90, -0.49] | [-1.91, -0.51] | yes |
| s42 | A | 2079 | 20.7 | 22.2 | -1.44 | [-2.84, -0.10] | [-2.88, +0.09] | [-2.81, -0.10] | yes |
| s42 | B | 2044 | 24.2 | 26.8 | -2.54 | [-3.77, -1.37] | [-3.85, -1.24] | [-3.93, -1.21] | yes |
| s42 | C | 1344 | 25.9 | 28.1 | -2.23 | [-3.50, -0.97] | [-3.70, -0.93] | [-3.60, -0.87] | yes |
| s42 | pooled | 5467 | 23.3 | 25.4 | -2.05 | [-2.78, -1.28] | [-2.83, -1.26] | [-2.86, -1.24] | yes |
| s43 | A | 2079 | 25.8 | 24.1 | +1.68 | [+0.29, +3.08] | [+0.34, +2.99] | [+0.29, +2.97] | no |
| s43 | B | 2044 | 26.5 | 26.7 | -0.20 | [-1.37, +1.03] | [-1.38, +1.10] | [-1.43, +1.00] | yes |
| s43 | C | 1344 | 28.9 | 30.4 | -1.41 | [-2.53, -0.22] | [-2.48, -0.30] | [-2.52, -0.24] | yes |
| s43 | pooled | 5467 | 26.8 | 26.6 | +0.22 | [-0.49, +0.93] | [-0.52, +0.95] | [-0.53, +0.97] | yes |
| s44 | A | 2079 | 23.2 | 24.1 | -0.87 | [-2.12, +0.43] | [-2.18, +0.40] | [-2.10, +0.41] | yes |
| s44 | B | 2044 | 23.6 | 24.3 | -0.73 | [-1.86, +0.34] | [-1.84, +0.40] | [-1.89, +0.42] | yes |
| s44 | C | 1344 | 26.0 | 26.3 | -0.30 | [-1.56, +0.82] | [-1.58, +0.94] | [-1.60, +1.04] | yes |
| s44 | pooled | 5467 | 24.1 | 24.7 | -0.68 | [-1.39, +0.00] | [-1.41, +0.07] | [-1.37, +0.05] | yes |
