# analysis/

Derived tables, recomputed from the per-prompt prediction files in the parent folder. No model is loaded by any of these scripts.

| File | Written by | Content | Paper |
|---|---|---|---|
| `results.json`, `tables.md` | `scripts/final_eval/analyse_final.py --study <study>` | Sections A (rates, Wilson CIs), pooled AUROC, paired AUROC differences, paired differences vs the cascade (bootstrap + exact McNemar), seed range, E (component ablation), B (matched thresholds), C (attack pass), D (learner English) | Tables I, III-VI, X-XII |
| `frontiers.json` | `frontiers.py` | threshold frontiers (macro over sources), native operating points, offline routing-band sweep | Fig. frontiers, Fig. operating points |
| `calibration.json` | `calibration.py` (reads `../../development/judge_tuning*.jsonl`) | ECE with bootstrap CIs and reliability bins, in distribution and on the final set | Table VII, Fig. reliability |
| `per_category.json`, `per_category.md` | `per_category.py` | rates by category with Wilson CIs | Table VIII |
| `guard_agreement.json` | `guard_agreement.py` | native verdict vs token-probability score per guard | guard-agreement paragraph |
| `judge_band.json` | `judge_band.py` | judge band sweep under both uncertain policies | Table IX |
| `paired_views.json` | `paired_views.py` (reads `../../development/redteam_all200*_variants.jsonl`) | paired effect of the leading-clause view, request-clustered bootstrap | Fig. views, Table XI text |

Bootstraps use B = 2,000 replicates, alpha = 0.05 and the study seed 20260926; each comparison draws from its own stream `numpy.random.default_rng([20260926, zlib.crc32(name)])`, named after the comparison, so an interval does not depend on how many arms the study contains (analyse_final.py `stream_rng`). Single rates carry 95% Wilson intervals; paired decisions an exact two-sided McNemar test.
