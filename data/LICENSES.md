# Data licences and redistribution policy

This file states which third-party data this repository redistributes, under
which terms, and which data it deliberately does not. The code of the
repository is licensed separately (see the top-level licence file).

## 1. Where each licence fact comes from

Three kinds of statement appear below, and they are marked so the reader can
weigh them:

* **(P)** taken from the study's pre-registration,
  `results/final_eval_20260926/PREREGISTRATION.md`, Section 3, written on
  26 September 2026 before the final sources were downloaded. These are the
  licences the study relied on when it decided to redistribute the final test set.
* **(L)** taken from the licence files distributed inside the W&I+LOCNESS
  archive (`licence.wi.txt`, `license.locness.txt`), whose presence the build
  recorded in `results/development/learner_arms_manifest.json`
  (`licence_files_seen`), together with the manifest's own statement
  `"non-commercial licence; not redistributed"`.
* **(U)** **not recorded anywhere in the study's manifests or pre-registration.**
  The pool and development-benchmark sources were used under the terms shown on
  their Hub dataset cards at download time, and the study wrote down their
  repository revisions and file digests but not their licences. The values below
  are given from memory of those dataset cards and have not been verified
  offline. None of these texts is redistributed here, so no licence obligation
  of this repository rests on them; verify them on the Hub before redistributing
  anything you build from them.

The manifests (`results/development/raw_sources_manifest.json`,
`raw_final_sources_manifest.json`, `pool_metadata.json`,
`learner_arms_manifest.json`, `parity_arms_manifest.json` and
`data/benchmarks_final/manifest.json`) are the source for every repository id,
revision, file digest and row count; they contain no licence fields.

## 2. Licence table

| Source | Repository / origin | Licence | Fact source | Redistributed here? |
|---|---|---|---|---|
| SimpleSafetyTests | `Bertievidgen/SimpleSafetyTests` | CC-BY-2.0 | (P) | Yes: 99 prompts in the final test set |
| CatQA (English) | `declare-lab/CategoricalHarmfulQA` | Apache-2.0 | (P) | Yes: 543 prompts |
| Do-Not-Answer | `LibrAI/do-not-answer` | Apache-2.0 | (P) | Yes: 937 prompts (573 core, 364 other) |
| CoCoNot (contrast, test) | `allenai/coconot` | ODC-BY | (P) | Yes: 378 prompts |
| PHTest (harmless) | `furonghuang-lab/PHTest` | MIT | (P) | Yes: 1,000 prompts |
| No Robots (test, first user turn) | `HuggingFaceH4/no_robots` | CC-BY-NC-4.0 | (P) | Yes: 496 prompts, **non-commercial** |
| W&I+LOCNESS v2.1 (BEA-2019) | University of Cambridge (CEWI) and UCLouvain (LOCNESS) | non-commercial research and educational use; no redistribution | (L) | **No.** Only per-sentence hashes, counts and edit statistics are recorded |
| JailbreakBench behaviors | `JailbreakBench/JBB-Behaviors` | MIT | (U) | No (rebuild) |
| XSTest | `walledai/XSTest` (gated) | CC-BY-4.0 | (U) | No (rebuild) |
| HarmBench standard | `walledai/HarmBench` (gated) | MIT | (U) | No (rebuild) |
| ToxicChat 0124 | `lmsys/toxic-chat` | CC-BY-NC-4.0 | (U) | No (rebuild) |
| Dolly 15k | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | (U) | No (rebuild) |
| BeaverTails 30k | `PKU-Alignment/BeaverTails` | CC-BY-NC-4.0 | (U) | No (rebuild) |
| WildJailbreak | `allenai/wildjailbreak` (gated) | ODC-BY with AI2 responsible-use terms | (U) | No (rebuild) |
| OR-Bench | `bench-llm/or-bench` | CC-BY-4.0 | (U) | No (rebuild) |

## 3. What this repository redistributes

**`data/benchmarks_final/final_test/test.jsonl`** (3,453 prompts) and its
`manifest.json`. The prompt texts are reproduced verbatim from six public
datasets under their own licences, with this attribution; the ids, labels,
categories, `source` field and `over_512_tokens` flags are this project's
annotations. Nothing else from any source is copied into the file.

Attribution, as the pre-registration lists them:

* **SimpleSafetyTests**, Bertie Vidgen et al., `Bertievidgen/SimpleSafetyTests`
  at revision `98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5`, CC-BY-2.0.
* **CategoricalHarmfulQA (CatQA)**, DeCLaRe Lab, `declare-lab/CategoricalHarmfulQA`
  at `321e068b391b7d1aeeb882ac468356a9513c9167`, Apache-2.0.
* **Do-Not-Answer**, LibrAI, `LibrAI/do-not-answer` at
  `74e74f2e4507ef256fe536f78a776f4a1ff67955`, Apache-2.0.
* **CoCoNot**, Allen Institute for AI, `allenai/coconot` (config `contrast`,
  split `test`) at `2cbe16aabf9069f17e48c8daad8aeabc29469eb7`, ODC-BY. The ODC-BY
  licence requires this notice to be kept with any redistribution of the data.
* **PHTest**, Furong Huang Lab, `furonghuang-lab/PHTest` at
  `edb80210812ecc7dab2219aa2e70921710c01888`, MIT.
* **No Robots**, Hugging Face H4, `HuggingFaceH4/no_robots` (split `test`) at
  `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, CC-BY-NC-4.0.

**Consequence of the No Robots terms.** 496 of the 3,453 prompts (`source ==
"no_robots"`) are licensed CC-BY-NC-4.0. The final test set as a whole may
therefore be used for **non-commercial purposes only**. A user who needs a
commercially usable subset can drop the `no_robots` rows (the remaining 2,957
prompts are under CC-BY-2.0, Apache-2.0, ODC-BY and MIT) and must then report
numbers on that subset, not on the pre-registered set.

The row counts above are after the pre-registered screen (within-source
duplicates, near-duplicates of the pool and development benchmarks,
cross-source duplicates, and the seeded sample of 1,000 PHTest prompts), so the
file is a filtered subset, not a copy, of each source.

## 4. What this repository does not redistribute, and why

* **The learner corpus (W&I+LOCNESS).** Its licence allows non-commercial
  research use and forbids redistribution. No sentence of it is under `data/`;
  the learner arms must be rebuilt from a copy obtained under the corpus licence
  (`data/README.md`, step 4). `results/development/learner_*.json` hold counts,
  scores and attributions only. **Known exception, to be resolved before
  publication:** the per-prompt prediction files
  `results/final_eval_20260926/learner_{original,corrected}_s{42,43,44}_predictions/*.jsonl`
  (24 files) were written with the full sentence text in each record, in both the
  learner-written and corrected form. Those `text` fields reproduce the corpus
  and must be stripped (keeping `id`, `label`, scores and decisions) or the
  files removed before this repository is made public. The same applies to
  `pooltest_layer1_predictions/test.jsonl` and the four `val_*_predictions/`
  files, which carry pool prompts, including gated WildJailbreak text.
* **The training pool (`data/distillation_pool/`).** 19,120 prompts drawn from
  WildJailbreak, BeaverTails, Dolly and ToxicChat; WildJailbreak is gated, and
  two of the four carry non-commercial terms on their dataset cards (unverified,
  Section 1). The pool is rebuilt deterministically from the pinned sources
  (`data/README.md`, steps 1 and 2) and its split hashes are recorded in
  `results/final_eval_20260926/frozen.json`, so a rebuild can be checked exactly.
* **The five development benchmarks (`data/benchmarks/`).** Direct relabelled
  extracts of JailbreakBench, XSTest, HarmBench, ToxicChat and OR-Bench. Rebuilt
  in the same run as the pool; hashes in `frozen.json`.
* **The parity arms (`data/benchmarks_parity/`).** Derived from the development
  benchmarks' safe prompts; rebuilt by `scripts/make_l2_benchmarks.py`.
* **Raw downloads (`data/raw/`).** The sources themselves, in their original
  files; 540 MB for the pool sources alone. Fetched with the two fetch scripts,
  which verify every file against the digest the Hub publishes.

The pool and benchmark texts are not withheld because their licences forbid
sharing in every case; they are withheld because the gated and non-commercial
sources among them make a mixed file the wrong thing to publish, and because
the pinned fetch plus seeded build reproduces them byte for byte.

## 5. Obligations when you use this data

* Keep this file and the attribution in Section 3 with any copy of the final
  test set.
* Treat the final test set as non-commercial while it contains the No Robots
  rows.
* Accept the Hub terms of the gated datasets yourself before fetching them; the
  fetch scripts use your own token and never embed one.
* Do not redistribute W&I+LOCNESS texts or anything that reproduces them.
* Cite the original datasets, not only this repository, in any publication.
