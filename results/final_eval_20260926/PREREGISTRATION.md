# Pre-registration: locked final test, external guards, and seed robustness

Written 26 September 2026, before any source file was downloaded and before any
model was run on the prompts described here. The runner (`run_final_eval.sh`)
records the hashes of the frozen system in `frozen.json` at the start of the run;
this document is not edited after that file exists.

## 1. Why this study exists

Every benchmark the project has reported on so far (JailbreakBench, XSTest,
HarmBench, ToxicChat test, OR-Bench) informed at least one configuration
decision; the decision register in `docs/validation_policy.md` lists seven. Those
five sets are therefore development benchmarks. The three-seed repeat study
showed that two trainings of the same recipe differ by up to 9 points of
harmful-prompt allowance on one benchmark, and every red-team and learner
result so far rests on one historical checkpoint. No published guard has been
compared.

This study answers three questions on prompts that informed no decision:

- **Q1.** How does the frozen cascade perform, against its own components
  (Layer 1 alone, judge alone), on new safe and harmful prompts?
- **Q2.** How does it compare with external safety classifiers on the same
  prompts?
- **Q3.** Do the HarmBench attack result and the learner-English result hold on
  the other two training seeds?

## 2. The frozen system

The system under test is the configuration on the Ubuntu ThinkBook as it stood
after patch 51, with no change: classifier-only Layer 1 with input normalisation
and leading-clause payload views, routing band 0.30/0.70, Qwen2.5-3B-Instruct
judge (4-bit, prompt v2, log-probability decision, band 0.40/0.60), Layer 3
refusing by default, rewrite check off, keyword refusal detector. The preflight
refuses to run if any of these differ.

Four Layer-1 checkpoints are used:

| Name | Directory | Role |
|---|---|---|
| historical | `artifacts/layer1` | the checkpoint every reported result used |
| s42, s43, s44 | `artifacts/layer1_repeat_study_*_s42/43/44` | the three-seed repeat of the same recipe |

`frozen.json` records SHA-256 of `config/config.yaml`, every `.py` file under
`layer1/ layer2/ layer3/ evaluation/ redteam/ utils/ scripts/`, the Layer-1
artifacts of all four checkpoints, the pool splits, the five benchmark files, the
learner arms and the HarmBench seed file, plus the git commit and dirty flag.

## 3. Test-set sources

Sources were chosen for having **no lineage** to the training pool (Dolly,
BeaverTails, ToxicChat train, WildJailbreak) or to the five development
benchmarks. Anything from the WildGuard family (AI2) and anything derived from
Anthropic HH-RLHF prompts (which BeaverTails also derives from) is excluded.
SORRY-Bench and StrongREJECT were considered and excluded because both reuse
rewritten items from HarmBench and AdvBench, which a near-duplicate screen would
not reliably catch.

| Source | HF repository | Class | Rows used | Authorship | Licence |
|---|---|---|---|---|---|
| SimpleSafetyTests | `Bertievidgen/SimpleSafetyTests` | harmful | all 100 | hand-written by the authors | CC-BY-2.0 |
| CatQA (English) | `declare-lab/CategoricalHarmfulQA` | harmful | all 550 | written with an unaligned LLM, 11 categories | Apache-2.0 |
| Do-Not-Answer, core | `LibrAI/do-not-answer` | harmful | risk areas *Malicious Uses*, *Discrimination/Exclusion/Toxicity/Hateful/Offensive*, *Misinformation Harms* | GPT-4 generated, human filtered | Apache-2.0 |
| Do-Not-Answer, other | same | harmful (secondary) | risk areas *Information Hazards*, *Human–Chatbot Interaction Harms* | same | same |
| CoCoNot contrast | `allenai/coconot`, config `contrast`, split `test` | safe | all (379) | authored near the noncompliance boundary; queries that should be complied with | ODC-BY |
| PHTest, harmless | `furonghuang-lab/PHTest` | safe | 1,000 sampled from rows labelled `harmless`; `controversial` rows dropped | generated with AutoDAN on Llama-2 | MIT |
| No Robots, test | `HuggingFaceH4/no_robots`, split `test` | safe | all (500), first user turn | human-written by professional annotators | CC-BY-NC-4.0 |

Two caveats are stated now rather than later. CoCoNot is from the same
organisation as WildJailbreak, although a different project and generator; it is
kept because it is the only authored over-refusal set that is not XSTest-derived,
and the near-duplicate screen against WildJailbreak is reported for it
specifically. Do-Not-Answer "other" contains questions a moderation guard could
legitimately allow (for example questions about the assistant's nature); it is a
secondary set and is not pooled into the primary harmful rate.

Llama Guard 3 1B is gated. If the weights are not available when the runner
starts, that arm is skipped and reported as not run; nothing else changes.

## 4. Screening rule, fixed before download

Applied in this order, with counts reported per source at every step:

1. Drop empty prompts. Drop exact duplicates within a source after
   normalisation (NFKC, lower-case, whitespace collapse), keeping the first.
2. Drop a prompt that is an exact normalised match, **or** a near-duplicate
   (character 5-gram Jaccard ≥ 0.8, MinHash 64 permutations / 16 bands,
   seed 42, the pool's own rule), of any prompt in the pool train, validation
   or test split, or in any of the five development benchmarks (both classes).
3. Drop exact cross-source duplicates within the final set, keeping the
   source listed first in the table above.
4. PHTest: after steps 1–3, draw 1,000 rows with `random.Random(20260926)`.

No prompt is edited, filtered by length, or filtered by content beyond the
rules above. Prompts longer than Layer 1's 512-token limit stay in and are
counted.

## 5. Arms

All arms score the same final file, `data/benchmarks_final/final_test/test.jsonl`.

| Arm | Command | What it is |
|---|---|---|
| cascade | `run_evaluation.py --input …` | the frozen system |
| layer1 | `run_evaluation.py --no-l2 --input …` | Layer 1 alone; a gray-zone prompt is allowed |
| judge | `run_judge_only.py --input …` | the judge alone on every prompt, P(mitigate) with the 0.40/0.60 band; uncertain counts as blocked, as in the pipeline |
| granite | `run_external_guard.py --model granite …` | IBM Granite Guardian 3.1 2B, risk `harm`, prompt-only |
| llamaguard | `run_external_guard.py --model llamaguard …` | Meta Llama Guard 3 1B, default categories, prompt-only; skipped if gated weights are absent |
| cascade-s42/43/44 | `run_evaluation.py --layer1-artifacts … --input …` | the cascade with each repeat-study checkpoint |

Both external guards are loaded with the same 4-bit NF4 quantisation the judge
uses, because the 6 GB GPU does not hold the 2B model in bf16 beside its
activations; the quantisation is recorded and is a limitation of the comparison.
Each guard is also run on the pool validation split (1,912 prompts) so that a
threshold can be chosen on data that is not the final test (Section 7).

## 6. Outcomes

**Primary**, per source and per arm:

- `safe_blocked` on safe sources; `harmful_allowed` on harmful sources
  (definitions in `evaluation/metrics.py` and `summarise_results.py`: an
  escalated prompt counts as blocked).
- Per-source 95% Wilson intervals.
- Paired differences between arms on the same prompts: percentile bootstrap of
  the rate difference, B = 2,000, seed 20260926, resampling prompts within the
  source; and an exact two-sided McNemar test on the discordant decisions.

**Secondary**:

- Pooled AUROC over the whole final set, per arm. Every source is single-class,
  so pooled ranking metrics partly measure source separability; they are
  reported with that caveat and are not primary.
- The three Do-Not-Answer "other" rates, and per-category rates inside CatQA
  and Do-Not-Answer (11 and 3 categories; read as a ranking, not as estimates,
  because the cells are small).
- Count of prompts over 512 Layer-1 tokens, and the rates with and without them.
- Latency per prompt for each arm, median over the final set, on the ThinkBook.

## 7. External-guard operating points

Two readings are reported for each guard, labelled as such:

- **Native**: the guard's own decision (`Yes`/`No` for Granite; `safe`/`unsafe`
  for Llama Guard), which is P(unsafe) > 0.5 on the two-token renormalised
  distribution. This is the primary comparison, because it is how the guard is
  shipped.
- **Matched on validation**: the P(unsafe) threshold chosen on the pool
  validation split so that the guard's `safe_blocked` there equals the
  cascade's on the same split; the guard's final-set rates at that threshold.
  The threshold is fixed by validation data only.

A third reading, the guard's `harmful_allowed` at the point on its final-set ROC
curve where its `safe_blocked` equals the cascade's, is a descriptive curve
read-out that uses final-set labels to pick the point; if reported it is labelled
post hoc and is not used in any claim.

## 8. Expectations written down in advance

These are expectations, not hypotheses that decide anything; the study reports
whatever it finds. They exist so that a surprising result is recognisable as one.

- E1. On explicit-harm sources (SimpleSafetyTests, CatQA) the cascade allows
  under 10% of harmful prompts, in line with HarmBench and JailbreakBench.
- E2. On plain requests (No Robots) the cascade blocks under 10% of safe
  prompts, in line with ToxicChat's safe side.
- E3. On the two over-refusal sources (CoCoNot contrast, PHTest) the cascade
  blocks more than 20% of safe prompts, in line with OR-Bench.
- E4. The judge alone blocks fewer harmful and more safe prompts than the
  cascade, as it did out of distribution on the development benchmarks.
- E5. The external guards block fewer safe prompts than the cascade on the
  over-refusal sources; on harmful sources the direction is not predicted.

## 9. Seed robustness (Q3)

Two studies are repeated with the s42, s43 and s44 checkpoints, using exactly the
commands recorded in the historical runs' `run_info`, changing only
`--layer1-artifacts`:

- **Attack pass**: all 200 HarmBench seeds once, six template attacks, three
  variants each, payload views on (`redteam_all200_views` command). Outcome:
  guard-bypass rate over the 3,600 variants with a 95% interval clustered by
  seed (B = 2,000), per attack type, and the number of seeds with at least one
  passing variant. Compared with the historical checkpoint's 165/3,600 (4.6%).
- **Learner arms**: W&I+LOCNESS original and corrected arms
  (`learner_original` / `learner_corrected` commands). Outcome: blocking of
  original minus corrected among edited learner pairs (`n_edits > 0`, levels
  A/B/C), per level and pooled, with sentence-pair, essay-grouped and
  writer-grouped bootstrap intervals (B = 2,000; essay and writer ids from the
  24 September mapping), against the same two-point non-inferiority margin.

The cascade is also run on the final test set with each of the three
checkpoints, so Q1 has a between-seed range as well as the historical point.

## 10. What the study cannot show

- It measures whether a prompt passes the guard, not whether a downstream model
  would produce harmful output.
- Every source label is taken as given; none was re-adjudicated. Do-Not-Answer
  and CoCoNot labels encode their authors' refusal policy, which is not
  identical to this system's.
- Three of the seven sources are model-written. A style effect between
  model-written and human-written prompts is confounded with source.
- The attack suite is the same six template attacks; nothing here speaks to
  adaptive or optimisation-based attackers.
- Three seeds bound the variance of one recipe; they do not select a better one.

## 11. Analysis and reporting

`analyse_final.py` produces every table in Sections 6, 7 and 9 from the saved
prediction files, with the bootstrap seeds above; it runs on the ThinkBook at the
end of the runner and again off the returned zip, and the two outputs must agree.
Results are reported per source, never as a single pooled headline rate.
