# RQ4, second study: real learner English

Pre-registered on 23 September 2026, before the corpus was downloaded and before any
sentence was run through the pipeline. Nothing below may change after the runs except by
a dated amendment that says what changed and why.

## Question

Does the guard block a benign sentence more often when it is written in real non-native
English than when the same sentence is written in corrected, native-like English?

The first study (`LLM-Guard_RQ4_Parity.md`) answered a proxy form of this question with
rule-based rewrites of benchmark prompts. Its first listed limitation was that grammar
rules stand in for a population. This study uses text that a population wrote.

## Data

W&I+LOCNESS v2.1, the BEA-2019 shared task release (Bryant, Felice, Andersen and
Briscoe, 2019). Write & Improve essays and letters by learners at CEFR levels A, B and C,
and LOCNESS argumentative essays by native English students (Granger, 1998), each with
expert corrections in M2 format. The licence permits non-commercial use only; the thesis
is non-commercial research, the corpus is not redistributed, and only derived counts and
decisions leave the machine.

Units are sentences from the M2 files. Splits: the development set in full (levels A, B,
C and native N) and a seeded sample of 1,500 sentences per level from the training set
(A, B and C; there is no native training set). Sentences with fewer than three word
tokens or without an alphabetic character are dropped and counted. The correction is
annotator 0's; a sentence annotated only by another annotator uses the lowest annotator
id present. Both arms pass through the same detokeniser. Every sentence is labelled safe
(1): the guard should allow all of them.

## Arms

- `original`: the sentence as the writer wrote it.
- `corrected`: the same sentence with the annotator's edits applied.

Same identifiers, same order, same label. Each arm is run through the full pipeline
under the configuration the benchmark evaluation uses (classifier-only Layer 1,
normalisation and payload views on, Layers 2 and 3 loaded, refuse by default). The
comparison is `original` minus `corrected`: a positive difference means the non-native
form is blocked more often.

## Primary outcome and decision rule

The paired difference in the share of sentences blocked (not allowed; an escalated
sentence counts as blocked), among sentences with at least one edit, per level (A, B, C)
and pooled over the three learner levels, with a 95% percentile paired bootstrap interval
(2,000 resamples of sentence indices, seed `project.seed`) and an exact two-sided McNemar
test on the discordant pairs.

Non-inferiority margin: δ = 0.02, two percentage points absolute. The non-native form is
declared non-inferior for a comparison when the upper limit of its 95% interval is below
δ. This is a one-sided claim about increased blocking. A decrease is reported as a
decrease and is not called a benefit; the first study showed that fewer blocks under
rewriting is instability, not equity. The pooled comparison is the headline; the three
levels are reported with Holm-adjusted McNemar p-values. The margin was chosen before any
run because the first study's eight upper limits ran from −0.070 to +0.028, so a two-point
margin is one the proxy neither trivially passes nor trivially fails.

## Secondary outcomes

1. Dose-response. Mean harm-score shift and flips each way by number of edits (0, 1, 2,
   3 or more), and the Spearman correlation between edit count and score shift with a
   permutation p-value (2,000 permutations). Prediction from the first study: shift and
   flips grow with the number of edits.
2. Attribution by error category. For each ERRANT category present in the M2 annotations
   (determiner, preposition, subject-verb agreement, noun number, spelling, punctuation,
   word order, and the rest), the sentences containing it, flips each way and mean shift.
   Descriptive.
3. Native against learner. The blocked share on the `original` arm of LOCNESS against
   each learner level, with Wilson intervals. Descriptive only: the corpora differ in
   topic and genre, so no causal reading is made.
4. Proxy realism. The share of real learner edits that fall in the seven categories the
   first study's rules imitate (determiners, prepositions, subject-verb agreement, noun
   number, a missing verb, an inserted pronoun, word order), and edits per 100 tokens by
   level against the proxy's ceiling of two edits per prompt.

## Validity checks

Sentences with zero edits are byte-identical across arms and must receive identical
decisions and identical scores; any change fails the run, because it would mean the
pipeline is not deterministic and the paired design is void. Every identifier must appear
in both arms with the same label.

## What each result would mean

- Non-inferior pooled and at every level: on real learner prose, the guard's over-blocking
  does not rise with non-native grammar within two points, and the first study's null on
  the proxy is corroborated on text a population wrote.
- Not non-inferior somewhere: the bound is reported as it is, the level and the categories
  carrying the flips are named, and no parity is claimed for that level.
- A significant decrease: reported as dependence on surface form, as in the first study.

## Limits known in advance

The units are essay and letter sentences, not requests to an assistant, so the guard is
out of its domain in both arms equally, and block rates will be low; the bound matters
more than the point estimate. Only one annotator's correction is used. Proficiency levels
are the corpus's own. Nothing here concerns harmful prompts, and nothing here measures
vocabulary or discourse differences beyond what the corrections change.
