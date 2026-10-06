# Second amendment, 28 September 2026

The second of two amendments dated 28 September. The first,
`AMENDMENT-2026-09-28.md`, records the Llama Guard arm and is not modified by
this one; the two are unrelated.

**The Layer-2 judge model was substituted and the judge arm re-scored.** This is
**not a pre-registered arm.** It is marked `[post hoc]` wherever it appears, on
the same footing as ShieldGemma. Nothing in the cascade, the Layer-1 arm, the
original judge arm, the three guard arms, the frozen test set or the
seed-robustness studies is altered. Every previously reported point estimate, decision and conclusion was checked
and is unchanged. Bootstrap intervals were later regenerated; see the
correction at the end of this file.

## 1. What was done and why

The pre-registration fixes the cascade's judge as Qwen2.5-3B-Instruct. Section
III reports that the judge alone outranks the full cascade, 0.878 against 0.811.
The obvious objection is that this says something about Qwen rather than about the
architecture: a different 3B model might not outrank the cascade at all, in which
case the finding is a quirk of one checkpoint.

`meta-llama/Llama-3.2-3B-Instruct` was substituted into the same slot to test
that. It is a close match to the incumbent in size and cost: 3.21 B parameters
against Qwen's 3.09 B, from an unrelated vendor and pretraining corpus, loaded in
the same 4-bit NF4 quantisation, the same `logprob` consistency mode and the
same 0.40/0.60 band. The weights and the prompt version differ; Section 2 fixes
how the version was chosen. Everything else is held fixed.

| | |
|---|---|
| Repository | `meta-llama/Llama-3.2-3B-Instruct` |
| Revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Quantisation | 4-bit NF4, as the incumbent |
| Decision mode | `logprob`, band 0.40 / 0.60, as the incumbent |
| Prompts scored | 3,453 frozen test prompts, as every other arm |

## 2. How the prompt version was chosen, and why that had to be settled first

The shipped judge runs prompt version `v2`. Choosing a version for Llama after
seeing its results would be precisely the freedom the pre-registration exists to
remove, so the rule was fixed in writing before any Llama test number existed.

`docs/validation_policy.md` records the original decision as "Judge prompt v2 over
v1, pool validation, 300 sampled prompts". `scripts/tune_judge.py` was therefore
re-run for Llama with identical parameters — the same 300-prompt validation
sample, seed 42, K=5, both versions, `data/distillation_pool/val.jsonl`, never the
evaluation benchmarks. A `--judge-model` flag was added to that script and to
`run_judge_only.py`; absent the flag both behave exactly as before.

**The criterion is not AUROC.** Inspecting the original tuning summary shows v2
was chosen for Qwen despite *worse* AUROC than v1. What separates them is accuracy
on decided prompts under the logprob band, together with a large reduction in
over-refusal. Had "pick the higher AUROC" been assumed, Llama would have been
selected on a different basis than Qwen and the comparison would have been
silently unfair.

The pre-specified rule was therefore: **the version with the higher logprob-band
accuracy on decided prompts wins.** Validation results, n=300:

| Judge | Version | Band accuracy | Benign blocked | Harmful allowed | Undecided | AUROC |
|---|---|---:|---:|---:|---:|---:|
| Qwen | v1 | 0.7450 | 0.4600 | 0.0467 | 0.0067 | 0.8869 |
| Qwen | **v2** | **0.7905** | 0.1600 | 0.2533 | 0.0133 | 0.8660 |
| Llama | **v1** | **0.8923** | 0.0933 | 0.1200 | 0.0100 | 0.9419 |
| Llama | v2 | 0.8639 | 0.0667 | 0.2000 | 0.0200 | 0.9328 |

Applied to Qwen the rule selects **v2**, which is what the shipped configuration
uses. The reconstructed criterion therefore reproduces the original decision
rather than inventing a new one, which is what makes the symmetric claim
checkable. Applied to Llama it selects **v1**.

That margin is 0.028 on 300 prompts and is not strongly determined. Both versions
were therefore scored on the test set, with their roles fixed in advance:

- **Primary**: Llama on **v1**, the version the rule selected.
- **Secondary**: Llama on **v2**, the prompt the shipped judge uses — a pure
  drop-in replacement reading.

Both are reported below whichever way they fell.

## 3. A format difference that would have corrupted a different comparison

During tuning, Llama outputs were repeatedly marked unparseable. They were not
wrong. They read, for example:

```
MITIGATE
CATEGORY: disinformation
SEVERITY: 0.9
RATIONALE: The user is asking the model to create and disseminate false information...
```

The verdict is correct and the format is otherwise right; Llama omits the literal
`DECISION:` label the prompt demands, and the parser requires it
(`re.search(r"DECISION:\s*(allow|mitigate)", raw, re.IGNORECASE)`). Qwen emits the
label, Llama does not.

This matters in two ways.

**It does not affect this comparison.** The `logprob` mode teacher-forces
`DECISION:` into the assistant turn and reads the next token, so it never depends
on the model reproducing the label. That is the mode the pipeline runs, the mode
the reported judge arm uses, and the mode of the selection criterion above. Both
test passes recorded **zero** unparsed outputs.

**It would have corrupted greedy or vote mode, asymmetrically.** Every one of the
discarded outputs was a `MITIGATE`; none was an `ALLOW`. Discarded verdicts are
therefore disproportionately blocks, which would have made Llama appear more
permissive than it is. Under greedy and vote the tuning run recorded 3 and 49
invalid outputs for Llama v1 against 0 and 0 for both Qwen versions — a
format-compliance count, not a judgement count, and it should never be read as
the latter.

The parser was **not** loosened to accept a bare leading verdict, although it
would parse these correctly. That is shared code, and changing it would silently
move the incumbent judge's greedy and vote numbers too.

## 4. Results on the frozen test set

Pooled AUROC, the same quantity and the same 3,089 prompts as every other arm:

| Arm | Pooled AUROC |
|---|---:|
| **Judge, Llama-3.2-3B v1** [post hoc] | **0.928** |
| Judge, Llama-3.2-3B v2 [post hoc] | 0.924 |
| Granite Guardian 3.1 2B | 0.923 |
| Judge, Qwen2.5-3B (shipped) | 0.878 |
| Llama Guard 3 1B | 0.834 |
| ShieldGemma 2B [post hoc] | 0.826 |
| Cascade | 0.811 |

The two prompt versions agree to 0.004, so the weakly-determined selection in
Section 2 does not carry the result.

Operational rates by source, percent, harmful sources reporting harmful-allowed
and safe sources safe-blocked, lower better throughout:

| Source | Cascade | Judge (Qwen) | Judge (Llama v1) | Judge (Llama v2) |
|---|---:|---:|---:|---:|
| sst | 5.1 | 2.0 | **0.0** | **0.0** |
| catqa | 19.2 | 2.4 | **0.9** | 1.8 |
| dna_core | 23.9 | 32.6 | **23.9** | 32.5 |
| dna_other (secondary) | 48.9 | 64.0 | 69.0 | 75.0 |
| coconot | **6.3** | 20.4 | 21.7 | 13.5 |
| phtest | 42.8 | 29.6 | 25.7 | **17.2** |
| no_robots | 12.7 | 7.1 | 5.4 | **2.4** |

Against the incumbent judge, Llama v1 is better on five of seven sources,
including all three primary harmful sources, and worse only on the secondary
Do-Not-Answer set and marginally on CoCoNot.

Paired against the cascade, arm minus cascade in percentage points, negative
better:

| Source | Llama v1 | 95% CI | McNemar p |
|---|---:|---|---:|
| sst | -5.1 | [-10.1, -1.0] | 0.0625 |
| catqa | **-18.2** | [-21.5, -15.1] | 1.06e-27 |
| dna_core | +0.0 | [-3.7, +3.7] | 1 |
| dna_other | +20.1 | [+14.3, +25.8] | 3.98e-10 |
| coconot | +15.3 | [+10.8, +19.8] | 4.39e-10 |
| phtest | **-17.1** | [-20.7, -13.4] | 1.18e-19 |
| no_robots | **-7.3** | [-10.3, -4.2] | 3.18e-06 |

The v2 reading is directionally the same and stronger on the safe sources:
phtest -25.6 [-29.1, -22.3], no_robots -10.3 [-13.3, -7.5], coconot +7.1, at the
cost of dna_core +8.6 rather than +0.0.

Wall clock is recorded but should not be compared closely: 884 s for v1 and 1,018 s
for v2 against 2,014 s for the incumbent, on the same 3,453 prompts, but the
incumbent ran on 26 September under different machine load.

## 5. What this changes, and what it does not

**It strengthens the judge-versus-cascade finding and removes the obvious
objection to it.** A single 3B model beats the whole three-layer cascade on four
of seven sources, three of them at p below 1e-05, and ties on a fifth. Two
unrelated 3B judges now both outrank the cascade, so that result is a property of
the architecture rather than of Qwen.

**It undermines the explanation given for the Granite gap.** Section III attributes
the cascade's deficit to Granite being a purpose-built published guard. A
general-purpose instruct model, prompted as a judge, now ranks at 0.928 against
Granite's 0.923 — a paired bootstrap difference of +0.0049 with a 95%
interval of [-0.0015, +0.0113] that covers zero, so this is parity, not a lead.
Purpose-built safety training is therefore not on its own what produced
Granite's advantage. What did produce it is not identified here.
The two sit at different operating points — Granite allows less harm (dna_core
16.9 against 23.9) and over-blocks far more (phtest 55.9 against 25.7) — but in
ranking quality they are level.

**It does not improve the cascade.** Only 2.8% of prompts reach Layer 2, so a
better judge barely moves the system's own rates. The gap between the cascade and
the component inside it widens from 0.067 to 0.117. This sharpens the existing
criticism rather than resolving it.

**One result runs the other way, and it is the most useful one here.** CoCoNot
contrast holds prompts authored at the compliance boundary that should be
answered. The cascade blocks 6.3% of them; the Qwen judge blocks 20.4% and the
Llama judge 21.7%. Both judges are substantially worse than Layer 1 on exactly the
prompts Layer 1 exists to handle cheaply, and they fail there in the same
direction despite being unrelated models. That is the clearest evidence in this
study that a cheap first layer contributes something an LLM judge does not, and it
argues that the cascade's premise is sound while its routing thresholds are wrong
— the thresholds Section IV already records as never having been validated.

## 6. Cautions

This arm is post hoc and cannot be used to rescue a conclusion any more than to
damage one.

It compares two models, not a population. It shows Qwen2.5-3B is a weak choice for
this slot; it does not show Llama-3.2-3B is a good one in general, and no claim
about "3B judges" follows from n=2.

The Do-Not-Answer "other" figure (69.0% allowed for Llama v1) carries the same
caution already applied to ShieldGemma's 92.9%: that secondary source contains
questions a guard may legitimately allow, which is why it is excluded from the
primary harmful rate.

The prompt versions were selected on the pool validation split, never on the
evaluation benchmarks, so no test-set information entered the choice.

## 7. Files and commands

```bash
python scripts/tune_judge.py --judge-model meta-llama/Llama-3.2-3B-Instruct \
    --output data/processed/judge_tuning_llama.jsonl \
    --summary data/processed/judge_tuning_llama_summary.json
```

```bash
python scripts/final_eval/run_judge_only.py --judge-model meta-llama/Llama-3.2-3B-Instruct \
    --prompt-version v1 --input data/benchmarks_final/final_test/test.jsonl \
    --output data/processed/final_eval_20260926/final_judge_llama_v1.json
```

The same with `--prompt-version v2` produces the secondary reading. Then re-run
`scripts/final_eval/analyse_final.py --study data/processed/final_eval_20260926`.

| File | Contents |
|---|---|
| `judge_tuning_llama_summary.json` | Llama v1/v2 validation metrics, alongside the untouched Qwen `judge_tuning_summary.json` |
| `final_judge_llama_v1.json` | Primary arm, test results and run info |
| `final_judge_llama_v2.json` | Secondary arm |
| `final_judge_llama_v{1,2}_predictions/` | Per-prompt rows |

---

### Correction, 28 September 2026 (bootstrap streams)

Every bootstrap interval above was regenerated after a defect was fixed in
`scripts/final_eval/analyse_final.py`. The script drew every bootstrap from one
shared generator created once in `main()`, so draws were consumed in analysis
order and an interval depended on how many arms had been resampled before it.
Adding an arm silently moved unrelated intervals; point estimates never moved,
because they are exact counts, which is what hid it.

Each bootstrap now derives its own stream from the fixed seed and a stable name,
and `grouped_boot` sorts its groups rather than taking them in file order. The
intervals in this file are the regenerated ones. No point estimate, decision,
McNemar p-value or conclusion changed. Most shifts are at the first decimal;
on SimpleSafetyTests they reach a full point, because with $n=99$ one
resampled prompt is worth 1.01 points and the interval can only move in
those steps.
