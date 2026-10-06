# Amendment, 28 September 2026

Supersedes item 1 of `AMENDMENT-2026-09-27.md`. That file is not edited; it
recorded what was true when it was written, and this records what changed.

*Hugging Face account usernames were redacted from this file on publication;
nothing else was changed.*

**Llama Guard 3 1B has now run.** The pre-registered two-guard comparison in
Section 5 is complete. Nothing in the cascade, the judge arm, the Layer-1 arm, the
Granite arm, the ShieldGemma arm, the frozen test set or the seed-robustness
studies is altered. No previously reported point estimate, decision or
conclusion changes. Bootstrap intervals were later regenerated; see the
correction at the end of this file.

## 1. How access was obtained

On 27 September the Hugging Face API returned, for the project's first account:

> Your request to access model meta-llama/Llama-Guard-3-1B has been rejected by
> the repo's authors.

That rejection still stands for that account; it was re-checked on 28 September
and is unchanged. A **separate request from a different, second account was
approved by Meta**, and the weights were downloaded through Meta's own gated
repository under the granted licence.

This is stated explicitly because the 27 September amendment recorded a decision
not to obtain these weights from ungated community re-uploads, on the grounds
that sourcing them elsewhere would defeat the authors' decision rather than
satisfy it. That reasoning is unchanged and was not circumvented. The gate was
passed, not bypassed: a new request was reviewed and granted by the repository's
authors through the same official channel that declined the first one.

| | |
|---|---|
| Repository | `meta-llama/Llama-Guard-3-1B` |
| Revision | `acf7aafa60f0410f8f42b1fa35e077d705892029` (pinned on the command line) |
| Gate type | `manual` — reviewed individually by Meta |
| Account granted | second account |
| Account rejected, still | first account |

## 2. This arm is pre-registered; one caveat is recorded anyway

Sections 3, 5 and 7 named this model, this scoring rule and both readings before
the study ran. Nothing about how it is scored was chosen after the fact, so it
carries the pre-registration's guarantee and is **not** marked post hoc, unlike
ShieldGemma.

Two facts are recorded so a reader can judge that for themselves rather than take
it on trust.

The arm ran after the other arms' results were known. That is a consequence of the
gate, not a choice, and it grants no analytic freedom here because the model, the
score, the decision rule and both readings were all fixed in advance. It would
matter if any of those had been selected now; none were.

The harm score is produced by a code path that did not exist when the
pre-registration was frozen (see item 3). That path implements what Section 7
already specified — P(unsafe) across the two answer tokens — for a model whose
output shape the previous code did not handle. It corrects an implementation, it
does not change a definition.

## 3. The first attempt on 28 September was withdrawn

The arm was run, completed cleanly, and produced a wrong number. It is kept in
`withdrawn_badstep_20260928/`, with a full account in that directory's README.

In short: `run_external_guard.py` read P(unsafe) from the first generated step.
Granite and ShieldGemma answer there. Llama Guard 3 emits a `\n\n` token first, so
the first step was the newline step, and P(unsafe) became a ratio between two
negligible numbers — shaped like a probability, made of rounding error. The median
probability mass on the answer words was **0.0000**, against 0.9997 for Granite and
1.0000 for ShieldGemma.

It was caught by the shape of the result rather than by any error: an accuracy
above both other guards next to the worst AUROC of the three. Those two readings
come from different sources — accuracy from the generated verdict, AUROC from the
stored score — so one being best while the other is worst means they disagree
about the same prompts, and one of them is not measuring the guard.

`answer_step()` now locates the step the verdict is stated on, rather than assuming
it. That this leaves the other guards untouched was checked, not asserted, by
running the patched and unpatched scripts over the same 60 prompts:

| Comparison | max difference in P(unsafe) | decisions changed |
|---|---:|---:|
| patched vs unpatched | **0.000e+00** | 0 |
| unpatched vs the 26 Sep reported run | 6.03e-03 | 0 |
| patched vs the 26 Sep reported run | 6.03e-03 | 0 |

The two scripts agree bitwise. Both drift from the September run by the same
negligible amount, which is run-to-run nondeterminism in the 4-bit kernels and
predates this change; no decision moves.

A run now fails loudly rather than quietly. `MIN_ANSWER_MASS = 0.5` raises a
warning when the median mass on the answer words is too low, and every result file
carries an `answer_reading` block recording that mass and which steps were used.
For this arm: median mass 0.999965 on the test set and 0.999975 on validation,
step 1 for every prompt in both.

## 4. The run

Both readings from Section 7 were produced, as they were for Granite and
ShieldGemma: the native decision, and a threshold matched on the pool validation
split so the guard's safe-blocked rate there equals the cascade's 3.8%.

| | Final test | Pool validation |
|---|---|---|
| Prompts | 3,453 | 1,912 |
| Guard calls | 3,453 | 1,912 |
| Wall clock | 356 s | 193 s |
| Median per prompt | 84 ms | 83 ms |
| Errors | 0 | 0 |
| Unparsed answers | 0 | 0 |
| Prompts truncated | 0 | 0 |

Loaded in 4-bit NF4, the same quantisation as the judge, the Granite arm and the
ShieldGemma arm. One call per prompt, against ShieldGemma's four.

## 5. What it found

Llama Guard does not sit cleanly above or below the cascade. It is better on two
sources, clearly worse on one, and indistinguishable on the rest. Paired
differences against the cascade, arm minus cascade, in percentage points; negative
is better on every row, since harmful sources report harmful-allowed and safe
sources report safe-blocked.

| Source | diff | 95% CI | McNemar p |
|---|---:|---|---:|
| sst (harmful) | -3.0 | [-8.1, +1.0] | 0.375 |
| catqa (harmful) | **-13.1** | [-16.6, -9.6] | 1.78e-12 |
| dna_core (harmful) | **+21.5** | [+16.9, +26.0] | 4.58e-18 |
| dna_other (harmful, secondary) | +16.2 | [+10.2, +22.3] | 1.49e-06 |
| coconot (safe) | -0.3 | [-3.4, +2.9] | 1 |
| phtest (safe) | **-15.1** | [-18.8, -11.4] | 9.63e-14 |
| no_robots (safe) | -1.4 | [-4.8, +1.8] | 0.483 |

It allows 13.1 points fewer harmful CatQA prompts and blocks 15.1 points fewer
harmless PHTest prompts — better on both axes at once, which is the comparison the
cascade loses. But it allows 21.5 points more Do-Not-Answer core prompts, the
largest single gap either direction in the table.

Pooled AUROC, which is ranking quality rather than any one threshold:

| Arm | Pooled AUROC |
|---|---:|
| Granite | 0.923 |
| Judge alone | 0.878 |
| **Llama Guard** | **0.834** |
| ShieldGemma [post hoc] | 0.826 |
| Cascade | 0.811 |

At the matched validation threshold, Llama Guard falls between the other two
guards and behind Granite on the primary harmful sources:

| Guard | threshold | sst | catqa | dna_core |
|---|---:|---:|---:|---:|
| Granite | 0.9669 | 12.1 | **8.3** | **43.6** |
| Llama Guard | 0.8706 | **5.1** | 14.5 | 57.2 |
| ShieldGemma [post hoc] | 0.7520 | 29.3 | 59.1 | 57.1 |

## 6. What this changes about the study's conclusion

It confirms the narrower conclusion, and moves it onto pre-registered ground.

On 27 September the claim "published guards outperform this system" was narrowed
to "this system is behind Granite Guardian specifically", on the strength of
ShieldGemma ranking only 0.015 above the cascade. That evidence was post hoc and
was explicitly described as unable to rescue a conclusion any more than damage one.

The pre-registered second guard now says the same thing. Llama Guard ranks 0.834
against the cascade's 0.811 — a gap of 0.023, inside the 0.060 range the cascade's
own training seeds move it across (0.751 to 0.811). Two independent guards from
two vendors, one of them pre-registered, are both within seed noise of the cascade
in ranking quality. Only Granite, at 0.923, is clearly ahead.

The defensible conclusion is unchanged and now rests on planned evidence: **the
cascade is clearly behind Granite Guardian, not behind published guards in
general.**

Llama Guard adds something ShieldGemma could not. ShieldGemma differed from the
cascade mainly in where its threshold sat, nearly reversing the error profile.
Llama Guard beats the cascade on both axes simultaneously on CatQA and PHTest,
while losing badly on Do-Not-Answer core. So the cascade's deficit is not uniform
across harm types: it is real on the sources these guards handle well, and the
guards are not uniformly better either.

One caution carries forward unchanged. Do-Not-Answer "other" is a secondary source
kept out of the primary harmful rate because it contains questions a guard may
legitimately allow; Llama Guard's 65.1% allowance there should be read with the
same caution already applied to ShieldGemma's 92.9%.

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
