# Validation policy and decision register

Which data may decide a design question, which data may only report, and which past
decisions broke that separation. Written because the thesis reports five benchmarks as
held-out while several configuration choices were in fact made by looking at them.

## The policy

| Set | Size | May select | May report |
|---|---|---|---|
| pool train | 15,296 | yes (it is the training data) | no |
| pool validation | 1,912 | **yes: this is the selection set** | only as a training diagnostic |
| pool test | 1,912 | no | yes, as an in-distribution score |
| the five benchmarks | 7,775 | **no** | **yes: this is the reporting set** |
| the 90 reviewed Layer-3 outputs | 90 | one or the other, never both, stated per use | as above |
| red-team runs | varies | yes | yes, since attacks are generated, not held out |

Two rules follow, and they are the whole policy.

1. **A number that decided something cannot also be a headline.** If a benchmark result was
   consulted to accept or reject a change, that benchmark is no longer held out for the
   claim that change supports, and the decision must be listed below.
2. **The pool validation split is where choices are made.** Thresholds, judge settings and
   feature flags are selected there and confirmed on the benchmarks once, at the end.
   `scripts/sweep_routing.py` already enforces this shape with `--select` and `--report`;
   `scripts/tune_judge.py` defaults to the validation split for the same reason.

## Register of decisions taken so far

"Selected on" is the evidence that actually decided the question. "Also reported" says
whether that same evidence appears in the thesis as a result.

| Decision | Selected on | Also reported | Status |
|---|---|---|---|
| Judge prompt v2 over v1 (patch 12) | pool validation, 300 sampled prompts | no | clean |
| Classifier-only stacker as default (patches 28, 29) | all five benchmarks | yes | **selection on the reporting set** |
| Input normalisation on (patches 25, 26) | red team, plus benign cost on the pool and the benchmarks | yes | **partly on the reporting set** |
| Leading-clause view, colon rule over sentence rule (patch 33) | pool validation **and pool test** decision counts | pool test is reported | **selection on a reporting set** |
| Leading-clause view adopted (patch 34) | all five benchmarks, against pre-registered cost limits | yes | **selection on the reporting set** |
| Opener augmentation rejected (patch 35) | all five benchmarks and the red team | yes | **selection on the reporting set** |
| Rewrite check not adopted (patches 36, 37) | the 90 reviewed outputs | the leak rate comes from the same 90 | **selection on the reporting set** |
| Refusal check not adopted (patches 43, 44) | the same 90 outputs | as above | **selection on the reporting set** |
| Layer-3 default mode refuse (patch 24) | red-team pilots and manual review | red team is reported | acceptable: attacks are generated per run |
| Refusal veto on decline (patch 23) | red-team pilots | as above | acceptable |
| Routing thresholds 0.3 and 0.7 | **nothing** | — | **never validated; see below** |
| Judge band 0.4 and 0.6 | nothing recorded | — | **never validated** |
| K' = 3 rewrite candidates, temperature 0.7 | nothing recorded | — | defaults, never varied |

Seven decisions were made by consulting data the thesis also reports. That does not make
the individual numbers wrong, and every one of them was taken against criteria written
before the run, which limits the freedom involved. It does mean the benchmark scores carry
some optimism that no confidence interval shows, and the honest framing is that they are
**the results of a configuration chosen with their help**, not an untouched test.

## What this costs, in the direction it costs it

The freedom is small per decision and not zero in total. Each of the seven was a binary
accept or reject against criteria fixed in advance, so the search was not over a large
space. Three of them, however, point the same way: the classifier-only stacker, input
normalisation and the leading-clause view were each adopted partly because the benchmark
cost was small, so the reported benchmark costs are the lower end of what those changes
might cost on fresh data. The two rejections, the opener retrain and the rewrite check,
reduce optimism rather than add to it, since a change that looked good on the reporting set
was still refused.

The honest quantification is that no benchmark number in this work should be read as a
clean held-out estimate to better than about a point, which is the scale of the margins
several of these decisions turned on.

## The unvalidated constants

The routing thresholds were the clearest gap: 0.3 and 0.7 appear in the configuration, in
the method chapter and in every figure, and nothing had chosen them. The sweep this policy
called for was run on 23 September, on the pool validation split, with no benchmark data.

| band | safe blocked | harmful allowed | to judge | ms/prompt |
|---|---|---|---|---|
| **0.30 / 0.70 (configured)** | 36 of 956, 3.8% | 34 of 956, 3.6% | 0.6% | 30 |
| 0.50 / 0.50 | 35, 3.7% | 33, 3.5% | 0.0% | 9 |
| 0.35 / 0.35 | 38, 4.0% | 29, 3.0% | 0.0% | 9 |
| 0.05 / 0.05 | 58, 6.1% | 19, 2.0% | 0.0% | 9 |

Three bands do beat the configured one at printed precision: 0.50/0.50 on both rates,
0.35/0.50 and 0.40/0.50 on harmful allowed. The margin is one prompt per class, 0.1
points, against a binomial standard error of 0.62 points, about six prompts. The frontier
is flat across this whole region because only 36 prompts of 1,912 score between 0.05 and
0.50, so **none of those bands is distinguishable from the configured one on this split**.

The latency column is not evidence here and an earlier version of this file treated it as
if it were. Every routed prompt is charged what it actually cost during collection, and
only the prompts the judge generatively answered carry a full call, so a narrow band
containing those prompts is priced against a different cost model from a wide band that is
not. The implied per-routed-prompt cost ran from about 700 ms to about 3,500 ms for that
reason alone.

The band is therefore kept, and the reason matters: nothing is distinguishable from it,
not that nothing beats it. The honest description is **insensitivity, not validation**: a
single split with no resampling on a 0.05 grid cannot validate a threshold, but it does
show that the original arbitrary choice costs at most one or two prompts per class.
Moving on a one-prompt margin would be chasing noise.

The sweep's larger finding concerns the band's width. Recovering the score distribution
from the share routed at each band gives 918 prompts at or below 0.05, 29 between 0.05 and
0.30, **11 inside the configured band**, and 955 above 0.70. No prompt scores exactly zero:
the band 0.00/0.00 routes precisely those and the sweep reports none, so the tau_low = 0
rows are a cumulative read-out rather than a point mass, which an earlier version of this
file got wrong.

By class, 898 of the 956 safe prompts score at or below 0.05 and 923 of the 956 harmful
score at or above 0.50; 98.1% of the split lies outside the interval between. The
calibrated score is almost perfectly bimodal, which is the saturation the first limitation
described, now counted rather than inferred.

**What the judge contributes here can be measured, and an earlier version of this file
wrongly said it could not.** The band 0.50/0.50 routes nothing to the judge and treats
every prompt outside the configured band identically, so the difference between the two is
the judge's whole effect on the eleven prompts it sees: one more safe prompt blocked and
one more harmful prompt allowed. That is one prompt the wrong way on each rate, which at
eleven prompts is at the split's resolution and so is a null result rather than a defect.
Widening the band to reach the low-score mass is not free either: against 0.05/0.05, the
band 0.00/0.05 routes 918 prompts and buys 8 more harmful prompts caught for 119 more safe
prompts blocked, about fifteen to one.

Two qualifiers travel with this. The split is balanced 50/50 and is not a deployment
traffic mix. And the reporting benchmarks route 0.5% to 3.7% of prompts to the judge, the
top of which is six times this split's rate, so the in-distribution inertness does not
transfer: whatever the cascade's middle layer is worth, it has to be worth it there.

The judge band and the rewrite-candidate count are in the same position, with less
consequence: the judge band only matters for the small share of prompts that reach Layer 2,
and K' only matters when `safe_rewrite` is chosen, which is not the default.

## What cannot be fixed retrospectively

Re-running the seven decisions on a set that was never used for selection would need a
benchmark none of this work has touched. None exists here: OR-Bench is fully consumed, and
the remaining sources are the pool's own. A fresh selection set can be drawn from the
unused WildJailbreak remainder, which would protect future decisions, but it shares a
generator with half the training pool and so cannot serve as a new reporting benchmark.
The defensible course is therefore disclosure, which this file is, plus the policy above
from here on.
