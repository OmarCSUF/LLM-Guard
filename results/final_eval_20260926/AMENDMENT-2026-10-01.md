# Fourth amendment, 1 October 2026

An amendment to `PREREGISTRATION.md`, which is not edited. Earlier amendments
are not modified. This one records analyses added in response to an expert
review of the manuscript. **No model was run and no prompt was re-scored**:
every analysis below is computed from the per-prompt prediction files the
pre-registered and post hoc arms already wrote. They are descriptive readings
of the final test set, marked post hoc wherever they appear, and **no
configuration decision was or will be taken from any of them**. Every
previously reported number is unchanged.

## 1. Operating-point frontiers and the routing-band sweep (`scripts/final_eval/frontiers.py`)

For every arm with a graded harm score (Layer 1 on all four checkpoints, the
two judges, the three guards) the threshold is swept over the stored scores and
two rates are read at each value, each a macro-average over sources so that no
source is weighted by its size: safe blocked over CoCoNot, PHTest and No Robots;
harmful allowed over SimpleSafetyTests, CatQA and Do-Not-Answer core.
Do-Not-Answer "other" is excluded as secondary. Native operating points are the
arms' stored decisions.

The cascade's band is swept offline: with each checkpoint's stored Layer-1
score and the judge arm's stored decision on every prompt (uncertain counted as
blocked, as in the pipeline), a symmetric band [0.5 − w, 0.5 + w] routes the
prompts inside it to the judge and decides the rest by Layer 1; w = 0.2 is the
configured band and w = 0.5 routes everything. This reads what widening the
band would have done on the final test set. It is a sensitivity analysis, not a
selection: the configured band is unchanged, and the reason it cannot be tuned
here is stated in the paper's Threats to Validity.

Output: `analysis/frontiers.json`. Figure and text: manuscript Section V-B.

**Results (item 1).** Native macro rates (safe blocked, harmful allowed), %:
cascade 20.6/16.0; Layer 1 20.3/18.3; judge 19.0/12.3; Llama judge 17.6/8.3;
Granite 32.2/5.7; Llama Guard 15.0/17.8; ShieldGemma 9.6/41.6. At the cascade's
macro safe-blocked level (20.6%) the frontiers allow: Layer 1 18.0, retrained
checkpoints 25.7/20.0/21.1, judge 11.5, Llama judge 7.6, Granite 7.7, Llama
Guard 16.0, ShieldGemma 23.0. Band sweep, configured checkpoint (routed %, SB,
HA): w=0.20 2.8/20.6/16.0; w=0.40 8.4/20.0/15.2; w=0.45 14.1/19.9/13.8;
w=0.46 17.5/20.1/12.8; w=0.47 61.5/28.8/6.5; w=0.48 100/19.0/12.3 (= judge
alone). Layer-1 scores span [0.034, 0.976]. Manuscript: Fig. 4 and the
paragraph after it in Section V-B; one sentence in Discussion VI-B adjusted.

## 2. Calibration in and out of distribution (`scripts/final_eval/calibration.py`)

ECE with the project's definition (`evaluation/metrics.py`, 10 equal-width bins)
and a percentile bootstrap over prompts (B = 2000, study seed, named stream),
plus reliability bins. In distribution: the configured Layer 1 (views on) was
scored once more on the pool test split (`pooltest_layer1.json`, Layer 1 only,
deterministic; AUROC 0.9913 against Table II's 0.9914, which was read from
cached raw-text signals without views); the judges' in-distribution reference is
the 300 pool-validation prompts of the tuning runs. Final test set: the 3,089
primary prompts, pooled (1,874 safe, 1,215 harmful), so those values are
conditional on the source mixture.

**Results (item 2).** ECE [95% CI]: Layer 1 pool test 0.013 [0.008, 0.022],
final 0.204 [0.191, 0.221]; retrained s42/s43/s44 final 0.252/0.226/0.222;
judge Qwen val300 0.191 [0.153, 0.242], final 0.172 [0.159, 0.186]; judge Llama
val300 0.102 [0.072, 0.141], final 0.129 [0.117, 0.141]; Granite 0.211, Llama
Guard 0.155, ShieldGemma 0.167 on the final set. Layer-1 extreme bins on the
final set: [0,0.1) n=1,481 mean 0.038, 14.2% harmful; [0.9,1] n=1,372 mean
0.973, 65.7% harmful (pool test: 0.035/2.6% and 0.976/97.1%). Output:
`analysis/calibration.json`. Manuscript: new Section V-E, Table VII, Fig. 6, one
sentence in Discussion VI-B.

## 3. Rates by category (`scripts/final_eval/per_category.py`)

The pre-registration's secondary outcome "per-category rates inside CatQA and
Do-Not-Answer, read as a ranking" is computed for every arm from the stored
predictions, and extended on the same footing to SimpleSafetyTests, CoCoNot
contrast and No Robots, which carry category labels; PHTest carries the single
label "harmless". CatQA is grouped by its 11 areas; Do-Not-Answer and CoCoNot
by subcategory. Wilson intervals per cell are in `analysis/per_category.json`;
the table is `analysis/per_category.md`. Manuscript: new Section V-E "Errors
by category" with a full-width table; two sentences in Discussion VI-C.

## 4. Native verdict versus harm score (`scripts/final_eval/guard_agreement.py`)

From the stored guard predictions: parse rate, agreement between the native
decision and the harm score at 0.5 with the direction of each disagreement,
answer-mass distribution, Llama Guard's answer step, and ShieldGemma's
any-policy-Yes versus max-probability agreement, on the final test set and the
pool validation split. **Results:** agreement 99.83% (Granite, 6 ties within
2.1e-5 of 0.5), 99.97% (Llama Guard, 1 tie at exactly 0.5), 100% (ShieldGemma);
validation 99.79% / 99.95% / 99.95%; all native outputs parsed; answer mass
median >= 0.9997 and minimum 0.93 (Granite), 0.96 (Llama Guard), 0.999
(ShieldGemma); Llama Guard answer step = 1 (second generated token) on all
prompts; ShieldGemma per-policy agreement 100% on the final set. Vendor rules
confirmed in the cited papers (Llama Guard: first-token probability; Granite:
softmax over safe/unsafe token logits aggregated over spellings; ShieldGemma:
Yes/No token scoring mode). Output: `analysis/guard_agreement.json`.
Manuscript: one paragraph each in Sections IV-D and V-B.

## 5. Judge band and uncertain policy (`scripts/final_eval/judge_band.py`)

Post hoc, from the judge arms' stored p_mitigate on the final test set: the
band [0.5 - u, 0.5 + u] swept for u in {0, 0.05, 0.10 (configured), 0.15, 0.20,
0.30, 0.40, 0.45} under both uncertain policies (block, as configured; allow),
with share uncertain, macro rates and per-source rates. Nothing was selected.
**Results (configured judge, SB/HA macro %):** stored uncertain 50 (1.45%);
8.5% of probabilities in (0.1, 0.9); 0.5 threshold 18.0/12.8; configured band
19.0/12.3 (block) vs 17.5/13.5 (allow); 0.05/0.95 23.4/9.8 (block) vs 13.5/17.2
(allow); CoCoNot blocked 19.3–21.4 and CatQA allowed <= 3.1 under every
setting. Substitute judge: 78 uncertain (2.26%), 16.7% in (0.1, 0.9); 0.05/0.95
31.5/4.7 (block) vs 8.1/13.9 (allow); CoCoNot blocked 21.4–28.6. Output:
`analysis/judge_band.json`. Manuscript: table and paragraph at the end of the
calibration subsection; one clause each in Discussion VI-B and Threats.
