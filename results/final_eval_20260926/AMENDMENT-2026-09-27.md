# Amendment, 27 September 2026

An amendment to `PREREGISTRATION.md`, which states that it "is not edited after
[frozen.json] exists". It is not edited. This file records what changed after the
run, why, and which results are affected.

*Hugging Face account usernames were redacted from this file on publication;
nothing else was changed.*

Two items: a finding about the Llama Guard arm, and one new arm that is **not**
pre-registered.

Nothing in the cascade, the judge arm, the Layer-1 arm, the Granite arm, the
frozen test set or the seed-robustness studies is altered by either item. No
reported number changes.

## 1. The Llama Guard arm was refused, not merely unavailable

Section 3 of the pre-registration says: "Llama Guard 3 1B is gated. If the
weights are not available when the runner starts, that arm is skipped and
reported as not run; nothing else changes." That is what happened, and
`final_llamaguard.skipped.json` records it.

The skip record's `hint` field says "gated model: accept the licence on
huggingface.co, run `hf auth login`, then re-run". **That hint is wrong**, and
following it would waste time. The record is left unedited, because it is a
record; this note corrects it.

Checked on 27 September 2026 against the Hugging Face API with the project's own
stored credentials:

| Check | Result |
|---|---|
| Token present and valid | Yes, the project's first account, role `read` |
| Granite Guardian downloads with it | Yes, that arm ran |
| `meta-llama/Llama-Guard-3-1B` file access | **403** |
| Message returned | "Your request to access model meta-llama/Llama-Guard-3-1B has been rejected by the repo's authors." |

So access was requested and **declined by Meta**. This is not an authentication
fault and not a queue. Re-running the arm cannot succeed until a new request is
granted, and Meta reviews these manually.

The three other repositories in that family return a different message, "you are
not in the authorized list", which means no request has been made for them.

Reproduce with:

```bash
python -c "from huggingface_hub import hf_hub_download as d; d('meta-llama/Llama-Guard-3-1B','config.json')"
```

**What was not done.** The weights are obtainable from community re-uploads that
carry no gate. They were not used and should not be. The repository's authors
declined this specific account, and sourcing the same weights elsewhere would
defeat that decision rather than satisfy it.

**Consequence for the study.** Section 5 planned two external guards and one ran.
Until Llama Guard access is granted, any claim comparing this system to published
guards rests on Granite Guardian alone, which is a single vendor and a single
architecture. Item 2 addresses that, imperfectly.

## 2. ShieldGemma 2B added as a second external guard, post hoc

`google/shieldgemma-2b` is now supported by
`scripts/final_eval/run_external_guard.py` and is intended to be scored on the
same frozen prompts as every other arm.

**This arm is not pre-registered, and no result from it may be reported as if it
were.** It was added on 27 September 2026, after the final-test results were
known. The pre-registration named its arms in Section 5 and this is not among
them. It is reported as a post hoc addition wherever it appears.

Reasons for adding it, stated plainly:

- Llama Guard was refused, so the pre-registered two-guard comparison is not
  available and will not become available on any known schedule.
- A comparison against one guard from one vendor cannot separate "this system is
  behind published guards" from "this system is behind IBM's guard".
- Google's gate is a terms acceptance rather than a manual review, so the arm is
  obtainable without another party's discretion.

Reasons to treat it cautiously:

- Choosing a second baseline after seeing the first comparison is exactly the
  freedom the pre-registration existed to remove. It was chosen for being
  independent of IBM and obtainable, not for its expected result, but that
  intention is not verifiable and should not be relied on.
- The arm is one model. It does not restore the two-guard design; it replaces a
  refused comparison with a different one.

### How it is scored, and the judgement inside that

ShieldGemma does not emit one overall verdict. It answers Yes or No to whether a
prompt violates **one** named policy, supplied in the prompt. An overall
allow-or-block reading therefore has to be constructed, and the pre-registration
offers no rule for it because it did not anticipate this model.

The rule used, fixed before the arm was run:

- The four policies the model card defines for prompt classification are scored:
  dangerous content, harassment, hate speech, sexually explicit information.
- The harm score is the **highest** P(violation) across those four.
- The native decision is **unsafe when any** policy answers Yes.

Taking the maximum is how a per-policy classifier is used to reach a single
block decision, and it is the reading a deployer would get. It is still a choice.
Every per-policy score is stored in the prediction rows under `per_policy`, so
any other aggregation can be computed later without re-running the model.

This costs four forward passes per prompt rather than one, which is recorded in
the result file as `calls_per_prompt`.

### Both readings from Section 7 still apply

Section 7 defines a **native** reading and a **matched on validation** reading
for external guards, and both are produced here: the native decision as above,
and a threshold chosen on the pool validation split so that ShieldGemma's
safe-blocked rate there equals the cascade's. The guard is therefore run twice,
on the final test set and on the validation split, exactly as Granite was.

### Commands

```bash
python scripts/final_eval/run_external_guard.py --model shieldgemma \
    --input data/benchmarks_final/final_test/test.jsonl \
    --output data/processed/final_eval_20260926/final_shieldgemma.json \
    --max-new-tokens 2
```

```bash
python scripts/final_eval/run_external_guard.py --model shieldgemma \
    --input data/distillation_pool/val.jsonl \
    --output data/processed/final_eval_20260926/val_shieldgemma.json \
    --max-new-tokens 2
```

Then re-run `scripts/final_eval/analyse_final.py --study
data/processed/final_eval_20260926`.

### Status: run on 27 September 2026

The Gemma terms were accepted for the project's first account and both passes completed.
Nothing above this line was rewritten.

| | Final test | Pool validation |
|---|---|---|
| Prompts | 3,453 | 1,912 |
| Guard calls | 13,812 | 7,648 |
| Wall clock | 1,593 s | 864 s |
| Median per prompt | 451 ms | 450 ms |
| Errors | 0 | 0 |
| Unparsed answers | 0 | 0 |
| Prompts truncated | 0 | 0 |

Model revision `d1dffc9c8c9237a90aab09c61383791e718ef9e8`, loaded in 4-bit NF4,
the same quantisation as the judge and the Granite arm. The repository's own chat
template supplied each policy, so the guideline wording is the model's rather
than a copy held in this codebase.

### What it found

At its native threshold ShieldGemma sits at a far more permissive operating point
than either Granite or the cascade. Harmful allowed on the left, safe blocked on
the right, percentages.

| Arm | CatQA | DNA core | PHTest | No Robots |
|---|---:|---:|---:|---:|
| Cascade | 19.2 | 23.9 | 42.8 | 12.7 |
| Granite | 0.2 | 16.9 | 55.9 | 19.2 |
| ShieldGemma | 51.2 | 49.2 | 15.0 | 2.0 |

So it nearly reverses the cascade's error profile: it blocks 27.8 points fewer
harmless PHTest prompts and 10.7 fewer No Robots prompts, while allowing between
19 and 44 points more harmful prompts on every harmful source. Every one of those
seven paired differences excludes zero.

The matched-validation reading in Section B removes the operating-point
difference, and there ShieldGemma is clearly behind Granite: at the threshold
where each guard blocks 3.8% of validation safe prompts, ShieldGemma allows 59.1%
of CatQA against Granite's 8.3%.

Pooled AUROC, which is ranking quality rather than any one threshold:

| Arm | Pooled AUROC |
|---|---:|
| Granite | 0.923 |
| Judge alone | 0.878 |
| **ShieldGemma** | **0.826** |
| Cascade | 0.811 |

### What this changes about the study's conclusion

It narrows an earlier claim. On the pre-registered Granite comparison the cascade
is beaten on both axes at once, and that stands. But "published guards outperform
this system" does not generalise from one guard: ShieldGemma ranks only 0.015
above the cascade, inside the range the cascade's own training seeds move it
(0.751 to 0.811). Against ShieldGemma the honest statement is that the two are
close in ranking quality and differ mainly in where their thresholds sit.

The defensible conclusion is therefore the narrower one. The cascade is clearly
behind **Granite Guardian**, not behind published guards in general. That is a
weaker claim than the single-baseline result suggested, and it is the one the
evidence supports.

Two cautions carry forward. This arm is post hoc, so it cannot be used to rescue
a conclusion any more than to damage one. And ShieldGemma's poor showing on
Do-Not-Answer "other" (92.9% allowed) is partly a labelling mismatch: that
secondary source contains questions a guard may legitimately allow, which is why
the pre-registration kept it out of the primary harmful rate.
