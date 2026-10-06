# Withdrawn: first Llama Guard attempt, 28 September 2026

These files are kept only so the record is complete. **Do not report anything from
them.** The harm score in them is not a harm score.

## What happened

Llama Guard access was granted on 28 September 2026 and the arm was run for the
first time. It completed without error: 3,453 prompts, 0 errors, 0 truncations, 0
unparsed answers. The metrics it printed were:

```
{"auroc": 0.6027, "accuracy": 0.7428, "recall": 0.6631, "precision": 0.7463}
```

That AUROC is wrong, and the pattern gave it away: an accuracy above both other
guards combined with the worst ranking of the three. Accuracy comes from the
guard's generated verdict; AUROC comes from the stored score. When one is the best
and the other the worst, the two disagree about the same prompts, which means one
of them is not measuring the guard.

## The cause

`run_external_guard.py` read `P(unsafe)` from `out.scores[0]`, the first generated
step, which is correct for Granite and ShieldGemma. Llama Guard 3 emits a `\n\n`
token before its verdict — every one of the 3,453 outputs begins with it:

```
2050  '\n\nsafe'
 378  '\n\nunsafe\nS1'
 375  '\n\nunsafe\nS2'
```

So step 0 was the newline step. Neither "safe" nor "unsafe" is a plausible token
there, and `P(unsafe)` became a ratio between two negligible numbers: a quantity
that has the shape of a probability and the content of rounding error. The
diagnostic already stored in each row shows it plainly:

| Arm | median `mass_on_answers` |
|---|---:|
| granite | 0.9997 |
| shieldgemma | 1.0000 |
| **llamaguard (this run)** | **0.0000** |

The scores collapsed into a 0.217–0.626 band and barely separated the guard's own
decisions: median 0.438 under a block, 0.377 under an allow.

The native decision was never affected. `native_parse_ok` is true for all 3,453
rows, because that reading comes from the decoded text, where the leading newlines
are simply stripped.

## The fix

`answer_step()` now locates the step the verdict is stated on — the first
generated token that is an answer token — instead of assuming step 0. For Granite
and ShieldGemma that resolves to step 0, so their numbers are untouched; this was
checked rather than asserted, by running the patched and unpatched scripts over
the same 60 prompts:

| Comparison | max difference in `p_unsafe` | decisions changed |
|---|---:|---:|
| patched vs unpatched | **0.000e+00** | 0 |
| unpatched vs the 26 Sep reported run | 6.03e-03 | 0 |
| patched vs the 26 Sep reported run | 6.03e-03 | 0 |

The patched and unpatched scripts agree bitwise. Both drift from the September run
by the same negligible amount, which is run-to-run nondeterminism in the 4-bit
kernels and predates this change.

A run now fails loudly instead of quietly: `MIN_ANSWER_MASS = 0.5` triggers a
warning when the median mass on the answer words is too low, and every result file
records `answer_reading` with the median mass and which steps were used.

## Files

| File | What it is |
|---|---|
| `final_llamaguard.json` | The withdrawn result. Its `metrics.auroc` is meaningless. |
| `final_llamaguard_predictions/` | The withdrawn rows. `harm_score` is noise; `final_pred` and `native_text` are sound. |
| `final_llamaguard.first-attempt.log` | The run's log, unedited. |

The corrected run is at the study root, in `final_llamaguard.json` and
`val_llamaguard.json`.
