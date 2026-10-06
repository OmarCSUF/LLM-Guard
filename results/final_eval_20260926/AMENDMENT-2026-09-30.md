# Third amendment, 30 September 2026

An amendment to `PREREGISTRATION.md`, which is not edited. The two earlier
amendments (`AMENDMENT-2026-09-27.md`, `AMENDMENT-2026-09-28.md` and
`AMENDMENT-2026-09-28b.md`) are not modified by this one.

**A post hoc component ablation was scored on the frozen final test set.**
Sixteen new arms are added. **None is pre-registered**, every one is marked
`[post hoc]` wherever it appears, and **no configuration decision was or will
be taken from any of them**: the cascade's configuration is the one frozen in
`frozen.json`, and the ablation exists to interpret the pre-registered results,
not to revise the system. Nothing in the eleven arms analysed before this date
is altered; their point estimates are exact counts and their bootstrap
intervals come from per-comparison streams, so adding arms cannot move them.

## 1. Why

The paper's central reading is that the cascade under-uses its middle layer
because the calibrated Layer-1 score is bimodal on the final test set. Two of
Layer 1's mechanisms, input normalisation (base64 and leetspeak recovery) and
the leading-clause payload view, were adopted during development against the
five development benchmarks and the red-team suite (`docs/validation_policy.md`,
patches 25, 26, 33 and 34). Their benign cost on prompts that informed no
decision had never been measured; the paper's System Design section promised
that measurement and the Results section did not deliver it. The temperature
scaler's effect on routing had likewise never been isolated.

## 2. Arms

Each arm is the frozen cascade (judge loaded, refusal default) with exactly one
change, scored on the identical 3,453 prompts with `scripts/run_evaluation.py`:

| Arm | Change | Flag |
|---|---|---|
| `cascade_noviews` | leading-clause payload view off | `--no-payload-views` |
| `cascade_nonorm` | input normalisation off | `--no-normalise-input` |
| `cascade_bare` | both off | both flags |
| `cascade_T1` | temperature scaler fixed at T = 1.0 | `--layer1-artifacts layer1_T1` |

and the same four for each repeat-study checkpoint (`_s42`, `_s43`, `_s44`,
via `--layer1-artifacts layer1_repeat_study_nciJu1oQ_s4x[_T1]`), so that every
ablation delta comes with its own spread across the four checkpoints.

## 3. Deviations from the frozen system, stated plainly

- `scripts/run_evaluation.py` gained the flags `--normalise-input` and
  `--no-normalise-input`, mirroring the existing `--payload-views` pair. Absent
  the flag, behaviour is unchanged. The file's SHA-256 therefore differs from
  the one in `frozen.json`, and every run in this amendment records
  `git_dirty: true` for that reason.
- The `_T1` artifact directories are copies of the corresponding Layer-1
  artifacts with `scaler_clf_only.json` and `scaler.json` rewritten to
  `T = 1.0` (`optimised: false`, with a note naming the source). The
  classifier weights and stacker are byte-identical copies.
- `scripts/final_eval/analyse_final.py` registers the sixteen arms, marks them
  post hoc, adds each historical ablation arm to the paired AUROC comparisons
  against the cascade, and writes a Section E table with the share of prompts
  routed to the judge per arm and checkpoint.

## 4. Order of events

All sixteen arms were run on 30 September 2026, after every result in the
paper's Results section, including the judge substitution of 28 September, was
known. They were run in one batch by a driver script, in the order listed above
(historical checkpoint first, then s42, s43, s44), and no arm was re-run.

## 5. Results

All sixteen arms completed (exit 0) between 13:50 and 14:36 on 30 September
2026; `analysis/results.json` and `analysis/tables.md` were regenerated with
them. Every previously reported point estimate and interval is unchanged, as
the per-comparison streams guarantee.

| Mechanism removed | Layer-1 scores changed | Decisions changed (historical / s42 / s43 / s44) | Pooled AUROC, historical |
|---|---:|---|---:|
| leading-clause view | 42 / 33 / 39 / 46 | 7 / 7 / 9 / 9, all safe prompts allowed once the view is off | 0.814 vs 0.811 (+0.0030 [+0.0016, +0.0047]) |
| input normalisation | 0 on every checkpoint | 0 / 0 / 0 / 0 | 0.811 (identical) |
| both | as view alone | as view alone | as view alone |
| temperature (T = 1) | all 3,453 (rescaled) | 7 / 10 / 10 / 5; gray zone 97→83, 79→60, 77→61, 82→71 | 0.811 (+0.0002 [-0.0018, +0.0021]) |

Reading, stated as observation: no prompt in the final test set has a
recoverable base64 or leetspeak form; the leading-clause view blocks seven to
nine safe prompts per checkpoint and catches no harmful one on any checkpoint,
so on plain prompts it is a small pure cost whose benefit lies entirely in the
framing attacks of the red-team suite; and the fitted temperature (1.19)
changes routing on 14 prompts and no rate by more than 0.3 points. The routing
band itself was not ablated on the final test set; the validation-split sweep
in `docs/validation_policy.md` and `data/processed/routing_sweep.md` remains
the only evidence on it.
