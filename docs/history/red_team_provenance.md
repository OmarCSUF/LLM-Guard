# Red-team provenance note (moved from `artifacts/red_team/PROVENANCE.md`)

This file is the audit note that lived at `artifacts/red_team/PROVENANCE.md` in the
original workspace. It is kept verbatim below the horizontal rule because the paper and
the old README refer to it. Two things changed in the release and are stated here rather
than by editing the note:

- The two uncitable files it describes, `artifacts/red_team/results.json` and
  `artifacts/red_team/results_fullpipeline.json`, and the third file
  `artifacts/redteam/results.json`, are **not shipped** in this repository. The note's
  checks (`md5sum ...`) therefore cannot be run here; they remain a record of what was found
  in the original tree.
- Paths of the form `data/processed/<file>` refer to the original workspace. In the release
  the files that were kept live under `results/`:

| Original path | Release path |
|---|---|
| `data/processed/redteam_all200_views.json` (+ `_variants.jsonl`) | `results/development/redteam_all200_views.json` (+ `_variants.jsonl`); byte-identical copy also at `results/final_eval_20260926/redteam_all200_historical.json` |
| `data/processed/redteam_all200.json` (+ `_variants.jsonl`) | `results/development/redteam_all200.json` (+ `_variants.jsonl`) |
| `data/processed/redteam_harmbench_only.json` | `results/development/redteam_harmbench_only.json` |
| `data/processed/redteam_coverage.json` | `results/development/redteam_coverage.json` |
| `data/processed/redteam_all200_control.json`, `redteam_all200_openers.json`, `redteam_200.json`, `redteam_clf_only.json`, `redteam_50_normoff.json`, `redteam_full.json` | not shipped (development runs; see `docs/history/development_runs.md`) |
| `data/processed/final_eval_20260926/analysis/tables.md` | `results/final_eval_20260926/analysis/tables.md` |

The `python3 -c` one-liner in the note works on any shipped file once the path is replaced,
for example:

```bash
python3 -c "import json;d=json.load(open('results/development/redteam_all200_views.json'));print(d['bypass_rate'], d['bypass_rate_verified'], d['run_info']['git_commit'])"
```

---

# Provenance of the files in this directory

**Do not cite `results.json` or `results_fullpipeline.json`. Neither can be traced
to a run, and the 0% bypass rate they report is not reproducible.** The
reproducible red-team results are in `data/processed/`, listed at the bottom of
this file.

Written after an audit of the artifact tree. Every claim below is a check anyone
can repeat with the command given.

## What is wrong with the two files here

### They are the same file

```bash
md5sum artifacts/red_team/results.json artifacts/red_team/results_fullpipeline.json
```

Both are `ef99f661dbf88e15d9a3b496245da77a`. They are byte-identical, down to the
recorded `elapsed_s` of 7360.37 seconds. One is named as though it were a
full three-layer evaluation and the other as though it were something else, but
only one run exists between them, and nothing records which.

### They carry no provenance

Neither file has a `run_info` block, so neither records the command, the git
commit, the seed, the configuration or the wall-clock time that produced it.
Provenance recording was added in patch 4, commit `f1c9d18`, on 15 September 2026.
These files predate it. Nothing in the repository establishes which code version,
which seed list or which attack settings produced the numbers they contain.

### Their headline number is not reproducible

They report 500 variants from 10 seeds at a 0.0% bypass rate. No run recorded
since has produced a bypass rate of zero. The measured rates under the shipped
configuration range from 4.6% to 22.6%, depending on the seed set and the
Layer-1 settings.

They also lack `bypass_rate_verified`, the rate computed only over variants that
provably still carry the harmful request. Without it a bypass rate cannot be
read, because a model-rewritten variant that lost its harmful intent counts as
"blocked" for the wrong reason.

### A third file disagrees with both

`artifacts/redteam/results.json`, in the near-identically named sibling
directory, has md5 `dd4d6be662237242607bf1e4a264c51a` and reports 1,800 variants
at a 100.0% bypass rate. It also has no `run_info`. The 0% and the 100% figures
have both been described as baselines for the other, and neither can be checked.

### They use attack names that no longer exist

Their `by_attack_type` keys include `paraphrase` and `translation`. Patch 19,
commit `38207e8`, renamed these to `prefix_rewrite` and `language_prefix` because
neither ever paraphrased or translated anything: the first swapped the opening
words of a prompt and the second prepended an instruction to answer in another
language, leaving the prompt in English. A file still using the old names reports
attacks under names that describe something the code did not do.

## What to cite instead

Every run below records its git commit, start time, seed and full configuration
in a `run_info` block, reports a verified bypass rate, and stores every variant
so a result can be re-read rather than trusted.

| Result file | Seeds | Variants | Bypass | Verified bypass | Commit |
|---|---:|---:|---:|---:|---|
| `data/processed/redteam_all200_views.json` | 200 | 3,600 | 4.6% | 4.6% | `6dead1c4` |
| `data/processed/redteam_harmbench_only.json` | 200 | 3,600 | 4.6% | 4.6% | `f9e14bf8` |
| `data/processed/redteam_all200_control.json` | 200 | 3,600 | 5.6% | 5.6% | `b3518409` |
| `data/processed/redteam_all200_openers.json` | 200 | 3,600 | 7.8% | 7.8% | `b3518409` |
| `data/processed/redteam_all200.json` | 200 | 3,600 | 13.3% | 13.3% | `6efe54d5` |
| `data/processed/redteam_coverage.json` | 307 | 5,526 | 16.6% | 16.6% | `f9e14bf8` |
| `data/processed/redteam_200.json` | 200 | 4,500 | 11.6% | 11.6% | `afddac7f` |
| `data/processed/redteam_clf_only.json` | 200 | 4,500 | 11.6% | 11.6% | `b1d7b43c` |
| `data/processed/redteam_50_normoff.json` | 50 | 900 | 22.6% | 22.6% | `afddac7f` |
| `data/processed/redteam_full.json` | 50 | 900 | 11.7% | 11.7% | `afddac7f` |

Reproduce any row with:

```bash
python3 -c "import json;d=json.load(open('data/processed/redteam_all200_views.json'));print(d['bypass_rate'], d['bypass_rate_verified'], d['run_info']['git_commit'])"
```

### The result to quote for the shipped configuration

`redteam_all200_views.json`: all 200 HarmBench seeds attacked once, six template
attacks, three variants each, 3,600 distinct variant texts, zero errors.

| Attack type | Bypass rate |
|---|---:|
| prefix_rewrite | 11.0% |
| decoding_manipulation | 4.3% |
| jailbreak_template | 3.7% |
| technical_prefix | 3.0% |
| language_prefix | 2.8% |
| role_play | 2.7% |
| **overall** | **4.6%** |

That 4.6% is 165 bypasses of 3,600. It is the same figure the pre-registered
final evaluation carries for the historical checkpoint in
`data/processed/final_eval_20260926/analysis/tables.md`, section C, which also
reports 6.3%, 5.0% and 7.8% for the three repeat-study seeds of the same training
recipe. The spread across seeds is wider than most differences the work compares,
so a single red-team number should not be reported without it.

## Why these files were kept

They are the artifacts the earlier results were reported from, so deleting them
would remove the evidence that the reported numbers came from somewhere. They are
left in place and labelled instead. `scripts/plot_evaluation.py` prints a warning
when it is pointed at a results file that is byte-identical to a sibling, so a
figure cannot be built from one of these without the warning appearing.
