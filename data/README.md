# Data: what is here, what is built, how to rebuild it

This directory holds the one dataset the repository redistributes (the locked
final test set) and the empty directories the build scripts fill. Everything
else is downloaded from pinned Hugging Face revisions and derived locally with
the scripts described below. The counts and hashes in this file are copied from
the manifests the recorded study wrote; those manifests are kept verbatim under
`results/development/` and `data/benchmarks_final/manifest.json`.

Licences and the redistribution policy are in [`LICENSES.md`](LICENSES.md).

## 1. Overview

| Path | Contents | In the repository? | How it is produced |
|---|---|---|---|
| `data/raw/hf/<repo>/<path>` | 11 source files from 8 Hub repositories (about 540 MB; WildJailbreak alone is 531 MB) | No | `scripts/fetch_datasets.py` |
| `data/raw/hf/manifest.json` | revision, size and digest of every fetched file | No (verbatim copy at `results/development/raw_sources_manifest.json`) | same |
| `data/distillation_pool/{train,val,test}.jsonl` + `metadata.json` | Layer-1 training pool, 15,296 / 1,912 / 1,912 prompts | No (metadata copy at `results/development/pool_metadata.json`) | `scripts/build_datasets.py` |
| `data/benchmarks/<name>/test.jsonl` | the five development benchmarks | No | `scripts/build_datasets.py` (same run) |
| `data/benchmarks_parity/{original,general,arabic}/<name>/test.jsonl` | RQ4 parity arms: the safe benchmark prompts and two rule-based non-native rewrites | No (manifest copy at `results/development/parity_arms_manifest.json`) | `scripts/make_l2_benchmarks.py` |
| `data/raw/wi_locness/` | W&I+LOCNESS v2.1 (BEA-2019), obtained manually | No, and never will be (non-commercial licence) | manual download |
| `data/benchmarks_learner/{original,corrected}/<level>/test.jsonl` | RQ4 learner-English arms | No (manifest copy at `results/development/learner_arms_manifest.json`) | `scripts/make_learner_arms.py` |
| `data/raw/hf_final/<repo>/<path>` | 6 source files from 6 Hub repositories (5.2 MB) | No (manifest copy at `results/development/raw_final_sources_manifest.json`) | `scripts/final_eval/fetch_final_sources.py` |
| `data/benchmarks_final/final_test/test.jsonl` | **the locked final test set, 3,453 prompts** | **Yes** | `scripts/final_eval/build_final_test.py` |
| `data/benchmarks_final/manifest.json` | counts at every screening stage, source revisions, reference-file hashes | **Yes** | same |
| `data/processed/` | run outputs | No (the recorded ones are under `results/`) | evaluation scripts |

Why the split: the final test set is small, its sources allow redistribution
with attribution (one of them non-commercially), and it is the file every
reported final number was computed on, so it ships. The pool and the
development benchmarks are large, are rebuilt deterministically from pinned
sources, and three of their sources are gated behind the Hub's terms, so they
are not copied here; rebuilding them is the way to get them. The learner
corpus may not be redistributed at all, so its arms are rebuilt from a copy
you obtain under its licence.

## 2. Prerequisites

* Run every command **from the repository root** with the project environment
  (`python` below means the interpreter of that environment; the scripts put
  `src/` on `sys.path` themselves). The `scripts/final_eval/*` scripts locate the
  repository with `LLMGUARD_REPO` when it is set, otherwise the current directory.
* `config/config.yaml` fixes the paths (`data.raw_dir`, `data.distillation_pool_dir`,
  `data.benchmarks_dir`) and the project seed (42). Leave them as they are.
* **Hugging Face token.** Three of the eight pool/benchmark sources are gated:
  `walledai/XSTest`, `walledai/HarmBench` and `allenai/wildjailbreak`. Open each
  dataset page on huggingface.co with your account, accept the terms, then either
  run `hf auth login` (`huggingface-cli login` with older `huggingface_hub`) or
  export `HF_TOKEN=...`. The fetch scripts read `HF_TOKEN` first and fall back to
  `huggingface_hub.get_token()`; the token is sent only to the `HF_ENDPOINT` host,
  never to the storage hosts that downloads redirect to, and is never printed.
  The study did not record any of the six final-test sources as gated; a gated
  repository answers 401/403 and the script says so.
* **Mirror.** Set `HF_ENDPOINT` (for example `HF_ENDPOINT=https://hf-mirror.com`)
  to download through a mirror. The scripts use plain HTTP against
  `<HF_ENDPOINT>/api/datasets/<repo>/tree/<revision>` and
  `<HF_ENDPOINT>/datasets/<repo>/resolve/<revision>/<path>`, follow 308
  redirects, and verify every file against the digest the Hub lists for it
  (sha256 for Git-LFS files, git blob sha1 for regular files), so a mirror cannot
  hand you a different file unnoticed. The endpoint used is recorded in the
  manifest.
* **Tokenizer cache (final set only).** The last step of `build_final_test.py`
  flags prompts longer than 512 `roberta-base` tokens. It loads the tokenizer
  from the local Hugging Face cache only. To reproduce the recorded file
  byte-for-byte, have the snapshot the study used in the cache before building:

  ```bash
  hf download roberta-base --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b
  ```

  (`hf` is the CLI of `huggingface_hub` 0.34+, the study environment has 0.36.2;
  older versions spell it `huggingface-cli download` and `huggingface-cli login`.)
  See step 6.

## 3. Pinned revisions: the `--revision-manifest` flag

Both fetch scripts fetch the **current** revision of each repository by default
and record it in the manifest they write. That is the behaviour the study ran
with on 15 and 26 September 2026, and it is unchanged. To fetch exactly the
files the study used, pin the revisions to the recorded manifests:

```bash
python scripts/fetch_datasets.py --revision-manifest results/development/raw_sources_manifest.json
python scripts/final_eval/fetch_final_sources.py --revision-manifest results/development/raw_final_sources_manifest.json
```

With a pin the script lists the repository tree at the recorded commit sha,
compares the digest the Hub publishes for each file with the digest the manifest
recorded, downloads from `.../resolve/<sha>/<path>`, verifies the content
against that digest, and marks the manifest entry `"pinned": true`. If the Hub's
digest at that revision differs from the recorded one (or the algorithm differs),
the source fails with `DIGEST MISMATCH ...`, nothing is written for it, and the
script exits 1 after the other sources. A repository that is not in the pin table
is fetched at its current revision with a warning.

`--revision REPO=SHA` (repeatable; `REPO` is the Hub id or the source name such
as `xstest`) pins a single repository and overrides the manifest for it. It
carries no digests, so only the Hub's own verification applies.

The manifest file may be any of: a `manifest.json` written by the fetch scripts
(`sources` block), the `source_revisions` block of `data/benchmarks_final/manifest.json`
(revisions and digests) or of `results/development/pool_metadata.json`
(revisions only).

## 4. The pipeline, in order

### Step 1: fetch the pool and benchmark sources

```bash
python scripts/fetch_datasets.py --revision-manifest results/development/raw_sources_manifest.json
# subset:  python scripts/fetch_datasets.py --only toxicchat jailbreakbench --revision-manifest ...
```

Writes `data/raw/hf/<repo>/<path>` and `data/raw/hf/manifest.json`. Expected
files (from `results/development/raw_sources_manifest.json`; `git-sha1` is the
git blob hash the Hub lists for non-LFS files):

| Source | Repository @ pinned revision | File | Bytes | Digest |
|---|---|---|---:|---|
| jailbreakbench | `JailbreakBench/JBB-Behaviors` @ `886acc352a31533ffbcf4ef22c744658688086fc` | `data/harmful-behaviors.csv` | 23,116 | git-sha1 `5a7549cd9de9bb327914de0800606a1af1ec8849` |
| | | `data/benign-behaviors.csv` | 20,570 | git-sha1 `659647d082ec98df45fdeb3f334ebd7c04e61638` |
| xstest (gated) | `walledai/XSTest` @ `f1d713187c61b6ae64e602d74f0b3d812cc2e8e8` | `data/train-00000-of-00001.parquet` | 16,964 | sha256 `ccb792dc92186eec69e8d2c59441113d43b2716ff2ca24bf6c5956df5997c7b1` |
| harmbench (gated) | `walledai/HarmBench` @ `fb6c2afd5a2a943d701d6db3efab87d077e81be5` | `standard/train-00000-of-00001.parquet` | 12,347 | sha256 `121425252baab536070b322fb9c1dc9f3ad119c5136003cdeac114a72ae3bd0d` |
| toxicchat | `lmsys/toxic-chat` @ `29df8e4dba60e1f4af4b4075c0705c5b313548a8` | `data/0124/toxic-chat_annotation_train.csv` | 8,199,182 | git-sha1 `d53e9317932905b4aa5e49784a046cbbcdf48462` |
| | | `data/0124/toxic-chat_annotation_test.csv` | 8,090,164 | git-sha1 `60a82d117e7767118c47b88b09486277eab0cb16` |
| dolly | `databricks/databricks-dolly-15k` @ `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | `databricks-dolly-15k.jsonl` | 13,085,339 | sha256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |
| beavertails | `PKU-Alignment/BeaverTails` @ `8401fe609d288129cc684a9b3be6a93e41cfe678` | `round0/30k/train.jsonl.gz` | 4,948,694 | sha256 `2e24b85dd599224eeb7379da297223bad50e11fea9fee6a7fdb4bd26aedec315` |
| wildjailbreak (gated) | `allenai/wildjailbreak` @ `5ddc12a7894f842b0619b8e1c7ee496b198af009` | `train/train.tsv` | 530,920,110 | sha256 `376719bfdb46ad1a19e7ba4f587f80cc7cb1368cc213ee647ef739c170550f7a` |
| orbench | `bench-llm/or-bench` @ `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272` | `or-bench-hard-1k.csv` | 169,255 | sha256 `a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d` |
| | | `or-bench-toxic.csv` | 64,721 | sha256 `3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057` |

The `sources` block of your `data/raw/hf/manifest.json` should equal the one in
`results/development/raw_sources_manifest.json` apart from `endpoint` (if you
used a mirror) and the added `"pinned": true`; the `run_info` block records your
own run and will differ.

### Step 2: build the pool and the five development benchmarks

```bash
python scripts/build_datasets.py
```

No options: the shipped pool was built with the script's defaults at commit
`c8f6c0da` on 15 September 2026 (`results/development/pool_metadata.json`,
`results/development/pool_build_2026-09-15.log`). Options added to the script
since then (`--source-balance`, staging of outputs) default to that behaviour.
The build is seeded (project seed 42; MinHash seed 42, 64 permutations, 16
bands, character 5-gram Jaccard threshold 0.8) and writes nothing to the final
paths until every overlap check is 0.

Expected outputs and sha256 (from `results/final_eval_20260926/frozen.json`,
blocks `pool` and `inputs`, and `data/benchmarks_final/manifest.json`,
`reference_files`):

| File | Rows | Allow / mitigate | sha256 |
|---|---:|---|---|
| `data/distillation_pool/train.jsonl` | 15,296 | 7,648 / 7,648 | `1cbd738c531105218588d5ec77b493c16c2752886c9675757d3382e9c71f4c05` |
| `data/distillation_pool/val.jsonl` | 1,912 | 956 / 956 | `16ce7eeab18f2e908256dc8b579c3a5060d54fcb06318c04d992ca04b91015e2` |
| `data/distillation_pool/test.jsonl` | 1,912 | 956 / 956 | `7c451ef4407aad7e5abf9b1dcbba4754657d6fdbf73a37f36fcaef7602168e70` |
| `data/benchmarks/jailbreakbench/test.jsonl` | 200 | 100 / 100 | `48f48ef492b950e0ed234b5431cf3fcf326da9dbea86936fb2f44f23b67010e2` |
| `data/benchmarks/xstest/test.jsonl` | 450 | 250 / 200 | `dbfae3fabba2bf5d2fc039ad6fd1f8477460968d251949c92cf3c6ed9ecf9719` |
| `data/benchmarks/harmbench/test.jsonl` | 200 | 0 / 200 | `a9db0c790bed0d0814244cc169c28644686338cb2b524df8074bae4c1d7a78ce` |
| `data/benchmarks/toxicchat/test.jsonl` | 4,951 | 4,597 / 354 | `0874a036e0628f07157ed06a0f03781a56894f3e270bfdadc68cf2d89fe90905` |
| `data/benchmarks/orbench/test.jsonl` | 1,974 | 1,319 / 655 | `fab7f52ace89d055eef84b3ec4c4a69206f6438cf7bf88628eed7906fbdbb463` |

Intermediate counts to compare against `data/distillation_pool/metadata.json`
(all from `pool_metadata.json`): ToxicChat test 5,083 loaded, 132 duplicates
dropped; pool candidates Dolly 10,544 (4,467 rows with a context passage
skipped), ToxicChat train 5,082, BeaverTails 4,278 harmful prompts kept out of
7,766 unique (3,085 mostly-safe excluded, 403 ties dropped), WildJailbreak
15,000 sampled per class out of 50,050 vanilla per class; 49,292 after
near-duplicate collapse (4 label-conflict groups, 8 prompts dropped); 48,995
after removing 297 evaluation near-duplicates; balanced to 9,560 per class with
5,000 WildJailbreak per class; every `overlap_checks` value 0. Pool composition
over the three splits: WildJailbreak 10,000, BeaverTails 4,255, Dolly 3,214,
ToxicChat train 1,651.

A later rebuild in the study workspace reproduced the five benchmark files with
these exact hashes. If a pool hash differs, compare `metadata.json` stage by
stage with `results/development/pool_metadata.json` before looking anywhere
else; a different source revision (step 1 without pins) is the usual cause.

### Step 3: parity arms (RQ4, rule-based non-native rewrites)

```bash
python scripts/make_l2_benchmarks.py --root data/benchmarks_parity
```

Reads the five benchmarks from step 2, keeps only the safe prompts (label 1),
and writes them unchanged plus two rewrites. HarmBench has no safe prompts and
is skipped. Recorded run: 22 September 2026, commit `b656744`, seed 42
(`results/development/parity_arms_manifest.json`).

| Benchmark | Safe prompts (rows in each arm) | general: edited / unchanged | arabic: edited / unchanged |
|---|---:|---|---|
| jailbreakbench | 100 | 90 / 10 | 89 / 11 |
| xstest | 250 | 149 / 101 | 149 / 101 |
| toxicchat | 4,597 | 3,229 / 1,368 | 3,183 / 1,414 |
| orbench | 1,319 | 1,190 / 129 | 1,186 / 133 |

sha256 of the twelve files as they were in the study workspace at release
(not recorded in a manifest; computed for this document):

| Arm | jailbreakbench | xstest | toxicchat | orbench |
|---|---|---|---|---|
| original | `32270075299432e3cd2975a7e7c3befb72e3f3f7dd11f9f7eacea71a014b1767` | `ee47bc3617b577fd1e6ed57b8a9bd3aec8f6aab2b4c94e713586238410ac7e00` | `e401bc98d860790b0360f2d30463aa538066a0aa4ec2dffb9c43d8d3f32c705d` | `d642118a9fce88d0743666f95a9164406ac44bc5dc951db69badbf85ccc53497` |
| general | `5e3fa2cb9372cce7a420543b1c4b85b68b882b20cf289f09e85fe610fca10c2c` | `010332302f97ca6d92a6208563cfedd254aa0a78d2b0622a95bff35ed30efd9f` | `976368a58a6e117403fcb61699cf588a848dbe17f2fa319947dbcde0bf8a5e63` | `108b6d6bc13eed1c850ad2fcfaadba7899dffa100c2b203543108e273f0582da` |
| arabic | `bda44f964996942f6449467caf98724df76c46053ddd0155b62affd08ce5d87e` | `79f8f8cb127178068598379aa8f9901984ac8049add2f8e917a63e0efcb8a923` | `e2c3515602b36578a6cbd6854057afa8ee373fa7368ae6486fd9ad281ddaffcd` | `e9f6cac5bd24f0c8eadc5c5721d097cad274ed40667276914f501192a6bcd7d8` |

### Step 4: learner-English arms (RQ4, manual corpus)

The W&I+LOCNESS v2.1 corpus (BEA-2019 shared task, University of Cambridge and
UCLouvain) is distributed by its owners under a non-commercial licence that
does not permit redistribution. Download `wi+locness_v2.1.bea19.tar.gz` from the
BEA-2019 shared task page (https://www.cl.cam.ac.uk/research/nl/bea2019st/),
accept its licence, and extract it so that the M2 files are somewhere below
`data/raw/wi_locness/`:

```bash
mkdir -p data/raw/wi_locness
tar -xzf wi+locness_v2.1.bea19.tar.gz -C data/raw/wi_locness     # -> data/raw/wi_locness/wi+locness/m2/*.m2
python scripts/make_learner_arms.py --source data/raw/wi_locness
```

The copy used by the study was 6,120,469 bytes with sha256
`d5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666` (computed
from the study workspace for this document; the M2 hashes below are from
`results/development/learner_arms_manifest.json`). The script finds
`[ABCN].{train,dev}.gold.bea19.m2` by name (the combined `ABC*`/`ABCN*` files
are ignored), takes annotator 0's edits, drops sentences under 3 word tokens,
samples 1,500 training sentences per level with seed 42 and keeps every dev
sentence. Defaults reproduce the recorded run (23 September 2026, commit `dc90a9e2`).

| M2 file | Bytes | sha256 |
|---|---:|---|
| `A.train.gold.bea19.m2` | 2,469,460 | `79cf56f7bd2ab22624e0308c205722ba9e2fe9a1b9dc1df7b35c3f9dba3a8497` |
| `A.dev.gold.bea19.m2` | 253,437 | `ddd7d1e98a1ccc3961660cc66ae44e254d52d10590a458e34b5b4c0adbf9866d` |
| `B.train.gold.bea19.m2` | 2,608,704 | `22465faee67ca87329d5fde7555e8a24fcf6d968511b13ebf74882eccd3a836f` |
| `B.dev.gold.bea19.m2` | 265,518 | `9b8ea2c7503a86d91f8f8c35f93b94532e7ebe4b1330f88af7798f67877c7469` |
| `C.train.gold.bea19.m2` | 1,837,840 | `66aea2af43dcf7e0022dcfd703743e802076149f4d6433d82b758d74e123ecd3` |
| `C.dev.gold.bea19.m2` | 189,549 | `d7cdfa7d3821f8ef4372f0cf0ba80d567e8d787338c35ca77db329f39c7fe4db` |
| `N.dev.gold.bea19.m2` | 194,267 | `02140717ca7fc3f3d7d4cff125220b7f160b369316e12e33c1fea59c1bbff6b5` |

Expected outputs (rows identical across the two arms; sha256 from
`results/final_eval_20260926/frozen.json`, block `inputs`):

| Level | Source sentences | Kept | `original/<level>/test.jsonl` | `corrected/<level>/test.jsonl` |
|---|---:|---:|---|---|
| wi_a | 11,530 | 2,494 | `2b3af012645deb66cb7b3d7c48b4b2d5f687c84f775054f25b76d2b69c74e49a` | `fb234e03bd586bc7605e4ab8352b04760c97b612b813963fbce4803ad2fa506b` |
| wi_b | 14,322 | 2,739 | `6b274fcc2b8b53b089781a8a90bf4293c7a5927fdeeaee109f12a89750746848` | `f7a34c2129d930dff05549950154edc9c260f2125b8a79c8b30ab187e325ec17` |
| wi_c | 11,852 | 2,527 | `3d2d1877e1a46a613a9cb8ea84e3155bbf2fd4e8d4098a6d7282a35cb0bf8c30` | `016954b09f64ba9fbc991905a20635d7a3c78d79f7620503a902301c03c2cf42` |
| locness_n | 988 | 986 | `1015fd04f99742918ffc6db3f7fcf6871c361fb22f3c350563b1257cfe8ba8a2` | `6f9fe251f868e096268a68a622b16513a6b86779ac6a33d7b052a3f5c8753f32` |

### Step 5: fetch the final-test sources

```bash
python scripts/final_eval/fetch_final_sources.py --revision-manifest results/development/raw_final_sources_manifest.json
```

Writes `data/raw/hf_final/<repo>/<path>` and `data/raw/hf_final/manifest.json`.
None of these repositories is gated. Expected files (from
`results/development/raw_final_sources_manifest.json`):

| Source | Repository @ pinned revision | File | Bytes | Digest |
|---|---|---|---:|---|
| sst | `Bertievidgen/SimpleSafetyTests` @ `98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5` | `sst_test_cases.csv` | 12,483 | git-sha1 `1a81f0d4c0eb5874cff00de49eb519693c848697` |
| catqa | `declare-lab/CategoricalHarmfulQA` @ `321e068b391b7d1aeeb882ac468356a9513c9167` | `data/catqa_english.json` | 106,279 | git-sha1 `17733209f5ca346070c4b0db31f89d9aca6ff7f0` |
| dna | `LibrAI/do-not-answer` @ `74e74f2e4507ef256fe536f78a776f4a1ff67955` | `data_en.csv` | 3,912,791 | git-sha1 `6c7ad83cf665b418229955e0abb602ab0cac6384` |
| coconot | `allenai/coconot` @ `2cbe16aabf9069f17e48c8daad8aeabc29469eb7` | `contrast/test-00000-of-00001.parquet` | 20,462 | sha256 `d2d4f9ea33eac017cfdd2b56669e417e979933418276083d3acb28be170a588f` |
| phtest | `furonghuang-lab/PHTest` @ `edb80210812ecc7dab2219aa2e70921710c01888` | `PHTest.csv` | 666,338 | git-sha1 `cb3b3e64e06637a5b6dfeae7c7bc041865e90df8` |
| no_robots | `HuggingFaceH4/no_robots` @ `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` | `data/test-00000-of-00001.parquet` | 570,905 | sha256 `60707b2636a46e37bb0c1e9ca263a18553f430317b7a53c691676d6a492fc0f2` |

### Step 6: build the final test set

Needs steps 1, 2 and 5: rule 2 of the pre-registered screen drops any prompt
that is an exact or near-duplicate of a prompt in the pool or in the five
development benchmarks, so those files must be present with the hashes of
step 2 (the build records their sha256 under `reference_files`).

```bash
python scripts/final_eval/build_final_test.py --require-tokenizer --out data/benchmarks_final_rebuild
sha256sum data/benchmarks_final_rebuild/final_test/test.jsonl data/benchmarks_final/final_test/test.jsonl
```

Writing to a separate `--out` keeps the shipped `manifest.json` intact (a rebuild
rewrites it with your own `run_info`). The shipped file is
`data/benchmarks_final/final_test/test.jsonl`, 3,453 rows, sha256
`39425a51fc671079dc341f57ec6a4821b1fb2732b50ae23cc95ccda3135e11a0`, built on
26 September 2026 at commit `dc90a9e2`.

Screening rule (PREREGISTRATION.md, Section 4), with the recorded counts:

| Source | Loaded | Rule 1 (within-source dedup) | Rule 2 (pool + benchmark screen) | Rule 3 (cross-source) | Rule 4 (PHTest sample) | Label |
|---|---:|---:|---:|---:|---:|---|
| sst | 100 | 100 | 99 (1 exact match in ToxicChat) | 99 | 99 | harmful |
| catqa | 550 | 543 (7 duplicates) | 543 | 543 | 543 | harmful |
| dna_core | 574 | 574 | 573 (1 near-duplicate of pool train) | 573 | 573 | harmful |
| dna_other | 365 | 364 (1 duplicate) | 364 | 364 | 364 | harmful (secondary) |
| coconot | 379 | 379 | 378 (1 near-duplicate of pool train) | 378 | 378 | safe |
| phtest | 2,077 harmless (1,192 controversial dropped at load) | 2,073 (4 duplicates) | 2,073 | 2,073 | 1,000 (seed 20260926) | safe |
| no_robots | 500 | 498 (2 duplicates) | 496 (1 exact, 1 near in ToxicChat) | 496 | 496 | safe |
| **total** | | | | | **3,453** | 1,874 safe, 1,215 harmful primary, 364 harmful secondary |

**The `over_512_tokens` flags.** After the four rules the script tokenises every
prompt with the `roberta-base` tokenizer from the local Hugging Face cache and
adds `"over_512_tokens": true` to the records over 512 tokens: 16 No Robots rows
in the recorded build. Prompts, ids, labels and order do not depend on this
step; only those 16 extra keys do, so a build without the tokenizer has the same
3,453 prompts but a different file hash, and `manifest.json` then records
`"long_prompts": {"skipped": ...}` instead of `{"limit": 512, "over_by_source": {"no_robots": 16}}`.

* Default (the recorded behaviour): if the tokenizer is not cached the step is
  skipped and a WARNING says so.
* `--require-tokenizer`: the skip becomes an error, so you cannot produce a
  file that silently differs from the recorded one.
* `--backbone-revision SHA`: load that cached snapshot of the tokenizer. The
  study's cache held `roberta-base` at `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
  Not needed in practice (the tokenizer files have not changed across snapshots),
  but it removes the question.

The script logs the output sha256 next to the recorded one at the end of the run.

## 5. Record schema

Every built file is JSON Lines, one object per prompt, in the layout
`scripts/run_evaluation.py` reads:

```json
{"id": "sst_1", "text": "...", "label": 0, "category": "...", "source": "sst"}
```

`label` follows `config.yaml`, `data.labels`: **1 = allow (safe), 0 = mitigate
(harmful)**. The final test set's `source` is one of `sst`, `catqa`, `dna_core`,
`dna_other`, `coconot`, `phtest`, `no_robots`; `dna_other` is the secondary
harmful set and is never pooled into the primary harmful rate. Final-set rows
longer than 512 RoBERTa tokens carry `"over_512_tokens": true`. Learner-arm rows
also carry the number of edits and their ERRANT categories.

## 6. Every source at a glance

Licence facts marked (P) are taken from `results/final_eval_20260926/PREREGISTRATION.md`,
Section 3; (L) from the licence files inside the corpus archive, as recorded in
`results/development/learner_arms_manifest.json`; (U) were not recorded by the
study and are given from the dataset card as remembered, unverified: check the
card before relying on them. Revisions and digests are from the manifests
named in each step above. See `LICENSES.md` for what this means in practice.

| Source | Hub repository | Pinned revision | Files | Licence | Rows used |
|---|---|---|---|---|---|
| JailbreakBench behaviors | `JailbreakBench/JBB-Behaviors` | `886acc35…86fc` | 2 CSV | MIT (U) | 200 benchmark prompts (100 harmful, 100 benign) |
| XSTest | `walledai/XSTest` (gated) | `f1d71318…e8e8` | 1 parquet | CC-BY-4.0 (U) | 450 benchmark prompts (250 safe, 200 unsafe) |
| HarmBench standard | `walledai/HarmBench` (gated) | `fb6c2afd…1be5` | 1 parquet | MIT (U) | 200 benchmark prompts, all harmful |
| ToxicChat 0124 | `lmsys/toxic-chat` | `29df8e4d…8a8` | 2 CSV | CC-BY-NC-4.0 (U) | test: 4,951 benchmark prompts; train: 1,651 pool prompts |
| Dolly 15k | `databricks/databricks-dolly-15k` | `bdd27f4d…ff1a` | 1 JSONL | CC-BY-SA-3.0 (U) | 3,214 pool prompts (instructions without context) |
| BeaverTails 30k | `PKU-Alignment/BeaverTails` | `8401fe60…e678` | 1 JSONL.gz | CC-BY-NC-4.0 (U) | 4,255 pool prompts (harmful only) |
| WildJailbreak | `allenai/wildjailbreak` (gated) | `5ddc12a7…f009` | 1 TSV | ODC-BY + AI2 responsible-use terms (U) | 10,000 pool prompts (5,000 vanilla per class) |
| OR-Bench | `bench-llm/or-bench` | `e36d8b80…e272` | 2 CSV | CC-BY-4.0 (U) | 1,974 benchmark prompts (1,319 hard-1k safe, 655 toxic) |
| W&I+LOCNESS v2.1 | manual (BEA-2019) | archive sha256 `d5cbf68c…c666` | 7 M2 files | non-commercial, no redistribution (L) | 8,746 sentences × 2 arms |
| SimpleSafetyTests | `Bertievidgen/SimpleSafetyTests` | `98223c5d…d3c5` | 1 CSV | CC-BY-2.0 (P) | 99 of 100 |
| CatQA (English) | `declare-lab/CategoricalHarmfulQA` | `321e068b…9167` | 1 JSON | Apache-2.0 (P) | 543 of 550 |
| Do-Not-Answer | `LibrAI/do-not-answer` | `74e74f2e…7955` | 1 CSV | Apache-2.0 (P) | 573 core + 364 other of 939 |
| CoCoNot contrast test | `allenai/coconot` | `2cbe16aa…9eb7` | 1 parquet | ODC-BY (P) | 378 of 379 |
| PHTest harmless | `furonghuang-lab/PHTest` | `edb80210…1888` | 1 CSV | MIT (P) | 1,000 sampled of 2,073 harmless (1,192 controversial dropped) |
| No Robots test | `HuggingFaceH4/no_robots` | `e6f9a4ac…105b` | 1 parquet | CC-BY-NC-4.0 (P) | 496 of 500 (first user turn) |

## 7. Checking what you built

```bash
sha256sum data/distillation_pool/*.jsonl data/benchmarks/*/test.jsonl          # against step 2
sha256sum data/benchmarks_learner/*/*/test.jsonl                                # against step 4
sha256sum data/benchmarks_final/final_test/test.jsonl                           # 39425a51...
python - <<'EOF'
import json; f = json.load(open("results/final_eval_20260926/frozen.json"))
print(json.dumps({**{f"pool_{k}": v["sha256"] for k, v in f["pool"].items()},
                  **{k: v["sha256"] for k, v in f["inputs"].items()}}, indent=1))
EOF
```

`results/final_eval_20260926/frozen.json` is the record the final evaluation
checked its inputs against; every hash in this file traces back to it, to the
fetch and build manifests, or (where stated) to the study workspace.
