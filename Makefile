# LLM-Guard release checkout: everyday targets.
#
#   make install        editable install with the test extra
#   make test           the pytest suite (no model loading, no network; ~1-2 min)
#   make lint           byte-compile every Python file
#   make analysis       regenerate the final-study analysis into a scratch copy and
#                       compare it with the stored results/final_eval_20260926/analysis/
#   make verify-frozen  re-hash the pinned checkpoints against frozen.json
#
# Override the interpreter with PYTHON=..., e.g.
#   make PYTHON=/path/to/venv/bin/python test

PYTHON  ?= python
STUDY   ?= results/final_eval_20260926
SCRATCH ?= build/analysis_check          # under build/, which .gitignore ignores

export PYTHONPATH := src

.PHONY: install test lint analysis verify-frozen

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m compileall -q src scripts tests app.py

# analyse_final.py writes <study>/analysis/{results.json,tables.md}; it is run on a
# copy so the stored study stays untouched. results.json is deterministic
# (fixed bootstrap seed), so the regenerated file is expected to be byte-identical;
# tests/test_analysis_regeneration.py does the same comparison with tolerances.
analysis:
	rm -rf "$(SCRATCH)/study"
	mkdir -p "$(SCRATCH)"
	cp -r "$(STUDY)" "$(SCRATCH)/study"
	rm -rf "$(SCRATCH)/study/analysis"
	$(PYTHON) scripts/final_eval/analyse_final.py --study "$(SCRATCH)/study" > "$(SCRATCH)/analyse_final.log"
	@cmp "$(STUDY)/analysis/results.json" "$(SCRATCH)/study/analysis/results.json" \
	    && echo "analysis: regenerated results.json is byte-identical to $(STUDY)/analysis/results.json" \
	    || { echo "analysis: results.json DIFFERS from the stored copy (see $(SCRATCH))"; exit 1; }

# Intended: scripts/final_eval/verify_frozen.py re-hashes every file listed in
# <study>/frozen.json (checkpoints, config, inputs) and reports mismatches. Until
# that script exists, the checkpoint hashes are covered by
# tests/test_frozen_checkpoints.py, which this target falls back to.
verify-frozen:
	@if [ -f scripts/final_eval/verify_frozen.py ]; then \
	    $(PYTHON) scripts/final_eval/verify_frozen.py --study "$(STUDY)"; \
	else \
	    echo "scripts/final_eval/verify_frozen.py not present; running the checkpoint-hash test instead"; \
	    $(PYTHON) -m pytest -q tests/test_frozen_checkpoints.py; \
	fi
