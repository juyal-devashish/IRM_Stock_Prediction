---
name: run-experiment
description: Run an IRM experiment (finance, colored-mnist, synthetic) via the irm-stock CLI, with a quick smoke mode, and report where outputs were written.
disable-model-invocation: true
---

Run the experiment named in $ARGUMENTS (default: ask which of `finance`, `colored-mnist`, `synthetic`).

1. Unless the user asked for a full run, use a **quick** configuration so it finishes in minutes:
   - finance: `uv run irm-stock finance --experiment N --n-restarts 1 --steps 100 --games-steps 100` (needs internet; prices are cached afterwards)
   - colored-mnist: `uv run --extra mnist irm-stock colored-mnist --n-restarts 1 --steps 100 --games-steps 100 --hidden-dim 64`
   - synthetic: `uv run irm-stock synthetic --configs FOU,PES --n-reps 2 --n-iterations 2000`
2. Full runs (defaults) take long (synthetic: hours). Confirm with the user first and run in the background.
3. Outputs go to `outputs/<experiment>/<timestamp>_<slug>/` (`results.json`, plot, `run.log`). Report that path and the printed summary (mean ± std train/test accuracy per method).
4. Do not copy into `results/` unless the user asks; that directory is curated.
