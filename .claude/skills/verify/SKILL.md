---
name: verify
description: Run lint, format check and the test suite for this repo and summarize failures. Use after code changes.
---

Run, in order, from the repo root, and stop to report on the first failing stage:

1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run pytest -q`

If `tests/test_regression_legacy.py` fails, the refactor changed numerical behaviour: say so prominently and do not "fix" it by editing `legacy/` or loosening tolerances. Report each stage as pass/fail with the relevant error output.
