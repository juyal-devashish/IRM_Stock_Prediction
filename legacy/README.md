# Legacy research scripts

Original, unmodified experiment code kept for reference and as the numerical
baseline for the regression tests in `tests/test_regression_legacy.py`.

Do not edit these files. The maintained implementation lives in `src/irm_stock/`.

The finance scripts do `from util_file import *`, so run them from this directory:

    cd legacy && uv run python finance_experiments_combined.py --experiment 1
