"""Superseded — delete this file.

This was a throwaway harness written while diagnosing the impulse-noise bug. It
has been replaced by scripts/check_impulse_robustness.py, which covers the same
ground as committed regression checks with explicit tolerances and a non-zero
exit code, using the shared fixtures in scripts/impulse_testkit.py.

Kept only because the session that superseded it had no shell available to remove
it. Safe to delete:

    rm scripts/_scratch_impulse_check.py
"""

raise SystemExit(
    "_scratch_impulse_check.py is superseded. "
    "Run: PYTHONPATH=backend python scripts/check_impulse_robustness.py"
)
