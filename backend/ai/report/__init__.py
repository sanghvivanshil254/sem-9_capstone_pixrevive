"""report/ — Restoration Report (T-2, plan 5.1): the honesty layer.

`build_restoration_report(context)` classifies a run's evidence into
recovered / reconstructed / flagged and is stored in
``context.image_metadata["restoration_report"]`` at the end of restore().
See restoration_report.py for the contract.
"""

from .restoration_report import build_restoration_report

__all__ = ["build_restoration_report"]
