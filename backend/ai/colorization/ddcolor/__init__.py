"""
Vendored DDColor inference package (ICCV 2023).

Fetched from https://github.com/piddnad/DDColor by
backend/ai/install_ddcolor.py — see _VENDOR_INFO.md for the pinned revision.
Licensed Apache-2.0. Do not hand-edit the vendored modules; re-run the
installer with --force to refresh them.

The upstream modules live in a `basicsr` package that would collide with the
site-packages basicsr 1.4.2 required by GFPGAN, so their imports are rewritten
to stay inside this directory.
"""

from .model import DDColor
from .pipeline import (
    ColorizationPipeline,
    build_ddcolor_model,
    load_checkpoint_state_dict,
)

__all__ = [
    "DDColor",
    "ColorizationPipeline",
    "build_ddcolor_model",
    "load_checkpoint_state_dict",
]
