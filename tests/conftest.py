"""Pytest-wide configuration for the PixRevive backend test suite.

Forces PIXREVIVE_DEVICE=cpu for the test process: several test modules each
construct a full RestorationPipeline, and on the 4 GB CUDA card those loaders
contend and OOM nondeterministically (a different test dies on every run,
with `memory allocation failed ... free: 0` in the captured stderr). The
suite asserts no CUDA-specific behavior — the heavy stages these tests
exercise are classical OpenCV — so CPU is both deterministic and sufficient.

Use `os.environ.setdefault` so an explicit `PIXREVIVE_DEVICE=cuda` from the
developer still wins when debugging GPU paths.
"""

import os

os.environ.setdefault("PIXREVIVE_DEVICE", "cpu")
