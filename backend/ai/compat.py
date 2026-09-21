"""
ai/compat.py — Compatibility shims for legacy vision restoration libraries.

PyTorch 2.0+ and modern torchvision removed `torchvision.transforms.functional_tensor`,
which `basicsr==1.4.2` and `gfpgan>=1.3.8` still attempt to import.
This module registers an alias to `torchvision.transforms.functional` in `sys.modules`.
"""

import sys
import torchvision.transforms.functional as F

if "torchvision.transforms.functional_tensor" not in sys.modules:
    sys.modules["torchvision.transforms.functional_tensor"] = F
