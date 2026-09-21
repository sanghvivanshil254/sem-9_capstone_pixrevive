import sys
import unittest
import torchvision.transforms.functional as F
import ai.compat


class TestCompat(unittest.TestCase):
    def test_functional_tensor_alias_installed(self):
        self.assertIn("torchvision.transforms.functional_tensor", sys.modules)
        self.assertIs(sys.modules["torchvision.transforms.functional_tensor"], F)
