import unittest

import torch

from src.utils import get_autocast, get_cast_dtype, set_seed


class TestUtils(unittest.TestCase):
    def test_set_seed(self):
        seed = 200
        set_seed(seed)
        self.assertEqual(torch.initial_seed(), seed)

    def test_get_cast_dtype(self):
        self.assertEqual(get_cast_dtype("fp16"), torch.float16)
        self.assertEqual(get_cast_dtype("bf16"), torch.bfloat16)

    def test_get_autocast(self):
        self.assertEqual(get_autocast("amp"), torch.cuda.amp.autocast)
