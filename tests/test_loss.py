import unittest

import torch

from src.loss import ClipLoss


class TestLoss(unittest.TestCase):
    def test_clip_loss(self):
        loss = ClipLoss()
        logits = torch.Tensor([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.assertEqual(loss(logits, logits), 0)
