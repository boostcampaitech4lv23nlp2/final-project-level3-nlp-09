import unittest

import torch


class TestClass(unittest.TestCase):
    def test(self):
        data = [[1, 2], [3, 4]]
        data_tensor = torch.tensor(data)
        self.assertEqual(data_tensor.size(), (2, 2))
