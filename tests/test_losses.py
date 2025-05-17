import torch
import torch.nn as nn
import unittest
from src.engine import losses

class TestLosses(unittest.TestCase):
    def test_get_loss_function_mse(self):
        loss_fn = losses.get_loss_function("MSELoss")
        self.assertIsInstance(loss_fn, nn.MSELoss)

    def test_get_loss_function_mae(self):
        loss_fn = losses.get_loss_function("MAELoss")
        self.assertIsInstance(loss_fn, nn.L1Loss)

    def test_get_loss_function_cross_entropy(self):
        loss_fn = losses.get_loss_function("CrossEntropyLoss")
        self.assertIsInstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_function_invalid(self):
        with self.assertRaises(ValueError):
            losses.get_loss_function("InvalidLoss")

if __name__ == '__main__':
    unittest.main()