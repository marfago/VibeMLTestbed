import pytest
import torch.nn as nn
from src.engine.losses import get_loss_function

def test_get_loss_function_mse():
    loss_fn = get_loss_function("MSELoss")
    assert isinstance(loss_fn, nn.MSELoss)

def test_get_loss_function_mae():
    loss_fn = get_loss_function("MAELoss")
    assert isinstance(loss_fn, nn.L1Loss)

def test_get_loss_function_cross_entropy():
    loss_fn = get_loss_function("CrossEntropyLoss")
    assert isinstance(loss_fn, nn.CrossEntropyLoss)

def test_get_loss_function_invalid_name():
    with pytest.raises(ValueError):
        get_loss_function("InvalidLoss")