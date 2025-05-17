import torch.nn as nn

def get_loss_function(loss_name):
    if loss_name == "MSELoss":
        return nn.MSELoss()
    elif loss_name == "MAELoss":
        return nn.L1Loss()
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function name: {loss_name}")