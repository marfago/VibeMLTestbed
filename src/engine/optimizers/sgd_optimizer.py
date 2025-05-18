import torch.optim as optim

class SGDOptimizer:
    def __init__(self, model_parameters, config):
        self.optimizer = optim.SGD(
            model_parameters,
            lr=config.get("lr", 0.01), # Default learning rate
            momentum=config.get("momentum", 0), # Default momentum
            weight_decay=config.get("weight_decay", 0) # Default weight decay
        )

    def get_optimizer(self):
        return self.optimizer