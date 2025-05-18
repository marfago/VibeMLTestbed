import torch.optim as optim

class RMSpropOptimizer:
    def __init__(self, model_parameters, config):
        self.optimizer = optim.RMSprop(
            model_parameters,
            lr=config.get("lr", 0.01), # Default learning rate
            alpha=config.get("alpha", 0.99), # Default alpha
            eps=config.get("eps", 1e-8), # Default epsilon
            weight_decay=config.get("weight_decay", 0), # Default weight decay
            momentum=config.get("momentum", 0), # Default momentum
            centered=config.get("centered", False) # Default centered
        )

    def get_optimizer(self):
        return self.optimizer