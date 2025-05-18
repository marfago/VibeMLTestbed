import torch.optim as optim

class AdamOptimizer:
    def __init__(self, model_parameters, config):
        self.optimizer = optim.Adam(
            model_parameters,
            lr=config.get("lr", 0.001), # Default learning rate
            betas=config.get("betas", (0.9, 0.999)), # Default betas
            eps=config.get("eps", 1e-8), # Default epsilon
            weight_decay=config.get("weight_decay", 0), # Default weight decay
            amsgrad=config.get("amsgrad", False) # Default amsgrad
        )

    def get_optimizer(self):
        return self.optimizer