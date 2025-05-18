# User Guide

## Configuring the Training Script

The training script can be configured using a YAML or JSON configuration file. This file allows you to specify parameters such as the model architecture, dataset, transformations, optimizer, learning rate, batch size, number of epochs, and device (CPU or GPU).

### Configuration File

The configuration file should be named `config.yaml` (or `config.json`) and placed in the root directory of the project.

The following parameters can be configured:

*   `device`: The device to use for training (either "cuda" or "cpu").
*   `dataset`:
    *   `name`: The name of the dataset to use (e.g., "MNIST", "CIFAR10", "CIFAR100").
    *   `batch_size`: The batch size to use for training.
*   `transformations`: A list of transformations to apply to the data. Available transformations: "ToTensor", "Normalize", "Resize". Transformations can be configured with parameters.
    *   `ToTensor`: No parameters.
    *   `Normalize`:
        *   `mean`: The mean for normalization (default: 0.1307).
        *   `std`: The standard deviation for normalization (default: 0.3081).
    *   `Resize`:
        *   `size`: The size to resize the image to (default: [28, 28]).
*   `optimizer`:
    *   `name`: The name of the optimizer to use (e.g., "Adam", "SGD", "RMSprop").
    *   `config`: A dictionary containing the optimizer-specific parameters.
        *   `Adam`:
            *   `lr`: The learning rate (default: 0.001).
            *   `betas`: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
            *   `eps`: Term added to improve numerical stability (default: 1e-08).
            *   `weight_decay`: Weight decay (L2 penalty) (default: 0).
            *   `amsgrad`: Whether to use the AMSGrad variant of this algorithm (default: False).
        *   `SGD`:
            *   `lr`: The learning rate.
            *   `momentum`: Momentum factor (default: 0).
            *   `weight_decay`: Weight decay (L2 penalty) (default: 0).
        *   `RMSprop`:
            *   `lr`: The learning rate.
            *   `alpha`: Smoothing constant (default: 0.99).
            *   `eps`: Term added to improve numerical stability (default: 1e-08).
            *   `weight_decay`: Weight decay (L2 penalty) (default: 0).
            *   `momentum`: Momentum factor (default: 0).
            *   `centered`: If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance (default: False).
*   `learning_rate`: The learning rate for the optimizer (deprecated, use optimizer.config.lr instead).
*   `epochs`: The number of epochs to train for.
*   `loss`: The loss function to use for training (e.g., "MSELoss", "MAELoss", "CrossEntropyLoss").
*   `metrics`: A list of metrics to track during training (e.g., "Accuracy", "F1", "ConfusionMatrix", "Precision", "Recall", "AUC", "ROC", "AveragePrecision").

### Example Configuration (config.yaml)

```yaml
# Configuration file for the ML Testbed Platform

# Device to use for training (cuda or cpu)
device: "cuda"

# Dataset configuration
dataset:
  name: "CIFAR10"
  batch_size: 64

# List of transformations to apply to the data
transformations:
  - name: "ToTensor"
  - name: "Normalize"
    mean: 0.4914
    std: 0.2023

# Optimizer configuration
optimizer:
  name: "Adam"
  config:
    lr: 0.001

# Number of epochs to train for
epochs: 5

# Loss function to use for training
loss: "CrossEntropyLoss"

# Metrics to track during training
metrics:
  - name: "Accuracy"
  - name: "F1"
```

### Running the Training Script

To run the training script with the configuration file, use the following command:

```bash
python src/main.py --config config.yaml
```

If no configuration file is specified, the script will use the default values in `config.yaml`.
