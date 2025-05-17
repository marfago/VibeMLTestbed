# User Guide

## Configuring the Training Script

The training script can be configured using a YAML or JSON configuration file. This file allows you to specify parameters such as the model architecture, dataset, transformations, optimizer, learning rate, batch size, number of epochs, and device (CPU or GPU).

### Configuration File

The configuration file should be named `config.yaml` (or `config.json`) and placed in the root directory of the project.

The following parameters can be configured:

*   `device`: The device to use for training (either "cuda" or "cpu").
*   `transformations`: A list of transformations to apply to the data. Available transformations: "ToTensor", "Normalize", "Resize". Transformations can be configured with parameters.
    *   `Normalize`:
        *   `mean`: The mean for normalization (default: 0.1307).
        *   `std`: The standard deviation for normalization (default: 0.3081).
    *   `Resize`:
        *   `size`: The size to resize the image to (default: [28, 28]).
*   `learning_rate`: The learning rate for the optimizer.
*   `epochs`: The number of epochs to train for.

### Example Configuration (config.yaml)

```yaml
# Configuration file for the ML Testbed Platform

# Device to use for training (cuda or cpu)
device: "cuda"

# List of transformations to apply to the data
# Available transformations: ToTensor, Normalize, Resize
# Transformations can be configured with parameters:
#   Normalize: mean, std
#   Resize: size
transformations:
  - name: "ToTensor"
  - name: "Normalize"
    mean: 0.1307
    std: 0.3081
  - name: "Resize"
    size: [28, 28]

# Learning rate for the optimizer
learning_rate: 0.001

# Number of epochs to train for
epochs: 5
```

### Running the Training Script

To run the training script with the configuration file, use the following command:

```bash
python src/main.py --config config.yaml
```

If no configuration file is specified, the script will use the default values in `config.yaml`.
