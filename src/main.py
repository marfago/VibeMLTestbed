import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms
import sys

from src.models.simple_nn import SimpleNN
from src.data.mnist_data import load_mnist_data
from src.data.cifar10_data import load_cifar10_data
from src.data.cifar100_data import load_cifar100_data
from src.engine.trainer import train, evaluate_model

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="ML Testbed Platform")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file (YAML or JSON)")

    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        elif args.config.endswith(".json"):
            config = json.load(f)
        else:
            raise ValueError("Invalid configuration file type: {}".format(args.config))

    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define available transformations
    available_transformations = {
        "ToTensor": transforms.ToTensor,
        "Normalize": lambda mean, std: transforms.Normalize(mean, std),
        "Resize": transforms.Resize
    }

    # Parse transformations
    transformations = []
    for transform_config in config["transformations"]:
        transform_name = transform_config["name"]
        if transform_name in available_transformations:
            if transform_name == "Normalize":
                mean = transform_config.get("mean", (0.1307,))
                std = transform_config.get("std", (0.3081,))
                transformations.append(available_transformations[transform_name](mean, std))
            elif transform_name == "Resize":
                size = transform_config.get("size", (28, 28))
                transformations.append(available_transformations[transform_name](size))
            else:
                transformations.append(available_transformations[transform_name]())
        else:
            raise ValueError(f"Invalid transformation: {transform_name}")

    # Load data
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    if dataset_name == "MNIST":
        train_loader, test_loader = load_mnist_data(batch_size=batch_size, transformations=transformations)
        num_classes = 10
        input_size = 28 * 28
    elif dataset_name == "CIFAR10":
        train_loader, test_loader = load_cifar10_data(batch_size=batch_size, transformations=transformations)
        num_classes = 10
        input_size = 32 * 32 * 3
    elif dataset_name == "CIFAR100":
        train_loader, test_loader = load_cifar100_data(batch_size=batch_size, transformations=transformations)
        num_classes = 100
        input_size = 32 * 32 * 3
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Initialize model, optimizer, and loss function
    model = SimpleNN(input_size=input_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training and testing loop
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, epoch, 0, float('inf'), 0, float('inf'), test_loader, num_classes=num_classes)
        test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion, num_classes=num_classes)

if __name__ == "__main__":
    main()