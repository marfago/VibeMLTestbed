from src.engine import optimizers
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms
import sys
import torchmetrics
from rich import print
from rich.table import Table
from rich.style import Style
import numpy as np

from src.models.simple_nn import SimpleNN
from src.data import get_dataset
from src.engine.trainer import train, evaluate_model
from src.transformations import get_transformation
from src.engine.metrics import initialize_metrics

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

    # Parse transformations
    transformations = []
    for transform_config in config["transformations"]:
        transformations.append(get_transformation(transform_config))

    # Load data
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    load_data, num_classes, input_size = get_dataset(dataset_name)
    train_loader, test_loader = load_data(batch_size=batch_size, transformations=transformations)

    # Initialize model, optimizer, and loss function
    model = SimpleNN(input_size=input_size, num_classes=num_classes).to(device)
    optimizer_name = config["optimizer"]["name"]
    optimizer_config = config["optimizer"].get("config", {})  # Get optimizer-specific config

    if optimizer_name == "Adam":
        optimizer = optimizers.adam_optimizer.AdamOptimizer(model.parameters(), optimizer_config).get_optimizer()
    elif optimizer_name == "SGD":
        optimizer = optimizers.sgd_optimizer.SGDOptimizer(model.parameters(), optimizer_config).get_optimizer()
    elif optimizer_name == "RMSprop":
        optimizer = optimizers.rmsprop_optimizer.RMSpropOptimizer(model.parameters(), optimizer_config).get_optimizer()
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics
    metrics = initialize_metrics(config, device, num_classes)

    final_test_metrics = {}
    final_train_metrics = {}
    # Training and testing loop
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _, _, _, _, train_metric_results, test_metric_results = train(model, device, train_loader, optimizer, criterion, epoch, 0, float('inf'), 0, float('inf'), test_loader, metrics=metrics, num_classes=num_classes)
        final_test_metrics = test_metric_results # Store the results of the last epoch
        final_train_metrics = train_metric_results

    # Create a table with two columns: one for training metrics and one for testing metrics
    table = Table(title="Final Metrics Summary")
    table.add_column("Metric", style="bold magenta")
    table.add_column("Training", style="cyan")
    table.add_column("Testing", style="green")

    # Add rows to the table
    for metric_name in metrics.keys():
        train_value = final_train_metrics.get(metric_name, "N/A")
        test_value = final_test_metrics.get(metric_name, "N/A")

        if isinstance(train_value, torch.Tensor):
            train_value = train_value.item()
        else:
            train_value = "N/A"

        if isinstance(test_value, torch.Tensor):
            test_value = test_value.item()
        else:
            test_value = "N/A"
        
        table.add_row(metric_name, str(train_value), str(test_value))

    print(table)

if __name__ == "__main__":
    main()