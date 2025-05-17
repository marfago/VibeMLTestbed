import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms
import sys

from src.models.simple_nn import SimpleNN
from src.data import get_dataset
from src.engine.trainer import train, evaluate_model
from src.transformations import get_transformation

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
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training and testing loop
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, epoch, 0, float('inf'), 0, float('inf'), test_loader, num_classes=num_classes)
        test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion, num_classes=num_classes)

if __name__ == "__main__":
    main()