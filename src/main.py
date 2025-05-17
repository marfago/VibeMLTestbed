import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from src.models.simple_nn import SimpleNN
from src.data.mnist_data import load_mnist_data
from src.engine.trainer import train, evaluate_model

def main():
    print("ML Testbed Platform - User Story 1: Train and test a simple fully connected neural network on MNIST.")

    # Argument parser
    parser = argparse.ArgumentParser(description="ML Testbed Platform")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_mnist_data()

    # Initialize model, optimizer, and loss function
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training and testing loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, epoch, 0, float('inf'), 0, float('inf'), test_loader)
        test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()