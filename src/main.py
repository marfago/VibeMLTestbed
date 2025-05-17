import torch
import torch.nn as nn
import torch.optim as optim

from src.models.simple_nn import SimpleNN
from src.data.mnist_data import load_mnist_data
from src.engine.trainer import train, evaluate_model

def main():
    print("ML Testbed Platform - User Story 1: Train and test a simple fully connected neural network on MNIST.")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()