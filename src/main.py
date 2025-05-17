import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        metric(output, target)

    avg_loss = running_loss / len(train_loader)
    accuracy = metric.compute()
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Testing function
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            metric(output, target)

    test_loss /= len(test_loader)
    accuracy = metric.compute()
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return test_loss, accuracy

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