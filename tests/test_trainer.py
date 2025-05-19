import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.engine.trainer import train, evaluate_model
from src.engine.metrics import initialize_metrics
import yaml

# Mock device for testing
device = torch.device("cpu")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create a mock model
class MockModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.linear(x)

# Create mock data loaders
def create_mock_data_loaders(num_classes):
    # Create some dummy data
    train_data = torch.randn(64, 10)
    # Ensure there are positive samples in all classes
    train_targets = torch.randint(0, num_classes, (64,))
    while len(torch.unique(train_targets)) < num_classes:
        train_targets = torch.randint(0, num_classes, (64,))
    test_data = torch.randn(32, 10)
    test_targets = torch.randint(0, num_classes, (32,))

    # Create TensorDatasets
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader

def test_train_function():
    num_classes = 3  # Reduced number of classes
    model = MockModel(num_classes).to(device)
    train_loader, test_loader = create_mock_data_loaders(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    metrics = initialize_metrics(config, device, num_classes)
    epoch = 1
    best_train_accuracy = 0.0
    best_train_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
    
    avg_train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, metric_results, test_metric_results = train(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epoch=epoch,
        best_train_accuracy=best_train_accuracy,
        best_train_loss=best_train_loss,
        best_test_accuracy=best_test_accuracy,
        best_test_loss=best_test_loss,
        test_loader=test_loader,
        metrics=metrics,
        wandb=None,
        num_classes=num_classes # Pass num_classes to train function
    )

    assert isinstance(avg_train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
def test_train_function_empty_data_loaders():
    num_classes = 10
    model = MockModel(num_classes).to(device)
    # Create empty data loaders
    train_loader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)), batch_size=32)
    test_loader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)), batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    metrics = initialize_metrics(config, device, num_classes)
    epoch = 1
    best_train_accuracy = 0.0
    best_train_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
    
    avg_train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, metric_results, test_metric_results = train(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epoch=epoch,
        best_train_accuracy=best_train_accuracy,
        best_train_loss=best_train_loss,
        best_test_accuracy=best_test_accuracy,
        best_test_loss=best_test_loss,
        test_loader=test_loader,
        metrics=metrics,
        wandb=None,
        num_classes=num_classes # Pass num_classes to train function
    )

    assert isinstance(avg_train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
def test_evaluate_model_function():
    num_classes = 10
    model = MockModel(num_classes).to(device)
    train_loader, test_loader = create_mock_data_loaders(num_classes)
    criterion = nn.CrossEntropyLoss()
    metrics = initialize_metrics(config, device, num_classes)
    
    test_loss, metric_results = evaluate_model(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion,
        metrics=metrics,
        wandb=None,
        num_classes=num_classes # Pass num_classes to evaluate_model function
    )

    assert isinstance(test_loss, float)
    assert isinstance(metric_results, dict)
def test_evaluate_model_function_empty_data_loaders():
    num_classes = 10
    model = MockModel(num_classes).to(device)
    # Create empty data loaders
    train_loader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)), batch_size=32)
    test_loader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)), batch_size=32)
    criterion = nn.CrossEntropyLoss()
    metrics = initialize_metrics(config, device, num_classes)
    
    test_loss, metric_results = evaluate_model(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion,
        metrics=metrics,
        wandb=None,
        num_classes=num_classes # Pass num_classes to evaluate_model function
    )

    assert isinstance(test_loss, float)
    assert isinstance(metric_results, dict)