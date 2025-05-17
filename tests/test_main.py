import torch.nn as nn
import torch.optim as optim
import pytest
import torch
from src.main import SimpleNN, load_mnist_data, train, evaluate_model

# Mock data for testing
@pytest.fixture
def dummy_data():
    # Create dummy data matching MNIST dimensions
    return torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,))

# Test the SimpleNN model
def test_simple_nn_forward(dummy_data):
    model = SimpleNN()
    data, _ = dummy_data
    output = model(data)
    assert output.shape == (10, 10) # Batch size x number of classes

# Test data loading
def test_load_mnist_data():
    # This test will download MNIST if not present, might take time
    # Consider mocking the dataset download for faster tests if needed
    train_loader, test_loader = load_mnist_data(batch_size=5)
    assert len(train_loader) > 0
    assert len(test_loader) > 0
    train_data, train_target = next(iter(train_loader))
    assert train_data.shape == (5, 1, 28, 28)
    assert train_target.shape == (5,)

# Test training function (basic check)
def test_train_function(dummy_data):
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(dummy_data[0].unsqueeze(1), dummy_data[1])] # Wrap dummy data in a list to simulate loader

    # Run one epoch
    loss, accuracy = train(model, device, train_loader, optimizer, criterion, 1)
    assert isinstance(loss, float)
    assert isinstance(accuracy, torch.Tensor) # torchmetrics returns a tensor

# Test testing function (basic check)
def test_test_function(dummy_data):
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    test_loader = [(dummy_data[0].unsqueeze(1), dummy_data[1])] # Wrap dummy data in a list to simulate loader

    # Run test
    loss, accuracy = evaluate_model(model, device, test_loader, criterion)
    assert isinstance(loss, float)
    assert isinstance(accuracy, torch.Tensor) # torchmetrics returns a tensor