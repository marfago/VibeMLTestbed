import torch.nn as nn
import torch.optim as optim
import pytest
import torch
from src.models.simple_nn import SimpleNN
from src.data.mnist_data import load_mnist_data
from src.engine.trainer import train, evaluate_model
from unittest.mock import patch, MagicMock

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
    test_loader_mock = MagicMock()
    test_loader_mock.__len__.return_value = 1
    loss, accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, 1, 0, float('inf'), 0, float('inf'), test_loader_mock)
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

# Test main function
@patch('src.main.load_mnist_data')
@patch('src.main.SimpleNN')
@patch('src.main.optim.Adam')
@patch('src.main.nn.CrossEntropyLoss')
@patch('src.main.train')
@patch('src.main.evaluate_model')
@patch('src.main.torch.cuda.is_available', return_value=False) # Mock CUDA availability
def test_main_function(mock_is_available, mock_evaluate_model, mock_train, mock_criterion, mock_optimizer, mock_simplenn, mock_load_mnist_data):
    from src.main import main

    # Configure mock SimpleNN to return a mock object with a .to() method, which in turn returns a mock with a .parameters() method
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = MagicMock()
    mock_model_instance.to.return_value.parameters.return_value = MagicMock()
    mock_simplenn.return_value = mock_model_instance

    # Mock return values for other functions
    mock_load_mnist_data.return_value = (MagicMock(), MagicMock()) # train_loader, test_loader
    mock_optimizer.return_value = MagicMock()
    mock_criterion.return_value = MagicMock()
    mock_train.return_value = (0.1, torch.tensor(0.9)) # loss, accuracy
    mock_evaluate_model.return_value = (0.2, torch.tensor(0.8)) # loss, accuracy

    main()

    # Assert calls
    mock_load_mnist_data.assert_called_once()
    mock_simplenn.assert_called_once()
    mock_model_instance.to.assert_called_once() # Assert .to() was called
    mock_model_instance.to.return_value.parameters.assert_called_once() # Assert .parameters() was called on the result of .to()
    mock_optimizer.assert_called_once_with(mock_model_instance.to.return_value.parameters.return_value, lr=0.001) # Assert optimizer was called with the result of .to().parameters()
    mock_criterion.assert_called_once()
    assert mock_train.call_count == 5 # Called for each epoch
    assert mock_evaluate_model.call_count == 5 # Called for each epoch
