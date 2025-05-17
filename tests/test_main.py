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

# Simple mock DataLoader for testing
class MockDataLoader:
    def __init__(self, data, batch_size=2):
        self.data = data[0] # Assuming data is a tuple (inputs, targets)
        self.targets = data[1]
        self.batch_size = batch_size
        self.length = len(self.data)
        # Correctly calculate the number of batches
        self._num_batches = (self.length + self.batch_size - 1) // self.batch_size if self.batch_size > 0 else 0


    def __iter__(self):
        for i in range(0, self.length, self.batch_size):
            yield self.data[i:i+self.batch_size], self.targets[i:i+self.batch_size]

    def __len__(self):
        return self._num_batches

# Test the MockDataLoader's __len__ method
def test_mock_dataloader_len(dummy_data):
    loader = MockDataLoader(dummy_data, batch_size=2)
    assert len(loader) == 5
    loader = MockDataLoader(dummy_data, batch_size=10)
    assert len(loader) == 1
    loader = MockDataLoader(dummy_data, batch_size=11)
    assert len(loader) == 1
    loader = MockDataLoader((torch.randn(0, 1, 28, 28), torch.randint(0, 10, (0,))), batch_size=2)
    assert len(loader) == 0
    loader = MockDataLoader(dummy_data, batch_size=0) # Test with batch_size 0
    assert len(loader) == 0


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

    # Use MockDataLoader
    train_loader = MockDataLoader(dummy_data, batch_size=2) # Specify batch size

    # Run one epoch
    test_loader_mock = MockDataLoader(dummy_data, batch_size=2)
    loss, accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, 1, 0, float('inf'), 0, float('inf'), test_loader_mock)
    assert isinstance(loss, float)
    assert isinstance(accuracy, torch.Tensor) # torchmetrics returns a tensor

# Test testing function (basic check)
def test_test_function(dummy_data):
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Use MockDataLoader
    test_loader = MockDataLoader(dummy_data, batch_size=2) # Specify batch size

    # Run test
    loss, accuracy = evaluate_model(model, device, test_loader, criterion)
    assert isinstance(loss, float)
    assert isinstance(accuracy, torch.Tensor) # torchmetrics returns a tensor

# Test main function
# Test main function
@patch('src.main.load_mnist_data')
@patch('src.main.SimpleNN')
@patch('src.main.train')
@patch('src.main.evaluate_model')
@patch('src.main.torch.device')
def test_main_function(mock_torch_device, mock_evaluate_model, mock_train, mock_simple_nn, mock_load_mnist_data):
    # Mock the device to always return cpu for testing
    mock_torch_device.return_value = torch.device("cpu")

    # Mock load_mnist_data to return mock loaders
    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)

    # Mock the model instance
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)] # Mock parameters to avoid empty list error

    # Mock train and evaluate_model return values
    mock_train.return_value = (0.1, torch.tensor(0.9), 0, float('inf'), 0, float('inf')) # Mock loss and accuracy
    mock_evaluate_model.return_value = (0.2, torch.tensor(0.8)) # Mock loss and accuracy

    # Call the main function
    from src.main import main
    main()

    # Assert that the mocked functions were called
    mock_load_mnist_data.assert_called_once()
    mock_simple_nn.assert_called_once()
    mock_train.assert_called() # Called for each epoch
    mock_evaluate_model.assert_called() # Called for each epoch

# Test training function with multiple epochs and loss decrease check
def test_train_function_loss_decrease(dummy_data):
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Use a slightly higher LR for faster convergence in mock
    criterion = nn.CrossEntropyLoss()

    train_loader = MockDataLoader(dummy_data, batch_size=2)
    test_loader_mock = MockDataLoader(dummy_data, batch_size=2)

    # Run multiple epochs and check if loss decreases
    previous_loss = float('inf')
    for epoch in range(3): # Run for 3 epochs
        loss, accuracy, _, _, _, _ = train(model, device, train_loader, optimizer, criterion, epoch + 1, 3, float('inf'), 0, float('inf'), test_loader_mock)
        assert isinstance(loss, float)
        assert isinstance(accuracy, torch.Tensor)
        # Check if loss decreased (basic check, not guaranteed for every epoch with small data)
        # A more robust test would check average loss over several runs or a larger dataset
        if epoch > 0:
             assert loss <= previous_loss + 1e-5 # Allow for minor fluctuations
        previous_loss = loss

# Test evaluate_model with empty data
def test_evaluate_model_empty_data():
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()

    empty_data = (torch.randn(0, 1, 28, 28), torch.randint(0, 10, (0,)))
    empty_loader = MockDataLoader(empty_data, batch_size=2)

    loss, accuracy = evaluate_model(model, device, empty_loader, criterion)

    # For an empty dataset, loss and accuracy should be 0 or similar indicator
    # Depending on implementation, loss might be NaN or 0, accuracy should be 0 or 1 (no samples)
    # Let's assume 0 loss and 0 accuracy for empty data based on typical metric behavior
    assert loss == 0.0
    assert accuracy == 0.0 # Or torch.tensor(0.0) depending on evaluate_model return type
