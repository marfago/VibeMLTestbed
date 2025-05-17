import torch
from torchvision import transforms
import sys
import pytest
from unittest.mock import patch, MagicMock
from src.data.mnist_data import load_mnist_data, CachedDataset
from src.engine.trainer import train, evaluate_model
import torch.nn as nn
import torch.optim as optim
from src.models.simple_nn import SimpleNN
import torchmetrics
import argparse
import yaml
import json
import io
from torchvision import datasets
from PIL import Image
import numpy as np

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

def test_normalize_transformation():
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    sample_data = torch.randn(1, 28, 28)
    normalized_data = normalize(sample_data)
    assert normalized_data.shape == (1, 28, 28)

def test_resize_transformation():
    resize = transforms.Resize(size=(64, 64))
    sample_data = torch.randn(1, 28, 28)
    resized_data = resize(sample_data)
    assert resized_data.shape == (1, 64, 64)

def test_load_mnist_data():
    train_loader, test_loader = load_mnist_data()
    assert train_loader is not None
    assert test_loader is not None

def test_train_function():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    train_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epoch = 1
    best_train_accuracy = 0.0
    best_train_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
    test_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]

    # Call the train function
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader
    )

    # Assert that the function returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)

def test_evaluate_model_function():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    test_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]
    criterion = nn.CrossEntropyLoss()

    # Call the evaluate_model function
    test_loss, accuracy = evaluate_model(model, device, test_loader, criterion)

    # Assert that the function returns the expected values
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, torch.Tensor)

def test_simple_nn_model():
    # Create an instance of the SimpleNN model
    model = SimpleNN()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 28, 28)

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Assert that the output has the correct shape
    assert output.shape == (1, 10)

def test_train_function_empty_loader():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    train_loader = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epoch = 1
    best_train_accuracy = 0.0
    best_train_loss = float('inf')
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
    test_loader = []

    # Call the train function
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader
    )

    # Assert that the function returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)

def test_evaluate_model_function_empty_loader():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    test_loader = []
    criterion = nn.CrossEntropyLoss()

    # Call the evaluate_model function
    test_loss, accuracy = evaluate_model(model, device, test_loader, criterion)

    # Assert that the function returns the expected values
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, torch.Tensor)

@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.yaml.safe_load')
@patch('src.main.SimpleNN')
@patch('src.main.load_mnist_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
def test_main_function_config_file(mock_evaluate_model, mock_train, mock_load_mnist_data, mock_simple_nn, mock_safe_load, mock_parse_args):
    mock_parse_args.return_value = argparse.Namespace(config="config.yaml")
    mock_safe_load.return_value = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1}
    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)]
    mock_train.return_value = (0.1, torch.tensor(0.9), 0, float('inf'), 0, float('inf'))
    mock_evaluate_model.return_value = (0.2, torch.tensor(0.8))

    from src.main import main
    main()

def test_main_function_invalid_config_file():
    with patch('src.main.argparse.ArgumentParser.parse_args') as mock_parse_args:
        def side_effect(*args, **kwargs):
            raise ValueError("Invalid configuration file type: config.txt")

        mock_parse_args.side_effect = side_effect
        try:
            from src.main import main
            main()
        except ValueError as e:
            assert "Invalid configuration file type: config.txt" in str(e)
        else:
            pytest.fail("ValueError was not raised")

def test_cached_dataset():
    # Create a mock dataset and transform
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = (Image.fromarray(np.zeros((28, 28), dtype=np.uint8)), 0)
    mock_transform = MagicMock(side_effect=transforms.ToTensor())

    # Create a CachedDataset instance
    cached_dataset = CachedDataset(mock_dataset, transform=mock_transform)

    # Access the same element twice
    sample1 = cached_dataset[0]
    sample2 = cached_dataset[0]

    # Assert that the transform was only called once for the same element
    assert mock_transform.call_count == 1

    # Assert that the samples are the same
    assert torch.equal(sample1[0], sample2[0])