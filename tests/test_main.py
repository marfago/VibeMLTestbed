import torch
from torchvision import transforms
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open
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
from src.transformations import get_transformation
import src.data
from src.data import get_dataset

@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.SimpleNN')
@patch('src.data.mnist_data.load_mnist_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
@patch('src.main.open', create=True)
@patch('src.main.torch.device')
def test_main_function(mock_torch_device, mock_open_file, mock_evaluate_model, mock_train, mock_load_mnist_data, mock_simple_nn, mock_parse_args):
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
    mock_train.return_value = (0.1, torch.tensor(0.9, dtype=torch.float32), 0.9, 0.1, 0.8, 0.2)
    mock_evaluate_model.return_value = (0.2, {"Accuracy": torch.tensor(0.8)}) # Mock loss and accuracy


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

def test_load_cifar10_data():
    from src.data.cifar10_data import load_cifar10_data
    train_loader, test_loader = load_cifar10_data()
    assert train_loader is not None
    assert test_loader is not None

def test_load_cifar100_data():
    from src.data.cifar100_data import load_cifar100_data
    train_loader, test_loader = load_cifar100_data()
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
    best_train_accuracy = -1.0
    best_train_loss = float('inf')
    best_test_accuracy = -1.0
    best_test_loss = float('inf')
    test_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10)}

    # Call the train function
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics=metrics
    )

    # Assert that the function returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
    assert train_accuracy.item() >= 0.0
    assert best_train_accuracy >= -1.0
    assert best_train_loss != float('inf')
    assert best_test_accuracy >= -1.0
    assert best_test_loss != float('inf')

def test_evaluate_model_function():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    test_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]
    criterion = nn.CrossEntropyLoss()
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10)}

    # Call the evaluate_model function
    test_loss, accuracy = evaluate_model(model, device, test_loader, criterion, metrics)

    # Assert that the function returns the expected values
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, dict)

def test_simple_nn_model():
    # Create an instance of the SimpleNN model
    model = SimpleNN(input_size=784, num_classes=10)

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
    best_train_accuracy = -1.0
    best_train_loss = float('inf')
    best_test_accuracy = -1.0
    best_test_loss = float('inf')
    test_loader = []
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10)}

    # Call the train function
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics=metrics
    )

    # Assert that the function returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
    assert train_accuracy.item() >= 0.0

def test_evaluate_model_function_empty_loader():
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    test_loader = []
    criterion = nn.CrossEntropyLoss()
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10)}

    # Call the evaluate_model function
    test_loss, accuracy = evaluate_model(model, device, test_loader, criterion, metrics)

    # Assert that the function returns the expected values
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, dict)

@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.SimpleNN')
@patch('src.data.mnist_data.load_mnist_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
@patch('src.main.open', create=True)
@patch('src.main.torch.device')
def test_main_function_config_file(mock_torch_device, mock_open_file, mock_evaluate_model, mock_train, mock_load_mnist_data, mock_simple_nn, mock_parse_args):
    # Mock the device to always return cpu for testing
    config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "mnist", "batch_size": 32}, "metrics": [{"name": "Accuracy"}]}
    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)]

    mock_train.return_value = (0.1, torch.tensor(0.9), 0, float('inf'), 0, float('inf'))
    mock_evaluate_model.return_value = (0.2, {"Accuracy": torch.tensor(0.8)})

    # Test YAML config
    mock_parse_args.return_value = argparse.Namespace(config="config.yaml")
    mock_open_file.return_value = mock_open(read_data=yaml.dump(config)).return_value
    mock_torch_device.return_value = config["device"]
    from src.main import main
    main()

    # Test JSON config
    mock_parse_args.return_value = argparse.Namespace(config="config.json")
    mock_open_file.return_value = mock_open(read_data=json.dumps(config)).return_value
    mock_torch_device.return_value = config["device"]
    from src.main import main
    main()

    # Test invalid config file
    mock_parse_args.return_value = argparse.Namespace(config="config.txt")
    mock_open_file.side_effect = ValueError("Invalid configuration file type: config.txt")
    mock_torch_device.return_value = "cpu"
    with pytest.raises(ValueError, match="Invalid configuration file type: config.txt"):
        from src.main import main
        main()

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

def test_cached_dataset_cifar10():
    # Create a mock dataset and transform
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = (Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)), 0)
    mock_transform = MagicMock(side_effect=transforms.ToTensor())

    # Create a CachedDataset instance
    from src.data.cifar10_data import CachedDataset
    cached_dataset = CachedDataset(mock_dataset, transform=mock_transform)

    # Access the same element twice
    sample1 = cached_dataset[0]
    sample2 = cached_dataset[0]

    # Assert that the transform was only called once for the same element
    assert mock_transform.call_count == 1

    # Assert that the samples are the same
    assert torch.equal(sample1[0], sample2[0])

def test_get_transformation_resize():
    transform_config = {"name": "Resize", "size": (64, 64)}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.Resize)
    assert transform.size == (64, 64)

def test_get_transformation_invalid():
    transform_config = {"name": "Invalid"}
    with pytest.raises(ValueError, match="Invalid transformation: Invalid"):
        get_transformation(transform_config)

def test_get_transformation_normalize_defaults():
    transform_config = {"name": "Normalize"}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.Normalize)
    assert transform.mean == (0.1307,)
    assert transform.std == (0.3081,)

def test_get_transformation_other():
    transform_config = {"name": "ToTensor"}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.ToTensor)

def test_get_transformation_normalize():
    transform_config = {"name": "Normalize", "mean": (0.5,), "std": (0.5,)}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.Normalize)
    assert transform.mean == (0.5,)
    assert transform.std == (0.5,)

def test_get_transformation_resize_defaults():
    transform_config = {"name": "Resize"}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.Resize)
    assert transform.size == (28, 28)

def test_get_dataset_invalid():
    with patch('src.data.importlib.import_module') as mock_import_module:
        mock_import_module.side_effect = ImportError("Invalid module")
        with pytest.raises(ValueError, match="Invalid dataset name: invalid_dataset"):
            get_dataset("invalid_dataset")
