import torch
from torchvision import transforms
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
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
import wandb
from src.engine.trainer import wandb_installed
from src.engine.optimizers import adam_optimizer
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

@pytest.mark.parametrize("config_file", ["config.yaml", "config_sgd.yaml", "config_cifar10.yaml"])
@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.SimpleNN')
@patch('src.data.mnist_data.load_mnist_data')
@patch('src.data.cifar10_data.load_cifar10_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
@patch('src.main.open', create=True)
@patch('src.main.torch.device')
def test_main_function(mock_torch_device, mock_open_file, mock_evaluate_model, mock_train, mock_load_cifar10_data, mock_load_mnist_data, mock_simple_nn, mock_parse_args, config_file):
    # Mock the device to always return cpu for testing
    mock_torch_device.return_value = torch.device("cpu")

    # Mock load_mnist_data to return mock loaders
    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    if config_file == "config_cifar10.yaml":
        mock_load_cifar10_data.return_value = (mock_train_loader, mock_test_loader)
    else:
        mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)

    # Mock the model instance
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)] # Mock parameters to avoid empty list error

    # Mock train and evaluate_model return values
    mock_train.return_value = (0.1, torch.tensor(0.9, dtype=torch.float32), 0.9, 0.1, 0.8, 0.2, {}, {})
    mock_evaluate_model.return_value = (0.2, {"Accuracy": torch.tensor(0.8)}) # Mock loss and accuracy

    # Mock the config file
    config = {}
    if config_file == "config_sgd.yaml":
        config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "mnist", "batch_size": 32}, "metrics": [{"name": "Accuracy"}], "optimizer": {"name": "SGD", "config": {"lr": 0.01, "momentum": 0.9}}, "wandb": {"enabled": False}}
    elif config_file == "config_cifar10.yaml":
        config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "cifar10", "batch_size": 32}, "metrics": [{"name": "Accuracy"}], "optimizer": {"name": "Adam", "config": {"lr": 0.001}}, "wandb": {"enabled": False}}
    else:
        config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "mnist", "batch_size": 32}, "metrics": [{"name": "Accuracy"}], "optimizer": {"name": "Adam", "config": {"lr": 0.001}}, "wandb": {"enabled": False}}

    mock_parse_args.return_value = argparse.Namespace(config=config_file)
    mock_open_file.return_value = mock_open(read_data=yaml.dump(config)).return_value
    mock_torch_device.return_value = config["device"]
    from src.main import main
    # Check if wandb is enabled in the config
    wandb_config = config.get("wandb", {})
    with patch('src.engine.trainer.wandb_installed', new=False):
        with patch('src.main.wandb') as mock_wandb:
            main()

@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.SimpleNN')
@patch('src.data.mnist_data.load_mnist_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
@patch('src.main.open', create=True)
@patch('src.main.torch.device')
def test_main_function_sgd_optimizer(mock_torch_device, mock_open_file, mock_evaluate_model, mock_train, mock_load_mnist_data, mock_simple_nn, mock_parse_args):
    # Mock the device to always return cpu for testing
    config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "mnist", "batch_size": 32}, "metrics": [{"name": "Accuracy"}], "optimizer": {"name": "SGD", "config": {"lr": 0.01, "momentum": 0.9}}, "wandb": {"enabled": False}}

    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)]

    mock_train.return_value = (0.1, torch.tensor(0.9), 0, float('inf'), 0, float('inf'), {}, {})
    mock_evaluate_model.return_value = (0.2, {"Accuracy": torch.tensor(0.8)})

    # Test YAML config
    mock_parse_args.return_value = argparse.Namespace(config="config.yaml")
    mock_open_file.return_value = mock_open(read_data=yaml.dump(config)).return_value
    mock_torch_device.return_value = config["device"]
    from src.main import main
    # Check if wandb is enabled in the config
    wandb_config = config.get("wandb", {})
    with patch('src.engine.trainer.wandb_installed', new=False):
        with patch('src.main.wandb') as mock_wandb:
            main()

@patch('src.main.argparse.ArgumentParser.parse_args')
@patch('src.main.SimpleNN')
@patch('src.data.mnist_data.load_mnist_data')
@patch('src.engine.trainer.train')
@patch('src.engine.trainer.evaluate_model')
@patch('src.main.open', create=True)
@patch('src.main.torch.device')
def test_main_function_rmsprop_optimizer(mock_torch_device, mock_open_file, mock_evaluate_model, mock_train, mock_load_mnist_data, mock_simple_nn, mock_parse_args):
    # Mock the device to always return cpu for testing
    config = {"device": "cpu", "transformations": [{"name": "ToTensor"}], "learning_rate": 0.001, "epochs": 1, "dataset": {"name": "mnist", "batch_size": 32}, "metrics": [{"name": "Accuracy"}], "optimizer": {"name": "RMSprop", "config": {"lr": 0.01, "alpha": 0.99}}, "wandb": {"enabled": False}}

    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_load_mnist_data.return_value = (mock_train_loader, mock_test_loader)
    mock_model_instance = MagicMock()
    mock_simple_nn.return_value.to.return_value = mock_model_instance
    mock_model_instance.parameters.return_value = [torch.randn(10, 10)]

    mock_train.return_value = (0.1, torch.tensor(0.9), 0, float('inf'), 0, float('inf'), {}, {})
    mock_evaluate_model.return_value = (0.2, {"Accuracy": torch.tensor(0.8)})

    # Test YAML config
    mock_parse_args.return_value = argparse.Namespace(config="config.yaml")
    mock_open_file.return_value = mock_open(read_data=yaml.dump(config)).return_value
    mock_torch_device.return_value = config["device"]
    from src.main import main
    # Check if wandb is enabled in the config
    wandb_config = config.get("wandb", {})
    with patch('src.engine.trainer.wandb_installed', new=False):
        with patch('src.main.wandb') as mock_wandb:
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
def test_cached_dataset_len():
    from src.data.cifar10_data import CachedDataset
    from torchvision import datasets, transforms
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    cached_dataset = CachedDataset(dataset)
    assert len(cached_dataset) == len(dataset)

def test_cached_dataset_getitem():
    from src.data.cifar10_data import CachedDataset
    from torchvision import datasets, transforms
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    cached_dataset = CachedDataset(dataset)
    image, target = cached_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, int)

def test_cached_dataset_transform():
    from src.data.cifar10_data import CachedDataset
    from torchvision import datasets, transforms
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    cached_dataset = CachedDataset(dataset, transform=transform)
    image, target = cached_dataset[0]
    assert torch.is_tensor(image)
    assert image.mean() >= -1.0 and image.mean() <= 1.0 # Check if normalization is applied

def test_cached_dataset_caching():
    from src.data.cifar10_data import CachedDataset
    from torchvision import datasets, transforms
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    cached_dataset = CachedDataset(dataset)
    image1, target1 = cached_dataset[0]
    image2, target2 = cached_dataset[0]
    assert image1 is image2 # Check if the same object is returned

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
@patch('src.engine.trainer.wandb')
def test_train_function_wandb_error(mock_wandb):
    # Mock wandb.log to raise an exception
    mock_wandb.log.side_effect = Exception("Wandb error")

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
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, _, _ = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics, wandb=mock_wandb
    )

    # Assert that the function still returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
    assert train_accuracy.item() >= 0.0

@patch('src.engine.trainer.wandb')
def test_train_function_no_accuracy(mock_wandb):
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
    metrics = {} # No accuracy metric

    # Call the train function
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, _, _ = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics, wandb=mock_wandb
    )

    # Assert that the function still returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
    assert train_accuracy.item() >= 0.0

@patch('src.engine.trainer.wandb')
def test_evaluate_model_function_no_accuracy(mock_wandb):
    # Create mock data and model
    model = nn.Linear(10, 10)
    device = torch.device("cpu")
    test_loader = [(torch.randn(1, 10), torch.randint(0, 10, (1,)))]
    criterion = nn.CrossEntropyLoss()
    metrics = {} # No accuracy metric

    # Call the evaluate_model function
    test_loss, accuracy = evaluate_model(model, device, test_loader, criterion, metrics)

    # Assert that the function returns the expected values
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, dict)

@patch('src.engine.trainer.wandb')
def test_evaluate_model_function(mock_wandb):
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
    train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, _, _ = train(
        model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics=metrics
    )

    # Assert that the function returns the expected values
    assert isinstance(train_loss, float)
    assert isinstance(train_accuracy, torch.Tensor)
    assert train_accuracy.item() >= 0.0

def test_simple_nn_forward_pass():
    # Create an instance of the SimpleNN model
    model = SimpleNN(input_size=784, num_classes=10)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 28, 28)

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Assert that the output has the correct shape
    assert output.shape == (1, 10)

def test_simple_nn_init():
    # Create an instance of the SimpleNN model
    model = SimpleNN(input_size=784, num_classes=10)
    assert model.fc1.in_features == 784
    assert model.fc2.out_features == 10

def test_get_transformation_defaults():
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

def test_get_transformation_resize():
    transform_config = {"name": "Resize", "size": (64, 64)}
    transform = get_transformation(transform_config)
    assert isinstance(transform, transforms.Resize)
    assert transform.size == (64, 64)

def test_get_transformation_invalid():
    transform_config = {"name": "Invalid"}
    with pytest.raises(ValueError, match="Invalid transformation: Invalid"):
        get_transformation(transform_config)

@patch('src.data.__init__.importlib')
def test_get_dataset_import_error(mock_importlib):
    mock_importlib.import_module.side_effect = ImportError
    with pytest.raises(ValueError, match="Invalid dataset name: invalid_dataset"):
        get_dataset("invalid_dataset")

@patch('src.data.cifar10_data.datasets.CIFAR10')
def test_cifar10_data_empty(mock_cifar10):
    mock_cifar10.return_value = MagicMock()
    mock_cifar10.return_value.__len__.return_value = 0
    from src.data.cifar10_data import load_cifar10_data
    with pytest.raises(ValueError) as excinfo:
        load_cifar10_data()
    assert str(excinfo.value) == "num_samples should be a positive integer value, but got num_samples=0"

@patch('src.engine.optimizers.adam_optimizer.optim.Adam')
def test_adam_optimizer_defaults(mock_adam):
    # Create a mock model
    model = nn.Linear(10, 10)
    # Create an AdamOptimizer instance
    optimizer = adam_optimizer.AdamOptimizer(model.parameters(), {}).get_optimizer()
    # Assert that the optimizer is an Adam optimizer
    assert isinstance(optimizer, MagicMock)

@patch('src.engine.optimizers.adam_optimizer.optim.Adam')
def test_adam_optimizer_config(mock_adam):
    # Create a mock model
    model = nn.Linear(10, 10)
    # Create an AdamOptimizer instance
    config = {"lr": 0.01, "betas": (0.8, 0.99), "eps": 1e-7, "weight_decay": 0.1, "amsgrad": True}
    optimizer = adam_optimizer.AdamOptimizer(model.parameters(), config).get_optimizer()
    # Assert that the optimizer is an Adam optimizer
    args, kwargs = mock_adam.call_args
    assert list(args[0]) == [p for p in model.parameters()]
    assert kwargs == {'lr': 0.01, 'betas': (0.8, 0.99), 'eps': 1e-07, 'weight_decay': 0.1, 'amsgrad': True}

def test_cached_dataset_mnist():
    from src.data.mnist_data import CachedDataset
    from torchvision import datasets, transforms
    
    # Create a dummy dataset
    dummy_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    
    # Create a CachedDataset
    cached_dataset = CachedDataset(dummy_dataset)
    
    # Access the same element twice
    image1, target1 = cached_dataset[0]
    image2, target2 = cached_dataset[0]
    
    # Check if the element is cached
    assert 0 in cached_dataset.cache
    
    # Check if the returned elements are the same
    assert torch.equal(image1, image2)
    assert target1 == target2
