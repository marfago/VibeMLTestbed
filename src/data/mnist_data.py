import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
def load_mnist_data(batch_size=64, transformations=None):
    if transformations is None:
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    transform = transforms.Compose(transformations)

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader