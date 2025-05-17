import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Dataset parameters
NUM_CLASSES = 10
INPUT_SIZE = 32 * 32 * 3

# Custom Dataset class to cache transformed samples
class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            image, target = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            sample = (image, target)
            self.cache[idx] = sample
            return sample

# Load CIFAR10 dataset
def load_cifar10_data(batch_size=64, transformations=None):
    if transformations is None:
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    transform = transforms.Compose(transformations)

    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)

    train_cached_dataset = CachedDataset(train_dataset, transform=transform)
    test_cached_dataset = CachedDataset(test_dataset, transform=transform)

    train_loader = DataLoader(train_cached_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_cached_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader