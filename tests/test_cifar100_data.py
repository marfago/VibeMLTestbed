import pytest
import torch
from torchvision import datasets, transforms
from src.data.cifar100_data import CachedDataset

def test_cached_dataset():
    # Create a dummy dataset
    dummy_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    
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