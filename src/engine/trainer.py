import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        metric(output, target)

    avg_loss = running_loss / len(train_loader)
    accuracy = metric.compute()
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Testing function
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            metric(output, target)

    test_loss /= len(test_loader)
    accuracy = metric.compute()
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return test_loss, accuracy