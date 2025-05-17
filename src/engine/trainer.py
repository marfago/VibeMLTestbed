import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm
import time

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader):
    model.train()
    running_loss = 0.0
    train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    start_time = time.time()

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}") as t:
        for batch_idx, (data, target) in enumerate(t):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_metric(output, target)
            t.set_postfix(loss=running_loss/(batch_idx+1))

    train_time = time.time() - start_time
    print(f"Length of train_loader: {len(train_loader)}") # Debug print
    if len(train_loader) == 0:
        avg_train_loss = 0.0
    else:
        avg_train_loss = running_loss / len(train_loader)

    if 'batch_idx' not in locals():
        train_accuracy = torch.tensor(0.0)
    else:
        train_accuracy = train_metric.compute()

    # Evaluate on test set
    start_time = time.time()
    test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)
    test_time = time.time() - start_time

    # Update best metrics
    if train_accuracy > best_train_accuracy:
        best_train_accuracy = train_accuracy
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
    if test_loss < best_test_loss:
        best_test_loss = test_loss

    # Print epoch summary
    print(f'{epoch:>3} - ({train_time:>6.2f},{test_time:>6.2f}) - training accuracy {train_accuracy:>5.2f} ({best_train_accuracy:>5.2f}) - training loss {avg_train_loss:>6.4f} ({best_train_loss:>6.4f}) - test accuracy {test_accuracy:>5.2f} ({best_test_accuracy:>5.2f}) - test loss {test_loss:>6.4f} ({best_test_loss:>6.4f})')

    return avg_train_loss, train_accuracy, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss

# Testing function
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            test_metric(output, target)

    print(f"Length of test_loader: {len(test_loader)}") # Debug print
    if len(test_loader) == 0:
        test_loss = 0.0
        accuracy = torch.tensor(0.0) # Assuming 0 accuracy for empty data
    else:
        test_loss /= len(test_loader)
        accuracy = test_metric.compute()
    return test_loss, accuracy