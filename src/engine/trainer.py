import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm
import time
import yaml
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
from src.engine import losses

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, num_classes=10):
    model.train()
    running_loss = 0.0
    
    # Initialize metrics
    metrics = {}
    for metric_config in config['metrics']:
        metric_name = metric_config['name']
        if metric_name == "Accuracy":
            metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        elif metric_name == "F1":
            metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(device)
        elif metric_name == "ConfusionMatrix":
            metrics[metric_name] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
        elif metric_name == "Precision":
            metrics[metric_name] = torchmetrics.Precision(task="multiclass", num_classes=num_classes).to(device)
        elif metric_name == "Recall":
            metrics[metric_name] = torchmetrics.Recall(task="multiclass", num_classes=num_classes).to(device)
        # Add other metrics here
        
    start_time = time.time()

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}", leave=False) as t:
        for batch_idx, (data, target) in enumerate(t):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update metrics
            for metric_name, metric in metrics.items():
                metric(output, target)
                
            t.set_postfix(loss=running_loss/(batch_idx+1))

    train_time = time.time() - start_time
    if len(train_loader) == 0:
        avg_train_loss = 0.0
    else:
        avg_train_loss = running_loss / len(train_loader)

    # Compute metrics
    metric_results = {}
    for metric_name, metric in metrics.items():
        if 'batch_idx' not in locals():
            metric_results[metric_name] = torch.tensor(0.0)
        else:
            metric_results[metric_name] = metric.compute()

    # Evaluate on test set
    start_time = time.time()
    test_loss, test_metric_results = evaluate_model(model, device, test_loader, criterion, metrics, num_classes=num_classes)
    test_time = time.time() - start_time

    # Update best metrics
    if metric_results.get("Accuracy", torch.tensor(0.0)) > best_train_accuracy:
        best_train_accuracy = metric_results.get("Accuracy", torch.tensor(0.0))
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
    if test_metric_results.get("Accuracy", torch.tensor(0.0)) > best_test_accuracy:
        best_test_accuracy = test_metric_results.get("Accuracy", torch.tensor(0.0))
    if test_loss < best_test_loss:
        best_test_loss = test_loss

    # Print epoch summary
    print_string = f'{epoch:>3} - ({train_time:>6.2f},{test_time:>6.2f})'
    for metric_name, value in metric_results.items():
        print_string += f' - training {metric_name} {value*100:>5.2f}'
    for metric_name, value in test_metric_results.items():
        print_string += f' - test {metric_name} {value*100:>5.2f}'
    print_string += f' - training loss {avg_train_loss:>6.4f} ({best_train_loss:>6.4f}) - test loss {test_loss:>6.4f} ({best_test_loss:>6.4f})'
    print(print_string)

    return avg_train_loss, metric_results.get("Accuracy", torch.tensor(0.0)), best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss

# Testing function
def evaluate_model(model, device, test_loader, criterion, metrics, num_classes=10):
    model.eval()
    test_loss = 0
    
    # Initialize metrics
    #metrics = {}
    #for metric_config in config['metrics']:
    #    metric_name = metric_config['name']
    #    metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", desc=f"Testing", leave=False) as t:
            for data, target in t:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                # Update metrics
                for metric_name, metric in metrics.items():
                    metric(output, target)
                    
                t.set_postfix(loss=test_loss/len(test_loader))

    if len(test_loader) == 0:
        test_loss = 0.0
        metric_results = {}
        for metric_name in metrics.keys():
            metric_results[metric_name] = torch.tensor(0.0) # Assuming 0 accuracy for empty data
    else:
        test_loss /= len(test_loader)
        metric_results = {}
        for metric_name, metric in metrics.items():
            metric_results[metric_name] = metric.compute()
    return test_loss, metric_results