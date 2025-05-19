import torch
import torch.nn as nn

import torchmetrics
from tqdm import tqdm
import time
import yaml
try:
    import wandb
    wandb_installed = True
except ImportError:
    wandb_installed = False

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
from src.engine import losses
from src.engine.metrics import compute_metrics

# Training function
from src.engine import optimizers
def train(model, device, train_loader, optimizer, criterion, epoch, best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, test_loader, metrics, wandb=None, num_classes=10):
    model.train()
    running_loss = 0.0
    
    start_time = time.time()

    all_preds = []
    all_targets = []

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}", leave=False) as t:
        for batch_idx, (data, target) in enumerate(t):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Accumulate predictions and targets
            all_preds.extend(output.cpu().tolist())
            all_targets.extend(target.cpu().tolist())
                
            t.set_postfix(loss=running_loss/(batch_idx+1))

    train_time = time.time() - start_time
    if len(train_loader) == 0:
        avg_train_loss = 0.0
    else:
        avg_train_loss = running_loss / len(train_loader)

    # Compute metrics
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)

    # Evaluate on test set
    start_time = time.time()
    test_loss, test_metric_results = evaluate_model(model, device, test_loader, criterion, metrics, wandb=wandb, num_classes=num_classes)
    test_time = time.time() - start_time

    # Update best metrics
    if "Accuracy" in metric_results and isinstance(metric_results["Accuracy"], torch.Tensor) and metric_results["Accuracy"].numel() == 1 and metric_results["Accuracy"].item() > best_train_accuracy:
        best_train_accuracy = metric_results["Accuracy"].item()
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
    if "Accuracy" in test_metric_results and isinstance(test_metric_results["Accuracy"], torch.Tensor) and test_metric_results["Accuracy"].numel() == 1 and test_metric_results["Accuracy"].item() > best_test_accuracy:
        best_test_accuracy = test_metric_results["Accuracy"].item()
    if test_loss < best_test_loss:
        best_test_loss = test_loss

    # Print epoch summary
    train_accuracy_val = metric_results.get("Accuracy", torch.tensor(0.0)).item() if isinstance(metric_results.get("Accuracy", torch.tensor(0.0)), torch.Tensor) and metric_results.get("Accuracy", torch.tensor(0.0)).numel() == 1 else 0.0
    test_accuracy_val = test_metric_results.get("Accuracy", torch.tensor(0.0)).item() if isinstance(test_metric_results.get("Accuracy", torch.tensor(0.0)), torch.Tensor) and test_metric_results.get("Accuracy", torch.tensor(0.0)).numel() == 1 else 0.0

    print_string = f'{epoch:>3} - ({train_time:>6.2f},{test_time:>6.2f}) - training accuracy {train_accuracy_val*100:>5.2f} ({best_train_accuracy*100:>5.2f}) - training loss {avg_train_loss:>6.4f} ({best_train_loss:>6.4f}) - test accuracy {test_accuracy_val*100:>5.2f} ({best_test_accuracy*100:>5.2f}) - test loss {test_loss:>6.4f} ({best_test_loss:>6.4f})'
    print(print_string)

    if wandb is not None:
        try:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy_val,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy_val,
                "best_train_accuracy": best_train_accuracy,
                "best_train_loss": best_train_loss,
                "best_test_accuracy": best_test_accuracy,
                "best_test_loss": best_test_loss,
                **{f"train_{k}": v.item() if isinstance(v, torch.Tensor) and k != "ConfusionMatrix" else v for k, v in metric_results.items()},
                **{f"test_{k}": v.item() if isinstance(v, torch.Tensor) and k != "ConfusionMatrix" else v for k, v in test_metric_results.items()}
            })
        except Exception as e:
            print(f"Error logging to wandb: {e}")
    return avg_train_loss, metric_results.get("Accuracy", torch.tensor(0.0)), best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, metric_results, test_metric_results

# Testing function
def evaluate_model(model, device, test_loader, criterion, metrics, wandb=None, num_classes=10):
    model.eval()
    test_loss = 0
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", desc=f"Testing", leave=False) as t:
            for data, target in t:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                # Accumulate predictions and targets
                all_preds.extend(output.cpu().tolist())
                all_targets.extend(target.cpu().tolist())
                
            if len(test_loader) > 0:
                t.set_postfix(loss=test_loss/len(test_loader))
            else:
                t.set_postfix(loss=0)

    if len(test_loader) == 0:
        test_loss = 0.0
        metric_results = {}
    else:
        test_loss /= len(test_loader)
        metric_results = compute_metrics(metrics, all_preds, all_targets, device)

    return test_loss, metric_results