import torch
import torchmetrics

def initialize_metrics(config, device, num_classes):
    metrics = {}
    if "metrics" in config:
        for metric_config in config['metrics']:
            metric_name = metric_config['name']
            if metric_name == "Accuracy":
                metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
            elif metric_name == "F1":
                metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
            elif metric_name == "ConfusionMatrix":
                metrics[metric_name] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
            elif metric_name == "Precision":
                metrics[metric_name] = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
            elif metric_name == "Recall":
                metrics[metric_name] = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
            elif metric_name == "AUC":
                metrics[metric_name] = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device)
            elif metric_name == "ROC":
                continue # Not implemented
            elif metric_name == "AveragePrecision":
                metrics[metric_name] = torchmetrics.AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
            # Add other metrics here
    return metrics

def compute_metrics(metrics, all_preds, all_targets, device):
    metric_results = {}
    if len(all_preds) > 0:
        preds_tensor = torch.tensor(all_preds).to(device)
        targets_tensor = torch.tensor(all_targets).to(device).long()
        
        # Create separate tensors for different metric types
        preds_class_tensor = preds_tensor.argmax(dim=1)
        
        for metric_name, metric in metrics.items():
            if metric_name in ["AUC", "ROC", "AveragePrecision"]:
                metric.update(preds_tensor, targets_tensor)
            else:
                metric.update(preds_class_tensor, targets_tensor)
            metric_results[metric_name] = metric.compute()
            metric.reset() # Reset metric after computing
    else:
        for metric_name in metrics.keys():
            metric_results[metric_name] = torch.tensor(0.0)
    return metric_results