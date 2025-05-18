import pytest
import torch
import torchmetrics
from src.engine.metrics import initialize_metrics, compute_metrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassAveragePrecision, MulticlassConfusionMatrix

# Mock device for testing
device = torch.device("cpu")

def test_initialize_metrics_empty_config():
    config = {}
    num_classes = 10
    metrics = initialize_metrics(config, device, num_classes)
    assert metrics == {}

def test_initialize_metrics_with_accuracy():
    config = {"metrics": [{"name": "Accuracy"}]}
    num_classes = 10
    metrics = initialize_metrics(config, device, num_classes)
    assert "Accuracy" in metrics
    assert isinstance(metrics["Accuracy"], MulticlassAccuracy)

def test_initialize_metrics_with_multiple_metrics():
    config = {"metrics": [{"name": "Accuracy"}, {"name": "F1"}, {"name": "Precision"}, {"name": "Recall"}, {"name": "AUC"}, {"name": "AveragePrecision"}, {"name": "ConfusionMatrix"}]}
    num_classes = 10
    metrics = initialize_metrics(config, device, num_classes)
    assert "Accuracy" in metrics
    assert "F1" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "AUC" in metrics
    assert "AveragePrecision" in metrics
    assert "ConfusionMatrix" in metrics
    assert isinstance(metrics["Accuracy"], MulticlassAccuracy)
    assert isinstance(metrics["F1"], MulticlassF1Score)
    assert isinstance(metrics["Precision"], MulticlassPrecision)
    assert isinstance(metrics["Recall"], MulticlassRecall)
    assert isinstance(metrics["AUC"], MulticlassAUROC)
    assert isinstance(metrics["AveragePrecision"], MulticlassAveragePrecision)
    assert isinstance(metrics["ConfusionMatrix"], MulticlassConfusionMatrix)


def test_initialize_metrics_unsupported_metric():
    config = {"metrics": [{"name": "UnsupportedMetric"}]}
    num_classes = 10
    metrics = initialize_metrics(config, device, num_classes)
    assert "UnsupportedMetric" not in metrics
    assert metrics == {}

def test_compute_metrics_empty_input():
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)}
    all_preds = []
    all_targets = []
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)
    assert "Accuracy" in metric_results
    assert torch.isclose(metric_results["Accuracy"], torch.tensor(0.0))

def test_compute_metrics_with_accuracy():
    metrics = {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)}
    all_preds = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]]
    all_targets = [1, 0, 2]
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)
    assert "Accuracy" in metric_results
    assert torch.isclose(metric_results["Accuracy"], torch.tensor(1.0))

def test_compute_metrics_with_f1_score():
    metrics = {"F1": torchmetrics.F1Score(task="multiclass", num_classes=3, average="macro").to(device)}
    all_preds = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]]
    all_targets = [1, 0, 2]
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)
    assert "F1" in metric_results
    # Note: The exact F1 score depends on the torchmetrics implementation,
    # but we can check if it's a tensor and not zero for a basic check.
    assert isinstance(metric_results["F1"], torch.Tensor)
    assert metric_results["F1"] > 0.0

def test_compute_metrics_with_auc():
    metrics = {"AUC": torchmetrics.AUROC(task="multiclass", num_classes=3).to(device)}
    all_preds = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]]
    all_targets = [1, 0, 2]
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)
    assert "AUC" in metric_results
    # Note: The exact AUC depends on the torchmetrics implementation,
    # but we can check if it's a tensor and not zero for a basic check.
    assert isinstance(metric_results["AUC"], torch.Tensor)
    assert metric_results["AUC"] > 0.0

def test_compute_metrics_with_confusion_matrix_skipped():
    metrics = {"ConfusionMatrix": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)}
    all_preds = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]]
    all_targets = [1, 0, 2]
    metric_results = compute_metrics(metrics, all_preds, all_targets, device)
    assert "ConfusionMatrix" not in metric_results # Confusion matrix is skipped in compute_metrics