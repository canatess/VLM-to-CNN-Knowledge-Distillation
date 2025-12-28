"""Model evaluation utilities."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    return_predictions: bool = False,
    compute_all_metrics: bool = True,
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        return_predictions: Whether to return all predictions and targets
        compute_all_metrics: Whether to compute F1-scores and top-5 accuracy
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import f1_score
    
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    top5_correct = 0
    
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(images)
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, targets, reduction='sum')
        
        # Get predictions
        predictions = logits.argmax(dim=1)
        
        # Update metrics
        total_loss += loss.item()
        correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        
        # Top-5 accuracy
        if compute_all_metrics:
            _, top5_preds = logits.topk(5, dim=1, largest=True, sorted=True)
            targets_expanded = targets.view(-1, 1).expand_as(top5_preds)
            top5_correct += (top5_preds == targets_expanded).sum().item()
        
        # Store for F1-score and other metrics
        if return_predictions or compute_all_metrics:
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    
    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total_samples
    }
    
    # Add top-5 accuracy
    if compute_all_metrics:
        top5_accuracy = 100.0 * top5_correct / total_samples
        results["top5_accuracy"] = top5_accuracy
    
    # Compute F1-scores
    if compute_all_metrics and all_predictions:
        predictions_np = torch.cat(all_predictions).numpy()
        targets_np = torch.cat(all_targets).numpy()
        
        # Compute F1-scores (macro, weighted, micro)
        results["f1_macro"] = f1_score(targets_np, predictions_np, average='macro') * 100
        results["f1_weighted"] = f1_score(targets_np, predictions_np, average='weighted') * 100
        results["f1_micro"] = f1_score(targets_np, predictions_np, average='micro') * 100
    
    if return_predictions:
        if all_predictions:
            results["predictions"] = torch.cat(all_predictions).numpy()
            results["targets"] = torch.cat(all_targets).numpy()
    
    return results


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 200
) -> Dict[str, float]:
    """
    Compute additional classification metrics.
    
    Args:
        predictions: Predicted labels [N]
        targets: Ground truth labels [N]
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics (accuracy, per-class accuracy, etc.)
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Overall accuracy
    accuracy = accuracy_score(targets, predictions) * 100
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    mean_class_acc = per_class_acc.mean() * 100
    
    # Top-5 accuracy (not straightforward without logits, skip for now)
    
    metrics = {
        "accuracy": accuracy,
        "mean_class_accuracy": mean_class_acc,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc
    }
    
    return metrics


@torch.no_grad()
def compute_top_k_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run on
        k: Top-k value
    
    Returns:
        Top-k accuracy as percentage
    """
    model.eval()
    
    correct = 0
    total = 0
    
    for images, targets in tqdm(data_loader, desc=f"Computing Top-{k}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(images)
        
        # Get top-k predictions
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        
        # Check if target is in top-k
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct += (top_k_preds == targets_expanded).sum().item()
        total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cuda",
    warmup_runs: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: Model to measure
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        warmup_runs: Number of warmup runs
        num_runs: Number of measurement runs
    
    Returns:
        Dictionary with timing statistics (ms)
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    
    # Synchronize before measurement
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    import time
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "median_ms": np.median(times)
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size in MB.
    
    Args:
        model: Model to analyze
    
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
