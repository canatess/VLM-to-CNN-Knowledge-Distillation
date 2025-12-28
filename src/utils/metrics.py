"""Metrics tracking utilities."""

from typing import Dict, List, Optional
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """
    Track multiple metrics over training.
    
    Stores history of metrics and provides easy access.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get(self, name: str) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            name: Metric name
        
        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """
        Get latest value of a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Latest value or None
        """
        history = self.get(name)
        return history[-1] if history else None
    
    def get_best(self, name: str, mode: str = "max") -> Optional[float]:
        """
        Get best value of a metric.
        
        Args:
            name: Metric name
            mode: 'max' or 'min'
        
        Returns:
            Best value or None
        """
        history = self.get(name)
        if not history:
            return None
        
        return max(history) if mode == "max" else min(history)
    
    def get_all(self) -> Dict[str, List[float]]:
        """
        Get all metrics history.
        
        Returns:
            Dictionary of all metrics
        """
        return self.metrics
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        for name, history in self.metrics.items():
            if history:
                summary[name] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "latest": history[-1]
                }
        return summary
    
    def __str__(self):
        lines = ["Metrics Summary:"]
        for name, stats in self.summary().items():
            lines.append(
                f"  {name}: mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, "
                f"best={stats['max']:.4f}"
            )
        return "\n".join(lines)


class ProgressTracker:
    """
    Track training progress and estimate remaining time.
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.step_times = []
    
    def start(self):
        """Start tracking time."""
        import time
        self.start_time = time.time()
    
    def step(self):
        """Record a step."""
        import time
        if self.start_time is None:
            self.start()
        
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        self.step_times.append(elapsed / self.current_step)
    
    def get_eta(self) -> float:
        """
        Get estimated time remaining in seconds.
        
        Returns:
            Estimated seconds remaining
        """
        if not self.step_times:
            return 0
        
        avg_step_time = np.mean(self.step_times[-100:])  # Use last 100 steps
        remaining_steps = self.total_steps - self.current_step
        return avg_step_time * remaining_steps
    
    def get_progress(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        return 100.0 * self.current_step / self.total_steps
