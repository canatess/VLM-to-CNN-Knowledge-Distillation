from typing import Dict, List, Optional
import numpy as np


class AverageMeter:

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get(self, name: str) -> List[float]:
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        history = self.get(name)
        return history[-1] if history else None
    
    def get_best(self, name: str, mode: str = "max") -> Optional[float]:
        history = self.get(name)
        if not history:
            return None
        
        return max(history) if mode == "max" else min(history)
    
    def get_all(self) -> Dict[str, List[float]]:
        return self.metrics
    
    def summary(self) -> Dict[str, Dict[str, float]]:
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

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.step_times = []
    
    def start(self):
        import time
        self.start_time = time.time()
    
    def step(self):
        import time
        if self.start_time is None:
            self.start()
        
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        self.step_times.append(elapsed / self.current_step)
    
    def get_eta(self) -> float:
        if not self.step_times:
            return 0
        
        avg_step_time = np.mean(self.step_times[-100:])
        remaining_steps = self.total_steps - self.current_step
        return avg_step_time * remaining_steps
    
    def get_progress(self) -> float:
        return 100.0 * self.current_step / self.total_steps
