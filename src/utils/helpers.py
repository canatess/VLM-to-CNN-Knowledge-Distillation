"""Helper utility functions."""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device: Optional[str] = None) -> str:
    """
    Get appropriate device for PyTorch.
    
    Args:
        device: Requested device ('cuda', 'cpu', or None for auto)
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        return "cpu"
    
    return device


def count_files(directory: str, extension: str = None) -> int:
    """
    Count files in directory.
    
    Args:
        directory: Directory path
        extension: File extension to filter (e.g., '.jpg')
    
    Returns:
        Number of files
    """
    path = Path(directory)
    if not path.exists():
        return 0
    
    if extension:
        return len(list(path.glob(f"*{extension}")))
    else:
        return len(list(path.iterdir()))


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(size_bytes: int) -> str:
    """
    Format bytes into human-readable size string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_config(config: dict, indent: int = 0):
    """
    Pretty print configuration dictionary.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils/
    return Path(__file__).parent.parent.parent


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of experiment
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir
