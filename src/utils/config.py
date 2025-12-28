"""Configuration management."""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    """
    Configuration class for experiments.
    
    All experimental settings are defined here for reproducibility.
    """
    # Experiment
    experiment_name: str = "cub_kd_experiment"
    seed: int = 42
    output_dir: str = "./outputs"
    
    # Data
    data_root: str = "CUB_200_2011"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_ratio: float = 0.1
    
    # Model
    teacher_model: str = "openai/clip-vit-base-patch32"
    student_architecture: str = "resnet18"
    num_classes: int = 200
    pretrained: bool = True
    
    # Training
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"  # 'adamw' or 'sgd'
    scheduler: Optional[str] = "cosine"  # 'cosine', 'step', or None
    
    # Distillation
    distillation_type: str = "combined"  # 'logit', 'attention', 'combined', or 'none'
    alpha_ce: float = 1.0  # Weight for cross-entropy loss
    alpha_kd: float = 1.0  # Weight for logit distillation
    alpha_attention: float = 0.1  # Weight for attention distillation
    temperature: float = 4.0  # Temperature for logit distillation
    attention_loss_type: str = "mse"  # 'mse', 'l1', or 'kl'
    attention_match_to: str = "teacher"  # 'teacher' or 'student'
    
    # Training settings
    use_amp: bool = False  # Mixed precision training
    log_interval: int = 10
    eval_interval: int = 1
    save_best_only: bool = True
    
    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


def load_config(path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file (.yaml or .json)
    
    Returns:
        Config object
    """
    return Config.load(path)


def save_config(config: Config, path: str):
    """
    Save configuration to file.
    
    Args:
        config: Config object
        path: Path to save configuration (.yaml or .json)
    """
    config.save(path)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge base configuration with overrides.
    
    Args:
        base_config: Base configuration
        override_dict: Dictionary with values to override
    
    Returns:
        New Config with merged values
    """
    config_dict = base_config.to_dict()
    config_dict.update(override_dict)
    return Config.from_dict(config_dict)
