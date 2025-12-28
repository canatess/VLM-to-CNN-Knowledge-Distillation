"""
Main training script for CUB-200-2011 Knowledge Distillation.

This script supports multiple training modes:
1. Standard training (from scratch or pretrained)
2. Logit-based knowledge distillation
3. Attention-based knowledge distillation
4. Combined distillation (logit + attention)

Usage:
    # Train with default config
    python scripts/train.py

    # Train with specific config
    python scripts/train.py --config configs/logit_kd.yaml

    # Override specific parameters
    python scripts/train.py --config configs/base.yaml --student_architecture mobilenetv3_small --num_epochs 30

    # Train from scratch (no pretrained weights)
    python scripts/train.py --pretrained false --distillation_type none
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.utils import Config, set_seed, ensure_dir, get_device
from src.data import build_dataloaders, load_class_names
from src.models import CLIPTeacher, StudentCNN
from src.distillation import KnowledgeDistiller
from src.training import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CUB-200-2011 Knowledge Distillation")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    
    # Experiment
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Data
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    
    # Model
    parser.add_argument("--student_architecture", type=str, default=None)
    parser.add_argument("--pretrained", type=lambda x: x.lower() == 'true', default=None)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    
    # Distillation
    parser.add_argument("--distillation_type", type=str, default=None,
                        choices=["none", "logit", "attention", "combined"])
    parser.add_argument("--alpha_kd", type=float, default=None)
    parser.add_argument("--alpha_attention", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    
    # Device
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def build_optimizer(model, config: Config):
    """Build optimizer from config."""
    if config.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(optimizer, config: Config, num_steps: int):
    """Build learning rate scheduler from config."""
    if config.scheduler is None or config.scheduler.lower() == "none":
        return None
    elif config.scheduler.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler.lower() == "step":
        return StepLR(optimizer, step_size=config.num_epochs // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load base configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(config, key, value)
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    config.device = device
    
    print("\n" + "="*80)
    print("CUB-200-2011 Knowledge Distillation Training")
    print("="*80)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Student Architecture: {config.student_architecture}")
    print(f"Distillation Type: {config.distillation_type}")
    print(f"Device: {device}")
    print(f"Seed: {config.seed}\n")
    
    # Create output directory
    output_dir = ensure_dir(config.output_dir) / config.experiment_name
    ensure_dir(output_dir)
    
    # Save configuration
    config.save(str(output_dir / "config.yaml"))
    print(f"Configuration saved to: {output_dir / 'config.yaml'}\n")
    
    # Load data
    print("Loading CUB-200-2011 dataset...")
    if config.val_ratio > 0:
        train_loader, val_loader, test_loader = build_dataloaders(
            root=config.data_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_ratio=config.val_ratio,
            seed=config.seed
        )
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val:   {len(val_loader.dataset)} samples")
        print(f"  Test:  {len(test_loader.dataset)} samples\n")
    else:
        train_loader, test_loader = build_dataloaders(
            root=config.data_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_ratio=0.0
        )
        val_loader = None
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Test:  {len(test_loader.dataset)} samples\n")
    
    # Build student model
    print(f"Building student model: {config.student_architecture}")
    need_features = config.distillation_type in ["attention", "combined"]
    student = StudentCNN(
        architecture=config.student_architecture,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        extract_features=need_features
    )
    
    param_info = student.get_num_parameters()
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  Trainable parameters: {param_info['trainable']:,}\n")
    
    # Build optimizer
    optimizer = build_optimizer(student, config)
    print(f"Optimizer: {config.optimizer.upper()}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}\n")
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Setup distillation if needed
    distiller = None
    if config.distillation_type != "none":
        print("Setting up knowledge distillation...")
        print(f"  Teacher: {config.teacher_model}")
        print(f"  Distillation type: {config.distillation_type}")
        
        # Load teacher model
        class_names = load_class_names(config.data_root)
        teacher = CLIPTeacher(
            model_name=config.teacher_model,
            class_names=class_names,
            extract_attention=(config.distillation_type in ["attention", "combined"])
        )
        teacher.to(device)
        
        # Create distiller
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=student,
            distillation_type=config.distillation_type,
            alpha_ce=config.alpha_ce,
            alpha_kd=config.alpha_kd,
            alpha_attention=config.alpha_attention,
            temperature=config.temperature,
            attention_loss_type=config.attention_loss_type,
            attention_match_to=config.attention_match_to
        )
        
        print(f"  Alpha CE: {config.alpha_ce}")
        print(f"  Alpha KD: {config.alpha_kd}")
        if config.alpha_attention > 0:
            print(f"  Alpha Attention: {config.alpha_attention}")
            print(f"  Attention loss: {config.attention_loss_type}")
        print()
    
    # Create trainer
    trainer = Trainer(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        distiller=distiller,
        use_amp=config.use_amp,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval
    )
    
    # Train
    save_path = str(output_dir / "best_model.pth")
    history = trainer.train(
        num_epochs=config.num_epochs,
        save_path=save_path,
        save_best_only=config.save_best_only
    )
    
    # Save final results
    import json
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Training completed successfully!\n")


if __name__ == "__main__":
    main()
