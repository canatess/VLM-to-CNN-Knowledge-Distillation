"""Training loop implementation."""

from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from ..distillation import KnowledgeDistiller
from .evaluator import evaluate_model


class Trainer:
    """
    Unified trainer for knowledge distillation experiments.
    
    Supports:
    - Standard supervised training (from scratch or pretrained)
    - Logit-based knowledge distillation
    - Attention-based knowledge distillation
    - Combined distillation
    
    Args:
        model: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        test_loader: Test data loader (optional)
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        device: Device to run training on
        distiller: KnowledgeDistiller instance (optional, for KD training)
        use_amp: Whether to use automatic mixed precision training
        log_interval: Steps between logging
        eval_interval: Epochs between evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        distiller: Optional[KnowledgeDistiller] = None,
        use_amp: bool = False,
        log_interval: int = 10,
        eval_interval: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.distiller = distiller
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # Initialize GradScaler for mixed precision if needed
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        # Loss components for KD
        loss_components = {
            "ce": 0.0,
            "kd": 0.0,
            "attention": 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            batch_size = images.size(0)
            
            if self.distiller is not None:
                # Knowledge distillation training
                losses = self.distiller.train_step(
                    images, targets, self.optimizer, self.scaler
                )
                loss = losses['total']
                
                # Accumulate loss components
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key] * batch_size
                
                # Get predictions for accuracy
                with torch.no_grad():
                    logits = self.model(images)
                    predictions = logits.argmax(dim=1)
            else:
                # Standard training
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = nn.functional.cross_entropy(logits, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss = nn.functional.cross_entropy(logits, targets)
                    loss.backward()
                    self.optimizer.step()
                
                predictions = logits.argmax(dim=1)
                loss = loss.item()
            
            # Update metrics
            total_loss += loss * batch_size
            correct += (predictions == targets).sum().item()
            total_samples += batch_size
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                current_acc = 100.0 * correct / total_samples
                pbar.set_postfix({
                    "loss": total_loss / total_samples,
                    "acc": f"{current_acc:.2f}%"
                })
            
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct / total_samples
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }
        
        # Add loss components if using KD
        if self.distiller is not None:
            for key, value in loss_components.items():
                if value > 0:
                    metrics[f"{key}_loss"] = value / total_samples
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        save_best_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model checkpoint
            save_best_only: Whether to save only the best model
        
        Returns:
            Dictionary with training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log training metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.2f}%")
            
            # Log KD components if available
            if "ce_loss" in train_metrics:
                print(f"  CE Loss:    {train_metrics['ce_loss']:.4f}")
            if "kd_loss" in train_metrics:
                print(f"  KD Loss:    {train_metrics['kd_loss']:.4f}")
            if "attention_loss" in train_metrics:
                print(f"  Attn Loss:  {train_metrics['attention_loss']:.4f}")
            
            # Save to history
            self.history["train_loss"].append(train_metrics['loss'])
            self.history["train_acc"].append(train_metrics['accuracy'])
            
            # Evaluate on validation set
            if self.val_loader and (epoch + 1) % self.eval_interval == 0:
                val_metrics = evaluate_model(
                    self.model,
                    self.val_loader,
                    self.device,
                    compute_all_metrics=True
                )
                print(f"  Val Loss:   {val_metrics['loss']:.4f}")
                print(f"  Val Acc:    {val_metrics['accuracy']:.2f}%")
                print(f"  Val Top-5:  {val_metrics.get('top5_accuracy', 0):.2f}%")
                print(f"  Val F1 (M): {val_metrics.get('f1_macro', 0):.2f}%")
                
                self.history["val_loss"].append(val_metrics['loss'])
                self.history["val_acc"].append(val_metrics['accuracy'])
                self.history.setdefault("val_top5_acc", []).append(val_metrics.get('top5_accuracy', 0))
                self.history.setdefault("val_f1_macro", []).append(val_metrics.get('f1_macro', 0))
                self.history.setdefault("val_f1_weighted", []).append(val_metrics.get('f1_weighted', 0))
                self.history.setdefault("val_f1_micro", []).append(val_metrics.get('f1_micro', 0))
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    if save_path and save_best_only:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'accuracy': val_metrics['accuracy'],
                            'top5_accuracy': val_metrics.get('top5_accuracy', 0),
                            'f1_macro': val_metrics.get('f1_macro', 0),
                            'history': self.history
                        }, save_path)
                        print(f"  Saved best model to {save_path}")
            
            # Evaluate on test set (final epoch)
            if self.test_loader and epoch == num_epochs - 1:
                test_metrics = evaluate_model(
                    self.model,
                    self.test_loader,
                    self.device,
                    compute_all_metrics=True
                )
                print(f"  Test Acc:   {test_metrics['accuracy']:.2f}%")
                print(f"  Test Top-5: {test_metrics.get('top5_accuracy', 0):.2f}%")
                print(f"  Test F1 (M):{test_metrics.get('f1_macro', 0):.2f}%")
                self.history["test_acc"].append(test_metrics['accuracy'])
                self.history.setdefault("test_top5_acc", []).append(test_metrics.get('top5_accuracy', 0))
                self.history.setdefault("test_f1_macro", []).append(test_metrics.get('f1_macro', 0))
                self.history.setdefault("test_f1_weighted", []).append(test_metrics.get('f1_weighted', 0))
                self.history.setdefault("test_f1_micro", []).append(test_metrics.get('f1_micro', 0))
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time:.2f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
