from typing import Literal, Optional, Dict, Any
import torch
import torch.nn as nn

from .losses import combined_loss
from .attention import match_attention_resolution


class KnowledgeDistiller:
    """
    High-level interface for knowledge distillation training.
    
    Supports multiple distillation strategies:
    - Logit-based distillation
    - Attention-based distillation
    - Combined distillation
    
    Args:
        teacher: Teacher model
        student: Student model
        distillation_type: Type of distillation ('logit', 'attention', 'combined')
        alpha_ce: Weight for cross-entropy loss
        alpha_kd: Weight for logit distillation loss
        alpha_attention: Weight for attention distillation loss
        temperature: Temperature for softmax in logit distillation
        attention_loss_type: Type of attention loss ('mse', 'l1', 'kl')
        attention_match_to: Match attention resolution to ('teacher' or 'student')
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        distillation_type: Literal["logit", "attention", "combined"] = "combined",
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        alpha_attention: float = 0.1,
        temperature: float = 4.0,
        attention_loss_type: str = "mse",
        attention_match_to: str = "teacher",
    ):
        self.teacher = teacher
        self.student = student
        self.distillation_type = distillation_type
        
        # Loss weights
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_attention = alpha_attention
        
        # Hyperparameters
        self.temperature = temperature
        self.attention_loss_type = attention_loss_type
        self.attention_match_to = attention_match_to
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def get_teacher_outputs(
        self,
        images: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
       
        self.teacher.eval()
        
        outputs = {}
        
        if return_attention and hasattr(self.teacher, 'get_attention_rollout'):
            # Get attention rollout from teacher
            try:
                attention = self.teacher.get_attention_rollout(images)
                outputs['attention'] = attention
            except:
                pass
        
        # Get logits
        if hasattr(self.teacher, 'get_logits'):
            logits = self.teacher.get_logits(images)
        else:
            logits = self.teacher(images)
        
        outputs['logits'] = logits
        
        return outputs
    
    def get_student_outputs(
        self,
        images: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        if return_attention and hasattr(self.student, 'get_attention_map'):
            # Get attention from student CNN features
            try:
                attention = self.student.get_attention_map(images)
                outputs['attention'] = attention
            except:
                pass
        
        # Get logits
        logits = self.student(images)
        outputs['logits'] = logits
        
        return outputs
    
    def compute_loss(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        # Determine if we need attention
        need_attention = (
            self.distillation_type in ["attention", "combined"] and
            self.alpha_attention > 0
        )
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.get_teacher_outputs(images, return_attention=need_attention)
        
        # Get student outputs
        student_outputs = self.get_student_outputs(images, return_attention=need_attention)
        
        # Extract components
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        student_attention = student_outputs.get('attention', None)
        teacher_attention = teacher_outputs.get('attention', None)
        
        # Match attention resolutions if both are available
        if student_attention is not None and teacher_attention is not None:
            student_attention, teacher_attention = match_attention_resolution(
                student_attention,
                teacher_attention,
                match_to=self.attention_match_to
            )
        
        # Compute combined loss
        losses = combined_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            alpha_ce=self.alpha_ce,
            alpha_kd=self.alpha_kd,
            temperature=self.temperature,
            student_attention=student_attention,
            teacher_attention=teacher_attention,
            alpha_attention=self.alpha_attention,
            attention_loss_type=self.attention_loss_type,
        )
        
        return losses
    
    def train_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
    ) -> Dict[str, float]:
       
        self.student.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                losses = self.compute_loss(images, targets)
            
            # Backward pass
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            losses = self.compute_loss(images, targets)
            losses['total'].backward()
            optimizer.step()
        
        # Convert losses to float for logging
        return {k: v.item() for k, v in losses.items()}
