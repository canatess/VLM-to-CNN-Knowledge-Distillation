import torch
import torch.nn.functional as F
from typing import Optional, Literal


def logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Compute KL divergence loss for logit-based knowledge distillation.
    
    This is the standard knowledge distillation loss from Hinton et al.
    
    Args:
        student_logits: Student model logits [B, num_classes]
        teacher_logits: Teacher model logits [B, num_classes]
        temperature: Temperature for softening probability distributions
    
    Returns:
        Distillation loss (scalar)
    """
    # Soften the distributions with temperature
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Compute KL divergence
    loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
    
    # Scale by temperature squared to maintain gradient magnitude
    return loss * (temperature ** 2)


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    return F.cross_entropy(logits, targets)


def feature_matching_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    MSE loss between student and teacher feature embeddings.
    
    Args:
        student_features: Student feature embeddings [B, D]
        teacher_features: Teacher feature embeddings [B, D]
        normalize: Whether to L2-normalize features before computing loss
    
    Returns:
        Feature matching loss (scalar)
    """
    if normalize:
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
    
    return F.mse_loss(student_features, teacher_features)


def attention_distillation_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    loss_type: Literal["mse", "l1", "kl"] = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute loss between student and teacher attention maps.
    
    Args:
        student_attention: Student attention map [B, 1, H, W]
        teacher_attention: Teacher attention map [B, 1, H, W]
        loss_type: Type of loss to use ('mse', 'l1', or 'kl')
        eps: Small constant for numerical stability
    
    Returns:
        Attention distillation loss (scalar)
    """
    if student_attention.shape != teacher_attention.shape:
        raise ValueError(
            f"Shape mismatch: student {student_attention.shape} "
            f"vs teacher {teacher_attention.shape}"
        )
    
    if loss_type == "mse":
        # Mean squared error
        return torch.mean((student_attention - teacher_attention) ** 2)
    
    elif loss_type == "l1":
        # L1 (MAE) loss
        return torch.mean(torch.abs(student_attention - teacher_attention))
    
    elif loss_type == "kl":
        # KL divergence treating attention maps as probability distributions
        # Flatten spatial dimensions
        student_flat = student_attention.flatten(1)  # [B, H*W]
        teacher_flat = teacher_attention.flatten(1)  # [B, H*W]
        
        # Normalize to probability distributions
        student_prob = student_flat / (student_flat.sum(dim=1, keepdim=True) + eps)
        teacher_prob = teacher_flat / (teacher_flat.sum(dim=1, keepdim=True) + eps)
        
        # Compute KL divergence
        return F.kl_div(
            (student_prob + eps).log(),
            teacher_prob,
            reduction="batchmean"
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_ce: float = 1.0,
    alpha_kd: float = 1.0,
    temperature: float = 4.0,
    student_attention: Optional[torch.Tensor] = None,
    teacher_attention: Optional[torch.Tensor] = None,
    alpha_attention: float = 0.0,
    attention_loss_type: str = "mse",
) -> dict:
    """
    Compute combined loss for knowledge distillation.
    
    Total loss = alpha_ce * CE + alpha_kd * KD + alpha_attention * Attention
    
    Args:
        student_logits: Student model logits [B, num_classes]
        teacher_logits: Teacher model logits [B, num_classes]
        targets: Ground truth labels [B]
        alpha_ce: Weight for cross-entropy loss
        alpha_kd: Weight for distillation loss
        temperature: Temperature for distillation
        student_attention: Student attention map [B, 1, H, W] (optional)
        teacher_attention: Teacher attention map [B, 1, H, W] (optional)
        alpha_attention: Weight for attention distillation loss
        attention_loss_type: Type of attention loss ('mse', 'l1', or 'kl')
    
    Returns:
        Dictionary containing:
            - 'total': Total weighted loss
            - 'ce': Cross-entropy loss
            - 'kd': Distillation loss
            - 'attention': Attention loss (if applicable)
    """
    losses = {}
    
    # Cross-entropy loss
    ce = cross_entropy_loss(student_logits, targets)
    losses['ce'] = ce
    
    # Distillation loss
    kd = logit_distillation_loss(student_logits, teacher_logits, temperature)
    losses['kd'] = kd
    
    # Total loss
    total = alpha_ce * ce + alpha_kd * kd
    
    # Attention distillation loss (if provided)
    if (student_attention is not None and 
        teacher_attention is not None and 
        alpha_attention > 0):
        attn_loss = attention_distillation_loss(
            student_attention,
            teacher_attention,
            loss_type=attention_loss_type
        )
        losses['attention'] = attn_loss
        total = total + alpha_attention * attn_loss
    
    losses['total'] = total
    
    return losses
