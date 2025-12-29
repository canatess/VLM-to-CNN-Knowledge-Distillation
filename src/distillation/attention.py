import torch
import torch.nn.functional as F


def compute_attention_rollout(
    attention_weights: torch.Tensor,
    discard_ratio: float = 0.9
) -> torch.Tensor:
    """
    Compute attention rollout from vision transformer attention weights.
    
    This aggregates attention across all layers to get a single attention map
    following the method from "Quantifying Attention Flow in Transformers".
    
    Args:
        attention_weights: Attention weights from all layers
                          Shape: [num_layers, B, num_heads, num_patches, num_patches]
        discard_ratio: Ratio of weakest attentions to discard (not used in basic version)
    
    Returns:
        Attention rollout map [B, 1, H, W]
    """
    # Average attention across heads
    # [num_layers, B, num_heads, num_patches, num_patches] -> [num_layers, B, num_patches, num_patches]
    attention = attention_weights.mean(dim=2)
    
    # Add identity matrix for residual connections
    num_patches = attention.shape[-1]
    device = attention.device
    eye = torch.eye(num_patches, device=device).unsqueeze(0).unsqueeze(0)
    
    # Add residual and normalize
    attention = attention + eye
    attention = attention / attention.sum(dim=-1, keepdim=True)
    
    # Compute rollout by recursive matrix multiplication
    rollout = attention[0]  # Start with first layer
    for i in range(1, len(attention)):
        rollout = torch.matmul(attention[i], rollout)
    
    # Extract attention from CLS token to all other patches
    # rollout shape: [B, num_patches, num_patches]
    # We want attention from CLS (index 0) to all image patches (excluding CLS)
    cls_attention = rollout[:, 0, 1:]  # [B, num_patches-1]
    
    # Reshape to spatial grid
    grid_size = int((num_patches - 1) ** 0.5)
    attention_map = cls_attention.reshape(-1, 1, grid_size, grid_size)
    
    # Normalize to [0, 1]
    attention_map = normalize_attention(attention_map)
    
    return attention_map


def normalize_attention(attention_map: torch.Tensor) -> torch.Tensor:
    # Min-max normalization per sample
    batch_size = attention_map.shape[0]
    attention_flat = attention_map.view(batch_size, -1)
    
    min_vals = attention_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = attention_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    
    normalized = (attention_map - min_vals) / (max_vals - min_vals + 1e-8)
    
    return normalized


def resize_attention_map(
    attention_map: torch.Tensor,
    target_size: tuple,
    mode: str = "bilinear"
) -> torch.Tensor:
    if attention_map.shape[2:] == target_size:
        return attention_map
    
    # Use align_corners=False for bilinear interpolation
    align_corners = False if mode == "bilinear" else None
    
    resized = F.interpolate(
        attention_map,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )
    
    return resized


def spatial_attention_from_features(
    feature_map: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:

    if method == "mean":
        # Average absolute activations across channels
        attention = feature_map.abs().mean(dim=1, keepdim=True)
    
    elif method == "max":
        # Max pooling across channels
        attention = feature_map.abs().max(dim=1, keepdim=True)[0]
    
    elif method == "std":
        # Standard deviation across channels
        attention = feature_map.std(dim=1, keepdim=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1]
    attention = normalize_attention(attention)
    
    return attention


def match_attention_resolution(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    match_to: str = "teacher"
) -> tuple:

    if match_to == "teacher":
        target_size = teacher_attention.shape[2:]
        student_attention = resize_attention_map(student_attention, target_size)
    
    elif match_to == "student":
        target_size = student_attention.shape[2:]
        teacher_attention = resize_attention_map(teacher_attention, target_size)
    
    else:
        raise ValueError(f"match_to must be 'teacher' or 'student', got {match_to}")
    
    return student_attention, teacher_attention
