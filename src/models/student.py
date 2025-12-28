"""CNN-based student models for knowledge distillation."""

import timm
import torch
import torch.nn as nn
from typing import Optional, Dict


class StudentCNN(nn.Module):
    """
    CNN-based student model using timm library.
    
    Supports various CNN architectures (ResNet, VGG, MobileNet, EfficientNet, etc.)
    for learning from teacher through knowledge distillation.
    
    Args:
        architecture: Model architecture name (e.g., 'resnet18', 'mobilenetv3_small')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        extract_features: Whether to enable feature extraction for attention distillation
    """
    
    # Mapping of common architecture names to timm model names
    ARCH_MAPPING = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "vgg16": "vgg16_bn",
        "vgg19": "vgg19_bn",
        "mobilenetv3_small": "mobilenetv3_small_100.lamb_in1k",
        "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "densenet121": "densenet121",
    }
    
    def __init__(
        self,
        architecture: str,
        num_classes: int = 200,
        pretrained: bool = True,
        extract_features: bool = False,
    ):
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.extract_features = extract_features
        
        # Map architecture name to timm model name
        timm_name = self.ARCH_MAPPING.get(architecture.lower(), architecture)
        
        # Create backbone model
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=False
        )
        
        # Store feature extraction hook if needed
        self._feature_maps = None
        if extract_features:
            self._register_feature_hook()
    
    def _register_feature_hook(self):
        """Register forward hook to extract intermediate features."""
        def hook_fn(module, input, output):
            self._feature_maps = output
        
        # Try to find the last convolutional layer
        # This varies by architecture, so we use a heuristic
        target_layer = None
        
        # For ResNet-style models
        if hasattr(self.backbone, 'layer4'):
            target_layer = self.backbone.layer4[-1]
        # For VGG-style models
        elif hasattr(self.backbone, 'features'):
            target_layer = self.backbone.features[-1]
        # For MobileNet/EfficientNet-style models
        elif hasattr(self.backbone, 'conv_head'):
            target_layer = self.backbone.conv_head
        elif hasattr(self.backbone, 'blocks'):
            target_layer = self.backbone.blocks[-1]
        
        if target_layer is not None:
            target_layer.register_forward_hook(hook_fn)
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass through student network.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features is False:
                logits: Classification logits [B, num_classes]
            If return_features is True:
                dict with keys:
                    - 'logits': Classification logits [B, num_classes]
                    - 'features': Feature maps [B, C, H, W] (if available)
        """
        # Reset feature maps
        self._feature_maps = None
        
        # Forward through backbone
        logits = self.backbone(x)
        
        if return_features and self._feature_maps is not None:
            return {
                "logits": logits,
                "features": self._feature_maps
            }
        
        return logits
    
    def get_feature_map(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract feature maps from the last convolutional layer.
        
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            Feature maps [B, C, H, W] or None if not available
        """
        if not self.extract_features:
            raise RuntimeError("Feature extraction not enabled. Set extract_features=True")
        
        _ = self.forward(x)
        return self._feature_maps
    
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate attention/saliency map from CNN features.
        
        Computes spatial attention by averaging absolute feature activations.
        
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            Attention map [B, 1, H, W] normalized to [0, 1]
        """
        feature_map = self.get_feature_map(x)
        
        if feature_map is None:
            raise RuntimeError("Could not extract feature map")
        
        # Compute spatial attention by averaging channel activations
        attention = feature_map.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize to [0, 1]
        attention = attention - attention.amin(dim=(2, 3), keepdim=True)
        attention = attention / (attention.amax(dim=(2, 3), keepdim=True) + 1e-8)
        
        return attention
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get number of parameters in the model.
        
        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params
        }


def build_student(
    architecture: str,
    num_classes: int = 200,
    pretrained: bool = True,
    extract_features: bool = False,
) -> StudentCNN:
    """
    Factory function to build student CNN model.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        extract_features: Whether to enable feature extraction
    
    Returns:
        StudentCNN instance
    """
    return StudentCNN(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        extract_features=extract_features
    )
