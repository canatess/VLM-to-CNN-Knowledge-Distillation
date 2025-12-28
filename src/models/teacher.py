"""CLIP-based teacher model for knowledge distillation."""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer


class CLIPTeacher(nn.Module):
    """
    CLIP Vision-Language Model as teacher for knowledge distillation.
    
    This model provides:
    - Logits for logit-based distillation
    - Attention maps for attention-based distillation
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'openai/clip-vit-base-patch32')
        class_names: List of class names for zero-shot classification
        prompt_template: Template for generating text prompts (e.g., 'a photo of a {}')
        extract_attention: Whether to extract attention maps from vision encoder
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        class_names: Optional[List[str]] = None,
        prompt_template: str = "a photo of a {}",
        extract_attention: bool = False,
    ):
        super().__init__()
        
        if class_names is None:
            raise ValueError("class_names must be provided for zero-shot classification")
        
        self.model_name = model_name
        self.class_names = class_names
        self.extract_attention = extract_attention
        
        # Load CLIP model with safetensors to avoid PyTorch security issue
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Prepare text prompts
        prompts = [
            prompt_template.format(name.replace("_", " "))
            for name in class_names
        ]
        
        # Tokenize text prompts
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Register as buffers (will be moved to device with model)
        self.register_buffer("text_input_ids", text_inputs["input_ids"], persistent=False)
        self.register_buffer("text_attention_mask", text_inputs["attention_mask"], persistent=False)
    
    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass through CLIP teacher.
        
        Args:
            pixel_values: Image tensor [B, 3, H, W]
            return_attention: Whether to return attention maps
        
        Returns:
            If return_attention is False:
                logits: Classification logits [B, num_classes]
            If return_attention is True:
                dict with keys:
                    - 'logits': Classification logits [B, num_classes]
                    - 'attention': Attention maps (if available)
        """
        # Get text and attention mask on the same device
        text_input_ids = self.text_input_ids.to(pixel_values.device)
        text_attention_mask = self.text_attention_mask.to(pixel_values.device)
        
        # Forward through CLIP
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_attentions=return_attention,
            return_dict=True
        )
        
        logits = outputs.logits_per_image
        
        if return_attention and outputs.vision_model_output.attentions is not None:
            # Extract attention maps from vision encoder
            attention_maps = outputs.vision_model_output.attentions
            return {
                "logits": logits,
                "attention": attention_maps,
                "image_embeds": outputs.image_embeds
            }
        
        return logits
    
    def get_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get classification logits only.
        
        Args:
            pixel_values: Image tensor [B, 3, H, W]
        
        Returns:
            Classification logits [B, num_classes]
        """
        return self.forward(pixel_values, return_attention=False)
    
    def get_attention_rollout(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get attention rollout from CLIP vision encoder.
        
        Args:
            pixel_values: Image tensor [B, 3, H, W]
        
        Returns:
            Attention map [B, 1, H, W]
        """
        outputs = self.forward(pixel_values, return_attention=True)
        
        if isinstance(outputs, dict) and "attention" in outputs:
            attention_maps = outputs["attention"]
            # Compute attention rollout
            rollout = self._compute_attention_rollout(attention_maps)
            return rollout
        
        raise RuntimeError("Attention maps not available")
    
    def _compute_attention_rollout(self, attention_maps: tuple) -> torch.Tensor:
        """
        Compute attention rollout from layer-wise attention maps.
        
        Args:
            attention_maps: Tuple of attention tensors from each layer
        
        Returns:
            Rolled out attention map [B, 1, H, W]
        """
        # Stack attention maps: [num_layers, B, num_heads, num_patches, num_patches]
        attentions = torch.stack(attention_maps)
        
        # Average over heads: [num_layers, B, num_patches, num_patches]
        attentions = attentions.mean(dim=2)
        
        # Add identity matrix for residual connections
        num_patches = attentions.shape[-1]
        eye = torch.eye(num_patches, device=attentions.device).unsqueeze(0).unsqueeze(0)
        attentions = attentions + eye
        
        # Normalize
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)
        
        # Compute rollout by matrix multiplication
        rollout = attentions[0]
        for i in range(1, len(attentions)):
            rollout = torch.matmul(attentions[i], rollout)
        
        # Extract CLS token attention to all patches (excluding CLS itself)
        # rollout: [B, num_patches, num_patches]
        cls_attention = rollout[:, 0, 1:]  # [B, num_patches-1]
        
        # Reshape to spatial dimensions
        patch_size = int((num_patches - 1) ** 0.5)
        attention_map = cls_attention.reshape(-1, 1, patch_size, patch_size)
        
        # Normalize to [0, 1]
        attention_map = attention_map - attention_map.amin(dim=(2, 3), keepdim=True)
        attention_map = attention_map / (attention_map.amax(dim=(2, 3), keepdim=True) + 1e-8)
        
        return attention_map
