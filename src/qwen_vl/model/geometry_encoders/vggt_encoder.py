"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional, List

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT and return the default (final) feature set."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        # Apply inverse reference frame transformation
        features = self._apply_inverse_reference_frame_transform(features)

        return features

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode images and return features from specific aggregator layers."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])

        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        tensor_features = []

        if layer_indices is None:
            layer_indices = [-2]

        for idx in layer_indices:
            tokens = aggregated_tokens_list[idx][0]
            tokens = self._apply_inverse_reference_frame_transform(tokens) # flip frames if ture
            patch_tokens = tokens[:, patch_start_idx:]
            camera_token = tokens[:, 0:1] # first token

            # reshape and trim
            patch_grid = patch_tokens.reshape(n_image, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(n_image, trimmed_h // spatial_merge_size, spatial_merge_size, trimmed_w // spatial_merge_size, spatial_merge_size, -1)
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(n_image, trimmed_h * trimmed_w, -1)

            if not include_camera_token:
                geo_feature = patch_tokens
            else:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)

            tensor_features.append(geo_feature.to(dtype).contiguous())

        return tensor_features
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
