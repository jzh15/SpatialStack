"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1

@dataclass
class MultiLayerFeatureFusionConfig:
    """Configuration for multi layer feature fusion."""
    fusion_method: str = "deepstack_vision_add"  # "add"
    vis_hidden_size: int = 1280
    geo_hidden_size: int = 2048
    lang_hidden_size: int = 2048
    geometry_fusion_layers: List[int] = None
    pos_encoding_type: str = "none"  # "none" | "rope2d" | "sincos2d"
    spatial_merge_size: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    include_camera_token: bool = False

class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        features_2d: torch.Tensor,     # [B, Nq, C]
        features_3d: torch.Tensor,     # [B, Nk, C]
        pos_embed_query: Optional[torch.Tensor] = None,  # [B, Nq, C]
        pos_embed_key: Optional[torch.Tensor] = None,    # [B, Nk, C]
    ) -> torch.Tensor:
        # Normalize features (LayerNorm under autocast returns fp32; cast back to original dtype)
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)

        if query.dtype != features_2d.dtype:
            query = query.to(features_2d.dtype)
        if key.dtype != features_3d.dtype:
            key = key.to(features_3d.dtype)
        if value.dtype != features_3d.dtype:
            value = value.to(features_3d.dtype)

        # Add externally-computed positional embeddings if provided
        if pos_embed_query is not None:
            query = query + pos_embed_query.to(dtype=query.dtype, device=query.device)
        if pos_embed_key is not None:
            key = key + pos_embed_key.to(dtype=key.dtype, device=key.device)
            
        # Cross-attention: 2D features as query, 3D features as key/value
        attn_output, _ = self.cross_attention(query, key, value)
        
        # First residual connection
        x = features_2d + attn_output
        
        # MLP with second residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x

class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features
        """

        _, h_grid, w_grid, _ = features_3d.shape
        if self.fusion_method == "add":
            return features_2d + features_3d
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            return self.projection(concat_features)
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * features_2d + (1 - gate) * features_3d
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            return norm_weight_2d * features_2d + norm_weight_3d * features_3d
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x

class MultiLayerFeatureFusionModule(nn.Module):
    """Enhanced multi feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: MultiLayerFeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.vis_hidden_size = config.vis_hidden_size
        self.geo_hidden_size = config.geo_hidden_size
        self.lang_hidden_size = config.lang_hidden_size
        self.geometry_fusion_layers = config.geometry_fusion_layers

        # Allow multiple fusion blocks per decoder layer (support duplicate layer indices)
        self.fusion_layers = nn.ModuleDict()
        for layer_num in self.geometry_fusion_layers:
            layer_key = str(layer_num)
            if layer_key not in self.fusion_layers:
                self.fusion_layers[layer_key] = nn.ModuleList()
            self.fusion_layers[layer_key].append(self._build_fusion_layer())
        self.reset_residual_branches_to_noop()
    
    def get_fusion_layer(self, layer_idx: int):
        fusion_layer = self.fusion_layers[str(layer_idx)]
        # Always return a ModuleList for consistent downstream handling
        if isinstance(fusion_layer, nn.ModuleList):
            return fusion_layer
        return nn.ModuleList([fusion_layer])

    def _build_fusion_layer(self):
        """Build fusion layers based on method."""
        fusion_layer = None

        if self.config.fusion_method == "deepstack_vision_add":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            geo_norm = Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6)
            geo_mlp = nn.Sequential(
                nn.Linear(self.geo_hidden_size, self.geo_hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.geo_hidden_size * 2, self.vis_hidden_size),
            )
            fusion_layer = nn.Sequential(
                geo_norm,
                geo_mlp
            )
        elif self.config.fusion_method == "deepstack_vision_cross_attn":

            fusion_layer = nn.ModuleDict({
                "geo_proj": nn.Sequential(
                    nn.LayerNorm(self.geo_hidden_size),
                    nn.Linear(self.geo_hidden_size, self.vis_hidden_size)
                ),
                # Unified cross attention with configurable position encoding
                "cross_attn": CrossAttentionBlock(
                    self.vis_hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout),
            })
        elif self.config.fusion_method == "deepstack_language_add":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            fusion_layer = nn.ModuleDict({
                "geo_ln":Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * self.config.spatial_merge_size ** 2, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                )
            })
        elif self.config.fusion_method == "deepstack_language_cross_attn":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            fusion_layer = nn.ModuleDict({
                "geo_ln": Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * self.config.spatial_merge_size ** 2, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "cam_proj": nn.Sequential(
                    Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                    nn.Linear(self.geo_hidden_size, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "cross_attn": CrossAttentionBlock(
                    self.lang_hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                ),
            })
        else:
            raise ValueError(f"Unknown fusion type: {self.config.fusion_method}")

        return fusion_layer

    def reset_residual_branches_to_noop(self) -> None:
        if self.config.fusion_method == "deepstack_language_add":
            for fusion_layers in self.fusion_layers.values():
                for fusion_layer in fusion_layers:
                    self._zero_init_last_linear(fusion_layer["geo_mlp"])

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return

    def forward(
        self, 
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        layer_num: int,
        vis_pos_embed: Optional[torch.Tensor] = None,
        geo_pos_embed: Optional[torch.Tensor] = None,
        fusion_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features (single tensor or list/tuple of tensors)
            layer_num: layer_num
            fusion_layer_idx: select which fusion block to use when multiple are attached to the same layer
        Returns:
            Fused features
        """
        fusion_layers = self.get_fusion_layer(layer_num)
        features_3d_list = list(features_3d) if isinstance(features_3d, (list, tuple)) else [features_3d]

        if fusion_layer_idx is not None:
            if fusion_layer_idx >= len(features_3d_list):
                raise ValueError(
                    f"fusion_layer_idx {fusion_layer_idx} out of range for provided features "
                    f"(available: {len(features_3d_list)})"
                )
            features_3d_list = [features_3d_list[fusion_layer_idx]]
            fusion_indices = [fusion_layer_idx]
        else:
            if len(features_3d_list) not in (1, len(fusion_layers)):
                raise ValueError(
                    f"Expected 1 or {len(fusion_layers)} geometry feature tensors for layer {layer_num}, "
                    f"got {len(features_3d_list)}"
                )
            if len(fusion_layers) == 1:
                fusion_indices = [0] * len(features_3d_list)
            else:
                fusion_indices = list(range(len(features_3d_list)))

        for geo_idx, fusion_idx in enumerate(fusion_indices):
            fusion_layer = fusion_layers[fusion_idx]

            if self.fusion_method == "deepstack_vision_add":
                geo_feats = fusion_layer(features_3d_list[geo_idx])
                assert features_2d.shape == geo_feats.shape, (
                    f"Shape mismatch: features_2d={features_2d.shape}, features_3d={geo_feats.shape}"
                )
                features_2d = features_2d + geo_feats

            # cross attention
            elif self.config.fusion_method == "deepstack_vision_cross_attn":
                geo_feats = fusion_layer['geo_proj'](features_3d_list[geo_idx])
                
                # cross attention
                features_2d = fusion_layer['cross_attn'](features_2d, geo_feats, vis_pos_embed, geo_pos_embed)

            elif self.config.fusion_method == "deepstack_language_add":
                geo_feats = fusion_layer['geo_ln'](features_3d_list[geo_idx])
                geo_feats = geo_feats.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                geo_feats = fusion_layer['geo_mlp'](geo_feats)
                features_2d = features_2d + geo_feats

            elif self.config.fusion_method == "deepstack_language_cross_attn":
                geo_feats = features_3d_list[geo_idx]
                num_imgs, num_geo_tokens = geo_feats.shape[:2]
                num_merged_patch_tokens = features_2d.shape[0] // num_imgs
                # split cam and patchs
                if num_merged_patch_tokens * self.config.spatial_merge_size ** 2 == num_geo_tokens:
                    geo_feats = fusion_layer['geo_ln'](geo_feats)
                    geo_feats = geo_feats.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                    geo_feats = fusion_layer['geo_mlp'](geo_feats)
                    geo_feats = geo_feats.reshape(num_imgs, num_merged_patch_tokens, -1)
                else:
                    features_3d_cam = fusion_layer['cam_proj'](geo_feats[:, 0:1])
                    features_3d_patchs = fusion_layer['geo_ln'](geo_feats[:, 1:])
                    features_3d_patchs = features_3d_patchs.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                    features_3d_patchs = fusion_layer['geo_mlp'](features_3d_patchs)
                    features_3d_patchs = features_3d_patchs.reshape(num_imgs, num_merged_patch_tokens, -1)
                    geo_feats = torch.cat([features_3d_cam, features_3d_patchs], dim=1)

                # cross attention
                features_2d = features_2d.reshape(num_imgs, num_merged_patch_tokens, -1)
                features_2d = fusion_layer['cross_attn'](features_2d, geo_feats, vis_pos_embed, geo_pos_embed)
                features_2d = features_2d.reshape(num_imgs * num_merged_patch_tokens, -1)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return features_2d
