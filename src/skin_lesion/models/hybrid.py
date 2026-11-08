"""EfficientNet features -> Transformer encoder -> SE block -> classifier."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

STRIDE = 32  # total downsampling of the EfficientNet feature extractor


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation over the token embedding dimension."""

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> reweighted (B, N, D)."""
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=1)))))
        return x * scale.unsqueeze(1)


class HybridModel(nn.Module):
    """CNN backbone + Transformer encoder + SE block hybrid classifier."""

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 7,
        pretrained: bool = True,
        img_size: int = 300,
        embed_dim: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        n_tokens = math.ceil(img_size / STRIDE) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        )
        self.se = SEBlock(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits `(B, num_classes)`."""
        feat = self.proj(self.backbone(x)[-1])  # (B, D, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, D)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Got {tokens.shape[1]} tokens, expected {self.pos_embed.shape[1]}; "
                "input size must match the img_size the model was built with."
            )
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.se(tokens).mean(dim=1))

    @property
    def cam_layer(self) -> nn.Module:
        """Layer whose output feature map Grad-CAM uses."""
        return self.proj

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of the CNN backbone."""
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of all newly added layers."""
        bb = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]
