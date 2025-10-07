"""Backbone implementations for the multi-task video model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask


class PatchEmbedding(nn.Module):
    """Image to patch embedding."""

    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerBackbone(nn.Module):
    """Simplified Vision Transformer backbone for video inputs."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, video: Tensor) -> Dict[str, Tensor | Dict[str, int]]:
        b, t, c, h, w = video.shape
        frames = video.view(b * t, c, h, w)
        tokens = self.patch_embed(frames)
        cls_tokens = self.cls_token.expand(frames.size(0), -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        patches = x[:, 1:]
        cls = cls.view(b, t, -1)
        patches = patches.view(b, t, self.patch_embed.num_patches, -1)
        return {
            "frame_tokens": cls,
            "patch_tokens": patches,
            "patch_metadata": {
                "grid_size": self.patch_embed.grid_size,
                "patch_size": self.patch_embed.patch_size,
                "image_size": self.patch_embed.image_size,
            },
        }


class VisionMambaBlock(nn.Module):
    """A lightweight Vision Mamba style block based on state space mixing."""

    def __init__(self, dim: int, state_dim: int, drop_path: float, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.state_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, drop=0.0)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm1(x)
        # cumulative integration imitates state-space propagation over the sequence
        state = torch.cumsum(self.state_proj(x_norm), dim=1)
        state = torch.tanh(state)
        projected = self.out_proj(state)
        gated = torch.sigmoid(self.gate_proj(x_norm)) * projected
        x = residual + self.drop_path(gated)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionMambaBackbone(nn.Module):
    """Simplified Vision Mamba backbone built from VisionMambaBlocks."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 24,
        state_dim: int = 64,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock(dim=embed_dim, state_dim=state_dim, drop_path=dpr[i], mlp_ratio=mlp_ratio)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, video: Tensor) -> Dict[str, Tensor | Dict[str, int]]:
        b, t, c, h, w = video.shape
        frames = video.view(b * t, c, h, w)
        tokens = self.patch_embed(frames)
        cls_tokens = self.cls_token.expand(frames.size(0), -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        patches = x[:, 1:]
        cls = cls.view(b, t, -1)
        patches = patches.view(b, t, self.patch_embed.num_patches, -1)
        return {
            "frame_tokens": cls,
            "patch_tokens": patches,
            "patch_metadata": {
                "grid_size": self.patch_embed.grid_size,
                "patch_size": self.patch_embed.patch_size,
                "image_size": self.patch_embed.image_size,
            },
        }
