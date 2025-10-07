"""Prediction heads for the multi-task model."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Aggregates frame tokens and predicts action classes."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, frame_tokens: Tensor) -> Tensor:
        pooled = frame_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class CompressionHead(nn.Module):
    """Projects patch tokens back into the pixel space."""

    def __init__(self, embed_dim: int, patch_size: int, in_chans: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    def forward(self, patch_tokens: Tensor, metadata: Dict[str, int]) -> Tensor:
        b, t, num_patches, dim = patch_tokens.shape
        patch_dim = self.patch_size * self.patch_size * self.in_chans
        patches = self.linear(patch_tokens)
        patches = patches.view(b, t, num_patches, self.in_chans, self.patch_size, self.patch_size)
        grid_size = metadata["grid_size"]
        image_size = metadata["image_size"]
        frames = patches.view(b, t, grid_size, grid_size, self.in_chans, self.patch_size, self.patch_size)
        frames = frames.permute(0, 1, 4, 2, 5, 3, 6)
        frames = frames.contiguous().view(b, t, self.in_chans, image_size, image_size)
        return frames


class KeyframeLocalizationHead(nn.Module):
    """Predicts keyframe logits for each temporal position."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, frame_tokens: Tensor) -> Tensor:
        x = self.proj(frame_tokens)
        x = self.act(x)
        logits = self.out(x).squeeze(-1)
        return logits
