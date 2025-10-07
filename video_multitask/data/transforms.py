"""Video-specific augmentation and preprocessing."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torchvision import transforms as T


class VideoTransform:
    """Applies image transforms frame-by-frame while preserving tensor layout."""

    def __init__(self, image_transform: Callable[[Tensor], Tensor]) -> None:
        self.image_transform = image_transform

    def __call__(self, clip: Tensor) -> Tensor:
        # clip: (T, C, H, W)
        processed = [self.image_transform(frame) for frame in clip]
        return torch.stack(processed, dim=0)


def build_default_transform(image_size: int = 224) -> VideoTransform:
    """Returns a default transform pipeline for training."""

    transform = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return VideoTransform(transform)
