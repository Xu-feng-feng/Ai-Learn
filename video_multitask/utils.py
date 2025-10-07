"""Helper utilities for training and evaluation."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import torch
from torch import Tensor


def multitask_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    videos = torch.stack([item["video"] for item in batch], dim=0)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    keyframes = torch.stack([item["keyframe_mask"] for item in batch], dim=0)
    task_masks = {
        "classification": torch.tensor([bool(item["tasks"]["classification"]) for item in batch], dtype=torch.bool),
        "compression": torch.tensor([bool(item["tasks"]["compression"]) for item in batch], dtype=torch.bool),
        "keyframe": torch.tensor([bool(item["tasks"]["keyframe"]) for item in batch], dtype=torch.bool),
    }
    return {
        "video": videos,
        "label": labels,
        "keyframe_mask": keyframes,
        "tasks": task_masks,
    }


def cycle_dataloaders(loaders: Dict[str, Iterable]) -> Iterator[Tuple[str, object]]:
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    while True:
        for name in loaders.keys():
            try:
                batch = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loaders[name])
                batch = next(iterators[name])
            yield name, batch
