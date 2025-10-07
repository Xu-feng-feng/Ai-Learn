"""Dataset utilities for multi-task video learning."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


@dataclass
class Annotation:
    """Single video sample description."""

    video_id: str
    frame_dir: Path
    label: int
    num_frames: int
    keyframe_indices: Sequence[int]

    @classmethod
    def from_json(cls, item: Dict[str, object], root: Path) -> "Annotation":
        frame_dir = root / str(item["frame_dir"])
        keyframes = item.get("keyframes", [])
        if isinstance(keyframes, list):
            keyframe_indices: Sequence[int] = [int(idx) for idx in keyframes]
        else:
            raise TypeError("keyframes must be a list of indices")

        return cls(
            video_id=str(item.get("video_id", item["frame_dir"])),
            frame_dir=frame_dir,
            label=int(item["label"]),
            num_frames=int(item.get("num_frames", 0)),
            keyframe_indices=keyframe_indices,
        )


def _load_annotation_file(path: Path, root: Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            annotations.append(Annotation.from_json(record, root))
    return annotations


def _list_frame_paths(frame_dir: Path) -> List[Path]:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frame images found in {frame_dir}")
    return frames


class FrameVideoDataset(Dataset):
    """Loads pre-extracted video frames and rich annotations.

    The dataset expects a JSONL annotation file where each line contains at
    least the following fields::

        {
          "video_id": "<unique identifier>",
          "frame_dir": "relative/path/to/frames",
          "label": <int>,
          "num_frames": <int>,  # optional but recommended
          "keyframes": [<int>, ...]  # optional; indices are zero-based
        }

    Frames inside ``frame_dir`` must follow a consistent naming scheme such as
    ``img_00001.jpg``. The dataset performs multi-task labelling by constructing
    frame-wise keyframe masks alongside the clip label.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_path: str | Path,
        num_frames: int,
        frame_stride: int,
        enable_compression: bool = True,
        enable_keyframe: bool = True,
        transform: Optional = None,
    ) -> None:
        self.root = Path(root)
        self.annotations = _load_annotation_file(Path(annotation_path), self.root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.enable_compression = enable_compression
        self.enable_keyframe = enable_keyframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            total_frames = (self.num_frames - 1) * self.frame_stride + 1

        max_start = max(total_frames - (self.num_frames - 1) * self.frame_stride, 1)
        start = random.randint(0, max_start - 1)
        indices = [start + i * self.frame_stride for i in range(self.num_frames)]
        return [min(idx, total_frames - 1) for idx in indices]

    def _load_frames(self, annotation: Annotation, indices: Sequence[int]) -> Tensor:
        frame_paths = _list_frame_paths(annotation.frame_dir)
        total_frames = len(frame_paths)
        selected = [frame_paths[min(idx, total_frames - 1)] for idx in indices]
        frames = [read_image(str(path)).float() / 255.0 for path in selected]
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        return clip

    def _keyframe_mask(self, indices: Sequence[int], keyframe_indices: Sequence[int]) -> Tensor:
        mask = torch.zeros(len(indices), dtype=torch.float32)
        keyframe_set = set(keyframe_indices)
        for i, idx in enumerate(indices):
            if idx in keyframe_set:
                mask[i] = 1.0
        return mask

    def __getitem__(self, index: int) -> Dict[str, Tensor | int | Dict[str, bool]]:
        annotation = self.annotations[index]
        frame_paths = _list_frame_paths(annotation.frame_dir)
        indices = self._sample_frame_indices(len(frame_paths))
        clip = self._load_frames(annotation, indices)

        if self.transform is not None:
            clip = self.transform(clip)

        keyframe_mask = self._keyframe_mask(indices, annotation.keyframe_indices)

        sample: Dict[str, Tensor | int | Dict[str, bool]] = {
            "video": clip,  # (T, C, H, W)
            "label": annotation.label,
            "keyframe_mask": keyframe_mask,
            "tasks": {
                "classification": True,
                "compression": self.enable_compression,
                "keyframe": self.enable_keyframe and keyframe_mask.sum() > 0,
            },
        }
        return sample


class SomethingSomethingV2(FrameVideoDataset):
    """Dataset wrapper pre-configured for Something-Something V2."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class UCF101(FrameVideoDataset):
    """Dataset wrapper pre-configured for UCF-101."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class Kinetics400(FrameVideoDataset):
    """Dataset wrapper pre-configured for Kinetics-400."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


DATASET_REGISTRY = {
    "ssv2": SomethingSomethingV2,
    "something-something-v2": SomethingSomethingV2,
    "ucf101": UCF101,
    "kinetics": Kinetics400,
    "kinetics400": Kinetics400,
}


def build_dataset(name: str, **kwargs) -> FrameVideoDataset:
    try:
        dataset_cls = DATASET_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
    return dataset_cls(**kwargs)
