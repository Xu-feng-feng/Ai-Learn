"""Configuration dataclasses for the multi-task video training pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """Configuration for a single video dataset.

    Attributes:
        name: Friendly identifier for the dataset (e.g. "ssv2", "ucf101").
        root: Path to the directory that contains extracted video frames.
        annotation_path: Path to a JSONL annotation file describing the split.
        split: Which split to use ("train", "val" or "test").
        num_frames: Number of frames sampled per video clip.
        frame_stride: Temporal stride between sampled frames.
        batch_size: Mini-batch size for this dataset.
        num_workers: Number of dataloader worker processes.
        enable_compression: Whether to compute the reconstruction loss.
        enable_keyframe: Whether to compute the keyframe localisation loss.
    """

    name: str
    root: str
    annotation_path: str
    split: str = "train"
    num_frames: int = 16
    frame_stride: int = 2
    batch_size: int = 4
    num_workers: int = 4
    enable_compression: bool = True
    enable_keyframe: bool = True


@dataclass
class OptimiserConfig:
    """Configuration for the optimiser and scheduler."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 2
    max_epochs: int = 20
    gradient_clip_norm: Optional[float] = 1.0


@dataclass
class ModelConfig:
    """Configuration for the multi-task model."""

    backbone: str = "vit"
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    state_dim: int = 64  # Only used for the Vision Mamba backbone.
    num_classes: int = 174  # Default to Something-Something V2.


@dataclass
class LossWeights:
    """Relative weights for the different objectives."""

    classification: float = 1.0
    compression: float = 1.0
    keyframe: float = 1.0


@dataclass
class ExperimentConfig:
    """Aggregated configuration used by the training script."""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    datasets: List[DatasetConfig] = field(default_factory=list)
    output_dir: str = "outputs"
    mixed_precision: bool = True
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1

    def dataset_by_name(self, name: str) -> DatasetConfig:
        for config in self.datasets:
            if config.name == name:
                return config
        raise KeyError(f"Unknown dataset name: {name}")


def create_default_config() -> ExperimentConfig:
    """Creates a minimal configuration that can serve as a starting point.

    The default configuration wires up the three datasets requested in the task
    (SSv2, UCF-101 and Kinetics-400) under the assumption that frame directories
    and annotation files follow a consistent JSONL schema. Paths should be
    overwritten by the user prior to training.
    """

    default_datasets: List[DatasetConfig] = [
        DatasetConfig(
            name="ssv2",
            root="/path/to/something_something_v2/frames",
            annotation_path="/path/to/something_something_v2/train.jsonl",
            split="train",
            num_frames=16,
            frame_stride=2,
            batch_size=4,
            num_workers=8,
            enable_compression=True,
            enable_keyframe=True,
        ),
        DatasetConfig(
            name="ucf101",
            root="/path/to/ucf101/frames",
            annotation_path="/path/to/ucf101/train.jsonl",
            split="train",
            num_frames=16,
            frame_stride=2,
            batch_size=4,
            num_workers=8,
            enable_compression=True,
            enable_keyframe=True,
        ),
        DatasetConfig(
            name="kinetics400",
            root="/path/to/kinetics400/frames",
            annotation_path="/path/to/kinetics400/train.jsonl",
            split="train",
            num_frames=16,
            frame_stride=2,
            batch_size=4,
            num_workers=8,
            enable_compression=True,
            enable_keyframe=True,
        ),
    ]

    return ExperimentConfig(datasets=default_datasets)
