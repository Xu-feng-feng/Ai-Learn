"""Entry point for training the multi-task VIM/VIT video model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from video_multitask.config import (
    DatasetConfig,
    ExperimentConfig,
    LossWeights,
    ModelConfig,
    OptimiserConfig,
    create_default_config,
)
from video_multitask.data.datasets import build_dataset
from video_multitask.data.transforms import build_default_transform
from video_multitask.models.multitask import MultiTaskVideoModel
from video_multitask.trainer import MultiTaskTrainer
from video_multitask.utils import multitask_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-task video model")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON configuration file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store checkpoints and logs")
    return parser.parse_args()


def _build_config_from_dict(cfg: Dict) -> ExperimentConfig:
    model = ModelConfig(**cfg.get("model", {}))
    optimiser = OptimiserConfig(**cfg.get("optimiser", {}))
    loss_weights = LossWeights(**cfg.get("loss_weights", {}))
    datasets = [DatasetConfig(**dataset_cfg) for dataset_cfg in cfg.get("datasets", [])]
    experiment = ExperimentConfig(
        model=model,
        optimiser=optimiser,
        loss_weights=loss_weights,
        datasets=datasets if datasets else create_default_config().datasets,
        output_dir=cfg.get("output_dir", "outputs"),
        mixed_precision=cfg.get("mixed_precision", True),
        log_every_n_steps=cfg.get("log_every_n_steps", 20),
        val_every_n_epochs=cfg.get("val_every_n_epochs", 1),
    )
    return experiment


def load_config(path: str | None) -> ExperimentConfig:
    if path is None:
        return create_default_config()
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    return _build_config_from_dict(cfg_dict)


def build_dataloaders(config: ExperimentConfig) -> tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    transform = build_default_transform(config.model.image_size)
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}

    for dataset_cfg in config.datasets:
        dataset = build_dataset(
            dataset_cfg.name,
            root=dataset_cfg.root,
            annotation_path=dataset_cfg.annotation_path,
            num_frames=dataset_cfg.num_frames,
            frame_stride=dataset_cfg.frame_stride,
            enable_compression=dataset_cfg.enable_compression,
            enable_keyframe=dataset_cfg.enable_keyframe,
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=dataset_cfg.split == "train",
            num_workers=dataset_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=multitask_collate_fn,
        )
        if dataset_cfg.split == "train":
            train_loaders[dataset_cfg.name] = loader
        else:
            val_loaders[dataset_cfg.name] = loader

    if not train_loaders:
        raise RuntimeError("No training datasets configured.")

    return train_loaders, val_loaders


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders, val_loaders = build_dataloaders(config)

    model = MultiTaskVideoModel(config.model)
    trainer = MultiTaskTrainer(model, config, device)
    trainer.fit(train_loaders, val_loaders if val_loaders else None)

    checkpoint_path = Path(config.output_dir) / "multitask_model.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
