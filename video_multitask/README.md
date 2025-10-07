# Multi-task Video Learning with Vision Mamba and Vision Transformer

This module provides a PyTorch implementation of a multi-task learning
framework that combines VIM (Vision Mamba) and ViT (Vision Transformer)
backbones for video understanding. The training pipeline jointly optimises
three objectives:

1. **Action classification** on datasets such as Something-Something V2,
   UCF-101 and Kinetics-400.
2. **Semantic video compression and reconstruction**, which learns to compress
   frame patches into the latent space and to reconstruct the original clip.
3. **Keyframe localisation and behaviour recognition** to detect the most
   informative frames for downstream action understanding.

## Key components

- `video_multitask/config.py` – dataclasses that capture experiment
  configuration, including dataset, model and optimiser settings.
- `video_multitask/data/datasets.py` – dataset wrappers that load
  pre-extracted RGB frames using a JSONL annotation file shared across all
  datasets.
- `video_multitask/data/transforms.py` – torchvision-based transforms applied
  frame-by-frame.
- `video_multitask/models/backbones.py` – simplified ViT and Vision Mamba
  backbones adapted for video clips.
- `video_multitask/models/heads.py` – prediction heads for classification,
  compression and keyframe localisation.
- `video_multitask/models/multitask.py` – combines the shared backbone with
  the multi-task heads and exposes a unified loss function.
- `video_multitask/trainer.py` – orchestration utilities for optimisation,
  logging and checkpointing.
- `train_multitask.py` – command line entry point that wires all components
  together.

## Dataset preparation

Each dataset split is described by a JSON lines file where each line contains::

```json
{
  "video_id": "<unique id>",
  "frame_dir": "relative/path/to/frames",
  "label": 12,
  "num_frames": 96,
  "keyframes": [3, 17, 42]
}
```

`frame_dir` must point to a directory containing extracted frames (e.g.
`img_00001.jpg`). `keyframes` lists the zero-based indices of behaviourally
salient frames used for the localisation objective. If a sample does not have
keyframe annotations, simply pass an empty list to disable the corresponding
loss for that item.

## Example configuration

The training script consumes a JSON configuration. The snippet below shows how
one might structure a multi-dataset experiment that leverages both ViT and VIM
backbones across train/validation splits:

```json
{
  "model": {
    "backbone": "vim",
    "image_size": 224,
    "patch_size": 16,
    "embed_dim": 512,
    "depth": 16,
    "mlp_ratio": 2.5,
    "state_dim": 96,
    "num_classes": 400
  },
  "optimiser": {
    "learning_rate": 0.0002,
    "max_epochs": 30,
    "gradient_clip_norm": 0.5
  },
  "loss_weights": {
    "classification": 1.0,
    "compression": 0.7,
    "keyframe": 0.5
  },
  "datasets": [
    {
      "name": "ssv2",
      "root": "/data/ssv2/frames",
      "annotation_path": "/data/ssv2/train.jsonl",
      "split": "train",
      "batch_size": 4,
      "num_workers": 8,
      "num_frames": 16,
      "frame_stride": 2,
      "enable_compression": true,
      "enable_keyframe": true
    },
    {
      "name": "ssv2",
      "root": "/data/ssv2/frames",
      "annotation_path": "/data/ssv2/val.jsonl",
      "split": "val",
      "batch_size": 4,
      "num_workers": 8,
      "num_frames": 16,
      "frame_stride": 2,
      "enable_compression": true,
      "enable_keyframe": true
    },
    {
      "name": "ucf101",
      "root": "/data/ucf101/frames",
      "annotation_path": "/data/ucf101/train.jsonl",
      "split": "train",
      "batch_size": 6,
      "num_workers": 6,
      "enable_keyframe": true
    },
    {
      "name": "kinetics400",
      "root": "/data/kinetics/frames",
      "annotation_path": "/data/kinetics/val.jsonl",
      "split": "val",
      "batch_size": 6,
      "num_workers": 6,
      "enable_keyframe": false
    }
  ]
}
```

Save the configuration to `config.json` and launch training with:

```bash
python train_multitask.py --config config.json --output-dir ./outputs
```

## Extending the framework

- Swap between ViT and VIM backbones by toggling the `model.backbone`
  parameter.
- Adjust loss weights to prioritise classification, compression or keyframe
  accuracy depending on downstream requirements.
- Implement custom dataset readers by extending `FrameVideoDataset` when
  dealing with alternative annotation schemas.
- Integrate evaluation metrics or logging backends (e.g. TensorBoard) in
  `MultiTaskTrainer`.

This code is intentionally modular to facilitate research on multi-task video
understanding and to provide a solid baseline for experiments that require
joint modelling of recognition, compression and temporal localisation.
