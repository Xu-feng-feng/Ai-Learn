"""Multi-task video model that fuses VIM/VIT backbones with specialised heads."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from video_multitask.config import LossWeights, ModelConfig
from .backbones import VisionMambaBackbone, VisionTransformerBackbone
from .heads import ClassificationHead, CompressionHead, KeyframeLocalizationHead


class MultiTaskVideoModel(nn.Module):
    """Joint model for action recognition, semantic compression and keyframe localisation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        backbone_name = config.backbone.lower()
        if backbone_name == "vit":
            self.backbone = VisionTransformerBackbone(
                image_size=config.image_size,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop_rate=config.drop_rate,
                attn_drop_rate=config.attn_drop_rate,
                drop_path_rate=config.drop_path_rate,
            )
        elif backbone_name == "vim":
            self.backbone = VisionMambaBackbone(
                image_size=config.image_size,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.depth,
                state_dim=config.state_dim,
                mlp_ratio=config.mlp_ratio,
                drop_path_rate=config.drop_path_rate,
            )
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        self.classification_head = ClassificationHead(config.embed_dim, config.num_classes, dropout=config.drop_rate)
        self.compression_head = CompressionHead(config.embed_dim, patch_size=config.patch_size)
        self.keyframe_head = KeyframeLocalizationHead(config.embed_dim)

    def forward(self, video: Tensor) -> Dict[str, Tensor | Dict[str, int]]:
        features = self.backbone(video)
        frame_tokens: Tensor = features["frame_tokens"]
        patch_tokens: Tensor = features["patch_tokens"]
        patch_metadata = features["patch_metadata"]

        logits = self.classification_head(frame_tokens)
        recon = self.compression_head(patch_tokens, patch_metadata)
        keyframe_logits = self.keyframe_head(frame_tokens)

        return {
            "logits": logits,
            "reconstruction": recon,
            "keyframe_logits": keyframe_logits,
            "features": features,
        }

    def compute_losses(
        self,
        batch: Dict[str, Tensor | Dict[str, Tensor]],
        loss_weights: LossWeights,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        video = batch["video"]
        targets = batch["label"]
        keyframes = batch["keyframe_mask"]
        task_masks = batch["tasks"]

        outputs = self(video)
        total_loss = torch.tensor(0.0, device=video.device)
        loss_dict: Dict[str, Tensor] = {}

        if task_masks["classification"].any():
            cls_mask = task_masks["classification"].bool()
            logits = outputs["logits"][cls_mask]
            cls_targets = targets[cls_mask]
            cls_loss = F.cross_entropy(logits, cls_targets)
            loss_dict["classification"] = cls_loss
            total_loss = total_loss + loss_weights.classification * cls_loss

        if task_masks["compression"].any():
            cmp_mask = task_masks["compression"].view(-1, 1, 1, 1, 1).to(video.dtype)
            reconstruction = outputs["reconstruction"]
            diff = (reconstruction - video) ** 2 * cmp_mask
            denom = cmp_mask.sum().clamp_min(1.0)
            compression_loss = diff.sum() / denom
            loss_dict["compression"] = compression_loss
            total_loss = total_loss + loss_weights.compression * compression_loss

        if task_masks["keyframe"].any():
            kf_mask = task_masks["keyframe"].bool()
            keyframe_logits = outputs["keyframe_logits"][kf_mask]
            keyframe_targets = keyframes[kf_mask]
            keyframe_loss = F.binary_cross_entropy_with_logits(keyframe_logits, keyframe_targets)
            loss_dict["keyframe"] = keyframe_loss
            total_loss = total_loss + loss_weights.keyframe * keyframe_loss

        return total_loss, loss_dict
