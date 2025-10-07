"""Training orchestration for the multi-task video model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from video_multitask.config import ExperimentConfig
from video_multitask.models.multitask import MultiTaskVideoModel


class MultiTaskTrainer:
    """Handles optimisation, logging and evaluation."""

    def __init__(self, model: MultiTaskVideoModel, config: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.optimiser.learning_rate,
            weight_decay=config.optimiser.weight_decay,
            betas=config.optimiser.betas,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        self.global_step = 0

    def _to_device(self, batch: Dict[str, object]) -> Dict[str, object]:
        def move(value: object):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value

        return {k: move(v) for k, v in batch.items()}

    def fit(self, train_loaders: Dict[str, DataLoader], val_loaders: Dict[str, DataLoader] | None = None) -> None:
        steps_per_epoch = max(len(loader) for loader in train_loaders.values())
        autocast = torch.cuda.amp.autocast
        for epoch in range(self.config.optimiser.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            data_iter = {name: iter(loader) for name, loader in train_loaders.items()}
            for step in range(steps_per_epoch):
                for name, loader in train_loaders.items():
                    try:
                        batch = next(data_iter[name])
                    except StopIteration:
                        data_iter[name] = iter(loader)
                        batch = next(data_iter[name])
                    batch = self._to_device(batch)
                    with autocast(enabled=self.config.mixed_precision):
                        loss, loss_dict = self.model.compute_losses(batch, self.config.loss_weights)
                    self.scaler.scale(loss).backward()

                    if self.config.optimiser.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.optimiser.gradient_clip_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    self.global_step += 1

                    if self.global_step % self.config.log_every_n_steps == 0:
                        losses_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                        print(f"Step {self.global_step}: loss={loss.item():.4f} ({losses_str})")

            avg_loss = epoch_loss / (steps_per_epoch * len(train_loaders))
            print(f"Epoch {epoch + 1} completed. Avg loss: {avg_loss:.4f}")

            if val_loaders and (epoch + 1) % self.config.val_every_n_epochs == 0:
                self.evaluate(val_loaders)

    @torch.no_grad()
    def evaluate(self, loaders: Dict[str, DataLoader]) -> None:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        for name, loader in loaders.items():
            for batch in loader:
                batch = self._to_device(batch)
                loss, loss_dict = self.model.compute_losses(batch, self.config.loss_weights)
                total_loss += loss.item() * batch["video"].size(0)
                total_samples += batch["video"].size(0)
                logits = self.model(batch["video"])["logits"]
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["label"]).sum().item()
        if total_samples > 0:
            print(f"Validation loss: {total_loss / total_samples:.4f}, accuracy: {correct / total_samples:.4f}")

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
