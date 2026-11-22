"""E02 trainer classes (PREREG §4). Imported by launch_train / predict; never copied into
site-packages.

Every trained arm (A, B, C2) shares:
- bf16 autocast with no GradScaler. torch.set_autocast_dtype("cuda", bfloat16) makes nnU-Net's
  stock `autocast(device, enabled=True)` calls run in bf16, so A and C2 keep nnU-Net's own
  train/validation steps unchanged; grad_scaler = None selects nnU-Net's unscaled branch.
- A NaN guard: a non-finite epoch train loss or weight raises NonFiniteTraining before that
  epoch's checkpoint is written (E01 D6).

Model B additionally changes the target, loss and output layer (trainer_heatmap.py).
Smoke classes differ only in num_epochs; their own class names keep their output folders apart.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)

SMOKE_EPOCHS = 2
AMP_DTYPE = torch.bfloat16


class NonFiniteTraining(RuntimeError):
    pass


class Bf16Mixin:
    """bf16 autocast, no GradScaler, NaN guard. Put before the nnU-Net trainer in the MRO."""

    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        torch.set_autocast_dtype("cuda", AMP_DTYPE)
        self.grad_scaler = None

    def train_step(self, batch: dict) -> dict:
        assert self.grad_scaler is None
        if self.device.type == "cuda":
            assert torch.get_autocast_dtype("cuda") == AMP_DTYPE
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        loss_here = self.logger.my_fantastic_logging["train_losses"][-1]
        bad = [n for n, p in self.network.named_parameters() if not torch.isfinite(p).all()]
        if not np.isfinite(loss_here) or bad:
            msg = (f"NON-FINITE TRAINING at epoch {self.current_epoch}: train_loss {loss_here}, "
                   f"{len(bad)} non-finite parameter tensors {bad[:3]}; stopping before checkpoint")
            self.print_to_log_file(msg)
            raise NonFiniteTraining(msg)


class E02SegTrainer(Bf16Mixin, nnUNetTrainer_250epochs):
    """Model A and C2: stock nnUNetTrainer_250epochs (Dice + CE) with bf16 and the NaN guard."""


class E02SegTrainer_smoke(E02SegTrainer):
    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS


class E02C2Trainer(E02SegTrainer):
    """C2: identical to A's trainer; its own class name keeps its folder distinct."""


class E02C2Trainer_smoke(E02C2Trainer):
    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS


def get(arm: str, smoke: bool):
    if arm == "A":
        return E02SegTrainer_smoke if smoke else E02SegTrainer
    if arm == "C2":
        return E02C2Trainer_smoke if smoke else E02C2Trainer
    if arm == "B":
        from e02 import trainer_heatmap as th

        return th.E02HeatmapTrainer_smoke if smoke else th.E02HeatmapTrainer
    raise ValueError(arm)
