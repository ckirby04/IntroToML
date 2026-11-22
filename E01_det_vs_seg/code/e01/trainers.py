"""Trainer classes for E01. Imported by launch_train.py; never copied into site-packages.

Model A is the stock nnU-Net class `nnUNetTrainer_250epochs`, unchanged.
Smoke classes differ only in num_epochs, and their own class names keep their output
folders separate from the full runs.
"""
from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)

SMOKE_EPOCHS = 2


class E01A_smoke(nnUNetTrainer_250epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS


class E01C2_smoke(nnUNetTrainer_250epochs):
    """C2 pre-flight (PREREG_C Amendment C-1.2): stock trainer, 2 epochs, separate folder."""

    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS


def get(arm: str, smoke: bool):
    if arm == "C2":                      # stock trainer on Dataset902 crops, fold "all"
        return E01C2_smoke if smoke else nnUNetTrainer_250epochs
    if arm == "A":
        return E01A_smoke if smoke else nnUNetTrainer_250epochs
    if arm == "B":
        from e01 import trainer_heatmap as th

        return th.E01HeatmapTrainer_smoke if smoke else th.E01HeatmapTrainer
    raise ValueError(arm)
