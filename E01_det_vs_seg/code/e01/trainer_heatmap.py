"""Model B trainer: nnUNetTrainer_250epochs with only the target, loss and output layer changed
(PREREG §3). Lives in E01\\code; never copied into site-packages.

Changes vs. stock:
- build_network_architecture: 1 output channel; output-conv bias init log(0.01/0.99).
- get_training_transforms / get_validation_transforms: nnU-Net's DownsampleSegForDSTransform
  is replaced by HeatmapDSTransform, which renders interior-anchor Gaussians per DS scale.
- _build_loss: HeatmapFocalLoss wrapped in nnU-Net's DeepSupervisionWrapper with nnU-Net's
  own scale weights.
- validation_step / on_validation_epoch_end: loss only (no Dice). 'Pseudo dice' is logged as
  NaN, so no fold-0 detection metric is ever printed; the NaN EMA means nnU-Net writes
  checkpoint_best once at epoch 0 and never again (unused: PREREG uses checkpoint_final only).
- perform_actual_validation: disabled. Fold-0 inference for BOTH arms goes through
  e01.predict (identical sliding-window/TTA path; PREREG §4).

Fixes after the 2026-11-11 NaN failure (PREREG Deviation D5; evidence: results/diag_nan/base,
training folds only):
- Output layers get the full focal-loss prior init: weights ~ N(0, 0.01) as well as the bias, so
  initial logits sit at the prior (-4.6) instead of +-15. Stock He init on the 1x1x1 output conv
  made the first-step loss 670-6800 and overflowed fp16 gradients.
- Mixed precision uses bfloat16 (fp32 range) with no GradScaler. With fp16 + GradScaler the
  scale collapsed to 0 after loss spikes and the weights became NaN.
- NaN guard: a non-finite epoch train loss or parameter raises NonFiniteTraining before the
  epoch's checkpoint is written, so the job exits non-zero and the queue stops.
The loss itself (alpha, beta, normalisation) and all other settings are unchanged.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import nn

from e01 import heatmap as hmod
from e01.loss import PRIOR, HeatmapFocalLoss

SMOKE_EPOCHS = 2
AMP_DTYPE = torch.bfloat16
OUT_WEIGHT_STD = 0.01


class NonFiniteTraining(RuntimeError):
    pass


class HeatmapDSTransform(SegOnlyTransform):   # SegOnly: BasicTransform would set image=None
    """Binary (augmented) segmentation (1, X, Y, Z) -> list of float heatmaps (1, X', Y', Z'),
    one per deep-supervision scale. Replaces DownsampleSegForDSTransform."""

    def __init__(self, ds_scales, spacing):
        super().__init__()
        self.ds_scales = [tuple(float(v) for v in s) for s in (ds_scales or [(1.0, 1.0, 1.0)])]
        self.spacing = tuple(float(v) for v in spacing)
        self.single = ds_scales is None

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params):
        assert segmentation.shape[0] == 1 and segmentation.ndim == 4, segmentation.shape
        mask = segmentation[0].numpy() > 0
        maps = hmod.heatmaps(mask, self.spacing, self.ds_scales)
        out = [torch.from_numpy(m[None]) for m in maps]
        return out[0] if self.single else out


def _swap_ds(compose, ds_scales, spacing):
    ts = compose.transforms
    idx = [i for i, t in enumerate(ts) if isinstance(t, DownsampleSegForDSTransform)]
    new = HeatmapDSTransform(ds_scales, spacing)
    if idx:
        assert len(idx) == 1
        ts[idx[0]] = new
    else:
        assert ds_scales is None
        ts.append(new)
    return compose


class E01HeatmapTrainer(nnUNetTrainer_250epochs):
    num_output_channels = 1

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.grad_scaler = None           # bf16 autocast: no loss scaling needed

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=AMP_DTYPE, enabled=True)
        return torch.autocast("cpu", enabled=False)

    # ---- network -------------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        net = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels,
            E01HeatmapTrainer.num_output_channels, enable_deep_supervision)
        bias = math.log(PRIOR / (1 - PRIOR))
        for conv in net.decoder.seg_layers:
            assert conv.out_channels == 1
            nn.init.normal_(conv.weight, 0.0, OUT_WEIGHT_STD)
            nn.init.constant_(conv.bias, bias)
        return net

    # ---- targets ---------------------------------------------------------------------------
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        assert not do_dummy_2d_data_aug and not is_cascaded and regions is None and ignore_label is None
        tr = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        return _swap_ds(tr, deep_supervision_scales, self.configuration_manager.spacing)

    def get_validation_transforms(self, deep_supervision_scales, is_cascaded=False, foreground_labels=None,
                                  regions=None, ignore_label=None):
        va = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded=is_cascaded,
                                                     foreground_labels=foreground_labels, regions=regions,
                                                     ignore_label=ignore_label)
        return _swap_ds(va, deep_supervision_scales, self.configuration_manager.spacing)

    # ---- loss ------------------------------------------------------------------------------
    def _build_loss(self):
        loss = HeatmapFocalLoss()
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(scales))])
            w[-1] = 0                     # nnU-Net default (non-DDP)
            loss = DeepSupervisionWrapper(loss, w / w.sum())
        return loss

    # ---- training step: bf16 autocast, no GradScaler; clipping as stock ---------------------
    def _to_device(self, batch: dict):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)
        return data, target

    def train_step(self, batch: dict) -> dict:
        assert self.grad_scaler is None
        data, target = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.network(data)
            l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        loss_here = self.logger.my_fantastic_logging["train_losses"][-1]
        bad = [n for n, p in self.network.named_parameters() if not torch.isfinite(p).all()]
        if not np.isfinite(loss_here) or bad:
            msg = (f"NON-FINITE TRAINING at epoch {self.current_epoch}: train_loss {loss_here}, "
                   f"{len(bad)} non-finite parameter tensors {bad[:3]}; stopping before checkpoint")
            self.print_to_log_file(msg)
            raise NonFiniteTraining(msg)

    # ---- validation: loss only -------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data, target = self._to_device(batch)
        with torch.no_grad():
            with self._autocast():
                output = self.network(data)
                l = self.loss(output, target)
        return {"loss": l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        loss_here = float(np.mean(collate_outputs(val_outputs)["loss"]))
        self.logger.log("mean_fg_dice", float("nan"), self.current_epoch)
        self.logger.log("dice_per_class_or_region", [float("nan")], self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        raise RuntimeError("E01 Model B: fold-0 inference runs through e01.predict, not nnU-Net validation")


class E01HeatmapTrainer_smoke(E01HeatmapTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS
