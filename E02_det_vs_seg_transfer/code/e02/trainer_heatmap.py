"""Model B trainer (PREREG §4, §4.1): E01's heatmap trainer ported to 2D, with targets from the
augmented INSTANCE map. Lives in E02\\code; never copied into site-packages.

Changes vs. E02SegTrainer (bf16 + NaN guard, trainers.py):
- build_network_architecture: 1 output channel; output convs weights N(0, 0.01), bias
  log(0.01 / 0.99) (E01 D6).
- Data: the instance map travels as a SECOND segmentation channel. InstanceDataset* append
  data\\instances\\Tr\\<case>.tif to the preprocessed seg in load_case; channel 0 (the gapped
  label) still drives nnU-Net's foreground oversampling, unchanged.
- Augmentation: nnU-Net's SpatialTransform interpolates segmentations per label value
  ('bilinear' one-hot), which is exact for the 2-label channel 0 but would cost one grid_sample
  per cell for channel 1. Channel 0 keeps the stock path; channel 1 uses nearest-neighbour with
  the same grid (InstanceSpatialTransform), as PREREG §4.1 requires.
- Targets: DownsampleSegForDSTransform is replaced by HeatmapDSTransform, which renders
  interior-anchor Gaussians from channel 1 per deep-supervision scale.
- Loss: HeatmapFocalLoss in nnU-Net's DeepSupervisionWrapper with nnU-Net's own scale weights.
- Validation: loss only (no Dice). Inference runs through e02.predict.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import nn

from e02 import common as c
from e02 import heatmap as hmod
from e02.loss import PRIOR, HeatmapFocalLoss
from e02.trainers import SMOKE_EPOCHS, Bf16Mixin

OUT_WEIGHT_STD = 0.01
INSTANCE_DIR = c.INSTANCES / "Tr"          # training images only


# ---------------------------------------------------------------- data ------------------------
def instance_for_case(identifier: str, properties: dict, seg_shape) -> np.ndarray:
    """Instance map of a TRAINING case, cropped like nnU-Net cropped the case, as (1, 1, y, x) int16."""
    import tifffile

    inst = tifffile.imread(INSTANCE_DIR / f"{identifier}.tif")
    assert inst.ndim == 2 and inst.max() < 2 ** 15, (identifier, inst.shape, inst.max())
    before = tuple(properties["shape_before_cropping"])
    assert before[-2:] == inst.shape, (identifier, before, inst.shape)
    bb = properties["bbox_used_for_cropping"]
    inst = inst[bb[-2][0]:bb[-2][1], bb[-1][0]:bb[-1][1]]
    out = inst.astype(np.int16)[None, None]
    assert out.shape[1:] == tuple(seg_shape[1:]), (identifier, out.shape, seg_shape)
    return out


class _InstanceMixin:
    def load_case(self, identifier):
        data, seg, seg_prev, properties = super().load_case(identifier)
        seg = np.asarray(seg[:])
        inst = instance_for_case(identifier, properties, seg.shape)
        # the gapped label lies inside the instances (gaps only remove pixels)
        assert not ((seg[0] > 0) & (inst[0] == 0)).any(), f"{identifier}: label outside instances"
        return data, np.concatenate([seg.astype(np.int16), inst], 0), seg_prev, properties


class InstanceDatasetBlosc2(_InstanceMixin, nnUNetDatasetBlosc2):
    pass


class InstanceDatasetNumpy(_InstanceMixin, nnUNetDatasetNumpy):
    pass


def instance_dataset_class(base):
    if issubclass(base, nnUNetDatasetBlosc2):
        return InstanceDatasetBlosc2
    if issubclass(base, nnUNetDatasetNumpy):
        return InstanceDatasetNumpy
    raise TypeError(base)


# ---------------------------------------------------------------- transforms ------------------
class InstanceSpatialTransform(SpatialTransform):
    """Stock SpatialTransform for seg channel 0; channel 1+ (instance ids) with nearest
    interpolation on the same sampling grid."""

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if segmentation.shape[0] == 1:
            return super()._apply_to_segmentation(segmentation, **params)
        s0 = super()._apply_to_segmentation(segmentation[:1], **params)
        mode = self.mode_seg
        try:
            self.mode_seg = "nearest"
            s1 = super()._apply_to_segmentation(segmentation[1:], **params)
        finally:
            self.mode_seg = mode
        return torch.cat([s0, s1.to(s0.dtype)], 0)


class HeatmapDSTransform(SegOnlyTransform):   # SegOnly: BasicTransform would set image=None
    """(2, Y, X) segmentation [gapped label, instance ids] -> list of float heatmaps (1, Y', X'),
    one per deep-supervision scale. Replaces DownsampleSegForDSTransform."""

    def __init__(self, ds_scales):
        super().__init__()
        self.ds_scales = [tuple(float(v) for v in s) for s in (ds_scales or [(1.0, 1.0)])]
        self.single = ds_scales is None

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params):
        assert segmentation.ndim == 3 and segmentation.shape[0] == 2, segmentation.shape
        inst = segmentation[1].numpy()
        maps = hmod.heatmaps(inst, self.ds_scales)
        out = [torch.from_numpy(m[None]) for m in maps]
        return out[0] if self.single else out


def _swap(compose, ds_scales):
    ts = compose.transforms
    for t in ts:
        if type(t) is SpatialTransform:
            t.__class__ = InstanceSpatialTransform
    idx = [i for i, t in enumerate(ts) if isinstance(t, DownsampleSegForDSTransform)]
    new = HeatmapDSTransform(ds_scales)
    if idx:
        assert len(idx) == 1
        ts[idx[0]] = new
    else:
        assert ds_scales is None
        ts.append(new)
    return compose


# ---------------------------------------------------------------- trainer ---------------------
class E02HeatmapTrainer(Bf16Mixin, nnUNetTrainer_250epochs):
    num_output_channels = 1

    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        sp = [float(v) for v in self.configuration_manager.spacing]
        assert len(sp) == 2 and np.allclose(sp, 1.0), f"E02 B expects 2D unit spacing, got {sp}"
        tf = self.plans_manager.transpose_forward
        assert list(tf) == sorted(tf), f"transpose_forward {tf}: instance maps would not align"

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        net = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels,
            E02HeatmapTrainer.num_output_channels, enable_deep_supervision)
        bias = math.log(PRIOR / (1 - PRIOR))
        for conv in net.decoder.seg_layers:
            assert conv.out_channels == 1
            nn.init.normal_(conv.weight, 0.0, OUT_WEIGHT_STD)
            nn.init.constant_(conv.bias, bias)
        return net

    def get_tr_and_val_datasets(self):
        self.dataset_class = instance_dataset_class(self.dataset_class)
        return super().get_tr_and_val_datasets()

    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        assert not do_dummy_2d_data_aug and not is_cascaded and regions is None and ignore_label is None
        tr = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        return _swap(tr, deep_supervision_scales)

    def get_validation_transforms(self, deep_supervision_scales, is_cascaded=False, foreground_labels=None,
                                  regions=None, ignore_label=None):
        va = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded=is_cascaded,
                                                     foreground_labels=foreground_labels, regions=regions,
                                                     ignore_label=ignore_label)
        return _swap(va, deep_supervision_scales)

    def _build_loss(self):
        loss = HeatmapFocalLoss()
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(scales))])
            w[-1] = 0                     # nnU-Net default (non-DDP)
            loss = DeepSupervisionWrapper(loss, w / w.sum())
        return loss

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                output = self.network(data)
                l = self.loss(output, target)
        return {"loss": l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        loss_here = float(np.mean(collate_outputs(val_outputs)["loss"]))
        self.logger.log("mean_fg_dice", float("nan"), self.current_epoch)
        self.logger.log("dice_per_class_or_region", [float("nan")], self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        raise RuntimeError("E02 Model B: inference runs through e02.predict, not nnU-Net validation")


class E02HeatmapTrainer_smoke(E02HeatmapTrainer):
    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = SMOKE_EPOCHS
