"""PREREG §4.1: the instance map goes through nnU-Net's augmentation as a second segmentation
channel with nearest-neighbour interpolation, and instance ids survive. Synthetic data only."""
import numpy as np
import pytest
import torch

from e02 import heatmap as h
from e02.trainer_heatmap import (E02HeatmapTrainer, HeatmapDSTransform, InstanceSpatialTransform,
                                 instance_dataset_class, InstanceDatasetBlosc2, InstanceDatasetNumpy)

PATCH = (128, 128)
SCALES = [[1.0, 1.0], [0.5, 0.5], [0.25, 0.25]]
IDS = np.arange(1, 41) * 7                       # non-consecutive: a blended id would be visible


def _sample(seed):
    rng = np.random.default_rng(seed)
    inst = np.zeros((180, 180), np.int16)
    k = 0
    for y in range(5, 170, 26):
        for x in range(5, 170, 26):
            if k == len(IDS):
                break
            hh, ww = rng.integers(12, 27, 2)       # up to 26 px: neighbours touch
            inst[y:y + hh, x:x + ww] = IDS[k]
            k += 1
    lab = (inst > 0).astype(np.int16)
    img = rng.normal(size=(1, 180, 180)).astype(np.float32) + lab
    return torch.from_numpy(img), torch.from_numpy(np.stack([lab, inst]))


def _transforms(force_spatial: bool):
    tr = E02HeatmapTrainer.get_training_transforms(None, PATCH, (-np.pi, np.pi), SCALES, (0, 1), False)
    sp = [t for t in tr.transforms if isinstance(t, InstanceSpatialTransform)]
    assert len(sp) == 1
    if force_spatial:
        sp[0].p_rotation = 1.0
        sp[0].p_scaling = 1.0
    return tr


def test_transform_pipeline_swaps():
    from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

    kinds = [type(t) for t in _transforms(False).transforms]
    assert HeatmapDSTransform in kinds and DownsampleSegForDSTransform not in kinds
    assert InstanceSpatialTransform in kinds


@pytest.mark.parametrize("seed", range(12))
def test_instance_ids_survive_augmentation(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = _transforms(force_spatial=True)
    pre = type(tr)(tr.transforms[:-1])            # everything except the heatmap target
    img, seg = _sample(seed)
    out = pre(image=img.clone(), segmentation=seg.clone())
    s = out["segmentation"]
    assert s.shape == (2, *PATCH) and not s.is_floating_point()
    ids = set(np.unique(s[1].numpy()).tolist()) - {0}
    assert ids and ids <= set(IDS.tolist()), sorted(ids - set(IDS.tolist()))[:5]
    fg0 = s[0].numpy() > 0
    on_inst = (s[1].numpy() > 0)[fg0].mean() if fg0.any() else 1.0
    assert on_inst > 0.97                          # label (bilinear path) and instances (nearest) agree


@pytest.mark.parametrize("seed", range(4))
def test_full_pipeline_targets(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = _transforms(force_spatial=bool(seed % 2))
    pre = type(tr)(tr.transforms[:-1])
    img, seg = _sample(seed)
    torch.manual_seed(100 + seed)
    np.random.seed(100 + seed)
    mid = pre(image=img.clone(), segmentation=seg.clone())
    n_ids = len(set(np.unique(mid["segmentation"][1].numpy()).tolist()) - {0})
    tg = HeatmapDSTransform(SCALES)(**mid)["segmentation"]
    assert isinstance(tg, list) and len(tg) == len(SCALES)
    for t, s in zip(tg, SCALES):
        assert t.dtype == torch.float32 and tuple(t.shape[1:]) == h.scaled_shape(PATCH, s)
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    assert int((tg[0] == 1.0).sum()) == n_ids      # one peak per instance present


def test_dataset_class_mapping():
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy

    assert instance_dataset_class(nnUNetDatasetBlosc2) is InstanceDatasetBlosc2
    assert instance_dataset_class(nnUNetDatasetNumpy) is InstanceDatasetNumpy
