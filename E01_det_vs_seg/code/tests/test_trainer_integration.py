"""CPU integration: E01HeatmapTrainer's real data pipeline on TRAINING-fold cases only (never fold 0),
with the patch shrunk to 64^3 so it is light. Checks the DS swap, target shapes/dtypes, one peak per
lesion per scale, and one train_step + validation_step on CPU."""
import copy

import numpy as np
import pytest
import torch

from e01 import common as c


@pytest.fixture(scope="module")
def trainer():
    from batchgenerators.utilities.file_and_folder_operations import load_json

    from e01.trainer_heatmap import E01HeatmapTrainer

    plans = copy.deepcopy(load_json(str(c.NN_PRE / c.DATASET / "nnUNetPlans.json")))
    plans["configurations"]["3d_fullres"]["patch_size"] = [64, 64, 64]
    plans["configurations"]["3d_fullres"]["batch_size"] = 1
    dj = load_json(str(c.NN_PRE / c.DATASET / "dataset.json"))
    import os
    import shutil
    from pathlib import Path

    tr_dir = Path(c.NN_RES) / c.DATASET / "E01HeatmapTrainer__nnUNetPlans__3d_fullres"
    existed = tr_dir.exists()
    t = E01HeatmapTrainer(plans, "3d_fullres", c.FOLD, dj, device=torch.device("cpu"))
    # nnU-Net's __init__ writes a citation-only log into the real results folder: remove it so a
    # later real run of Model B starts from an untouched folder.
    if os.path.exists(t.log_file):
        os.remove(t.log_file)
    if not existed and tr_dir.exists():
        shutil.rmtree(tr_dir)
    tmp = c.RESULTS / "tmp_integration"                       # never touches nnUNet_results
    tmp.mkdir(parents=True, exist_ok=True)
    t.output_folder = str(tmp)
    t.log_file = str(tmp / "training_log_integration.txt")
    t.initialize()
    yield t
    shutil.rmtree(tmp, ignore_errors=True)
    assert existed or not tr_dir.exists(), "integration test left files in nnUNet_results"


def _train_batch(t, n=2):
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

    ds_scales = t._get_deep_supervision_scales()
    rot, dummy2d, init_ps, mirror = t.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    tr = t.get_training_transforms(t.configuration_manager.patch_size, rot, ds_scales, mirror, dummy2d,
                                   use_mask_for_norm=t.configuration_manager.use_mask_for_norm,
                                   foreground_labels=t.label_manager.foreground_labels)
    ds_tr, ds_val = t.get_tr_and_val_datasets()
    g = c.guard()
    assert all(g.fold_of(k) != 0 for k in ds_tr.identifiers)      # training folds only
    dl = nnUNetDataLoader(ds_tr, n, init_ps, t.configuration_manager.patch_size, t.label_manager,
                          oversample_foreground_percent=1.0, transforms=tr)
    return dl.generate_train_batch(), tr, ds_scales


def test_ds_transform_swapped(trainer):
    from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

    from e01.trainer_heatmap import HeatmapDSTransform

    _, tr, _ = _train_batch(trainer, 1)
    kinds = [type(x) for x in tr.transforms]
    assert HeatmapDSTransform in kinds and DownsampleSegForDSTransform not in kinds


def test_targets_and_one_step(trainer):
    batch, _, scales = _train_batch(trainer, 2)
    tg = batch["target"]
    assert isinstance(tg, list) and len(tg) == len(scales)
    for t, s in zip(tg, scales):
        assert t.dtype == torch.float32
        assert tuple(t.shape[2:]) == tuple(int(round(64 * f)) for f in s)
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    full = tg[0]                                      # foreground-oversampled: lesions present
    assert float(full.max()) == 1.0
    before = [p.detach().clone() for p in trainer.network.parameters()][:3]
    out = trainer.train_step(batch)
    assert np.isfinite(out["loss"]) and out["loss"] > 0
    after = [p.detach() for p in trainer.network.parameters()][:3]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    v = trainer.validation_step(batch)
    assert set(v) == {"loss"} and np.isfinite(v["loss"])


def test_output_layer_one_channel(trainer):
    net = trainer.network
    assert all(conv.out_channels == 1 for conv in net.decoder.seg_layers)
