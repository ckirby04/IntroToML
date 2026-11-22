"""CPU integration (ported from E01): the real B and A data pipelines on preprocessed TRAINING images
(Dataset921 Tr only), patch shrunk to 256 x 256. Checks the instance channel, the target swap,
one peak per instance, bf16 settings, and one train_step + validation_step. Skipped until
e02.plan_preprocess has run."""
import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from e02 import common as c

PLANS_FILE = c.NN_PRE / c.DATASET / "nnUNetPlans.json"
pytestmark = pytest.mark.skipif(not PLANS_FILE.exists(), reason="run e02.plan_preprocess first")


def _make(cls_name):
    from batchgenerators.utilities.file_and_folder_operations import load_json

    from e02 import trainers

    cls = trainers.get({"E02HeatmapTrainer": "B", "E02SegTrainer": "A"}[cls_name], False)
    plans = copy.deepcopy(load_json(str(PLANS_FILE)))
    plans["configurations"]["2d"]["patch_size"] = [256, 256]
    plans["configurations"]["2d"]["batch_size"] = 2
    dj = load_json(str(c.NN_PRE / c.DATASET / "dataset.json"))
    tr_dir = Path(c.NN_RES) / c.DATASET / f"{cls.__name__}__nnUNetPlans__2d"
    existed = tr_dir.exists()
    t = cls(plans, "2d", c.FOLD, dj, device=torch.device("cpu"))
    # nnU-Net's __init__ writes a log into the real results folder: remove it so a real run
    # starts from an untouched folder.
    if os.path.exists(t.log_file):
        os.remove(t.log_file)
    if not existed and tr_dir.exists():
        shutil.rmtree(tr_dir)
    tmp = c.RESULTS / f"tmp_integration_{cls_name}"          # never touches nnUNet_results
    tmp.mkdir(parents=True, exist_ok=True)
    t.output_folder = str(tmp)
    t.log_file = str(tmp / "training_log_integration.txt")
    t.initialize()
    return t, tmp, existed, tr_dir


@pytest.fixture(scope="module")
def trainer_b():
    t, tmp, existed, tr_dir = _make("E02HeatmapTrainer")
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
    ds_tr, _ = t.get_tr_and_val_datasets()
    train_cases = {r["case"] for r in c.image_rows("Tr")}
    assert set(ds_tr.identifiers) <= train_cases           # training images only
    dl = nnUNetDataLoader(ds_tr, n, init_ps, t.configuration_manager.patch_size, t.label_manager,
                          oversample_foreground_percent=1.0, transforms=tr)
    return dl.generate_train_batch(), ds_tr, ds_scales


def test_instance_channel_loaded(trainer_b):
    _, ds, _ = _train_batch(trainer_b, 1)
    case = ds.identifiers[0]
    data, seg, _, props = ds.load_case(case)
    assert seg.shape[0] == 2 and seg.dtype == np.int16
    import tifffile

    inst = tifffile.imread(c.INSTANCES / "Tr" / f"{case}.tif")
    assert int(seg[1].max()) == int(inst.max()) and seg.shape[-2:] == inst.shape


def test_targets_and_one_step(trainer_b):
    batch, _, scales = _train_batch(trainer_b, 2)
    tg = batch["target"]
    assert isinstance(tg, list) and len(tg) == len(scales)
    for t, s in zip(tg, scales):
        assert t.dtype == torch.float32
        assert tuple(t.shape[2:]) == tuple(int(round(256 * f)) for f in s)
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    assert float(tg[0].max()) == 1.0
    assert trainer_b.grad_scaler is None
    before = [p.detach().clone() for p in trainer_b.network.parameters()][:3]
    out = trainer_b.train_step(batch)
    assert np.isfinite(out["loss"]) and out["loss"] > 0
    after = [p.detach() for p in trainer_b.network.parameters()][:3]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    v = trainer_b.validation_step(batch)
    assert set(v) == {"loss"} and np.isfinite(v["loss"])
    assert all(conv.out_channels == 1 for conv in trainer_b.network.decoder.seg_layers)


def test_segmenter_bf16_settings():
    t, tmp, existed, tr_dir = _make("E02SegTrainer")
    try:
        assert t.grad_scaler is None and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ds_tr, _ = t.get_tr_and_val_datasets()
        _, seg, _, _ = ds_tr.load_case(ds_tr.identifiers[0])
        assert seg.shape[0] == 1                            # A sees the gapped label only
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        assert existed or not tr_dir.exists()
