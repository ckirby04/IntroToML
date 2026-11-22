"""Ported from E01 tests/test_loss.py to 2D (instances instead of lesions)."""
import math

import numpy as np
import pytest
import torch

from e02 import heatmap as h
from e02.loss import PRIOR, HeatmapFocalLoss


def test_loss_ordering():
    """Minimum is the sharp peak; p == soft target is NOT zero-loss under CenterNet focal loss;
    the bias-initialised prior is worst."""
    cells = [h.Cell(id=1, anchor=(16, 16), centroid=(16.0, 16.0), n_px=300)]
    t = torch.from_numpy(h.render((32, 32), cells))[None, None]
    sharp = torch.where(t >= 1, torch.tensor(12.0), torch.tensor(-12.0))
    soft = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    prior = torch.full_like(t, math.log(PRIOR / (1 - PRIOR)))
    f = HeatmapFocalLoss()
    assert f(sharp, t).item() < 1e-3 < f(soft, t).item() < f(prior, t).item()


def test_loss_matches_hand_computation():
    t = torch.tensor([1.0, 0.5, 0.0]).view(1, 1, 3, 1)
    x = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 3, 1)      # p = 0.5 everywhere
    pos = -(0.5 ** 2) * math.log(0.5)
    neg = -((1 - 0.5) ** 4) * (0.5 ** 2) * math.log(0.5) - (1.0 ** 4) * (0.5 ** 2) * math.log(0.5)
    assert HeatmapFocalLoss()(x, t).item() == pytest.approx(pos + neg, rel=1e-5)


def test_toy_overfit_loss_decreases_on_real_backbone():
    """The actual Model B network (E02 2d plans, 1 output channel, DS) overfits one toy image on
    CPU: total DS loss falls by > 50% in 40 steps."""
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    from e02 import common as c
    from e02.trainer_heatmap import E02HeatmapTrainer

    torch.manual_seed(0)
    # The trained model keeps an exact copy of the plans; data\nnUNet_preprocessed was deleted at close-out.
    plans_file = c.NN_PRE / c.DATASET / "nnUNetPlans.json"
    if not plans_file.exists():
        plans_file = c.NN_RES / c.DATASET / "E02HeatmapTrainer__nnUNetPlans__2d" / "plans.json"
    plans = load_json(str(plans_file))
    arch = plans["configurations"]["2d"]["architecture"]
    net = E02HeatmapTrainer.build_network_architecture(arch["network_class_name"], arch["arch_kwargs"],
                                                       arch["_kw_requires_import"], 1, 2, True)
    assert all(abs(float(conv.bias.detach()) - math.log(PRIOR / (1 - PRIOR))) < 1e-6 for conv in net.decoder.seg_layers)
    pool = np.vstack(arch["arch_kwargs"]["strides"])
    side = int(np.prod(pool, axis=0).max()) * 2
    shape = (side, side)
    inst = np.zeros(shape, np.int16)
    inst[20:34, 20:34] = 1
    inst[20:34, 34:46] = 2                      # touching cell
    inst[side - 20:side - 12, 10:18] = 3
    scales = [list(s) for s in 1 / np.cumprod(pool, axis=0)][:-1]
    tg = [torch.from_numpy(x)[None, None] for x in h.heatmaps(inst, scales)]
    x = torch.randn(1, 1, *shape)
    x[0, 0][torch.from_numpy(inst > 0)] += 3.0
    w = np.array([1 / 2 ** i for i in range(len(scales))])
    w[-1] = 0
    loss = DeepSupervisionWrapper(HeatmapFocalLoss(), w / w.sum())
    opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = net(x)
        assert [tuple(o.shape[2:]) for o in out] == [tuple(t.shape[2:]) for t in tg]
        l = loss(out, tg)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        opt.step()
        losses.append(l.item())
    assert losses[-1] < 0.5 * losses[0], losses[::5]
