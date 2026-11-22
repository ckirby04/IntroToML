import math

import numpy as np
import pytest
import torch

from e01 import heatmap as h
from e01.loss import PRIOR, HeatmapFocalLoss


def test_loss_ordering():
    """Minimum is the sharp peak (p=1 at peak, 0 elsewhere); p == soft target is NOT zero-loss
    under CenterNet focal loss (negatives with 0 < t < 1 still pay (1-t)^4 p^2 log(1-p));
    the bias-initialised prior is worst."""
    les = [h.Lesion(anchor=(8, 8, 8), centroid_vox=(8, 8, 8), n_vox=20, esd_mm=4.0)]
    t = torch.from_numpy(h.render((16, 16, 16), les, (1, 1, 1)))[None, None]
    sharp = torch.where(t >= 1, torch.tensor(12.0), torch.tensor(-12.0))
    soft = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    prior = torch.full_like(t, math.log(PRIOR / (1 - PRIOR)))
    f = HeatmapFocalLoss()
    assert f(sharp, t).item() < 1e-3 < f(soft, t).item() < f(prior, t).item()


def test_loss_matches_hand_computation():
    t = torch.tensor([1.0, 0.5, 0.0]).view(1, 1, 3, 1, 1)
    x = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 3, 1, 1)      # p = 0.5 everywhere
    pos = -(0.5 ** 2) * math.log(0.5)
    neg = -((1 - 0.5) ** 4) * (0.5 ** 2) * math.log(0.5) - (1.0 ** 4) * (0.5 ** 2) * math.log(0.5)
    assert HeatmapFocalLoss()(x, t).item() == pytest.approx(pos + neg, rel=1e-5)


def test_toy_overfit_loss_decreases_on_real_backbone():
    """The actual Model B network (plans architecture, 1 output channel, DS) overfits one toy
    volume on CPU: total DS loss falls by > 50% in 40 steps."""
    from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform  # noqa
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    from e01 import common as c
    from e01.trainer_heatmap import E01HeatmapTrainer

    torch.manual_seed(0)
    plans = load_json(str(c.NN_PRE / c.DATASET / "nnUNetPlans.json"))
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]
    net = E01HeatmapTrainer.build_network_architecture(arch["network_class_name"], arch["arch_kwargs"],
                                                       arch["_kw_requires_import"], 1, 2, True)
    assert all(abs(float(conv.bias.detach()) - math.log(PRIOR / (1 - PRIOR))) < 1e-6 for conv in net.decoder.seg_layers)
    shape = (32, 32, 32)
    m = np.zeros(shape, bool)
    m[8:12, 8:12, 8:12] = True
    m[22, 24, 20] = True
    pool = np.vstack(arch["arch_kwargs"]["strides"])       # = ConfigurationManager.pool_op_kernel_sizes
    scales = [list(s) for s in 1 / np.cumprod(pool, axis=0)][:-1]
    tg = [torch.from_numpy(x)[None, None] for x in h.heatmaps(m, (1, 1, 1), scales)]
    x = torch.randn(1, 1, *shape)
    x[0, 0][torch.from_numpy(m)] += 3.0
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
