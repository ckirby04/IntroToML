"""Phase 3 step 7: does nnU-Net's deep-supervision downsampling keep a usable float heatmap?

Answer: it keeps the float dtype, but it resamples with mode='nearest-exact', so a single-voxel
peak (target == 1) is kept only if it lands on the sampled lattice. Peaks are therefore lost at
coarse scales, and E01 renders heatmaps per scale instead (e01.heatmap.render).
"""
import numpy as np
import torch
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

from e01 import heatmap as h

SCALES = [(1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]


def _peaks_after_nnunet_ds(anchor):
    les = [h.Lesion(anchor=anchor, centroid_vox=anchor, n_vox=1, esd_mm=1.24)]
    hm = h.render((32, 32, 32), les, (1.0, 1.0, 1.0))
    out = DownsampleSegForDSTransform(SCALES)(segmentation=torch.from_numpy(hm[None]))["segmentation"]
    return [o.dtype for o in out], [int((o >= 1.0 - 1e-6).sum()) for o in out], [float(o.max()) for o in out]


def test_nnunet_ds_keeps_float_dtype():
    dtypes, _, _ = _peaks_after_nnunet_ds((10, 10, 10))
    assert all(d == torch.float32 for d in dtypes)


def test_nnunet_ds_drops_off_lattice_peaks():
    # nearest-exact at scale 1/2 samples source index floor((i + 0.5) * 2) = 2i + 1, so odd
    # coordinates survive and even ones are lost.
    _, n_odd, _ = _peaks_after_nnunet_ds((11, 11, 11))
    _, n_even, max_even = _peaks_after_nnunet_ds((10, 10, 10))
    assert n_odd[1] == 1
    assert n_even[1] == 0 and max_even[1] < 1.0          # the positive voxel is gone at 1/2
    # over all 8 parity classes of an anchor, the peak survives at 1/2 in only 1 of them
    surv = sum(_peaks_after_nnunet_ds((10 + a, 10 + b, 10 + c))[1][1]
               for a in (0, 1) for b in (0, 1) for c in (0, 1))
    assert surv == 1


def test_per_scale_render_keeps_exactly_one_peak_per_lesion():
    for anchor in [(10, 10, 10), (11, 10, 13), (5, 20, 30)]:
        les = [h.Lesion(anchor=anchor, centroid_vox=anchor, n_vox=1, esd_mm=1.24)]
        for s in SCALES:
            m = h.render((32, 32, 32), les, (1.0, 1.0, 1.0), s)
            assert m.shape == h.scaled_shape((32, 32, 32), s)
            assert int((m == 1.0).sum()) == 1
