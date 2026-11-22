"""Synthetic tests for C2 (PREREG_C_two_stage.md Amendment C-1). No real data."""
import numpy as np
from scipy import ndimage

from e01 import c2
from e01 import two_stage as ts

SP = (1.0, 1.0, 1.0)


def _ball(shape, cen, r):
    g = np.indices(shape)
    return ((g - np.asarray(cen)[:, None, None, None]) ** 2).sum(0) <= r * r


def test_crop_centre_padding_and_paste_roundtrip():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(50, 60, 40)).astype(np.float32)
    q = (3, 58, 20)                                     # near two borders -> padding
    cr = c2.extract_crop(vol, q, size=16)
    assert cr.shape == (16, 16, 16)
    assert cr[8, 8, 8] == vol[q]                        # centre voxel = start + size // 2
    assert np.all(cr[:5] == 0)                          # x < 0 padded
    m = np.zeros(vol.shape, bool)
    c2.paste_or(m, np.ones((16, 16, 16), bool), c2.crop_start(q, 16))
    lo = np.maximum(c2.crop_start(q, 16), 0)
    hi = np.minimum(c2.crop_start(q, 16) + 16, vol.shape)
    assert m.sum() == np.prod(hi - lo)


def test_crop_affine_maps_crop_origin_to_scan_voxel():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = [-90, -120, -70]
    st = np.array([-4, 10, 7])
    a2 = c2.crop_affine(aff, st)
    assert np.allclose(a2 @ [0, 0, 0, 1], aff @ [*st, 1])


def test_jitter_halfwidth():
    assert list(c2.jitter_halfwidth(2.0, SP)) == [4, 4, 4]
    assert list(c2.jitter_halfwidth(28.0, SP)) == [7, 7, 7]
    assert list(c2.jitter_halfwidth(80.0, SP)) == [10, 10, 10]


def test_negative_plan_totals_and_proportionality():
    plan = c2.negative_plan([10, 0, 30], n_pos=40, seed=1)
    assert plan.sum() == 20 and plan[1].sum() == 0
    assert plan[:, 0].sum() == 10 and plan[:, 1].sum() == 10


def test_sample_scan_crops_positives_and_negative_distance():
    shape = (80, 80, 80)
    img = np.zeros(shape, np.float32)
    brain = _ball(shape, (40, 40, 40), 35)
    rng = np.random.default_rng(3)
    img[brain] = rng.normal(size=brain.sum())
    lab = _ball(shape, (30, 40, 40), 4) | _ball(shape, (55, 40, 40), 2)
    img[lab] = 8.0                                       # lesions are the brightest voxels
    out = c2.sample_scan_crops(img, lab, SP, n_rand=5, n_bright=5, rng=np.random.default_rng(1))
    pos = [r for r in out if r["kind"] == "pos"]
    assert len(pos) == 2
    for r in pos:
        assert np.all(np.abs(r["offset"]) <= 4)
    dist = ndimage.distance_transform_edt(~lab)
    negs = [r for r in out if r["kind"] in ("neg_rand", "neg_bright")]
    assert len(negs) == 10
    for r in negs:
        assert dist[r["center"]] >= c2.NEG_MIN_DIST_MM and img[r["center"]] != 0


def test_stage2_from_crop_assigns_centre_component():
    p2 = np.zeros((32, 32, 32), np.float32)
    p2[_ball(p2.shape, (16, 16, 16), 4)] = 0.9
    p2[_ball(p2.shape, (16, 16, 16), 2)] = 0.97
    p2[_ball(p2.shape, (4, 4, 4), 2)] = 0.99             # other lesion in the crop, not assigned
    a2, reg, ok = c2.stage2_from_crop(p2, SP)
    assert ok and np.isclose(a2, 0.97) and reg.sum() == _ball(p2.shape, (16, 16, 16), 4).sum()
    a2, reg, ok = c2.stage2_from_crop(np.zeros((32, 32, 32), np.float32), SP)
    assert not ok and a2 == 0.0 and not reg.any()


def test_c2_scores_and_mask_paste_and_fallback():
    from e01 import heatmap as h

    shape = (60, 60, 60)
    les = [h.Lesion(anchor=(20, 20, 20), centroid_vox=(20, 20, 20), n_vox=30, esd_mm=6.0),
           h.Lesion(anchor=(40, 40, 40), centroid_vox=(40, 40, 40), n_vox=30, esd_mm=8.0)]
    heat = h.render(shape, les, SP)
    reg0 = np.zeros((96, 96, 96), bool)
    reg0[46:51, 46:51, 46:51] = True                    # around crop centre 48
    out = {"points": np.array([[20, 20, 20], [40, 40, 40]]), "b": np.array([0.6, 0.9]),
           "a2": np.array([0.8, 0.0]), "starts": np.array([c2.crop_start(p) for p in [(20, 20, 20), (40, 40, 40)]]),
           "regions": np.stack([reg0, np.zeros_like(reg0)])}
    s, m = ts.c2_scores_and_mask(out, heat, SP, shape)
    assert np.allclose(s, [0.7, 0.45])
    assert m[18:23, 18:23, 18:23].all()                 # pasted crop region
    assert m[40, 40, 40]                                # fallback (s = 0.45 >= 0.4, empty region)
