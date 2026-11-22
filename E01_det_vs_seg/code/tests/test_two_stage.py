"""Synthetic tests for the C1 cascade (PREREG_C_two_stage.md §2). No fold-0 data."""
import numpy as np

from e01 import heatmap as h
from e01 import two_stage as ts

SP = (1.0, 1.0, 1.0)


def _ball(shape, c, r):
    g = np.indices(shape)
    return ((g - np.asarray(c)[:, None, None, None]) ** 2).sum(0) <= r * r


def test_assignment_contains_near_and_far():
    shape = (40, 40, 40)
    prob = np.zeros(shape, np.float32)
    prob[_ball(shape, (10, 10, 10), 3)] = 0.9            # comp near peak 1 (inside)
    prob[_ball(shape, (30, 10, 10), 2)] = 0.6            # comp 2 mm from peak 2's location
    les = [h.Lesion(anchor=a, centroid_vox=a, n_vox=30, esd_mm=6.0)
           for a in [(10, 10, 10), (30, 14, 10), (10, 30, 30)]]
    heat = h.render(shape, les, SP) * np.array(1.0, np.float32)
    cas = ts.cascade(prob, heat, SP)
    by_pt = {tuple(p): (b, a, s, k) for p, b, a, s, k in zip(cas["points"], cas["b"], cas["a"], cas["s"], cas["comp"])}
    b, a, s, k = by_pt[(10, 10, 10)]
    assert k > 0 and np.isclose(a, 0.9) and np.isclose(s, (1.0 + 0.9) / 2)
    b, a, s, k = by_pt[(30, 14, 10)]                     # nearest comp voxel (30,12,10) at 2.0 mm
    assert k > 0 and np.isclose(a, 0.6)
    b, a, s, k = by_pt[(10, 30, 30)]                     # nothing within 2 mm
    assert k == 0 and a == 0.0 and np.isclose(s, 0.5)


def test_assignment_tie_prefers_higher_max():
    lab = np.zeros((9, 9, 9), np.int64)
    lab[4, 2, 4] = 1
    lab[4, 6, 4] = 2
    k = ts.assign_peaks(np.array([[4, 4, 4]]), lab, np.array([0.3, 0.8]), SP)
    assert k[0] == 2


def test_mask_uses_A_region_once_and_fallback_radius():
    shape = (48, 48, 48)
    prob = np.zeros(shape, np.float32)
    big = _ball(shape, (14, 14, 14), 6)
    prob[big] = 0.95
    # two split B peaks on the big lesion + one B-only lesion with ESD 8 mm (sigma = 2 mm)
    les = [h.Lesion(anchor=(12, 14, 14), centroid_vox=(12, 14, 14), n_vox=30, esd_mm=6.0),
           h.Lesion(anchor=(17, 14, 14), centroid_vox=(17, 14, 14), n_vox=30, esd_mm=6.0),
           h.Lesion(anchor=(34, 34, 34), centroid_vox=(34, 34, 34), n_vox=30, esd_mm=8.0)]
    heat = 0.9 * h.render(shape, les, SP)
    cas = ts.cascade(prob, heat, SP)
    assert (cas["s"] >= 0.4).all()
    m = ts.cascade_mask(cas, prob, heat, SP)
    assert (m & big).sum() == big.sum()
    fb = m & ~big
    vox = np.argwhere(fb)
    r = np.sqrt(((vox - 34) ** 2).sum(1)).max()
    assert 3.0 <= r <= 4.5                               # ~ 2 sigma = ESD/2 = 4 mm
    assert np.sqrt(((np.argwhere(m) - 34) ** 2).sum(1)).min() == 0


def test_low_score_peaks_are_dropped_from_mask():
    shape = (30, 30, 30)
    prob = np.zeros(shape, np.float32)
    les = [h.Lesion(anchor=(15, 15, 15), centroid_vox=(15, 15, 15), n_vox=30, esd_mm=6.0)]
    heat = 0.7 * h.render(shape, les, SP)                # s = 0.35 < 0.4, no A support
    cas = ts.cascade(prob, heat, SP)
    assert not ts.cascade_mask(cas, prob, heat, SP).any()


def test_lesion_dice():
    gt = np.zeros((20, 20, 20), np.int32)
    gt[2:6, 2:6, 2:6] = 1
    gt[12:16, 12:16, 12:16] = 2
    mask = np.zeros(gt.shape, bool)
    mask[2:6, 2:6, 2:6] = True
    mask[2:6, 6:8, 2:6] = True                           # extra 32 voxels attached to lesion 1
    d = ts.lesion_dice(mask, gt, 2)
    assert np.isclose(d[0], 2 * 64 / (96 + 64)) and d[1] == 0.0


def test_c3_routes_large_to_a_and_small_to_c2():
    """Amendment C-2.1: A component ESD > 10 mm -> C1 score/A region; else C2 score/crop region."""
    shape = (60, 40, 40)
    prob = np.zeros(shape, np.float32)
    big, small = _ball(shape, (15, 20, 20), 7), _ball(shape, (45, 20, 20), 3)   # ESD ~14.6 / ~6.6 mm
    prob[big] = 0.95
    prob[small] = 0.8
    les = [h.Lesion(anchor=a, centroid_vox=a, n_vox=30, esd_mm=6.0) for a in [(15, 20, 20), (45, 20, 20)]]
    heat = h.render(shape, les, SP) * np.array(0.6, np.float32)
    cas = ts.cascade(prob, heat, SP)
    large = ts.c3_route(cas, prob, SP)
    idx = {tuple(p): i for i, p in enumerate(cas["points"])}
    ib, is_ = idx[(15, 20, 20)], idx[(45, 20, 20)]
    assert large[ib] and not large[is_]
    reg = np.zeros((3, 3, 3), bool)
    reg[1, 1, 1] = True
    out2 = {"points": cas["points"], "b": cas["b"], "a2": np.full(len(cas["b"]), 0.1),
            "regions": [reg.copy() for _ in cas["b"]], "starts": [np.asarray(p) - 1 for p in cas["points"]]}
    out2["a2"][is_] = 0.9
    s, mask, lg = ts.c3_scores_and_mask(cas, out2, prob, heat, SP, shape)
    assert np.isclose(s[ib], cas["s"][ib])                           # A route keeps C1's score
    assert np.isclose(s[is_], (cas["b"][is_] + 0.9) / 2)              # C2 route uses a2
    assert (mask & big).sum() == big.sum()                            # A region (p >= 0.5) pasted
    assert mask[45, 20, 20] and (mask & small).sum() == 1             # only the crop region for the small one


def test_c4_uses_c1_scores_and_c2_regions():
    """Amendment C-3.1: s = C1's s; kept peaks get the C2 crop region, else the heatmap fallback."""
    shape = (60, 40, 40)
    prob = np.zeros(shape, np.float32)
    prob[_ball(shape, (15, 20, 20), 4)] = 0.9
    les = [h.Lesion(anchor=a, centroid_vox=a, n_vox=30, esd_mm=6.0) for a in [(15, 20, 20), (45, 20, 20)]]
    heat = h.render(shape, les, SP) * np.array(0.7, np.float32)
    cas = ts.cascade(prob, heat, SP)
    idx = {tuple(p): i for i, p in enumerate(cas["points"])}
    i1, i2 = idx[(15, 20, 20)], idx[(45, 20, 20)]          # s = 0.8 (kept) / 0.35 (dropped)
    reg = np.zeros((3, 3, 3), bool)
    reg[1, 1, 1] = True
    out2 = {"points": cas["points"], "b": cas["b"], "a2": np.full(len(cas["b"]), 1.0),
            "regions": [reg.copy() for _ in cas["b"]], "starts": [np.asarray(p) - 1 for p in cas["points"]]}
    s, mask = ts.c4_scores_and_mask(cas, out2, heat, SP, shape)
    assert np.allclose(s, cas["s"])                         # high a2 does not change the score
    assert mask[15, 20, 20] and mask.sum() == 1             # C2 crop region only, not A's ball
    out2["regions"][i1] = np.zeros((3, 3, 3), bool)         # empty crop region -> heatmap fallback
    _, mask = ts.c4_scores_and_mask(cas, out2, heat, SP, shape)
    assert mask[15, 20, 20] and mask.sum() > 1 and not mask[45, 20, 20]
