import numpy as np

from e01 import decode as d
from e01 import heatmap as h


def test_decoder_recovers_planted_peaks():
    sp = (1.0, 1.0, 1.0)
    anchors = [(10, 10, 10), (30, 12, 25), (12, 35, 8), (20, 20, 30)]
    les = [h.Lesion(anchor=a, centroid_vox=a, n_vox=30, esd_mm=e) for a, e in
           zip(anchors, [2.0, 6.0, 12.0, 20.0])]
    hm = h.render((40, 45, 40), les, sp)
    pts, sc = d.points_from_heatmap(hm, sp)
    assert sorted(map(tuple, pts)) == sorted(anchors)
    assert np.all(sc == 1.0)


def test_decoder_with_scores_and_noise():
    rng = np.random.default_rng(0)
    sp = (0.9, 0.9, 1.5)
    anchors = [(10, 10, 10), (30, 30, 15)]
    les = [h.Lesion(anchor=a, centroid_vox=a, n_vox=30, esd_mm=8.0) for a in anchors]
    hm = 0.7 * h.render((40, 40, 25), les, sp) + rng.uniform(0, 0.02, (40, 40, 25)).astype(np.float32)
    pts, sc = d.points_from_heatmap(hm, sp)
    strong = {tuple(p) for p, s in zip(pts, sc) if s > 0.5}
    assert strong == set(anchors)
    assert np.all(sc[[tuple(p) not in strong for p in pts]] < 0.05 + 0.02)


def test_nms_merges_peaks_closer_than_radius_but_not_farther():
    sp = (1.0, 1.0, 1.0)
    hm = np.zeros((30, 30, 30), np.float32)
    hm[10, 10, 10], hm[12, 10, 10] = 0.9, 0.8        # 2 mm apart -> one point
    hm[10, 10, 20], hm[10, 10, 24] = 0.9, 0.8        # 4 mm apart -> two points
    pts, sc = d.points_from_heatmap(hm, sp)
    got = {tuple(p): s for p, s in zip(pts, sc)}
    assert (10, 10, 10) in got and (12, 10, 10) not in got
    assert (10, 10, 20) in got and (10, 10, 24) in got


def test_nms_halfwidth_in_mm():
    assert d.nms_halfwidth((1.0, 1.0, 1.0)) == (3, 3, 3)
    assert d.nms_halfwidth((0.5, 1.0, 2.0)) == (6, 3, 2)


def test_plateau_gives_one_point_first_in_c_order():
    hm = np.zeros((10, 10, 10), np.float32)
    hm[4:6, 4, 4] = 0.6
    pts, sc = d.points_from_heatmap(hm, (1, 1, 1))
    assert [tuple(p) for p in pts] == [(4, 4, 4)]


def test_model_a_points_one_per_component_at_argmax():
    p = np.zeros((20, 20, 20), np.float32)
    p[2:5, 2:5, 2:5] = 0.3
    p[3, 3, 4] = 0.9
    p[10:12, 10, 10] = 0.04                          # below floor: no component
    p[15, 15, 15] = 0.2
    p[15, 15, 16] = 0.2                              # tie -> first in C order
    pts, sc, labels, n = d.points_from_prob(p)
    assert n == 2
    got = sorted(zip(map(tuple, pts), sc))
    assert got == [((3, 3, 4), np.float32(0.9)), ((15, 15, 15), np.float32(0.2))]
