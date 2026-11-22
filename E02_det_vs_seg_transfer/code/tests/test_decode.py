import numpy as np

from e02 import decode as d


def test_a_points_one_per_8cc_at_max():
    p = np.zeros((10, 10))
    p[1, 1], p[2, 2] = 0.3, 0.9                   # diagonal neighbours: one 8-connected component
    p[7, 7] = 0.06
    p[5, 0] = 0.04                                # below the floor
    pts, sc, lab, n = d.points_from_prob(p)
    assert n == 2 and pts.tolist() == [[2, 2], [7, 7]] and np.allclose(sc, [0.9, 0.06])


def test_a_ties_first_in_c_order():
    p = np.zeros((5, 5))
    p[1, 3] = p[2, 1] = p[1, 2] = 0.7
    p[2, 2] = 0.7
    pts, sc, _, n = d.points_from_prob(p)
    assert n == 1 and pts.tolist() == [[1, 2]]


def test_b_nms_halfwidth_4px():
    hm = np.zeros((40, 40))
    hm[10, 10], hm[10, 15] = 0.9, 0.8             # Chebyshev distance 5 > 4: both kept
    hm[30, 10], hm[30, 14] = 0.9, 0.8             # distance 4: the weaker one is suppressed
    pts, sc = d.points_from_heatmap(hm)
    assert sorted(map(tuple, pts.tolist())) == [(10, 10), (10, 15), (30, 10)]


def test_b_plateau_one_point_and_floor():
    hm = np.zeros((10, 10))
    hm[4, 4:6] = 0.5                              # plateau of two pixels
    hm[8, 8] = 0.04                               # below the floor
    pts, sc = d.points_from_heatmap(hm)
    assert pts.tolist() == [[4, 4]] and sc.tolist() == [0.5]
    assert d.points_from_heatmap(np.zeros((5, 5)))[0].shape == (0, 2)
