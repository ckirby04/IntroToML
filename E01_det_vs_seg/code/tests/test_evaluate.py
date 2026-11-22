import numpy as np
import pytest

from e01 import evaluate as ev
from jannus.evaluation.detection.matching import extract_components


def gt_two_lesions(sp=(1.0, 1.0, 1.0)):
    m = np.zeros((40, 40, 40), bool)
    m[10:13, 10:13, 10:13] = True                    # lesion 1
    m[30, 30, 30] = True                             # lesion 2 (1 voxel)
    return extract_components(m, sp, connectivity=26)


def test_point_rule_tolerance_and_fp():
    gt = gt_two_lesions()
    pts = np.array([[11, 11, 11],                    # inside L1
                    [30, 30, 32],                    # 2.0 mm from L2 -> hit
                    [30, 33, 30],                    # 3.0 mm from L2 -> miss -> FP
                    [12, 14, 12],                    # 2.0 mm from L1 -> hit (redundant)
                    [0, 0, 0]])                      # FP
    sc = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    m = ev.match_points(pts, sc, gt)
    assert m.pred_hits == [(1,), (2,), (), (1,), ()]
    assert list(m.pred_is_fp) == [False, False, True, False, True]
    assert list(m.gt_score) == [0.9, 0.8]
    assert m.n_redundant() == 1
    assert m.n_fp(tau=0.6) == 1 and m.n_fp() == 2


def test_point_rule_uses_physical_spacing():
    gt = gt_two_lesions(sp=(1.0, 1.0, 3.0))
    m = ev.match_points(np.array([[30, 30, 31]]), np.array([1.0]), gt)   # 3 mm in z -> miss
    assert m.pred_hits == [()]
    m = ev.match_points(np.array([[32, 30, 30]]), np.array([1.0]), gt)   # 2 mm in x -> hit
    assert m.pred_hits == [(2,)]


def test_point_can_hit_two_close_lesions():
    m = np.zeros((20, 20, 20), bool)
    m[5, 5, 5] = m[5, 5, 8] = True
    gt = extract_components(m, (1, 1, 1))
    r = ev.match_points(np.array([[5, 5, 6]]), np.array([1.0]), gt)
    assert r.pred_hits == [(1, 2)]


def test_size_bins_half_open_upper():
    from jannus.evaluation.detection.geometry import assign_size_bin

    assert assign_size_bin(2.999, ev.BINS) == "<3"
    assert assign_size_bin(3.0, ev.BINS) == "3-10"
    assert assign_size_bin(10.0, ev.BINS) == ">10"


def test_merged_components_and_centroid_outside():
    m = np.zeros((30, 30, 30), bool)
    m[5, 5, 5] = m[5, 5, 9] = True
    gt = extract_components(m, (1, 1, 1))
    pred = np.zeros_like(m, dtype=np.int32)
    pred[5, 5, 4:11] = 1                             # one component over both lesions
    pred[20, 20, 20] = 2
    assert ev.merged_components(pred, gt) == 1
    # hollow shell: centroid > 2 mm from lesion
    g = np.meshgrid(*[np.arange(30) - 15] * 3, indexing="ij")
    r2 = sum(x ** 2 for x in g)
    shell = (r2 <= 81) & (r2 > 36)
    assert ev.centroid_outside(extract_components(shell, (1, 1, 1))) == 1
    assert ev.centroid_outside(gt) == 0


def test_froc_cpm_and_paired_bootstrap_small():
    gt = gt_two_lesions()
    res = []
    for i in range(6):
        pts = np.array([[11, 11, 11], [0, 0, i]])
        res.append(ev.case_result(f"c{i}", f"p{i // 2}", pts, np.array([0.9, 0.5]), gt))
    data = ev.to_data(res)
    s = ev.stats(data, fixed_tau=0.5)
    assert s["sens@1.0"] == pytest.approx(0.5)       # L1 always found, L2 never
    assert s["fixed.fp_per_scan"] == pytest.approx(1.0)
    b = ev.paired_bootstrap({"A": data, "B": data}, {"A": 0.5, "B": 0.5}, n=50)
    assert b["n_patients"] == 3
    assert b["B_minus_A"]["cpm"]["point"] == 0.0
