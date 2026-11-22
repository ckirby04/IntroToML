import numpy as np
import pytest
from scipy import ndimage

from e02 import evaluate as ev
from e02 import params as P


def _gt():
    inst = np.zeros((40, 40), np.int64)
    inst[5:15, 5:15] = 1
    inst[5:15, 15:25] = 2                         # touches 1
    inst[30:33, 30:33] = 3
    lab = inst > 0
    lab[5:15, 14] = False                         # gap
    return ev.make_gt(inst, lab)


def test_one_hit_inside_and_nearest_and_tie():
    gt = _gt()
    pts = np.array([[10, 14],                     # inside 1 (on its border pixel) -> 1 only
                    [10, 26],                     # 2 px right of 2 -> 2
                    [20, 15],                     # below the 1|2 border: 1 at (14,14) and 2 at (14,15),
                                                  # distance 6.08 vs 6.0 -> 2
                    [36, 36],                     # 5.66 from 3 -> 3
                    [0, 39]])                     # far -> FP
    m = ev.match_points(pts, np.array([0.9, 0.8, 0.7, 0.6, 0.5]), gt)
    assert m.pred_hits == [(1,), (2,), (2,), (3,), ()]
    assert m.pred_is_fp.tolist() == [False, False, False, False, True]
    inst = np.zeros((9, 9), np.int64)
    inst[4, 1] = 2
    inst[4, 7] = 1
    tie = ev.match_points(np.array([[4, 4]]), np.array([1.0]), ev.make_gt(inst, inst > 0))
    assert tie.pred_hits == [(1,)]                # equal distance 3 -> lower id


def test_multi_hit_rule_and_redundant():
    gt = _gt()
    m = ev.match_points(np.array([[10, 14]]), np.array([0.9]), gt, rule="multi")
    assert m.pred_hits == [(1, 2)]
    m = ev.match_points(np.array([[8, 8], [9, 9]]), np.array([0.9, 0.3]), gt)
    assert m.n_redundant() == 1 and m.n_fp() == 0 and np.isfinite(m.gt_score[0])


def test_hit_tolerance_boundary():
    inst = np.zeros((30, 30), np.int64)
    inst[0, 0] = 1
    near = ev.match_points(np.array([[0, 8]]), np.array([1.0]), ev.make_gt(inst, inst > 0))
    far = ev.match_points(np.array([[0, 9]]), np.array([1.0]), ev.make_gt(inst, inst > 0))
    assert P.HIT_TOL_PX > 8 and near.pred_hits == [(1,)] and far.pred_hits == [()]


def test_bins_upper_on_edge():
    assert P.size_bin(P.BIN_EDGES_PX[0]) == "medium" and P.size_bin(P.BIN_EDGES_PX[1]) == "large"
    assert P.size_bin(P.BIN_EDGES_PX[0] - 1e-9) == "small"


def test_object_dice_matches_brute_force():
    rng = np.random.default_rng(1)
    gt = _gt()
    for _ in range(5):
        mask = ndimage.binary_dilation(rng.random(gt.inst.shape) > 0.97, iterations=2)
        fast = ev.object_dice(mask, gt)
        pl, _ = ndimage.label(mask, structure=ev.S8)
        for g in range(1, gt.n + 1):
            L = (gt.inst == g) & gt.label
            ids = np.unique(pl[L])
            ids = ids[ids > 0]
            want = 0.0 if not len(ids) else 2 * (np.isin(pl, ids) & L).sum() / (np.isin(pl, ids).sum() + L.sum())
            assert fast[g - 1] == pytest.approx(want)


def test_dice_and_merged():
    gt = _gt()
    assert ev.dice(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0
    assert ev.dice(gt.label, gt.label) == 1.0
    pl, _ = ndimage.label(gt.inst > 0, structure=ev.S8)
    assert ev.merged_components(pl, gt) == 1      # the 1|2 blob covers two cells


def test_make_gt_rejects_non_compact_ids():
    inst = np.zeros((4, 4), np.int64)
    inst[0, 0] = 2
    with pytest.raises(AssertionError):
        ev.make_gt(inst, inst > 0)


def _toy_data(shift):
    rng = np.random.default_rng(0)
    res = []
    for i in range(12):
        gt = _gt()
        pts = np.array([[10, 10], [10, 20], [31, 31], [0, 39]])
        sc = np.clip(rng.random(4) + shift, 0, 1)
        res.append(ev.CaseResult(f"c{i}", f"w{i % 4}", ev.match_points(pts, sc, gt), gt.bins))
    return ev.to_data(res)


def test_paired_bootstrap_contrasts():
    a, b = _toy_data(0.0), _toy_data(0.2)
    seg = ev.SegData(np.full(12, 0.8), np.full(36, 0.5), np.repeat(np.arange(12), 3),
                     np.array(["small", "medium", "large"] * 12, dtype=object))
    out = ev.paired_bootstrap({"A": {"det": a, "seg": seg, "tau": 0.5}, "B": {"det": b, "seg": None, "tau": 0.3}},
                              [("B", "A")], n=50, seed=1)
    assert out["n_wells"] == 4
    ci = out["contrasts"]["B-A"]["cpm"]
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1]
    assert out["arms"]["A"]["dice_image"]["point"] == pytest.approx(0.8)


def test_verdict_rules():
    lines8 = {f"L{i}": 0.03 for i in range(8)}
    v = ev.verdict("H4", {"point": 0.03, "ci95": [0.01, 0.05, 2000]}, lines8)
    assert v["replicated"] and v["lines_in_direction"] == 8
    two_bad = dict(lines8, L0=-0.01, L1=-0.02)
    assert not ev.verdict("H4", {"point": 0.03, "ci95": [0.01, 0.05, 2000]}, two_bad)["replicated"]
    assert not ev.verdict("H4", {"point": 0.03, "ci95": [-0.01, 0.05, 2000]}, lines8)["replicated"]
    # H1: CI must include 0 and lie within +-0.05
    assert ev.verdict("H1", {"point": 0.0, "ci95": [-0.04, 0.04, 2000]}, {k: 0.0 for k in lines8})["replicated"]
    assert not ev.verdict("H1", {"point": 0.0, "ci95": [-0.06, 0.04, 2000]}, {k: 0.0 for k in lines8})["replicated"]
    # H5: lower bound > -0.02
    assert ev.verdict("H5", {"point": 0.0, "ci95": [-0.01, 0.01, 2000]}, {k: -0.01 for k in lines8})["replicated"]
    # H2: lines with < 200 large cells do not take part; 6 taking part, 1 against -> assessed, ok
    counts = {f"L{i}": (500 if i < 6 else 10) for i in range(8)}
    d = {f"L{i}": (0.1 if i else -0.1) for i in range(8)}
    v = ev.verdict("H2", {"point": 0.1, "ci95": [0.05, 0.2, 2000]}, d, counts)
    assert v["lines_taking_part"] == 6 and v["condition_B"] and v["replicated"]
    counts4 = {f"L{i}": (500 if i < 4 else 10) for i in range(8)}
    v = ev.verdict("H2", {"point": 0.1, "ci95": [0.05, 0.2, 2000]}, d, counts4)
    assert v["condition_B_state"] == "not assessable" and not v["replicated"]
