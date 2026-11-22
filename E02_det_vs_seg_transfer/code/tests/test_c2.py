import numpy as np

from e02 import c2
from e02 import params as P


def test_jitter_halfwidth_clip():
    assert c2.jitter_halfwidth(10.0) == round(P.C2_JITTER_PX[0]) == 16
    assert c2.jitter_halfwidth(1000.0) == round(P.C2_JITTER_PX[1]) == 41


def test_negative_plan_totals_and_halves():
    plan = c2.negative_plan([10, 0, 30, 60], n_pos=8, seed=3)
    assert plan.sum() == 4 and plan[1].sum() == 0
    assert plan[:, 0].sum() == 2 and plan[:, 1].sum() == 2


def test_sample_positive_and_negative_rules():
    rng = np.random.default_rng(0)
    inst = np.zeros((300, 300), np.int16)
    inst[20:40, 20:40] = 1
    inst[20:40, 40:60] = 2
    inst[100:110, 100:110] = 3
    img = np.full((300, 300), 100, np.uint8)
    img[230:260, 230:260] = 255                    # brightest >= 0.5% of pixels, far from every cell
    rows = c2.sample_image_crops(img, inst, n_rand=3, n_bright=2, rng=rng)
    pos = [r for r in rows if r["kind"] == "pos"]
    assert len(pos) == 2 and len({r["cell_id"] for r in pos}) == 2
    for r in pos:
        assert all(abs(o) <= 16 for o in r["offset"])
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(inst == 0)
    for r in rows:
        if r["kind"] in ("neg_rand", "neg_bright"):
            assert dist[r["center"]] >= P.C2_NEG_MIN_DIST_PX
        if r["kind"] == "neg_bright":
            assert img[r["center"]] == 255


def test_dense_image_negatives_are_skipped():
    rng = np.random.default_rng(0)
    inst = np.ones((100, 100), np.int16)           # every pixel is a cell
    img = np.zeros((100, 100), np.uint8)
    rows = c2.sample_image_crops(img, inst, n_rand=1, n_bright=1, rng=rng)
    assert [r["kind"] for r in rows] == ["pos", "neg_rand_skipped", "neg_bright_skipped"]
