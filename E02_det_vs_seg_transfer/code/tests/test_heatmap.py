import numpy as np

from e02 import heatmap as h
from e02 import params as P


def test_anchor_is_interior_max_edt_and_tie_rules():
    comp = np.zeros((7, 9), bool)
    comp[1:6, 1:8] = True                         # 5 x 7 rectangle: max EDT on the middle row
    a, cen = h.interior_anchor(comp)
    assert a == (3, 4) and cen == (3.0, 4.0)
    ring = np.zeros((11, 11), bool)               # ring: centroid lies outside the instance
    ring[2:9, 2:9] = True
    ring[4:7, 4:7] = False
    a, cen = h.interior_anchor(ring)
    assert ring[a] and not ring[tuple(int(round(v)) for v in cen)]


def test_touching_instances_get_one_peak_each():
    inst = np.zeros((40, 60), np.int16)
    inst[10:30, 10:30] = 1
    inst[10:30, 30:50] = 2                        # touches 1: one connected component, two cells
    hm = h.heatmaps(inst, None)[0]
    cells = h.find_cells(inst)
    assert len(cells) == 2
    assert (hm == 1.0).sum() == 2
    for cl in cells:
        assert hm[cl.anchor] == 1.0 and inst[cl.anchor] == cl.id


def test_split_instance_is_one_cell():
    inst = np.zeros((30, 30), np.int16)
    inst[2:8, 2:8] = 1
    inst[20:28, 20:28] = 1                        # same id, two pieces (overlap rule / augmentation)
    cells = h.find_cells(inst)
    assert len(cells) == 1 and cells[0].n_px == 36 + 64
    assert inst[cells[0].anchor] == 1


def test_sigma_clip_and_truncation():
    small = h.Cell(id=1, anchor=(50, 50), centroid=(50.0, 50.0), n_px=10)
    big = h.Cell(id=2, anchor=(50, 50), centroid=(50.0, 50.0), n_px=40000)
    assert small.sigma == P.SIGMA_MIN_PX and big.sigma == P.SIGMA_MAX_PX
    mid = h.Cell(id=3, anchor=(0, 0), centroid=(0.0, 0.0), n_px=int(np.pi * 20 ** 2))   # ESD ~ 40 px
    assert abs(mid.sigma - mid.esd / 4) < 1e-9
    hm = h.render((101, 101), [small])
    r = np.hypot(*np.meshgrid(np.arange(101) - 50, np.arange(101) - 50, indexing="ij"))
    assert (hm[r > 3 * small.sigma + 1e-6] == 0).all()
    assert np.isclose(hm[50, 54], np.exp(-16 / (2 * small.sigma ** 2)), atol=1e-6)


def test_ds_scales_keep_every_peak():
    inst = np.zeros((64, 64), np.int16)
    inst[4:8, 4:8] = 1                            # tiny cell
    inst[30:50, 30:50] = 2
    scales = [(1.0, 1.0), (0.5, 0.5), (0.25, 0.25), (0.125, 0.125)]
    maps = h.heatmaps(inst, scales)
    for m, s in zip(maps, scales):
        assert m.shape == h.scaled_shape(inst.shape, s)
        assert (m == 1.0).sum() == 2 or (s[0] <= 0.125 and (m == 1.0).sum() >= 1)


def test_empty_and_anchors_and_areas():
    assert h.find_cells(np.zeros((5, 5), np.int16)) == []
    inst = np.zeros((10, 10), np.int16)
    inst[1:4, 1:4] = 1
    inst[6:9, 6:9] = 3                            # id 2 absent
    anc, area = h.anchors_and_areas(inst)
    assert area.tolist() == [9, 0, 9] and anc[0].tolist() == [2, 2] and anc[1].tolist() == [-1, -1]


def test_matches_train_stats_anchor_rule():
    """train_stats.py fixed the NMS width with its own anchor code: both must agree."""
    from e02.train_stats import anchors_and_areas as ts_anchors

    rng = np.random.default_rng(0)
    inst = np.zeros((80, 80), np.int64)
    for i in range(1, 15):
        y, x = rng.integers(0, 70, 2)
        inst[y:y + rng.integers(3, 12), x:x + rng.integers(3, 12)] = i
    ids = np.unique(inst[inst > 0])
    remap = np.zeros(inst.max() + 1, np.int64)
    remap[ids] = np.arange(1, len(ids) + 1)
    inst = remap[inst]
    a1, s1 = ts_anchors(inst)
    a2, s2 = h.anchors_and_areas(inst)
    assert np.array_equal(a1, a2) and np.array_equal(s1, s2)
