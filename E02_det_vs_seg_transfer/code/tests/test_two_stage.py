import numpy as np

from e02 import params as P
from e02 import two_stage as ts
from e02.c2_predict import load_c2, pack_regions


def test_assignment_contains_nearest_ties():
    comp = np.zeros((30, 30), np.int64)
    comp[5:10, 5:10] = 1
    comp[5:10, 21:26] = 2
    cmax = np.array([0.6, 0.9])
    pts = np.array([[7, 7],            # inside 1
                    [7, 15],           # 6 from 1 and 6 from 2: tie -> higher component max (2)
                    [25, 25],          # far from both
                    [7, 12]])          # 3 from 1, 9 from 2 -> 1
    k = ts.assign_peaks(pts, comp, cmax)
    assert k.tolist() == [1, 2, 0, 1]


def test_cascade_scores_and_unassigned():
    pa = np.zeros((40, 40))
    pa[5:15, 5:15] = 0.8
    hb = np.zeros((40, 40))
    hb[10, 10] = 0.6                   # inside A's component
    hb[30, 30] = 0.4                   # no A component nearby
    cas = ts.cascade(pa, hb)
    s = dict(zip(map(tuple, cas["points"].tolist()), cas["s"]))
    assert np.isclose(s[(10, 10)], 0.7) and np.isclose(s[(30, 30)], 0.2)


def test_cascade_mask_uses_a_region_and_fallback():
    pa = np.zeros((80, 80))
    pa[5:15, 5:15] = 0.9               # A region for peak 1
    pa[40:45, 40:45] = 0.3             # component without p >= 0.5 -> fallback for peak 2
    hb = np.zeros((80, 80))
    yy, xx = np.mgrid[:80, :80]
    hb = np.maximum(hb, 0.9 * np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 4.0 ** 2)))
    hb = np.maximum(hb, 0.9 * np.exp(-((yy - 42) ** 2 + (xx - 42) ** 2) / (2 * 4.0 ** 2)))
    cas = ts.cascade(pa, hb)
    m = ts.cascade_mask(cas, pa, hb)
    assert m[5:15, 5:15].all() and not m[16:30, 16:30].any()
    fb = ts.fallback_region(hb, (42, 42))
    assert m[fb].all() and fb[42, 42]
    r = np.hypot(yy - 42, xx - 42)
    assert (r[fb] <= 2 * 4.0 + 1).all()        # e^-2 level of a sigma-4 Gaussian: radius 2 sigma


def test_fallback_clipped_to_disc():
    hb = np.full((100, 100), 0.5)
    fb = ts.fallback_region(hb, (50, 50))
    yy, xx = np.mgrid[:100, :100]
    assert fb.sum() == int(((yy - 50) ** 2 + (xx - 50) ** 2 <= P.FALLBACK_RADIUS_PX ** 2).sum())


def test_crop_paste_roundtrip_and_padding():
    img = np.arange(30 * 40).reshape(30, 40)
    cr = ts.extract_crop(img, (2, 3), size=8, fill=-1)
    assert cr[0, 0] == -1 and cr[2, 1] == img[0, 0] and cr[4, 4] == img[2, 3]
    full = np.zeros((30, 40), bool)
    ts.paste_or(full, np.ones((8, 8), bool), ts.crop_start((2, 3), 8))
    assert full.sum() == 6 * 7 and full[:6, :7].all()


def test_stage2_from_crop_centre():
    p2 = np.zeros((P.C2_CROP, P.C2_CROP))
    c = P.C2_CROP // 2
    p2[c - 5:c + 5, c - 5:c + 5] = 0.7
    p2[0:5, 0:5] = 0.9
    a2, reg, ok = ts.stage2_from_crop(p2)
    assert ok and a2 == 0.7 and reg.sum() == 100 and reg[c, c]
    a2, reg, ok = ts.stage2_from_crop(np.zeros((P.C2_CROP, P.C2_CROP)))
    assert not ok and a2 == 0.0 and not reg.any()


def test_region_pack_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    regs = [np.zeros((16, 16), bool), rng.random((16, 16)) > 0.7, np.zeros((16, 16), bool)]
    regs[2][3:5, 7:12] = True
    box, bits, off = pack_regions(regs)
    np.savez(tmp_path / "x.npz", points=np.zeros((3, 2), int), b=np.zeros(3), a2=np.zeros(3),
             assigned=np.zeros(3, bool), starts=np.zeros((3, 2), int), rbox=box, rbits=bits, roff=off)
    back = load_c2(tmp_path / "x.npz", size=16)["regions"]
    assert all(np.array_equal(a, b) for a, b in zip(regs, back))


def test_c4_equals_c1_scores_and_uses_c2_regions():
    hb = np.zeros((50, 50))
    hb[10, 10], hb[40, 40] = 0.8, 0.8
    cas = {"points": np.array([[10, 10], [40, 40]]), "s": np.array([0.6, 0.2])}
    reg = np.zeros((2, P.C2_CROP, P.C2_CROP), bool)
    reg[0, P.C2_CROP // 2, P.C2_CROP // 2] = True
    out2 = {"points": cas["points"], "b": np.array([0.8, 0.8]), "a2": np.array([0.0, 0.9]),
            "starts": np.array([ts.crop_start(q) for q in cas["points"]]), "regions": reg}
    s4, m4 = ts.c4_scores_and_mask(cas, out2, hb)
    assert np.array_equal(s4, cas["s"]) and m4.sum() == 1 and m4[10, 10]
    s2, m2 = ts.c2_scores_and_mask(out2, hb)
    assert np.allclose(s2, [0.4, 0.85]) and m2[10, 10] and m2[40, 40]   # peak 2: empty region -> fallback
