import numpy as np
import pytest

from e01 import heatmap as h


def ball(shape, center, radius_mm, spacing):
    g = np.meshgrid(*[(np.arange(n) - c) * s for n, c, s in zip(shape, center, spacing)], indexing="ij")
    return sum(x ** 2 for x in g) <= radius_mm ** 2


def test_gaussian_centered_and_mm_scaled_with_anisotropic_spacing():
    sp = (0.5, 1.0, 2.0)
    shape = (80, 40, 20)
    m = ball(shape, (40, 20, 10), 4.0, sp)          # ESD ~ 8 mm -> sigma = 2 mm
    les = h.find_lesions(m, sp)
    assert len(les) == 1
    L = les[0]
    assert L.anchor == (40, 20, 10)                  # symmetric ball: anchor == center
    assert L.sigma_mm == pytest.approx(np.clip(L.esd_mm / 4, 1, 4))
    hm = h.render(shape, les, sp)
    assert hm[40, 20, 10] == 1.0 and np.unravel_index(hm.argmax(), shape) == (40, 20, 10)
    sig = L.sigma_mm
    for ax, s in enumerate(sp):
        for k in (1, 2):
            idx = [40, 20, 10]
            idx[ax] += k
            exp = np.exp(-((k * s) ** 2) / (2 * sig ** 2))
            assert hm[tuple(idx)] == pytest.approx(exp, rel=1e-5)
            idx[ax] -= 2 * k
            assert hm[tuple(idx)] == pytest.approx(exp, rel=1e-5)   # symmetric
    # truncation at 3 sigma (in mm) along the coarsest axis
    r = int(np.ceil(3 * sig / sp[2]))
    assert hm[40, 20, 10 + r + 1] == 0.0


@pytest.mark.parametrize("vox", [[(5, 5, 5)], [(5, 5, 5), (5, 5, 6)]])
def test_tiny_lesions_get_a_peak(vox):
    m = np.zeros((12, 12, 12), bool)
    for v in vox:
        m[v] = True
    les = h.find_lesions(m, (1.0, 1.0, 1.0))
    assert len(les) == 1 and les[0].sigma_mm == 1.0
    assert m[les[0].anchor]
    for s in [(1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.125, 0.125, 0.125)]:
        hm = h.render(m.shape, les, (1.0, 1.0, 1.0), s)
        assert int((hm == 1.0).sum()) == 1


def test_hollow_shell_anchor_is_inside_lesion():
    """Amendment 1: a ring-enhancing (hollow) lesion's centroid lies in the hole; the anchor
    must lie on the lesion and within the 2 mm match tolerance of it."""
    sp = (1.0, 1.0, 1.0)
    shape = (40, 40, 40)
    shell = ball(shape, (20, 20, 20), 9.0, sp) & ~ball(shape, (20, 20, 20), 6.0, sp)
    les = h.find_lesions(shell, sp)
    assert len(les) == 1
    L = les[0]
    cen = np.asarray(L.centroid_vox)
    vox = np.argwhere(shell)
    d_centroid = np.sqrt((((vox - cen) * sp) ** 2).sum(1)).min()
    assert d_centroid > 2.0                          # raw centroid would fail the 2 mm rule
    assert shell[L.anchor]                           # anchor is a lesion voxel
    hm = h.render(shape, les, sp)
    assert hm[L.anchor] == 1.0 and int((hm == 1.0).sum()) == 1


def test_anchor_ties_go_to_centroid_then_c_order():
    m = np.zeros((10, 10, 10), bool)
    m[2:5, 2:5, 2:8] = True                          # elongated box: EDT ties along the long axis
    (L,) = h.find_lesions(m, (1.0, 1.0, 1.0))
    assert L.anchor == (3, 3, 4)                    # centroid z = 4.5 -> tie 4 vs 5 -> C order -> 4


def test_two_lesions_overlapping_gaussians_take_max():
    sp = (1.0, 1.0, 1.0)
    m = ball((40, 40, 40), (15, 20, 20), 3.0, sp) | ball((40, 40, 40), (24, 20, 20), 3.0, sp)
    les = h.find_lesions(m, sp)
    assert len(les) == 2
    hm = h.render(m.shape, les, sp)
    assert int((hm == 1.0).sum()) == 2
    assert hm.max() == 1.0


def test_empty_mask():
    assert h.find_lesions(np.zeros((8, 8, 8), bool), (1, 1, 1)) == []
    assert h.render((8, 8, 8), [], (1, 1, 1)).max() == 0.0
