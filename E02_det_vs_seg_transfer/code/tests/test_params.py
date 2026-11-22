"""params.py must equal PREREG §5, i.e. results\\train_stats.json (training images only)."""
import json

import pytest

from e02 import common as c
from e02 import params as P


@pytest.fixture(scope="module")
def stats():
    return json.loads((c.RESULTS / "train_stats.json").read_text())


def test_params_equal_train_stats(stats):
    p = stats["params"]
    assert P.K == pytest.approx(p["k_px_per_mm"], abs=0)
    assert [P.SIGMA_MIN_PX, P.SIGMA_MAX_PX] == pytest.approx(p["sigma_px"], abs=0)
    assert P.NMS_HALFWIDTH_PX == p["nms_halfwidth_px"] == 4
    assert P.HIT_TOL_PX == pytest.approx(p["hit_tol_px"], abs=0)
    assert P.FALLBACK_RADIUS_PX == pytest.approx(p["c1_fallback_radius_px"], abs=0)
    assert P.C2_CROP == p["c2_crop_px"] == 384
    assert list(P.BIN_EDGES_PX) == pytest.approx(p["size_bin_edges_px"], abs=0)
    assert list(P.FP_TARGETS) == pytest.approx(p["fp_targets_per_image"], abs=0)


def test_k_rule(stats):
    med = stats["train"]["esd_px_quantiles"]["0.5"]
    assert P.K == pytest.approx(med / c.E01_TRAIN_MEDIAN_ESD_MM, rel=1e-12)


def test_prereg_values_rounded():
    assert round(P.HIT_TOL_PX, 2) == 8.17 and round(P.FALLBACK_RADIUS_PX, 1) == 32.7
    assert [round(x, 2) for x in P.BIN_EDGES_PX] == [13.63, 29.45]
    assert [round(x, 2) for x in P.FP_TARGETS] == [14.95, 29.90, 59.81, 89.71]
    assert round(P.SIGMA_MIN_PX, 2) == 4.09 and round(P.SIGMA_MAX_PX, 2) == 16.34
    assert P.TAU == {"A": 0.5, "B": 0.3, "C1": 0.4, "C2": 0.4, "C4": 0.4}
    assert P.BOOT_SEED == 20261116 and P.C2_SEED == 20261117 and c.SEED == 1600


def test_prereg_hash_recorded():
    assert c.assert_prereg()
