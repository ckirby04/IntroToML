"""Fixed E02 parameters (PREREG §3, §5, §7, §8). Every pixel value comes from
results\\train_stats.json (training images only); tests/test_params.py checks that they agree.
Nothing here may be changed without a dated PREREG amendment.
"""
from __future__ import annotations

import math

K = 4.085432402224472                     # px per E01 mm (median training ESD 17.33 px / 4.243 mm)

# Model B target (PREREG §4.1)
SIGMA_FRAC = 0.25                         # sigma = ESD / 4
SIGMA_MIN_PX = max(1.0, K)                # 4.085
SIGMA_MAX_PX = 4.0 * K                    # 16.34
TRUNC_SIGMAS = 3.0

# Points (PREREG §6)
CANDIDATE_FLOOR = 0.05
NMS_HALFWIDTH_PX = 4                      # collision-matched to E01 (owner decision 2026-11-16)

# Matching (PREREG §7)
HIT_TOL_PX = 2.0 * K                      # 8.17

# Two-stage (PREREG §7.1)
TAU = {"A": 0.5, "B": 0.3, "C1": 0.4, "C2": 0.4, "C4": 0.4}
MASK_P = 0.5
ASSIGN_PX = HIT_TOL_PX
FALLBACK_REL = math.exp(-2.0)
FALLBACK_RADIUS_PX = 8.0 * K              # 32.68

# C2 crops (PREREG §7.1)
C2_CROP = 384
C2_JITTER_PX = (4.0 * K, 10.0 * K)        # 16.34 .. 40.85
C2_NEG_MIN_DIST_PX = 10.0 * K             # 40.85
C2_POS_PER_IMAGE = 2
C2_NEG_PER_POS = 0.5
C2_BRIGHT_FRAC = 0.005
C2_NEG_RETRIES = 100
C2_SEED = 20261117

# Size bins on ESD in px (PREREG §3); an object exactly on an edge goes up
BIN_EDGES_PX = (13.63425735166143, 29.44615645486938)
BINS = (("small", 0.0, BIN_EDGES_PX[0]), ("medium", BIN_EDGES_PX[0], BIN_EDGES_PX[1]),
        ("large", BIN_EDGES_PX[1], float("inf")))
BIN_NAMES = tuple(b[0] for b in BINS)

# FP targets per image (PREREG §5); "1 FP" = the second
FP_TARGETS = (14.951551739558179, 29.903103479116357, 59.806206958232714, 89.70931043734907)
ONE_FP = FP_TARGETS[1]

# Uncertainty (PREREG §8)
N_BOOT = 2000
BOOT_SEED = 20261116

# Descriptives (PREREG §10): confluence tertile edges, cells per training image
CONFLUENCE_EDGES = (143.0, 323.0)

# Replication (PREREG §2)
LINE_MIN_BIN_CELLS = 200
H1_MARGIN = 0.05
H5_MARGIN = -0.02


def esd_px(area_px):
    import numpy as np

    return np.sqrt(4.0 * np.asarray(area_px, float) / np.pi)


def size_bin(esd):
    """'small' | 'medium' | 'large' (upper bin on an edge)."""
    if esd < BIN_EDGES_PX[0]:
        return "small"
    return "medium" if esd < BIN_EDGES_PX[1] else "large"


def sigma_px(esd: float) -> float:
    return float(min(max(SIGMA_FRAC * esd, SIGMA_MIN_PX), SIGMA_MAX_PX))
