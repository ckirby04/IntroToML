"""Map -> scored point predictions (PREREG §5). Both arms produce (points_vox (n,3) int, scores (n,)).

Model A: 26-connected components of p >= 0.05; one point per component at its max-probability
         voxel (first in C order on ties); score = that max.
Model B: local maxima over a box of half-width round(r_NMS / spacing) voxels per axis
         (r_NMS = 3.0 mm), value >= 0.05; plateaus -> one voxel (first in C order); score = value.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

CANDIDATE_FLOOR = 0.05
R_NMS_MM = 3.0
_S26 = np.ones((3, 3, 3), dtype=bool)


def points_from_prob(prob: np.ndarray, floor: float = CANDIDATE_FLOOR):
    """Model A. Returns (points, scores, labels, n_components)."""
    mask = prob >= floor
    labels, n = ndimage.label(mask, structure=_S26)
    if n == 0:
        return np.zeros((0, 3), int), np.zeros(0), labels, 0
    flat = prob.ravel()
    lab = labels.ravel()
    idx = np.flatnonzero(lab)
    # per component: index of max prob, ties -> smallest flat index (C order)
    order = np.lexsort((idx, -flat[idx], lab[idx]))     # by label, then -prob, then flat index
    first = np.ones(len(order), bool)
    first[1:] = lab[idx[order[1:]]] != lab[idx[order[:-1]]]
    best = idx[order[first]]
    pts = np.stack(np.unravel_index(best, prob.shape), axis=1)
    return pts.astype(int), flat[best].astype(np.float64), labels, n


def nms_halfwidth(spacing: Sequence[float], r_mm: float = R_NMS_MM) -> tuple[int, ...]:
    return tuple(max(int(round(r_mm / s)), 1) for s in spacing)


def points_from_heatmap(hm: np.ndarray, spacing: Sequence[float], floor: float = CANDIDATE_FLOOR,
                        r_mm: float = R_NMS_MM):
    """Model B. Returns (points, scores)."""
    hw = nms_halfwidth(spacing, r_mm)
    mx = ndimage.maximum_filter(hm, size=tuple(2 * h + 1 for h in hw), mode="constant", cval=-np.inf)
    peak = (hm == mx) & (hm >= floor)
    if not peak.any():
        return np.zeros((0, 3), int), np.zeros(0)
    lab, n = ndimage.label(peak, structure=_S26)          # plateaus: one voxel each
    flat = np.flatnonzero(lab)
    labs = lab.ravel()[flat]
    _, first = np.unique(labs, return_index=True)          # flat is sorted -> first in C order
    best = flat[first]
    pts = np.stack(np.unravel_index(best, hm.shape), axis=1)
    return pts.astype(int), hm.ravel()[best].astype(np.float64)
