"""Map -> scored point predictions (PREREG §6). Both arms give (points (n, 2) int, scores (n,)).

Model A: 8-connected components of p >= 0.05; one point per component at its max-probability
         pixel (first in C order on ties); score = that max.
Model B: local maxima over a box of half-width 4 px, value >= 0.05; plateaus -> one pixel (first
         in C order); score = the peak value.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from e02 import params as P

S8 = np.ones((3, 3), dtype=bool)


def points_from_prob(prob: np.ndarray, floor: float = P.CANDIDATE_FLOOR):
    """Model A. Returns (points, scores, labels, n_components)."""
    assert prob.ndim == 2
    labels, n = ndimage.label(prob >= floor, structure=S8)
    if n == 0:
        return np.zeros((0, 2), int), np.zeros(0), labels, 0
    flat = prob.ravel()
    lab = labels.ravel()
    idx = np.flatnonzero(lab)
    order = np.lexsort((idx, -flat[idx], lab[idx]))     # by label, then -prob, then flat index
    first = np.ones(len(order), bool)
    first[1:] = lab[idx[order[1:]]] != lab[idx[order[:-1]]]
    best = idx[order[first]]
    pts = np.stack(np.unravel_index(best, prob.shape), axis=1)
    return pts.astype(int), flat[best].astype(np.float64), labels, n


def points_from_heatmap(hm: np.ndarray, floor: float = P.CANDIDATE_FLOOR, hw: int = P.NMS_HALFWIDTH_PX):
    """Model B. Returns (points, scores)."""
    assert hm.ndim == 2
    mx = ndimage.maximum_filter(hm, size=2 * hw + 1, mode="constant", cval=-np.inf)
    peak = (hm == mx) & (hm >= floor)
    if not peak.any():
        return np.zeros((0, 2), int), np.zeros(0)
    lab, _ = ndimage.label(peak, structure=S8)              # plateaus: one pixel each
    flat = np.flatnonzero(lab)
    _, first = np.unique(lab.ravel()[flat], return_index=True)   # flat is sorted -> first in C order
    best = flat[first]
    pts = np.stack(np.unravel_index(best, hm.shape), axis=1)
    return pts.astype(int), hm.ravel()[best].astype(np.float64)
