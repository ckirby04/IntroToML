"""Two-stage arms C1, C2, C4 (PREREG §7.1), 2D, pure functions. E01's two_stage.py rules with
26 -> 8 connectivity and mm -> px parameters (params.py).

C1: B's peaks q (decode.points_from_heatmap), each assigned to the 8-cc of A's p >= 0.05 that
    contains q, else the nearest within 8.17 px (ties: higher component max, then lower label).
    a(q) = max p over the component (0 if unassigned); s(q) = (b(q) + a(q)) / 2.
    Mask: union over s >= 0.4 of (component ∩ {p >= 0.5}) or, if that is empty or q is
    unassigned, the heatmap fallback (8-cc containing q of {heat >= e^-2 b(q)} in a disc of
    radius 32.7 px).
C2: the same rules on each peak's 384 x 384 crop of C2's map; the region is pasted back.
C4: C1's scores; C2's regions (else the fallback) for kept peaks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from e02 import decode as dc
from e02 import params as P

S8 = dc.S8


def assign_peaks(points: np.ndarray, comp_labels: np.ndarray, comp_max: np.ndarray,
                 r: float = P.ASSIGN_PX) -> np.ndarray:
    """Component label (1-based) per peak, 0 if unassigned."""
    w = int(np.ceil(r))
    shape = np.asarray(comp_labels.shape)
    out = np.zeros(len(points), dtype=np.int64)
    for i, q in enumerate(np.asarray(points, int).reshape(-1, 2)):
        k = int(comp_labels[tuple(q)])
        if k:
            out[i] = k
            continue
        lo, hi = np.maximum(q - w, 0), np.minimum(q + w + 1, shape)
        sub = comp_labels[lo[0]:hi[0], lo[1]:hi[1]]
        nz = np.argwhere(sub > 0)
        if not len(nz):
            continue
        d = np.sqrt(((nz + lo - q) ** 2).sum(1))
        ok = d <= r + 1e-9
        if not ok.any():
            continue
        dk: dict[int, float] = {}
        for lab, dist in zip(sub[tuple(nz[ok].T)], d[ok]):
            dk[int(lab)] = min(dk.get(int(lab), np.inf), float(dist))
        out[i] = min(dk, key=lambda L: (round(dk[L], 9), -float(comp_max[L - 1]), L))
    return out


def stage2(prob: np.ndarray, points: np.ndarray):
    """(a per peak, assigned component per peak, component labels) for one stage-2 map."""
    comp, n = ndimage.label(prob >= P.CANDIDATE_FLOOR, structure=S8)
    cmax = np.asarray(ndimage.maximum(prob, comp, index=np.arange(1, n + 1)), float) if n else np.zeros(0)
    k = assign_peaks(points, comp, cmax) if len(points) else np.zeros(0, np.int64)
    a = np.zeros(len(points)) if not n else np.where(k > 0, cmax[np.maximum(k, 1) - 1], 0.0)
    return a, k, comp


def cascade(prob_a: np.ndarray, heat_b: np.ndarray) -> dict:
    """C1 points and scores."""
    pts, b = dc.points_from_heatmap(heat_b)
    a, k, comp = stage2(prob_a, pts)
    return {"points": pts, "b": b, "a": a, "s": (b + a) / 2.0, "comp": k, "comp_labels": comp}


def fallback_region(heat_b: np.ndarray, q) -> np.ndarray:
    """8-cc containing q of {heat >= e^-2 heat[q]}, clipped to a disc of radius 32.7 px around q."""
    q = np.asarray(q, int)
    w = int(np.ceil(P.FALLBACK_RADIUS_PX))
    lo, hi = np.maximum(q - w, 0), np.minimum(q + w + 1, np.asarray(heat_b.shape))
    sub = heat_b[lo[0]:hi[0], lo[1]:hi[1]]
    gy, gx = np.meshgrid(np.arange(lo[0], hi[0]) - q[0], np.arange(lo[1], hi[1]) - q[1], indexing="ij")
    disc = gy ** 2 + gx ** 2 <= P.FALLBACK_RADIUS_PX ** 2 + 1e-9
    cand = (sub >= P.FALLBACK_REL * heat_b[tuple(q)]) & disc
    lab, _ = ndimage.label(cand, structure=S8)
    out = np.zeros(heat_b.shape, bool)
    lq = lab[tuple(q - lo)]
    if lq:
        out[lo[0]:hi[0], lo[1]:hi[1]] = lab == lq
    else:                                    # cannot happen (q is in cand); kept for safety
        out[tuple(q)] = True
    return out


def cascade_mask(cas: dict, prob_a: np.ndarray, heat_b: np.ndarray, tau: float = P.TAU["C1"]) -> np.ndarray:
    """C1 mask: union of the regions of peaks with s >= tau."""
    keep = np.flatnonzero(cas["s"] >= tau)
    comps = np.unique(cas["comp"][keep])
    comps = comps[comps > 0]
    hi_a = prob_a >= P.MASK_P
    mask = np.isin(cas["comp_labels"], comps) & hi_a if len(comps) else np.zeros(prob_a.shape, bool)
    has_hi = np.zeros(int(cas["comp_labels"].max()) + 1, bool)
    if len(comps):
        has_hi[np.unique(cas["comp_labels"][hi_a & (cas["comp_labels"] > 0)])] = True
    for i in keep:
        k = int(cas["comp"][i])
        if k == 0 or not has_hi[k]:
            mask |= fallback_region(heat_b, cas["points"][i])
    return mask


# ---------------------------------------------------------------- crops ---------------------------
def crop_start(center, size: int = P.C2_CROP) -> np.ndarray:
    """Start of a size x size crop whose centre pixel is `center` (centre = start + size // 2)."""
    return np.asarray(center, int) - size // 2


def extract_crop(img: np.ndarray, center, size: int = P.C2_CROP, fill=0) -> np.ndarray:
    st = crop_start(center, size)
    out = np.full((size, size), fill, dtype=img.dtype)
    lo = np.maximum(st, 0)
    hi = np.minimum(st + size, img.shape)
    if np.any(hi <= lo):
        return out
    out[lo[0] - st[0]:hi[0] - st[0], lo[1] - st[1]:hi[1] - st[1]] = img[lo[0]:hi[0], lo[1]:hi[1]]
    return out


def paste_or(full: np.ndarray, crop: np.ndarray, start) -> None:
    """full |= crop placed at start; parts in the padding are dropped."""
    st = np.asarray(start, int)
    lo = np.maximum(st, 0)
    hi = np.minimum(st + np.asarray(crop.shape), full.shape)
    if np.any(hi <= lo):
        return
    full[lo[0]:hi[0], lo[1]:hi[1]] |= crop[lo[0] - st[0]:hi[0] - st[0], lo[1] - st[1]:hi[1] - st[1]]


def stage2_from_crop(p2: np.ndarray):
    """PREREG §7.1 rules on one C2 crop map; the peak is the crop centre. -> (a2, region, assigned)."""
    q = np.full((1, 2), p2.shape[0] // 2)
    a, k, comp = stage2(p2, q)
    if k[0] == 0:
        return 0.0, np.zeros(p2.shape, bool), False
    return float(a[0]), (comp == k[0]) & (p2 >= P.MASK_P), True


def c2_scores_and_mask(out2: dict, heat_b: np.ndarray, tau: float = P.TAU["C2"]):
    """C2: s = (b + a2) / 2; mask = union over s >= tau of the crop region pasted back, or the
    heatmap fallback when the region is empty."""
    s = (np.asarray(out2["b"], float) + np.asarray(out2["a2"], float)) / 2.0
    return s, _regions_mask(out2, heat_b, np.flatnonzero(s >= tau))


def c4_scores_and_mask(cas: dict, out2: dict, heat_b: np.ndarray, tau: float = P.TAU["C4"]):
    """C4: C1's scores; kept peaks get C2's crop region, else the fallback."""
    s = np.asarray(cas["s"], float)
    assert np.array_equal(cas["points"], out2["points"]), "C1 and C2 must share B's peaks"
    return s, _regions_mask(out2, heat_b, np.flatnonzero(s >= tau))


def _regions_mask(out2: dict, heat_b: np.ndarray, idx) -> np.ndarray:
    mask = np.zeros(heat_b.shape, bool)
    for i in idx:
        if out2["regions"][i].any():
            paste_or(mask, out2["regions"][i], out2["starts"][i])
        else:
            mask |= fallback_region(heat_b, out2["points"][i])
    return mask
