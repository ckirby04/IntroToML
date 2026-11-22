"""Model B targets (PREREG §4.1): one 2D Gaussian per INSTANCE at its interior anchor, sigma in px.

Pure numpy/scipy, so it runs inside nnU-Net's augmentation workers and in tests.
- Instances: the ids of the (augmented) instance map, not connected components (PREREG §3.1 of
  the plan): touching cells stay separate.
- Anchor: the instance pixel with maximum Euclidean distance transform (unit spacing, instance
  mask padded with background); ties -> nearest to the instance centroid, then first in C order.
  A cell cut by the patch border gets the anchor of its visible part. An instance split into
  pieces (overlap rule or augmentation) is still one instance with one anchor.
- Gaussian sigma = clip(ESD / 4, 4.09, 16.34) px, ESD from the visible area; truncated at 3 sigma;
  exactly 1.0 at the anchor; overlaps -> elementwise max.
- Deep supervision: rendered per scale (as E01): at scale s the anchor maps to the pixel holding
  its position, floor((a + 0.5) * s), and the grid step is 1/s full-resolution px.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from e02 import params as P


@dataclass(frozen=True)
class Cell:
    id: int
    anchor: tuple[int, int]
    centroid: tuple[float, float]
    n_px: int

    @property
    def esd(self) -> float:
        return float(np.sqrt(4.0 * self.n_px / np.pi))

    @property
    def sigma(self) -> float:
        return P.sigma_px(self.esd)


def interior_anchor(comp: np.ndarray, offset=(0, 0)):
    """Anchor of one instance given as a boolean crop -> (anchor, centroid) in crop + offset coords."""
    edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
    edt[~comp] = -1.0
    cand = np.argwhere(edt >= edt.max() - 1e-9)              # C order
    cen = np.argwhere(comp).mean(axis=0)
    if len(cand) > 1:
        d = ((cand - cen) ** 2).sum(axis=1)
        cand = cand[[int(np.argmin(d))]]                     # first minimum -> C order
    off = np.asarray(offset)
    return tuple(int(v) for v in cand[0] + off), tuple(float(v) for v in cen + off)


def find_cells(inst: np.ndarray) -> list[Cell]:
    inst = np.asarray(inst)
    assert inst.ndim == 2, inst.shape
    if not (inst > 0).any():
        return []
    inst = np.where(inst > 0, inst, 0).astype(np.int64)
    out = []
    for i, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        comp = inst[sl] == i
        a, cen = interior_anchor(comp, offset=(sl[0].start, sl[1].start))
        out.append(Cell(id=i, anchor=a, centroid=cen, n_px=int(comp.sum())))
    return out


def scaled_shape(shape: Sequence[int], scale: Sequence[float]) -> tuple[int, ...]:
    # identical to nnU-Net's DownsampleSegForDSTransform: round(i * s)
    return tuple(int(round(i * s)) for i, s in zip(shape, scale))


def render(shape: Sequence[int], cells: Sequence[Cell], scale: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
    shp = scaled_shape(shape, scale)
    sc = np.asarray(scale, dtype=np.float64)
    step = 1.0 / sc                                         # full-res px per pixel at this scale
    hm = np.zeros(shp, dtype=np.float32)
    for cell in cells:
        c = np.floor((np.asarray(cell.anchor) + 0.5) * sc).astype(int)
        c = np.clip(c, 0, np.asarray(shp) - 1)
        sig = cell.sigma
        r = np.ceil(P.TRUNC_SIGMAS * sig / step).astype(int)
        lo, hi = np.maximum(c - r, 0), np.minimum(c + r + 1, shp)
        gy, gx = np.meshgrid((np.arange(lo[0], hi[0]) - c[0]) * step[0],
                             (np.arange(lo[1], hi[1]) - c[1]) * step[1], indexing="ij")
        d2 = gy ** 2 + gx ** 2
        g = np.exp(-d2 / (2.0 * sig * sig)).astype(np.float32)
        g[d2 > (P.TRUNC_SIGMAS * sig) ** 2] = 0.0
        sub = hm[lo[0]:hi[0], lo[1]:hi[1]]
        np.maximum(sub, g, out=sub)
        hm[tuple(c)] = 1.0
    return hm


def heatmaps(inst: np.ndarray, scales: Sequence[Sequence[float]] | None) -> list[np.ndarray]:
    cells = find_cells(inst)
    scales = scales or [(1.0, 1.0)]
    return [render(inst.shape, cells, s) for s in scales]


def anchors_and_areas(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(anchors (n, 2), areas (n,)) indexed by instance id - 1; absent ids get area 0."""
    inst = np.asarray(inst)
    n = int(inst.max()) if inst.size else 0
    anchors = np.full((n, 2), -1, np.int64)
    areas = np.zeros(n, np.int64)
    for cell in find_cells(inst):
        anchors[cell.id - 1] = cell.anchor
        areas[cell.id - 1] = cell.n_px
    return anchors, areas
