"""Model B targets (PREREG §3.1 + Amendment 1): interior-anchor 3D Gaussians, sigma in mm.

Pure numpy/scipy/cc3d so it runs inside nnU-Net's data-augmentation workers and in tests.

- Lesions: 26-connected components of the (augmented) mask.
- Anchor: the component voxel with maximum Euclidean distance transform (in mm, on the
  component mask padded with background); ties -> nearest to the component centroid (mm),
  then first in C order.
- Peak exactly 1.0 at the anchor voxel; Gaussian sigma_L = clip(ESD_L / 4, 1, 4) mm,
  truncated at 3 sigma (ellipsoidal, in mm); overlaps -> elementwise max.
- Deep supervision: heatmaps are rendered per scale (not downsampled), because nnU-Net's
  DownsampleSegForDSTransform (nearest-exact) drops single-voxel peaks
  (tests/test_ds_downsampling.py). At scale s the anchor maps to the voxel containing its
  physical position, floor((a + 0.5) * s), and the scale's spacing is spacing / s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cc3d
import numpy as np
from scipy import ndimage

SIGMA_FRAC = 0.25        # sigma = ESD / 4
SIGMA_MIN_MM = 1.0
SIGMA_MAX_MM = 4.0
TRUNC_SIGMAS = 3.0


@dataclass(frozen=True)
class Lesion:
    anchor: tuple[int, int, int]      # voxel index at full resolution
    centroid_vox: tuple[float, float, float]
    n_vox: int
    esd_mm: float

    @property
    def sigma_mm(self) -> float:
        return float(np.clip(SIGMA_FRAC * self.esd_mm, SIGMA_MIN_MM, SIGMA_MAX_MM))


def esd_mm(n_vox: int, spacing: Sequence[float]) -> float:
    return float(np.cbrt(6.0 * n_vox * float(np.prod(spacing)) / np.pi))


def interior_anchor(comp: np.ndarray, spacing: Sequence[float], offset=(0, 0, 0)):
    """Anchor of one component given as a boolean crop. Returns (anchor_vox, centroid_vox), both
    in the coordinates of the crop + offset."""
    pad = np.pad(comp, 1)
    edt = ndimage.distance_transform_edt(pad, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt[~comp] = -1.0
    m = edt.max()
    cand = np.argwhere(edt >= m - 1e-9)                      # C order
    centroid = np.argwhere(comp).mean(axis=0)
    if len(cand) > 1:
        d = (((cand - centroid) * np.asarray(spacing)) ** 2).sum(axis=1)
        cand = cand[[int(np.argmin(d))]]                      # argmin -> first minimum (C order)
    off = np.asarray(offset)
    return tuple(int(v) for v in cand[0] + off), tuple(float(v) for v in centroid + off)


def find_lesions(mask: np.ndarray, spacing: Sequence[float]) -> list[Lesion]:
    mask = np.ascontiguousarray(mask > 0)
    if not mask.any():
        return []
    labels = cc3d.connected_components(mask, connectivity=26)
    out = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == k
        anchor, cen = interior_anchor(comp, spacing, offset=tuple(s.start for s in sl))
        n = int(comp.sum())
        out.append(Lesion(anchor=anchor, centroid_vox=cen, n_vox=n, esd_mm=esd_mm(n, spacing)))
    return out


def scaled_shape(shape: Sequence[int], scale: Sequence[float]) -> tuple[int, ...]:
    # identical to nnU-Net's DownsampleSegForDSTransform: round(i * s)
    return tuple(int(round(i * s)) for i, s in zip(shape, scale))


def render(shape: Sequence[int], lesions: Sequence[Lesion], spacing: Sequence[float],
           scale: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Heatmap at one scale. spacing is the FULL-resolution spacing in mm."""
    shp = scaled_shape(shape, scale)
    sp = np.asarray(spacing, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    hm = np.zeros(shp, dtype=np.float32)
    for les in lesions:
        c = np.floor((np.asarray(les.anchor) + 0.5) * np.asarray(scale)).astype(int)
        c = np.clip(c, 0, np.asarray(shp) - 1)
        sig = les.sigma_mm
        r = np.ceil(TRUNC_SIGMAS * sig / sp).astype(int)
        lo, hi = np.maximum(c - r, 0), np.minimum(c + r + 1, shp)
        grids = np.meshgrid(*[(np.arange(l, h) - ci) * s for l, h, ci, s in zip(lo, hi, c, sp)],
                            indexing="ij")
        d2 = sum(g ** 2 for g in grids)
        g = np.exp(-d2 / (2.0 * sig * sig)).astype(np.float32)
        g[d2 > (TRUNC_SIGMAS * sig) ** 2] = 0.0
        sub = hm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(sub, g, out=sub)
        hm[tuple(c)] = 1.0                                   # exact peak (also guards fp rounding)
    return hm


def heatmaps(mask: np.ndarray, spacing: Sequence[float],
             scales: Sequence[Sequence[float]] | None) -> list[np.ndarray]:
    les = find_lesions(mask, spacing)
    scales = scales or [(1.0, 1.0, 1.0)]
    return [render(mask.shape, les, spacing, s) for s in scales]
