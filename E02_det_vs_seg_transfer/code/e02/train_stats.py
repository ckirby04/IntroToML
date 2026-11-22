"""Training-set statistics that fix E02's pixel parameters (PLAN.md §5). Reads Tr only.

Run:  env\\run.cmd -m e02.train_stats [--workers 8]

Writes results\\train_stats.json (console log: results\\train_stats.log). Test images are touched only for GT
counts (instances per bin with the Tr-derived edges), as E01 reported its fold-0 bin counts.
- ESD (2D) = sqrt(4 A / pi), A = instance pixel count in the instance map.
- k = median Tr ESD (px) / E01 median training ESD (mm).
- Bin edges = Tr ESD quantiles at E01's training bin shares.
- Interior anchor = max-EDT pixel of the instance (ties -> nearest centroid, then first in C order),
  as E01 heatmap.interior_anchor, in 2D with unit spacing.
- Neighbour distance = distance from each instance's anchor to the nearest other anchor in its image.
- NMS half-width = the largest h at which the share of instances with another anchor within
  Chebyshev distance h is <= E01's training share at its own half-width (0.94%). Size scaling
  (round(3k)) is reported but not used: cells are packed far tighter than metastases.
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool

import numpy as np

from e02 import common as c


def anchors_and_areas(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage

    n = int(inst.max())
    areas = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    anchors = np.zeros((n, 2), np.int64)
    for i, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        comp = inst[sl] == i
        edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
        cand = np.argwhere(edt == edt.max())
        if len(cand) > 1:
            cen = np.argwhere(comp).mean(0)
            d = ((cand - cen) ** 2).sum(1)
            cand = cand[d == d.min()][:1]
        anchors[i - 1] = cand[0] + [sl[0].start, sl[1].start]
    return anchors, areas


def _one(case: str) -> dict:
    import tifffile
    from scipy.spatial import cKDTree

    inst = tifffile.imread(c.INSTANCES / "Tr" / f"{case}.tif")
    anc, area = anchors_and_areas(inst)
    nn = np.full(len(anc), np.nan)
    cheb = np.full(len(anc), np.inf)
    if len(anc) > 1:
        t = cKDTree(anc)
        nn = t.query(anc, k=2)[0][:, 1]
        cheb = t.query(anc, k=2, p=np.inf)[0][:, 1]
    return {"case": case, "esd": np.sqrt(4 * area / np.pi), "nn": nn, "cheb": cheb}


def main(workers: int) -> None:
    rows = list(csv.DictReader(open(c.META / "images.csv")))
    tr = [r for r in rows if r["split"] == "Tr"]
    line_of = {r["case"]: r["cell_line"] for r in tr}
    with Pool(workers) as pool:
        res = pool.map(_one, [r["case"] for r in tr], chunksize=16)
    esd = np.concatenate([r["esd"] for r in res])
    nn = np.concatenate([r["nn"] for r in res])
    cheb = np.concatenate([r["cheb"] for r in res])
    # NMS half-width (owner decision 2026-11-16): the largest h whose collision share
    # (another anchor within Chebyshev distance h) does not exceed E01's.
    shares = {h: float(np.mean(cheb <= h)) for h in range(1, 41)}
    h_match = max([h for h, s in shares.items() if s <= c.E01_TRAIN_NMS_COLLISION_SHARE], default=1)
    lines = np.concatenate([[line_of[r["case"]]] * len(r["esd"]) for r in res])

    med = float(np.median(esd))
    k = med / c.E01_TRAIN_MEDIAN_ESD_MM
    q1, q2 = c.E01_TRAIN_BIN_SHARES[0], c.E01_TRAIN_BIN_SHARES[0] + c.E01_TRAIN_BIN_SHARES[1]
    e_lo, e_hi = (float(x) for x in np.quantile(esd, [q1, q2]))
    density = len(esd) / len(tr)
    f = density / c.E01_TRAIN_LESIONS_PER_SCAN
    params = {
        "k_px_per_mm": k,
        "sigma_px": [max(1.0, k), 4 * k],
        "nms_halfwidth_px": h_match,
        "nms_halfwidth_px_size_scaled_unused": max(int(round(3.0 * k)), 1),
        "nms_collision_share_by_h": shares,
        "hit_tol_px": 2.0 * k,
        "c1_fallback_radius_px": 8.0 * k,
        "c2_crop_px": int(min(max(32, 32 * round(96 * k / 32)), 512)),
        "c2_crop_px_unclipped": 96 * k,
        "size_bin_edges_px": [e_lo, e_hi],
        "fp_targets_per_image": [t * f for t in (0.5, 1, 2, 3)],
    }
    nn_med = float(np.nanmedian(nn))
    check = {"median_nn_anchor_dist_px": nn_med,
             "nn_dist_quantiles_px": {str(q): float(np.nanquantile(nn, q)) for q in (0.05, 0.25, 0.5)},
             "nms_halfwidth_lt_median_nn": params["nms_halfwidth_px"] < nn_med,
             "hit_tol_lt_median_nn": params["hit_tol_px"] < nn_med,
             "share_nn_le_nms_halfwidth": float(np.nanmean(nn <= params["nms_halfwidth_px"])),
             "share_nn_le_hit_tol": float(np.nanmean(nn <= params["hit_tol_px"]))}
    bins = np.digitize(esd, [e_lo, e_hi])            # boundary -> upper bin, as E01
    per_line = {}
    for ln in sorted(set(lines)):
        m = lines == ln
        per_line[ln] = {"instances": int(m.sum()), "median_esd_px": float(np.median(esd[m])),
                        "bin_shares": [float(np.mean(bins[m] == b)) for b in range(3)]}

    # test GT counts only (no predictions exist)
    import tifffile
    te_esd = []
    for r in (r for r in rows if r["split"] == "Ts"):
        inst = tifffile.imread(c.INSTANCES / "Ts" / f"{r['case']}.tif")
        a = np.bincount(inst.ravel())[1:]
        te_esd.append(np.sqrt(4 * a[a > 0] / np.pi))
    te_esd = np.concatenate(te_esd)
    te_bins = np.digitize(te_esd, [e_lo, e_hi])

    out = {
        "train": {"images": len(tr), "instances": int(len(esd)), "instances_per_image": density,
                  "esd_px_quantiles": {str(q): float(np.quantile(esd, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                  "esd_spread_p95_over_p5": float(np.quantile(esd, .95) / np.quantile(esd, .05)),
                  "bin_counts": [int((bins == b).sum()) for b in range(3)], "per_cell_line": per_line},
        "e01_reference": {"median_esd_mm": c.E01_TRAIN_MEDIAN_ESD_MM, "bin_shares": c.E01_TRAIN_BIN_SHARES,
                          "lesions_per_scan": c.E01_TRAIN_LESIONS_PER_SCAN,
                          "esd_spread_p95_over_p5": 16.71 / 1.97},
        "params": params, "neighbour_check": check,
        "test_gt": {"images": sum(r["split"] == "Ts" for r in rows), "instances": int(len(te_esd)),
                    "bin_counts": [int((te_bins == b).sum()) for b in range(3)]},
    }
    c.RESULTS.mkdir(exist_ok=True)
    (c.RESULTS / "train_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
