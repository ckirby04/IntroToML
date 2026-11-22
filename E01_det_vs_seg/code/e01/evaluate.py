"""E01 evaluation glue on jannus.evaluation.detection (PREREG §6–§8). Never uses the legacy
jannus.evaluation.metrics lesion matching.

- GT lesions: jannus extract_components (26-connected) on the label, spacing from the header.
- Matching (both arms): a point hits lesion L if its distance (mm) to the nearest voxel of L is
  <= R_TOL_MM; returned as a jannus MatchResult(rule="point") so jannus FROC/CPM/bootstrap apply.
- Size bins: ESD < 3, [3, 10), >= 10 mm (jannus assign_size_bin, half-open -> boundary goes up).

CLI:
    run.cmd -m e01.evaluate --gt-check            descriptive: raw centroid > 2 mm from lesion
    run.cmd -m e01.evaluate --dryrun-pilotc       pipeline dry run on PilotC fold-0 exports
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from e01 import common as c  # noqa: F401  (environment first)

from jannus.evaluation.detection.bootstrap import patient_case_weights  # noqa: E402
from jannus.evaluation.detection.geometry import assign_size_bin  # noqa: E402
from jannus.evaluation.detection.matching import ComponentTable, MatchResult, extract_components  # noqa: E402
from jannus.evaluation.detection.metrics import (DetectionData, FROC, concat_cases, cpm, froc,  # noqa: E402
                                                 operating_point)

R_TOL_MM = 2.0
BINS = (("<3", 0.0, 3.0), ("3-10", 3.0, 10.0), (">10", 10.0, float("inf")))
BIN_NAMES = tuple(b[0] for b in BINS)
FP_TARGETS = (0.5, 1.0, 2.0, 3.0)
N_BOOT = 2000
BOOT_SEED = 20261109


# ---------------------------------------------------------------- I/O ---------------------------
def load_label(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    cols = tuple(float(v) for v in np.linalg.norm(img.affine[:3, :3], axis=0))
    assert np.allclose(zooms, cols, atol=1e-3), f"{path.name}: pixdim {zooms} != affine {cols}"
    return np.asarray(img.dataobj) > 0.5, zooms


def load_nnunet_prob(npz: Path, shape_xyz) -> np.ndarray:
    """nnU-Net export (C, z, y, x) -> class-1 probability in nibabel (x, y, z) order."""
    p = np.load(npz)["probabilities"][1].transpose(2, 1, 0)
    assert p.shape == tuple(shape_xyz), f"{npz.name}: prob {p.shape} != label {tuple(shape_xyz)}"
    return np.ascontiguousarray(p, dtype=np.float32)


# ---------------------------------------------------------------- matching ----------------------
def gt_bins(gt: ComponentTable) -> list[str]:
    return [assign_size_bin(float(d), BINS) for d in gt.esd_mm]


def match_points(points: np.ndarray, scores: np.ndarray, gt: ComponentTable,
                 r_tol_mm: float = R_TOL_MM) -> MatchResult:
    """PREREG §6 point rule. A point may hit several lesions; FP = hits none."""
    sp = np.asarray(gt.spacing, dtype=np.float64)
    w = np.ceil(r_tol_mm / sp).astype(int)
    shape = np.asarray(gt.labels.shape)
    pred_hits, touches = [], np.zeros(len(points), bool)
    gt_hit_by: list[list[int]] = [[] for _ in range(gt.n)]
    for i, p in enumerate(np.asarray(points, int)):
        lo, hi = np.maximum(p - w, 0), np.minimum(p + w + 1, shape)
        sub = gt.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        touches[i] = gt.labels[tuple(p)] > 0
        nz = np.argwhere(sub > 0)
        hits: list[int] = []
        if len(nz):
            d = np.sqrt((((nz + lo - p) * sp) ** 2).sum(axis=1))
            ids = sub[tuple(nz.T)]
            for g in np.unique(ids[d <= r_tol_mm + 1e-9]):
                hits.append(int(g))
                gt_hit_by[int(g) - 1].append(i + 1)
        pred_hits.append(tuple(hits))
    z = np.zeros(gt.n)
    return MatchResult(rule="point", pred_scores=np.asarray(scores, float), pred_hits=pred_hits,
                       pred_touches=touches, gt_hit_by=[tuple(h) for h in gt_hit_by],
                       gt_overlap_vox=z.copy(), gt_overlap_frac=z.copy(), gt_best_bbox_iou=z.copy(),
                       params={"r_tol_mm": float(r_tol_mm)})


def merged_components(pred_labels: np.ndarray, gt: ComponentTable) -> int:
    """Predicted components overlapping >= 2 distinct GT lesions (PREREG §7 descriptive)."""
    both = (pred_labels > 0) & (gt.labels > 0)
    if not both.any():
        return 0
    pairs = np.unique(np.stack([pred_labels[both], gt.labels[both]], 1), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return int((counts >= 2).sum())


def centroid_outside(gt: ComponentTable, tol_mm: float = R_TOL_MM) -> int:
    """GT lesions whose raw centroid is > tol_mm from every voxel of the lesion."""
    sp = np.asarray(gt.spacing)
    n = 0
    for k in range(gt.n):
        lo, hi = gt.bbox[k, :3], gt.bbox[k, 3:]
        sub = gt.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == (k + 1)
        vox = (np.argwhere(sub) + lo) * sp
        if np.sqrt(((vox - gt.centroid_mm[k]) ** 2).sum(1)).min() > tol_mm:
            n += 1
    return n


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


# ---------------------------------------------------------------- per case ----------------------
@dataclass
class CaseResult:
    case_id: str
    patient_id: str
    match: MatchResult
    bins: list[str]
    extra: dict


def case_result(case_id, patient_id, points, scores, gt: ComponentTable, extra=None) -> CaseResult:
    return CaseResult(case_id, patient_id, match_points(points, scores, gt), gt_bins(gt), extra or {})


def to_data(results: Sequence[CaseResult]) -> DetectionData:
    rows = []
    for r in results:
        m = r.match
        rows.append((r.case_id, r.patient_id, m.gt_score, np.asarray(r.bins, dtype=object),
                     m.pred_scores[m.pred_is_fp]))
    return concat_cases(rows)


# ---------------------------------------------------------------- statistics --------------------
def curves(data: DetectionData, w=None) -> dict[str, FROC]:
    out = {"all": froc(data, case_weight=w)}
    for b in BIN_NAMES:
        out[b] = froc(data, case_weight=w, gt_mask=(data.gt_bin == b))
    return out


def stats(data: DetectionData, w=None, fixed_tau: float | None = None) -> dict[str, float]:
    cv = curves(data, w)
    s: dict[str, float] = {"cpm": cpm(cv["all"], FP_TARGETS)}
    for t in FP_TARGETS:
        v, _, reached = cv["all"].sensitivity_at_fp(t)
        s[f"sens@{t}"] = v
        s[f"reached@{t}"] = float(reached)
    for b in BIN_NAMES:
        s[f"sens@1[{b}]"] = cv[b].sensitivity_at_fp(1.0)[0]
        s[f"cpm[{b}]"] = cpm(cv[b], FP_TARGETS)
    if fixed_tau is not None:
        op = operating_point(data, fixed_tau, bins=BIN_NAMES, case_weight=w)
        s["fixed.sensitivity"] = op["sensitivity"]
        s["fixed.fp_per_scan"] = op["fp_per_scan"]
        for b in BIN_NAMES:
            s[f"fixed.sensitivity[{b}]"] = op[f"sensitivity[{b}]"]
    return s


def paired_bootstrap(arms: dict[str, DetectionData], fixed_taus: dict[str, float | None],
                     n: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Same patient draw for every arm; also B - A differences when both are present."""
    names = list(arms)
    ref = arms[names[0]]
    for a in names[1:]:
        assert arms[a].case_ids == ref.case_ids and arms[a].patient_ids == ref.patient_ids
    W, n_pat = patient_case_weights(ref, n, seed)
    point = {a: stats(arms[a], None, fixed_taus[a]) for a in names}
    samples: dict[str, dict[str, list]] = {a: {k: [] for k in point[a]} for a in names}
    for w in W:
        for a in names:
            st = stats(arms[a], w, fixed_taus[a])
            for k in point[a]:
                samples[a][k].append(st[k])

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)] if v.size else [None, None, 0]

    out = {"n_resamples": n, "seed": seed, "n_patients": n_pat, "arms": {}}
    for a in names:
        out["arms"][a] = {k: {"point": point[a][k], "ci95": ci(samples[a][k])} for k in point[a]
                          if not k.startswith("reached")}
        out["arms"][a].update({k: point[a][k] for k in point[a] if k.startswith("reached")})
    if {"A", "B"} <= set(names):
        diff = {}
        for k in point["A"]:
            if k.startswith(("reached", "fixed")):
                continue
            d = np.asarray(samples["B"][k], float) - np.asarray(samples["A"][k], float)
            diff[k] = {"point": point["B"][k] - point["A"][k], "ci95": ci(d)}
        out["B_minus_A"] = diff
    return out


# ---------------------------------------------------------------- CLI tasks ---------------------
D161 = Path(r"G:\BrainMetShare\1.60_artifacts\nnUNet_raw\Dataset161_PilotC")
PILOTC_F0 = Path(r"G:\BrainMetShare\1.60_artifacts\nnUNet_results\Dataset161_PilotC"
                 r"\nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres\fold_0\validation")


def gt_check() -> dict:
    g = c.guard()
    lab = c.NN_RAW / c.DATASET / "labelsTr"
    out = {}
    for name, folds in (("fold0_test", [0]), ("folds1-4_train", [1, 2, 3, 4])):
        n_les = n_out = 0
        per_bin = {b: 0 for b in BIN_NAMES}
        for sid in g.pool_ids(folds=folds):
            m, sp = load_label(lab / f"{sid}.nii.gz")
            t = extract_components(m, sp, connectivity=26)
            n_les += t.n
            n_out += centroid_outside(t)
            for b in gt_bins(t):
                per_bin[b] += 1
        out[name] = {"lesions": n_les, "centroid_gt_2mm_outside": n_out, "lesions_per_bin_1mm_grid": per_bin}
    print(json.dumps(out, indent=1))
    (c.RESULTS / "gt_anchor_check.json").write_text(json.dumps(out, indent=1))
    return out


def dryrun_pilotc() -> dict:
    """Model-A path end to end on PilotC fold-0 exports vs Dataset161 labels (pipeline test)."""
    from e01.decode import points_from_prob

    g = c.guard()
    ids = g.pool_ids(folds=[0])
    results, dices, merged, n_comp = [], [], 0, 0
    for sid in ids:
        m, sp = load_label(D161 / "labelsTr" / f"{sid}.nii.gz")
        p = load_nnunet_prob(PILOTC_F0 / f"{sid}.npz", m.shape)
        gt = extract_components(m, sp, connectivity=26)
        pts, sc, plabels, ncomp = points_from_prob(p)
        merged += merged_components(plabels, gt)
        n_comp += ncomp
        dices.append(dice(p >= 0.5, m))
        results.append(case_result(sid, g.patient_of(sid), pts, sc, gt))
    data = to_data(results)
    boot = paired_bootstrap({"A": data}, {"A": 0.5})
    red = sum(r.match.n_redundant() for r in results)
    out = {"what": "PIPELINE DRY RUN: PilotC fold_0 (1.60 pilot model, Dataset161 grid) - not an E01 result",
           "n_scans": len(ids), "n_patients": len(set(data.patient_ids)),
           "n_lesions": int(len(data.gt_score)),
           "lesions_per_bin": {b: int((data.gt_bin == b).sum()) for b in BIN_NAMES},
           "mean_dice@0.5": float(np.mean(dices)), "components@0.05": n_comp,
           "merged_components@0.05": merged, "redundant_points@0.05": int(red), "bootstrap": boot}
    od = c.RESULTS / "dryrun_pilotc"
    od.mkdir(parents=True, exist_ok=True)
    (od / "metrics.json").write_text(json.dumps(out, indent=1))
    from jannus.evaluation.detection.report import plot_froc_comparison

    cv = curves(data)
    panels = [("All lesions", [("PilotC (A path)", cv["all"])], int(len(data.gt_score)))]
    panels += [(f"ESD {b} mm", [("PilotC (A path)", cv[b])], int((data.gt_bin == b).sum())) for b in BIN_NAMES]
    plot_froc_comparison(panels, od / "froc_dryrun.png", x_max=4.0, targets=FP_TARGETS)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-check", action="store_true")
    ap.add_argument("--dryrun-pilotc", action="store_true")
    a = ap.parse_args()
    if a.gt_check:
        gt_check()
    if a.dryrun_pilotc:
        r = dryrun_pilotc()
        print(json.dumps({k: v for k, v in r.items() if k != "bootstrap"}, indent=1))
