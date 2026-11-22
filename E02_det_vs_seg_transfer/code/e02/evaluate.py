"""E02 evaluation on jannus.evaluation.detection (PREREG §2, §3, §7, §8, §10).

- GT objects: the instances of the instance map (data\\instances\\Ts). Size = ESD of the
  instance's pixel count; bins from params (upper bin on an edge).
- Matching: E02's own 2D matcher, returned as jannus MatchResult(rule="point"), so jannus
  FROC / CPM / bootstrap weights apply unchanged. jannus works on records only; case = image
  ("FP per scan" = FP per image) and "patient id" = well (the bootstrap cluster).
  * one-hit rule (primary): a point hits at most one instance: the one whose pixel it lies on,
    else the instance with the nearest pixel (Euclidean, pixel centres) within 8.17 px, ties ->
    lower id.
  * multi-hit rule (sensitivity analysis): a point hits every instance with a pixel within 8.17 px.
- Dice GT: the gapped binary label. Per-object Dice uses L = the instance's pixels in the gapped
  label; instances with no such pixel are left out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from e02 import common as c  # noqa: F401  (environment first)
from e02 import params as P

from jannus.evaluation.detection.bootstrap import patient_case_weights  # noqa: E402
from jannus.evaluation.detection.matching import MatchResult  # noqa: E402
from jannus.evaluation.detection.metrics import (DetectionData, FROC, concat_cases, cpm, froc,  # noqa: E402
                                                 operating_point)

S8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------- GT ------------------------------
@dataclass
class GT:
    inst: np.ndarray            # (y, x) instance ids, 0 = background
    label: np.ndarray           # (y, x) bool, gapped binary label
    areas: np.ndarray           # (n,) instance pixel counts (id - 1)

    @property
    def n(self) -> int:
        return len(self.areas)

    @property
    def bins(self) -> list[str]:
        return [P.size_bin(e) for e in P.esd_px(self.areas)]


def make_gt(inst: np.ndarray, label: np.ndarray) -> GT:
    inst = np.asarray(inst).astype(np.int64)
    n = int(inst.max())
    areas = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    assert (areas > 0).all(), "instance ids must be compact (every id 1..n present)"
    label = np.asarray(label) > 0
    assert not (label & (inst == 0)).any(), "gapped label outside instances"
    return GT(inst=inst, label=label, areas=areas)


def load_gt(case: str, split: str = "Ts") -> GT:
    import tifffile
    from PIL import Image

    inst = tifffile.imread(c.INSTANCES / split / f"{case}.tif")
    lab = np.asarray(Image.open(c.NN_RAW / c.DATASET / f"labels{split}" / f"{case}.png")) > 0
    return make_gt(inst, lab)


# ---------------------------------------------------------------- matching ------------------------
def _near(inst: np.ndarray, p: np.ndarray, r: float):
    """[(distance, id)] of instances with a pixel within r of p (one entry per id, min distance)."""
    w = int(np.ceil(r))
    lo = np.maximum(p - w, 0)
    hi = np.minimum(p + w + 1, inst.shape)
    sub = inst[lo[0]:hi[0], lo[1]:hi[1]]
    nz = np.argwhere(sub > 0)
    if not len(nz):
        return []
    d = np.sqrt(((nz + lo - p) ** 2).sum(1))
    ok = d <= r + 1e-9
    best: dict[int, float] = {}
    for g, dd in zip(sub[tuple(nz[ok].T)], d[ok]):
        g = int(g)
        if dd < best.get(g, np.inf):
            best[g] = float(dd)
    return sorted((dd, g) for g, dd in best.items())


def match_points(points: np.ndarray, scores: np.ndarray, gt: GT, rule: str = "one",
                 r: float = P.HIT_TOL_PX) -> MatchResult:
    assert rule in ("one", "multi")
    pts = np.asarray(points, int).reshape(-1, 2)
    pred_hits: list[tuple[int, ...]] = []
    touches = np.zeros(len(pts), bool)
    gt_hit_by: list[list[int]] = [[] for _ in range(gt.n)]
    for i, p in enumerate(pts):
        inside = int(gt.inst[tuple(p)])
        touches[i] = inside > 0
        if rule == "one":
            if inside:
                hits = (inside,)
            else:
                near = _near(gt.inst, p, r)
                hits = (min(near)[1],) if near else ()        # (distance, id): ties -> lower id
        else:
            hits = tuple(sorted(g for _, g in _near(gt.inst, p, r)))
        for g in hits:
            gt_hit_by[g - 1].append(i + 1)
        pred_hits.append(hits)
    z = np.zeros(gt.n)
    return MatchResult(rule="point", pred_scores=np.asarray(scores, float).reshape(-1), pred_hits=pred_hits,
                       pred_touches=touches, gt_hit_by=[tuple(h) for h in gt_hit_by],
                       gt_overlap_vox=z.copy(), gt_overlap_frac=z.copy(), gt_best_bbox_iou=z.copy(),
                       params={"r_tol_px": float(r), "hit_rule": rule})


# ---------------------------------------------------------------- masks -----------------------------
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


def object_dice(mask: np.ndarray, gt: GT) -> np.ndarray:
    """Per instance: L = instance pixels in the gapped label; P_L = union of the mask's 8-cc that
    overlap L; Dice_L = 2|P_L ∩ L| / (|P_L| + |L|), 0 if none overlaps; NaN if L is empty."""
    mask = np.asarray(mask, bool)
    L = np.where(gt.label, gt.inst, 0)
    nL = np.bincount(L.ravel(), minlength=gt.n + 1)[1:].astype(float)
    inter = np.bincount(L[mask].ravel(), minlength=gt.n + 1)[1:].astype(float)   # |L ∩ mask| = |P_L ∩ L|
    pl, k = ndimage.label(mask, structure=S8)
    size = np.bincount(pl.ravel(), minlength=k + 1).astype(float)
    both = (L > 0) & (pl > 0)
    pairs = np.unique(np.stack([L[both], pl[both]], 1), axis=0) if both.any() else np.zeros((0, 2), int)
    nP = np.zeros(gt.n)
    np.add.at(nP, pairs[:, 0] - 1, size[pairs[:, 1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(nL > 0, 2.0 * inter / (nP + nL), np.nan)
    return out


def merged_components(pred_labels: np.ndarray, gt: GT) -> int:
    """Predicted components overlapping >= 2 distinct GT instances (PREREG §10)."""
    both = (pred_labels > 0) & (gt.inst > 0)
    if not both.any():
        return 0
    pairs = np.unique(np.stack([pred_labels[both], gt.inst[both]], 1), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return int((counts >= 2).sum())


# ---------------------------------------------------------------- records -------------------------
@dataclass
class CaseResult:
    case_id: str
    well: str
    match: MatchResult
    bins: list[str]
    extra: dict = field(default_factory=dict)


def to_data(results: Sequence[CaseResult]) -> DetectionData:
    rows = [(r.case_id, r.well, r.match.gt_score, np.asarray(r.bins, dtype=object),
             r.match.pred_scores[r.match.pred_is_fp]) for r in results]
    return concat_cases(rows)


@dataclass
class SegData:
    """Per-image Dice (n_cases,) and per-object Dice with each object's case index and bin."""
    image_dice: np.ndarray
    obj_dice: np.ndarray
    obj_case: np.ndarray
    obj_bin: np.ndarray


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


# ---------------------------------------------------------------- statistics ----------------------
def det_stats(data: DetectionData, w=None, fixed_tau: float | None = None) -> dict[str, float]:
    cv = {"all": froc(data, case_weight=w)}
    for b in P.BIN_NAMES:
        cv[b] = froc(data, case_weight=w, gt_mask=(data.gt_bin == b))
    s: dict[str, float] = {"cpm": cpm(cv["all"], P.FP_TARGETS)}
    for i, t in enumerate(P.FP_TARGETS):
        v, _, reached = cv["all"].sensitivity_at_fp(t)
        s[f"sens@t{i + 1}"] = v
        s[f"reached@t{i + 1}"] = float(reached)
    for b in P.BIN_NAMES:
        s[f"sens@1FP[{b}]"] = cv[b].sensitivity_at_fp(P.ONE_FP)[0]
        s[f"cpm[{b}]"] = cpm(cv[b], P.FP_TARGETS)
    if fixed_tau is not None:
        op = operating_point(data, fixed_tau, bins=P.BIN_NAMES, case_weight=w)
        s["fixed.sensitivity"] = op["sensitivity"]
        s["fixed.fp_per_image"] = op["fp_per_scan"]
        for b in P.BIN_NAMES:
            s[f"fixed.sensitivity[{b}]"] = op[f"sensitivity[{b}]"]
    return s


def seg_stats(seg: SegData, w=None) -> dict[str, float]:
    w = np.ones(len(seg.image_dice)) if w is None else np.asarray(w, float)
    wo = w[seg.obj_case]
    s = {"dice_image": _wmean(seg.image_dice, w), "dice_object": _wmean(seg.obj_dice, wo)}
    for b in P.BIN_NAMES:
        m = seg.obj_bin == b
        s[f"dice_object[{b}]"] = _wmean(seg.obj_dice[m], wo[m])
    return s


def arm_stats(det: DetectionData | None, seg: SegData | None, w, tau) -> dict[str, float]:
    s = {}
    if det is not None:
        s.update(det_stats(det, w, tau))
    if seg is not None:
        s.update(seg_stats(seg, w))
    return s


def bootstrap_weights(arms: dict[str, dict], n: int = P.N_BOOT, seed: int = P.BOOT_SEED):
    """(W (n, n_cases), n_wells, point stats per arm). One well draw shared by every arm."""
    ref = next(a["det"] for a in arms.values() if a["det"] is not None)
    for a in arms.values():
        if a["det"] is not None:
            assert a["det"].case_ids == ref.case_ids and a["det"].patient_ids == ref.patient_ids
        if a["seg"] is not None:
            assert len(a["seg"].image_dice) == ref.n_cases
    W, n_wells = patient_case_weights(ref, n, seed)
    point = {a: arm_stats(arms[a]["det"], arms[a]["seg"], None, arms[a]["tau"]) for a in arms}
    return W, n_wells, point


def bootstrap_samples(arms: dict[str, dict], point: dict, W_rows: np.ndarray) -> dict:
    """Statistics for a block of resample rows: {arm: {key: (len(W_rows),)}}. Chunks of rows can be
    computed separately (and saved) and concatenated in row order: the result is identical."""
    out = {a: {k: np.empty(len(W_rows)) for k in point[a]} for a in arms}
    for i, w in enumerate(W_rows):
        for a in arms:
            st = arm_stats(arms[a]["det"], arms[a]["seg"], w, arms[a]["tau"])
            for k in point[a]:
                out[a][k][i] = st.get(k, np.nan)
    return out


def paired_bootstrap(arms: dict[str, dict], contrasts: Sequence[tuple[str, str]],
                     n: int = P.N_BOOT, seed: int = P.BOOT_SEED) -> dict:
    """arms[name] = {"det": DetectionData | None, "seg": SegData | None, "tau": float | None}.
    Same well draw for every arm and metric; differences for each (x, y) contrast as x - y."""
    W, n_wells, point = bootstrap_weights(arms, n, seed)
    samples = bootstrap_samples(arms, point, W)
    return summarise_bootstrap(point, samples, contrasts, n, seed, n_wells)


def summarise_bootstrap(point: dict, samples: dict, contrasts: Sequence[tuple[str, str]],
                        n: int, seed: int, n_wells: int) -> dict:
    names = list(point)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)] if v.size else [None, None, 0]

    out = {"n_resamples": n, "seed": seed, "n_wells": n_wells, "arms": {}, "contrasts": {}}
    for a in names:
        out["arms"][a] = {k: ({"point": point[a][k], "ci95": ci(samples[a][k])} if not k.startswith("reached")
                              else point[a][k]) for k in point[a]}
    for x, y in contrasts:
        keys = [k for k in point[x] if k in point[y] and not k.startswith("reached")]
        out["contrasts"][f"{x}-{y}"] = {k: {"point": point[x][k] - point[y][k],
                                            "ci95": ci(samples[x][k] - samples[y][k])} for k in keys}
    return out


# ---------------------------------------------------------------- hypotheses (PREREG §2) ----------
# id: (x, y, metric, kind, bin or None)
HYPOTHESES = {
    "H1": ("B", "A", "cpm", "equiv", None),
    "H2": ("A", "B", "sens@1FP[large]", "pos", "large"),
    "H3": ("B", "A", "cpm[small]", "pos", "small"),
    "H4": ("C1", "B", "cpm", "pos", None),
    "H5": ("C1", "A", "dice_image", "noninf", None),
    "H6": ("C4", "A", "dice_object", "pos", None),
}


def _cond_a(kind: str, point: float, lo, hi) -> bool:
    if lo is None or hi is None:
        return False
    if kind == "equiv":
        return lo <= 0.0 <= hi and lo >= -P.H1_MARGIN and hi <= P.H1_MARGIN
    if kind == "pos":
        return lo > 0.0
    if kind == "noninf":
        return lo > P.H5_MARGIN
    raise ValueError(kind)


def _line_ok(kind: str, d: float) -> bool:
    if not np.isfinite(d):
        return False
    if kind == "equiv":
        return abs(d) <= P.H1_MARGIN
    if kind == "pos":
        return d > 0.0
    return d > P.H5_MARGIN


def verdict(hid: str, contrast_ci: dict, per_line_diff: dict[str, float],
            per_line_bin_count: dict[str, int] | None = None) -> dict:
    """Condition A (CI) and condition B (per-line direction) of PREREG §2."""
    x, y, metric, kind, b = HYPOTHESES[hid]
    pt, (lo, hi, _) = contrast_ci["point"], contrast_ci["ci95"]
    a_ok = _cond_a(kind, pt, lo, hi)
    lines = sorted(per_line_diff)
    if b is not None:
        assert per_line_bin_count is not None
        lines = [ln for ln in lines if per_line_bin_count.get(ln, 0) >= P.LINE_MIN_BIN_CELLS]
    n_part = len(lines)
    n_ok = sum(_line_ok(kind, per_line_diff[ln]) for ln in lines)
    if n_part >= 8:
        b_ok, b_state = n_ok >= 7 and n_part - n_ok <= 1, "assessed"
    elif n_part >= 5:
        b_ok, b_state = n_part - n_ok <= 1, "assessed"
    else:
        b_ok, b_state = False, "not assessable"
    desc = None
    if hid == "H1" and not (a_ok and b_ok):          # PREREG §2: description only
        desc = ("B better" if lo is not None and lo > 0 else
                "A better" if hi is not None and hi < 0 else "inconclusive")
    return {"hypothesis": hid, "contrast": f"{x}-{y}", "metric": metric, "point": pt, "ci95": [lo, hi],
            "h1_description": desc,
            "condition_A": a_ok, "lines_taking_part": n_part, "lines_in_direction": n_ok,
            "condition_B": b_ok, "condition_B_state": b_state,
            "replicated": bool(a_ok and b_ok),
            "per_line": {ln: per_line_diff[ln] for ln in sorted(per_line_diff)}}
