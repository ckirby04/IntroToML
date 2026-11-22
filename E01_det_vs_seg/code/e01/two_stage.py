"""Addendum C, variant C1: two-stage "detect (B) then segment (A)" cascade (PREREG_C_two_stage.md).

    run.cmd -m e01.two_stage            # fold-0 evaluation of C1 vs B (detection) and vs A (Dice)

Every rule here is fixed by PREREG_C_two_stage.md §2-§3; the file's SHA-256 is checked against
results\\prereg_C_sha256.txt before anything is computed. No new inference: uses the saved E01
fold-0 maps in results\\pred\\{A,B}.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

from e01 import common as c  # first: environment
from e01 import decode as dc
from e01 import evaluate as ev

ASSIGN_MM = 2.0                    # = E01 match tolerance
TAU = {"A": 0.5, "B": 0.3, "C1": 0.4, "C2": 0.4, "C3": 0.4, "C4": 0.4}   # fixed operating points; C = (0.3 + 0.5) / 2
MASK_P = 0.5                       # A's fixed-op mask threshold
FALLBACK_REL = float(np.exp(-2.0))  # heatmap >= e^-2 * peak  <=>  2 sigma  <=>  radius ESD/2
FALLBACK_RADIUS_MM = 8.0           # 2 * sigma_max
_S26 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------- C1 core (pure) ---------------
def assign_peaks(points: np.ndarray, comp_labels: np.ndarray, comp_max: np.ndarray,
                 spacing: Sequence[float], r_mm: float = ASSIGN_MM) -> np.ndarray:
    """Component label (1-based) per peak, 0 if unassigned (PREREG_C §2 'Assignment').

    Contains q -> that component; else nearest component within r_mm (mm), ties -> higher
    component max, then lower label."""
    sp = np.asarray(spacing, float)
    w = np.ceil(r_mm / sp).astype(int)
    shape = np.asarray(comp_labels.shape)
    out = np.zeros(len(points), dtype=np.int64)
    for i, q in enumerate(np.asarray(points, int)):
        k = int(comp_labels[tuple(q)])
        if k:
            out[i] = k
            continue
        lo, hi = np.maximum(q - w, 0), np.minimum(q + w + 1, shape)
        sub = comp_labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        nz = np.argwhere(sub > 0)
        if not len(nz):
            continue
        d = np.sqrt((((nz + lo - q) * sp) ** 2).sum(1))
        ok = d <= r_mm + 1e-9
        if not ok.any():
            continue
        labs = sub[tuple(nz[ok].T)]
        dk = {}
        for lab, dist in zip(labs, d[ok]):
            dk[int(lab)] = min(dk.get(int(lab), np.inf), float(dist))
        out[i] = min(dk, key=lambda L: (round(dk[L], 9), -float(comp_max[L - 1]), L))
    return out


def cascade(prob_a: np.ndarray, heat_b: np.ndarray, spacing: Sequence[float]):
    """C1 points and scores. Returns dict with points, b, a, s, assigned component, comp labels."""
    pts, b = dc.points_from_heatmap(heat_b, spacing)
    comp, n = ndimage.label(prob_a >= dc.CANDIDATE_FLOOR, structure=_S26)
    cmax = np.asarray(ndimage.maximum(prob_a, comp, index=np.arange(1, n + 1)), float) if n else np.zeros(0)
    k = assign_peaks(pts, comp, cmax, spacing) if len(pts) else np.zeros(0, np.int64)
    a = np.where(k > 0, cmax[np.maximum(k, 1) - 1] if n else 0.0, 0.0)
    return {"points": pts, "b": b, "a": a, "s": (b + a) / 2.0, "comp": k, "comp_labels": comp}


def fallback_region(heat_b: np.ndarray, q, spacing: Sequence[float]) -> np.ndarray:
    """26-cc containing q of {heat >= e^-2 * heat[q]}, clipped to a ball of 8 mm around q."""
    sp = np.asarray(spacing, float)
    q = np.asarray(q, int)
    w = np.ceil(FALLBACK_RADIUS_MM / sp).astype(int)
    lo, hi = np.maximum(q - w, 0), np.minimum(q + w + 1, np.asarray(heat_b.shape))
    sub = heat_b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    grid = np.stack(np.meshgrid(*[(np.arange(l, h) - qq) * s for l, h, qq, s in zip(lo, hi, q, sp)],
                                indexing="ij"), 0)
    ball = (grid ** 2).sum(0) <= FALLBACK_RADIUS_MM ** 2 + 1e-9
    cand = (sub >= FALLBACK_REL * heat_b[tuple(q)]) & ball
    lab, _ = ndimage.label(cand, structure=_S26)
    out = np.zeros(heat_b.shape, bool)
    lq = lab[tuple(q - lo)]
    if lq:
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = lab == lq
    else:                                    # cannot happen (q is in cand), kept for safety
        out[tuple(q)] = True
    return out


def cascade_mask(cas: dict, prob_a: np.ndarray, heat_b: np.ndarray, spacing, tau: float = TAU["C1"]) -> np.ndarray:
    """Union of regions of peaks with s >= tau (PREREG_C §2 'C1 mask')."""
    mask = np.zeros(prob_a.shape, bool)
    done_comp: set[int] = set()
    hi_a = prob_a >= MASK_P
    for q, s, k in zip(cas["points"], cas["s"], cas["comp"]):
        if s < tau:
            continue
        k = int(k)
        if k:
            if k in done_comp:
                continue
            reg = (cas["comp_labels"] == k) & hi_a
            if reg.any():
                mask |= reg
                done_comp.add(k)
                continue
        mask |= fallback_region(heat_b, q, spacing)
    return mask


def c2_scores_and_mask(out: dict, heat_b: np.ndarray, spacing, shape, tau: float = TAU["C1"]):
    """Amendment C-1.3: s = (b + a2) / 2; mask = union over s >= tau of the crop region pasted
    back (padding dropped), or the §2 heatmap fallback when the region is empty."""
    from e01.c2 import paste_or

    s = (np.asarray(out["b"], float) + np.asarray(out["a2"], float)) / 2.0
    mask = np.zeros(shape, bool)
    for i in np.flatnonzero(s >= tau):
        if out["regions"][i].any():
            paste_or(mask, out["regions"][i], out["starts"][i])
        else:
            mask |= fallback_region(heat_b, out["points"][i], spacing)
    return s, mask


C3_LARGE_ESD_MM = 10.0             # Amendment C-2.1: E01's large-lesion bin edge, not tuned


def c3_route(cas: dict, prob_a: np.ndarray, spacing) -> np.ndarray:
    """Amendment C-2.1 gate: True where q's A component restricted to p >= 0.5 has ESD > 10 mm."""
    k = np.asarray(cas["comp"], np.int64)
    n = int(cas["comp_labels"].max()) if k.size else 0
    if not n or not k.size:
        return np.zeros(k.shape, bool)
    vox = float(np.prod(np.asarray(spacing, float)))
    vol = np.asarray(ndimage.sum(prob_a >= MASK_P, cas["comp_labels"], index=np.arange(1, n + 1)), float) * vox
    esd = (6.0 * vol / np.pi) ** (1.0 / 3.0)
    return (k > 0) & (esd[np.maximum(k, 1) - 1] > C3_LARGE_ESD_MM)


def c3_scores_and_mask(cas: dict, out2: dict, prob_a: np.ndarray, heat_b: np.ndarray, spacing, shape,
                       tau: float = TAU["C1"]):
    """Amendment C-2.1: large -> C1 score and A region; else -> C2 score and crop region."""
    from e01.c2 import paste_or

    large = c3_route(cas, prob_a, spacing)
    s2 = (np.asarray(out2["b"], float) + np.asarray(out2["a2"], float)) / 2.0
    s = np.where(large, cas["s"], s2)
    mask = np.zeros(shape, bool)
    hi_a = prob_a >= MASK_P
    done: set[int] = set()
    for i in np.flatnonzero(s >= tau):
        if large[i]:
            k = int(cas["comp"][i])
            if k not in done:
                mask |= (cas["comp_labels"] == k) & hi_a
                done.add(k)
        elif out2["regions"][i].any():
            paste_or(mask, out2["regions"][i], out2["starts"][i])
        else:
            mask |= fallback_region(heat_b, out2["points"][i], spacing)
    return s, mask, large


def c4_scores_and_mask(cas: dict, out2: dict, heat_b: np.ndarray, spacing, shape, tau: float = TAU["C1"]):
    """Amendment C-3.1: C1's score for every peak; kept peaks get C2's crop region, else the fallback."""
    from e01.c2 import paste_or

    s = np.asarray(cas["s"], float)
    mask = np.zeros(shape, bool)
    for i in np.flatnonzero(s >= tau):
        if out2["regions"][i].any():
            paste_or(mask, out2["regions"][i], out2["starts"][i])
        else:
            mask |= fallback_region(heat_b, out2["points"][i], spacing)
    return s, mask


def lesion_dice(mask: np.ndarray, gt_labels: np.ndarray, n_gt: int) -> np.ndarray:
    """Per-GT-lesion Dice: P_L = union of mask components (26-cc) overlapping L; 0 if none."""
    out = np.zeros(n_gt)
    if n_gt == 0:
        return out
    pl, _ = ndimage.label(mask, structure=_S26)
    for g in range(1, n_gt + 1):
        L = gt_labels == g
        ids = np.unique(pl[L])
        ids = ids[ids > 0]
        if not len(ids):
            continue
        P = np.isin(pl, ids)
        out[g - 1] = 2.0 * (P & L).sum() / (P.sum() + L.sum())
    return out


# ---------------------------------------------------------------- evaluation -------------------
def _check_prereg():
    # hash log: one "<sha256> <utc> [label]" line per version; the last line is the current one
    # (line 1 = the version C1 was run under; later lines = dated amendments)
    lines = [ln for ln in (c.RESULTS / "prereg_C_sha256.txt").read_text().splitlines()
             if ln.strip() and len(ln.split()[0].lstrip("*")) == 64]
    rec = lines[-1].split()[0]
    got = c.sha256(c.E01 / "PREREG_C_two_stage.md")
    assert got == rec, f"PREREG_C_two_stage.md changed since its hash was recorded ({got} != {rec})"
    return got


def _wmean(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    return float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def main() -> None:
    import argparse

    from jannus.evaluation.detection.bootstrap import patient_case_weights
    from jannus.evaluation.detection.matching import extract_components

    from e01.final import lab_path, pred_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--c2", action="store_true", help="also evaluate C2 (needs results\\two_stage\\c2_pred)")
    ap.add_argument("--c3", action="store_true", help="also evaluate C3 (Amendment C-2; implies --c2)")
    ap.add_argument("--c4", action="store_true", help="also evaluate C3 and C4 (Amendment C-3; implies --c2)")
    args = ap.parse_args()
    with_c4 = args.c4
    with_c3 = args.c3 or with_c4
    with_c2 = args.c2 or with_c3
    if with_c2:
        from e01.c2_predict import load_c2, out_dir as c2_out_dir

        c2_dir = c2_out_dir("test")
        run = json.loads((c2_dir / "run.json").read_text())
        assert run["cases"] == "test" and not run["missing"] and run["epoch"] == 250, run
    prereg_sha = _check_prereg()
    c.assert_paths()
    g = c.guard()
    ids = g.pool_ids(folds=[0])
    out_dir = c.RESULTS / "two_stage"
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_arms = (["C2"] if with_c2 else []) + (["C3"] if with_c3 else []) + (["C4"] if with_c4 else [])
    seg_arms = ["A", "C1"] + extra_arms
    res = {"A": [], "B": [], "C1": [], **{k: [] for k in extra_arms}}
    dice = {k: [] for k in seg_arms}
    ldice = {k: [] for k in seg_arms}    # per case: array over that case's lesions
    rows = []
    t0 = time.time()
    for sid in ids:
        m, sp = ev.load_label(lab_path(sid))
        gt = extract_components(m, sp, connectivity=26)
        pa = ev.load_nnunet_prob(pred_path("A", sid), m.shape)
        pb = ev.load_nnunet_prob(pred_path("B", sid), m.shape)
        pid = g.patient_of(sid)
        pts_a, sc_a, _, _ = dc.points_from_prob(pa)
        res["A"].append(ev.case_result(sid, pid, pts_a, sc_a, gt))
        cas = cascade(pa, pb, sp)
        res["B"].append(ev.case_result(sid, pid, cas["points"], cas["b"], gt))
        res["C1"].append(ev.case_result(sid, pid, cas["points"], cas["s"], gt))
        mask_a = pa >= MASK_P
        mask_c = cascade_mask(cas, pa, pb, sp)
        gl = gt.labels > 0
        dice["A"].append(ev.dice(mask_a, gl))
        dice["C1"].append(ev.dice(mask_c, gl))
        ldice["A"].append(lesion_dice(mask_a, gt.labels, gt.n))
        ldice["C1"].append(lesion_dice(mask_c, gt.labels, gt.n))
        extra = {}
        if with_c2:
            o2 = load_c2(c2_dir / f"{sid}.npz")
            assert np.array_equal(o2["points"], cas["points"]) and np.allclose(o2["b"], cas["b"]), sid
            s2, mask_2 = c2_scores_and_mask(o2, pb, sp, m.shape)
            res["C2"].append(ev.case_result(sid, pid, cas["points"], s2, gt))
            dice["C2"].append(ev.dice(mask_2, gl))
            ldice["C2"].append(lesion_dice(mask_2, gt.labels, gt.n))
            m2 = res["C2"][-1].match
            k2 = s2 >= TAU["C1"]
            extra = {"C2_assigned": int(o2["assigned"].sum()), "C2_kept@0.4": int(k2.sum()),
                     "C2_kept_fallback_mask": int(sum(not o2["regions"][i].any() for i in np.flatnonzero(k2))),
                     "C2_det@0.4": int((m2.gt_score >= TAU["C1"]).sum()),
                     "C2_fp@0.4": int((m2.pred_scores[m2.pred_is_fp] >= TAU["C1"]).sum()),
                     "C2_dice": round(dice["C2"][-1], 4)}
            if with_c3:
                s3, mask_3, large = c3_scores_and_mask(cas, o2, pa, pb, sp, m.shape)
                res["C3"].append(ev.case_result(sid, pid, cas["points"], s3, gt))
                dice["C3"].append(ev.dice(mask_3, gl))
                ldice["C3"].append(lesion_dice(mask_3, gt.labels, gt.n))
                m3 = res["C3"][-1].match
                k3 = s3 >= TAU["C3"]
                extra.update({"C3_large_route": int(large.sum()), "C3_kept@0.4": int(k3.sum()),
                              "C3_kept_large_route": int((k3 & large).sum()),
                              "C3_det@0.4": int((m3.gt_score >= TAU["C3"]).sum()),
                              "C3_fp@0.4": int((m3.pred_scores[m3.pred_is_fp] >= TAU["C3"]).sum()),
                              "C3_dice": round(dice["C3"][-1], 4)})
            if with_c4:
                s4, mask_4 = c4_scores_and_mask(cas, o2, pb, sp, m.shape)
                res["C4"].append(ev.case_result(sid, pid, cas["points"], s4, gt))
                dice["C4"].append(ev.dice(mask_4, gl))
                ldice["C4"].append(lesion_dice(mask_4, gt.labels, gt.n))
                k4 = s4 >= TAU["C4"]
                extra.update({"C4_kept@0.4": int(k4.sum()),
                              "C4_kept_fallback_mask": int(sum(not o2["regions"][i].any() for i in np.flatnonzero(k4))),
                              "C4_dice": round(dice["C4"][-1], 4)})
        keep = cas["s"] >= TAU["C1"]
        mC = res["C1"][-1].match
        rows.append({**extra,"case": sid, "patient": pid, "n_gt": gt.n, "B_points": len(cas["b"]),
                     "B_points_assigned": int((cas["comp"] > 0).sum()),
                     "C1_kept@0.4": int(keep.sum()),
                     "C1_kept_fallback_mask": int((keep & (cas["comp"] == 0)).sum()),
                     "C1_det@0.4": int((mC.gt_score >= TAU["C1"]).sum()),
                     "C1_fp@0.4": int((mC.pred_scores[mC.pred_is_fp] >= TAU["C1"]).sum()),
                     "A_dice@0.5": round(dice["A"][-1], 4), "C1_dice": round(dice["C1"][-1], 4)})
    print(f"per-case done in {time.time() - t0:.0f}s", flush=True)

    data = {k: ev.to_data(v) for k, v in res.items()}
    for k in res:
        assert data[k].case_ids == data["A"].case_ids and data[k].patient_ids == data["A"].patient_ids
    bins_per_case = [np.asarray(r.bins, dtype=object) for r in res["A"]]

    def seg_stats(arm, w):
        s = {"dice_scan": _wmean(dice[arm], w)}
        lw = np.concatenate([np.full(len(ld), wi) for ld, wi in zip(ldice[arm], w)])
        lv = np.concatenate(ldice[arm])
        lb = np.concatenate(bins_per_case)
        s["dice_lesion"] = _wmean(lv, lw)
        for b in ev.BIN_NAMES:
            s[f"dice_lesion[{b}]"] = _wmean(lv[lb == b], lw[lb == b])
        return s

    def all_stats(w):
        st = {k: ev.stats(data[k], w, TAU[k]) for k in res}
        for k in seg_arms:
            st[k].update(seg_stats(k, w))
        return st

    n_cases = len(ids)
    W, n_pat = patient_case_weights(data["A"], ev.N_BOOT, ev.BOOT_SEED)
    point = all_stats(np.ones(n_cases))
    point_unw = {k: ev.stats(data[k], None, TAU[k]) for k in res}      # sanity: matches E01
    for k in point_unw["A"]:
        assert np.isclose(point_unw["A"][k], point["A"][k], equal_nan=True), k
    samples = {k: {m: [] for m in point[k]} for k in point}
    t0 = time.time()
    for w in W:
        st = all_stats(w)
        for k in st:
            for m in st[k]:
                samples[k][m].append(st[k][m])
    print(f"bootstrap done in {time.time() - t0:.0f}s", flush=True)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)] if v.size else [None, None, 0]

    def diff(x, y, keys):
        return {m: {"point": point[x][m] - point[y][m],
                    "ci95": ci(np.asarray(samples[x][m]) - np.asarray(samples[y][m]))} for m in keys}

    det_keys = [m for m in point["B"] if not m.startswith(("reached", "fixed"))]
    seg_keys = [m for m in point["C1"] if m.startswith("dice")]
    arms = {k: {m: ({"point": point[k][m], "ci95": ci(samples[k][m])} if not m.startswith("reached")
                    else point[k][m]) for m in point[k]} for k in point}
    summary = {
        "addendum": "PREREG_C_two_stage.md", "addendum_sha256": prereg_sha, "variant": "C1 cascade",
        "status": "EXPLORATORY (fold-0 aggregates of A and B seen before design)",
        "n_scans": n_cases, "n_patients": n_pat, "n_lesions": int(len(data["A"].gt_score)),
        "fixed_operating_points": TAU,
        "bootstrap": {"n": ev.N_BOOT, "seed": ev.BOOT_SEED, "paired": "A, B, C1 share draws"},
        "arms": arms,
        "C1_minus_B_detection": diff("C1", "B", det_keys),
        "C1_minus_A_detection_context": diff("C1", "A", det_keys),
        "C1_minus_A_dice": diff("C1", "A", seg_keys),
        "C1_counts": {"B_points_total": int(sum(r["B_points"] for r in rows)),
                      "B_points_assigned_to_A_component": int(sum(r["B_points_assigned"] for r in rows)),
                      "kept@0.4": int(sum(r["C1_kept@0.4"] for r in rows)),
                      "kept@0.4_with_fallback_mask": int(sum(r["C1_kept_fallback_mask"] for r in rows)),
                      "redundant_points_per_scan": float(np.mean([r.match.n_redundant() for r in res["C1"]]))},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if with_c4:
        write_variant("C4", summary, rows, diff, det_keys, seg_keys, out_dir)
        return
    if with_c3:
        write_variant("C3", summary, rows, diff, det_keys, seg_keys, out_dir)
        return
    if with_c2:
        write_c2(summary, data, rows, res, diff, det_keys, seg_keys, out_dir)
        return
    (out_dir / "summary_C1.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(out_dir / "per_case_C1.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    fig_froc(data, c.REPORT / "two_stage_C1_froc.png")
    write_table(summary, c.REPORT / "two_stage_C1_table.md")
    print((c.REPORT / "two_stage_C1_table.md").read_text(encoding="utf-8"))


def write_c2(summary, data, rows, res, diff, det_keys, seg_keys, out_dir):
    """C2 outputs (Amendment C-1.4); C1's files from the original run are left untouched."""
    summary = {**summary, "variant": "C2 trained crop segmenter (Amendment C-1), with A, B, C1 on shared draws",
               "bootstrap": {**summary["bootstrap"], "paired": "A, B, C1, C2 share draws"},
               "C2_minus_B_detection": diff("C2", "B", det_keys),
               "C2_minus_A_dice": diff("C2", "A", seg_keys),
               "C2_minus_C1_detection_context": diff("C2", "C1", det_keys),
               "C2_minus_C1_dice_context": diff("C2", "C1", seg_keys),
               "C2_counts": {"B_points_total": int(sum(r["B_points"] for r in rows)),
                             "assigned": int(sum(r["C2_assigned"] for r in rows)),
                             "kept@0.4": int(sum(r["C2_kept@0.4"] for r in rows)),
                             "kept@0.4_with_fallback_mask": int(sum(r["C2_kept_fallback_mask"] for r in rows)),
                             "redundant_points_per_scan": float(np.mean([r.match.n_redundant() for r in res["C2"]]))}}
    (out_dir / "summary_C2.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(out_dir / "per_case_C2.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    fig_froc(data, c.REPORT / "two_stage_C2_froc.png")
    write_table_c2(summary, c.REPORT / "two_stage_C2_table.md")
    print((c.REPORT / "two_stage_C2_table.md").read_text(encoding="utf-8"))


VARIANTS = {
    "C3": {"amend": "C-2", "title": "B detects; size gate routes each peak to A or the C2 crop model",
           "variant": "C3 size-gated router (Amendment C-2, post hoc)",
           "status": "EXPLORATORY, POST HOC (rule written after fold-0 C1/C2 results were seen)",
           "counts": ("B_points", "C3_large_route", "C3_kept@0.4", "C3_kept_large_route")},
    "C4": {"amend": "C-3", "title": "C1 scores every B peak; kept peaks get the C2 crop model's outline",
           "variant": "C4 C1 scores + C2 outlines (Amendment C-3, post hoc)",
           "status": "EXPLORATORY, POST HOC (rule written after fold-0 C1/C2/C3 results were seen)",
           "counts": ("B_points", "C4_kept@0.4", "C4_kept_fallback_mask")},
}


def write_variant(arm, summary, rows, diff, det_keys, seg_keys, out_dir):
    """Post-hoc variant outputs (Amendments C-2.2 / C-3.2); earlier variants' files are left untouched."""
    from jannus.evaluation.detection.report import fmt_ci, md_table

    v = VARIANTS[arm]
    shown = [k for k in ("C1", "C2", "C3", "C4") if k in summary["arms"] and k <= arm]
    s = {**summary, "variant": v["variant"] + ", with A, B, " + ", ".join(shown[:-1]) + " on shared draws",
         "status": v["status"],
         "bootstrap": {**summary["bootstrap"], "paired": "A, B, " + ", ".join(shown) + " share draws"},
         f"{arm}_minus_B_detection": diff(arm, "B", det_keys), f"{arm}_minus_A_dice": diff(arm, "A", seg_keys),
         f"{arm}_minus_C1_detection": diff(arm, "C1", det_keys), f"{arm}_minus_C1_dice": diff(arm, "C1", seg_keys),
         f"{arm}_minus_C2_detection": diff(arm, "C2", det_keys), f"{arm}_minus_C2_dice": diff(arm, "C2", seg_keys),
         f"{arm}_counts": {k: int(sum(r[k] for r in rows)) for k in v["counts"]}}
    (out_dir / f"summary_{arm}.json").write_text(json.dumps(s, indent=1, default=float))
    with open(out_dir / f"per_case_{arm}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    ar = s["arms"]

    def f(e):
        return fmt_ci(e["point"], e["ci95"][0], e["ci95"][1])

    det = [("CPM (primary)", "cpm")] + [(f"Sensitivity @ {t} FP/scan", f"sens@{t}") for t in ev.FP_TARGETS]
    det += [(f"Sensitivity @ 1 FP/scan, ESD {b} mm", f"sens@1[{b}]") for b in ev.BIN_NAMES]
    det += [(f"CPM, ESD {b} mm", f"cpm[{b}]") for b in ev.BIN_NAMES]
    det += [("Fixed op: sensitivity", "fixed.sensitivity"), ("Fixed op: FP/scan", "fixed.fp_per_scan")]
    r1 = [[n, f(ar["B"][k])] + [f(ar[a][k]) for a in shown]
          + [f(s[f"{arm}_minus_B_detection"][k]) if k in det_keys else "-",
             f(s[f"{arm}_minus_C1_detection"][k]) if k in det_keys else "-"] for n, k in det]
    seg = [("Mean per-scan Dice (primary)", "dice_scan"), ("Mean per-lesion Dice, all", "dice_lesion")]
    seg += [(f"Mean per-lesion Dice, ESD {b} mm", f"dice_lesion[{b}]") for b in ev.BIN_NAMES]
    r2 = [[n, f(ar["A"][k])] + [f(ar[a][k]) for a in shown]
          + [f(s[f"{arm}_minus_A_dice"][k]), f(s[f"{arm}_minus_C2_dice"][k])] for n, k in seg]
    cnt = s[f"{arm}_counts"]
    if arm == "C3":
        note = (f"{cnt['C3_large_route']} of {cnt['B_points']} B points take the A route (A component ESD > 10 mm); "
                f"C3 keeps {cnt['C3_kept@0.4']} points at s >= 0.4, {cnt['C3_kept_large_route']} of them on the A route.")
    else:
        note = (f"C4 keeps {cnt['C4_kept@0.4']} of {cnt['B_points']} B points at s >= 0.4 (C1's score), "
                f"{cnt['C4_kept_fallback_mask']} of them with the heatmap fallback mask. "
                "C4 detection equals C1 by construction.")
    parts = [
        f"# Two-stage {arm} ({v['title']}): fold 0 "
        f"({s['n_scans']} scans, {s['n_patients']} patients, {s['n_lesions']} lesions)",
        "",
        f"**EXPLORATORY, POST HOC** - `PREREG_C_two_stage.md` Amendment {v['amend']} (SHA-256 "
        f"`{s['addendum_sha256'][:16]}...`). The rule was written after earlier fold-0 results were seen, so these "
        f"numbers are optimistic. 95% patient-bootstrap CIs, 2,000 paired resamples shared by A, B, "
        + ", ".join(shown) + ". Two-stage variants operate at s >= 0.4.",
        "", "## Detection", "",
        md_table(["Metric", "B"] + shown + [f"{arm} - B", f"{arm} - C1"], r1),
        "", "## Segmentation", "",
        md_table(["Metric", "A"] + shown + [f"{arm} - A", f"{arm} - C2"], r2),
        "", note, "",
    ]
    path = c.REPORT / f"two_stage_{arm}_table.md"
    path.write_text(chr(10).join(parts), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


def write_table_c2(s, path: Path):
    from jannus.evaluation.detection.report import fmt_ci, md_table

    A, B, C1, C2 = (s["arms"][k] for k in ("A", "B", "C1", "C2"))
    dB, dA = s["C2_minus_B_detection"], s["C2_minus_A_dice"]

    def f(e):
        return fmt_ci(e["point"], e["ci95"][0], e["ci95"][1])

    def flag(arm, k):
        if k.startswith("sens@") and "[" not in k:
            return "" if arm.get(f"reached@{k[5:]}", 1.0) else " (unreached)"
        return ""

    det = [("CPM (primary)", "cpm")] + [(f"Sensitivity @ {t} FP/scan", f"sens@{t}") for t in ev.FP_TARGETS]
    det += [(f"Sensitivity @ 1 FP/scan, ESD {b} mm", f"sens@1[{b}]") for b in ev.BIN_NAMES]
    det += [(f"CPM, ESD {b} mm", f"cpm[{b}]") for b in ev.BIN_NAMES]
    r1 = [[n, f(B[k]) + flag(B, k), f(C2[k]) + flag(C2, k), f(dB[k]), f(C1[k]), f(A[k]) + flag(A, k)] for n, k in det]
    for n, k in (("Fixed op: sensitivity", "fixed.sensitivity"), ("Fixed op: FP/scan", "fixed.fp_per_scan")):
        r1.append([n, f(B[k]), f(C2[k]), "-", f(C1[k]), f(A[k])])
    seg = [("Mean per-scan Dice (primary)", "dice_scan"), ("Mean per-lesion Dice, all", "dice_lesion")]
    seg += [(f"Mean per-lesion Dice, ESD {b} mm", f"dice_lesion[{b}]") for b in ev.BIN_NAMES]
    r2 = [[n, f(A[k]), f(C2[k]), f(dA[k]), f(C1[k])] for n, k in seg]
    cnt = s["C2_counts"]
    parts = [
        f"# Two-stage C2 (B detects, trained crop model segments): fold 0 ({s['n_scans']} scans, "
        f"{s['n_patients']} patients, {s['n_lesions']} lesions)",
        "",
        f"**EXPLORATORY** - see `PREREG_C_two_stage.md` Amendment C-1 (SHA-256 `{s['addendum_sha256'][:16]}...`). "
        "95% patient-bootstrap CIs, 2,000 paired resamples shared by A, B, C1, C2. Fixed operating points: "
        "A p >= 0.5, B peak >= 0.3, C1/C2 s >= 0.4.",
        "", "## Detection: C2 vs B", "",
        md_table(["Metric", "B: heatmap", "C2: two-stage", "C2 - B", "C1 (context)", "A (context)"], r1),
        "", "## Segmentation: C2 vs A", "",
        md_table(["Metric", "A: segmentation", "C2: two-stage", "C2 - A", "C1 (context)"], r2),
        "",
        f"C2 keeps {cnt['kept@0.4']} points at s >= 0.4 ({cnt['kept@0.4_with_fallback_mask']} with the "
        f"heatmap fallback mask); {cnt['assigned']} of {cnt['B_points_total']} B points are assigned to a "
        "stage-2 component.",
        "",
    ]
    txt = chr(10).join(parts)
    path.write_text(txt, encoding="utf-8")


def fig_froc(data, path: Path):
    from jannus.evaluation.detection.report import plot_froc_comparison

    lab = {"A": "A: segmentation", "B": "B: heatmap", "C1": "C1: B detect -> A segment",
           "C2": "C2: B detect -> crop model segment"}
    cv = {a: ev.curves(data[a]) for a in data}
    panels = [("All lesions", [(lab[a], cv[a]["all"]) for a in data], int(len(data["A"].gt_score)))]
    for b in ev.BIN_NAMES:
        panels.append((f"ESD {b} mm", [(lab[a], cv[a][b]) for a in data], int((data["A"].gt_bin == b).sum())))
    plot_froc_comparison(panels, path, x_max=4.0, targets=ev.FP_TARGETS)


def write_table(s, path: Path):
    from jannus.evaluation.detection.report import fmt_ci, md_table

    A, B, C = (s["arms"][k] for k in ("A", "B", "C1"))
    dB, dA = s["C1_minus_B_detection"], s["C1_minus_A_dice"]

    def f(e):
        return fmt_ci(e["point"], e["ci95"][0], e["ci95"][1])

    def flag(arm, k):
        if k.startswith("sens@") and "[" not in k:
            return "" if arm.get(f"reached@{k[5:]}", 1.0) else " (unreached)"
        return ""

    det = [("CPM (primary)", "cpm")] + [(f"Sensitivity @ {t} FP/scan", f"sens@{t}") for t in ev.FP_TARGETS]
    det += [(f"Sensitivity @ 1 FP/scan, ESD {b} mm", f"sens@1[{b}]") for b in ev.BIN_NAMES]
    det += [(f"CPM, ESD {b} mm", f"cpm[{b}]") for b in ev.BIN_NAMES]
    r1 = [[n, f(B[k]) + flag(B, k), f(C[k]) + flag(C, k), f(dB[k]), f(A[k]) + flag(A, k)] for n, k in det]
    for n, k in (("Fixed op: sensitivity", "fixed.sensitivity"), ("Fixed op: FP/scan", "fixed.fp_per_scan")):
        r1.append([n, f(B[k]), f(C[k]), "-", f(A[k])])
    seg = [("Mean per-scan Dice (primary)", "dice_scan"), ("Mean per-lesion Dice, all", "dice_lesion")]
    seg += [(f"Mean per-lesion Dice, ESD {b} mm", f"dice_lesion[{b}]") for b in ev.BIN_NAMES]
    r2 = [[n, f(A[k]), f(C[k]), f(dA[k])] for n, k in seg]
    cnt = s["C1_counts"]
    txt = (f"# Two-stage C1 (B detects, A segments): fold 0 ({s['n_scans']} scans, {s['n_patients']} patients, "
           f"{s['n_lesions']} lesions)\n\n**EXPLORATORY** - see `PREREG_C_two_stage.md` (SHA-256 "
           f"`{s['addendum_sha256'][:16]}...`). 95% patient-bootstrap CIs, 2,000 paired resamples. "
           "Fixed operating points: A p >= 0.5, B peak >= 0.3, C1 s >= 0.4.\n\n"
           "## Detection: C1 vs B\n\n"
           + md_table(["Metric", "B: heatmap", "C1: two-stage", "C1 - B", "A (context)"], r1)
           + "\n\n## Segmentation: C1 vs A\n\n"
           + md_table(["Metric", "A: segmentation", "C1: two-stage", "C1 - A"], r2)
           + f"\n\nB has no mask (Dice n/a). C1 keeps {cnt['kept@0.4']} points at s >= 0.4, of which "
             f"{cnt['kept@0.4_with_fallback_mask']} use the heatmap fallback mask; "
             f"{cnt['B_points_assigned_to_A_component']} of {cnt['B_points_total']} B points are assigned to an A "
             "component.\n")
    path.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
