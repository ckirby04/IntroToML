"""Phase 4: fold-0 evaluation of both arms, summary.json and report figures (PREREG §5–§9).

    run.cmd -m e01.final

Order (so nothing is chosen after seeing predictions):
1. Example scans are selected from GT only (PREREG §9) and written to
   results\\example_selection.json BEFORE any prediction is loaded.
2. Each fold-0 scan: GT components (26-conn) on the Dataset901 grid; A points (components at
   p >= 0.05, argmax), B points (3 mm NMS, >= 0.05); point-rule matching at 2.0 mm.
3. Paired patient bootstrap (2,000, seed 20261109), fixed operating points A p>=0.5, B >=0.3.
4. results\\summary.json, results\\per_case.csv, report\\*.png, report\\results_table.md.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from e01 import common as c  # first: environment
from e01 import decode as dc
from e01 import evaluate as ev

from jannus.evaluation.detection.matching import extract_components, match_by_overlap  # noqa: E402

FIXED = {"A": 0.5, "B": 0.3}
LABEL = {"A": "A: segmentation (nnU-Net)", "B": "B: heatmap detection"}
SERIES = {"A": "#2a78d6", "B": "#eb6834"}        # jannus report slots 1-2; validated (CVD dE 24.7)
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GT_COLOR = "#1baf7a"                                # jannus slot 3


def lab_path(sid):
    return c.NN_RAW / c.DATASET / "labelsTr" / f"{sid}.nii.gz"


def img_path(sid):
    return c.NN_RAW / c.DATASET / "imagesTr" / f"{sid}_0000.nii.gz"


def pred_path(arm, sid):
    return c.RESULTS / "pred" / arm / f"{sid}.npz"


# ---------------------------------------------------------------- 1. selection (GT only) -------
def select_examples(ids, gts) -> dict:
    small = [(sum(b == "<3" for b in ev.gt_bins(gts[s])), s) for s in ids]
    most_small = sorted(small, key=lambda t: (-t[0], t[1]))[0]
    counts = sorted(((gts[s].n, s) for s in ids))
    median = counts[(len(counts) - 1) // 2]
    largest = sorted(((-int(gts[s].n_vox.max()) if gts[s].n else 0, s) for s in ids))[0]
    sel = {"rule": "PREREG §9, GT only, chosen before predictions are loaded; ties -> case id order",
           "most_lesions_lt3mm": {"case": most_small[1], "n_lt3mm": most_small[0]},
           "median_lesion_count": {"case": median[1], "n_lesions": median[0],
                                   "how": "sorted by (count, id), index (n-1)//2"},
           "largest_single_lesion": {"case": largest[1], "n_vox": -largest[0]},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (c.RESULTS / "example_selection.json").write_text(json.dumps(sel, indent=1))
    return sel


# ---------------------------------------------------------------- 2. per case --------------------
def evaluate_cases(ids, g, gts, spacings):
    from scipy import ndimage

    res = {"A": [], "B": [], "A_overlap": []}
    rows = []
    for sid in ids:
        gt, sp = gts[sid], spacings[sid]
        shape = gt.labels.shape
        pa = ev.load_nnunet_prob(pred_path("A", sid), shape)
        pb = ev.load_nnunet_prob(pred_path("B", sid), shape)
        assert np.isfinite(pa).all() and np.isfinite(pb).all(), sid
        pid = g.patient_of(sid)
        pts, sc, plabels, ncomp = dc.points_from_prob(pa)
        ra = ev.case_result(sid, pid, pts, sc, gt)
        ptb, scb = dc.points_from_heatmap(pb, sp)
        rb = ev.case_result(sid, pid, ptb, scb, gt)
        # 1.60 overlap rule, reference only (A): components at p >= 0.05, score = component max
        comp = extract_components(pa >= dc.CANDIDATE_FLOOR, sp, connectivity=26)
        osc = np.asarray(ndimage.maximum(pa, comp.labels, index=np.arange(1, comp.n + 1))) if comp.n else np.zeros(0)
        ro = ev.CaseResult(sid, pid, match_by_overlap(comp, gt, osc), ev.gt_bins(gt), {})
        merged = ev.merged_components(plabels, gt)
        d = ev.dice(pa >= 0.5, gt.labels > 0)
        for k, r in (("A", ra), ("B", rb), ("A_overlap", ro)):
            res[k].append(r)
        mA, mB = ra.match, rb.match
        rows.append({"case": sid, "patient": pid, "n_gt": gt.n,
                     "n_lt3": sum(b == "<3" for b in ev.gt_bins(gt)),
                     "A_points": len(mA.pred_scores), "B_points": len(mB.pred_scores),
                     "A_det@0.5": int((mA.gt_score >= FIXED["A"]).sum()),
                     "B_det@0.3": int((mB.gt_score >= FIXED["B"]).sum()),
                     "A_fp@0.5": int((mA.pred_scores[mA.pred_is_fp] >= FIXED["A"]).sum()),
                     "B_fp@0.3": int((mB.pred_scores[mB.pred_is_fp] >= FIXED["B"]).sum()),
                     "A_redundant": mA.n_redundant(), "B_redundant": mB.n_redundant(),
                     "A_merged@0.05": merged, "A_components@0.05": ncomp, "A_dice@0.5": round(d, 4)})
    return res, rows


# ---------------------------------------------------------------- 4. figures --------------------
def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_froc(data, path):
    from jannus.evaluation.detection.report import plot_froc_comparison

    cv = {a: ev.curves(data[a]) for a in ("A", "B")}
    n_all = int(len(data["A"].gt_score))
    panels = [("All lesions", [(LABEL[a], cv[a]["all"]) for a in ("A", "B")], n_all)]
    for b in ev.BIN_NAMES:
        nb = int((data["A"].gt_bin == b).sum())
        panels.append((f"ESD {b} mm", [(LABEL[a], cv[a][b]) for a in ("A", "B")], nb))
    pts = []
    for b in ("all",) + ev.BIN_NAMES:
        row = []
        for a in ("A", "B"):
            from jannus.evaluation.detection.metrics import operating_point
            op = operating_point(data[a], FIXED[a], bins=ev.BIN_NAMES)
            sens = op["sensitivity"] if b == "all" else op[f"sensitivity[{b}]"]
            row.append((f"{a} fixed op ({'p' if a == 'A' else 'peak'} ≥ {FIXED[a]})", op["fp_per_scan"], sens))
        pts.append(row)
    plot_froc_comparison(panels, path, x_max=4.0, targets=ev.FP_TARGETS, points=pts)


def fig_bins(boot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [("All", "sens@1.0")] + [(f"ESD {b} mm", f"sens@1[{b}]") for b in ev.BIN_NAMES]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax, "Lesion sensitivity at 1 FP/scan, with 95% patient-bootstrap CIs")
    x = np.arange(len(keys))
    for j, a in enumerate(("A", "B")):
        off = (j - 0.5) * 0.22
        for i, (_, k) in enumerate(keys):
            e = boot["arms"][a][k]
            lo, hi = e["ci95"][0], e["ci95"][1]
            ax.plot([x[i] + off] * 2, [lo, hi], color=SERIES[a], linewidth=2, solid_capstyle="round")
            ax.plot([x[i] + off], [e["point"]], marker="o", markersize=8, color=SERIES[a],
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                    label=LABEL[a] if i == 0 else None)
            ax.annotate(f"{e['point']:.2f}", (x[i] + off, hi), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(x, [k[0] for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Sensitivity", color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _to_ras(vol, affine):
    import nibabel as nib
    ornt = nib.orientations.io_orientation(affine)
    return nib.orientations.apply_orientation(vol, ornt)


def fig_examples(sel, gts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.lines import Line2D

    picks = [("Most lesions < 3 mm", sel["most_lesions_lt3mm"]["case"]),
             ("Median lesion count", sel["median_lesion_count"]["case"]),
             ("Largest single lesion", sel["largest_single_lesion"]["case"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (title, sid) in zip(axes, picks):
        im = nib.load(str(img_path(sid)))
        gt = gts[sid]
        aff = im.affine
        img = _to_ras(np.asarray(im.dataobj, dtype=np.float32), aff)
        glab = _to_ras(gt.labels, aff)
        pa = _to_ras(ev.load_nnunet_prob(pred_path("A", sid), gt.labels.shape), aff)
        pb = ev.load_nnunet_prob(pred_path("B", sid), gt.labels.shape)
        ptb, scb = dc.points_from_heatmap(pb, gt.spacing)
        keep = scb >= FIXED["B"]
        pvol = np.zeros(gt.labels.shape, np.int32)
        for i, p in enumerate(ptb[keep]):
            pvol[tuple(p)] = i + 1
        pvol = _to_ras(pvol, aff)
        big = int(np.argmax(gt.n_vox)) + 1
        z = int(round(np.argwhere(glab == big)[:, 2].mean()))
        sl = img[:, :, z]
        lo, hi = np.percentile(sl[sl > 0], [1, 99.5]) if (sl > 0).any() else (0, 1)
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
        ax.contour((glab[:, :, z] > 0).T, levels=[0.5], colors=[GT_COLOR], linewidths=1.6)
        ax.contour((pa[:, :, z] >= FIXED["A"]).T, levels=[0.5], colors=[SERIES["A"]], linewidths=1.2,
                   linestyles="dashed")
        near = np.argwhere(pvol[:, :, max(z - 2, 0):z + 3] > 0)          # B peaks within +-2 mm of slice
        if len(near):
            ax.plot(near[:, 0], near[:, 1], "+", color=SERIES["B"], markersize=11, markeredgewidth=2)
        ax.set_title(f"{title}\n{sid}: {gt.n} GT lesion{'' if gt.n == 1 else 's'}, axial slice {z}", color=TEXT, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Line2D([], [], color=GT_COLOR, linewidth=1.6, label="GT lesion"),
               Line2D([], [], color=SERIES["A"], linewidth=1.2, linestyle="dashed", label="A mask (p ≥ 0.5)"),
               Line2D([], [], color=SERIES["B"], marker="+", markersize=11, markeredgewidth=2, linestyle="none",
                      label="B peak ≥ 0.3 (within ±2 slices)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("Fold-0 examples chosen by GT only (PREREG §9); slice through the largest lesion; "
                 "neurological orientation", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main ---------------------------
def main() -> None:
    c.assert_paths()
    g = c.guard()
    ids = g.pool_ids(folds=[0])
    gts, spacings = {}, {}
    for sid in ids:
        m, sp = ev.load_label(lab_path(sid))
        gts[sid] = extract_components(m, sp, connectivity=26)
        spacings[sid] = sp
    sel = select_examples(ids, gts)                              # before any prediction is read
    for a in ("A", "B"):
        run = json.loads((c.RESULTS / "pred" / a / "run.json").read_text())
        assert run["n_cases"] == len(ids) and not run["missing"], (a, run)
    res, rows = evaluate_cases(ids, g, gts, spacings)
    data = {k: ev.to_data(v) for k, v in res.items()}
    boot = ev.paired_bootstrap({"A": data["A"], "B": data["B"]}, FIXED)
    ref = ev.stats(data["A_overlap"], None, None)
    n_scans, n_pat = len(ids), len(set(data["A"].patient_ids))
    fixed_fp_patient = {a: boot["arms"][a]["fixed.fp_per_scan"]["point"] * n_scans / n_pat for a in ("A", "B")}
    gt_check = json.loads((c.RESULTS / "gt_anchor_check.json").read_text())
    summary = {
        "experiment": "E01_det_vs_seg", "fold": 0, "n_scans": n_scans, "n_patients": n_pat,
        "n_lesions": int(len(data["A"].gt_score)),
        "lesions_per_bin_1mm_grid": {b: int((data["A"].gt_bin == b).sum()) for b in ev.BIN_NAMES},
        "rules": {"match": f"point within {ev.R_TOL_MM} mm of lesion voxel", "candidate_floor": dc.CANDIDATE_FLOOR,
                  "b_nms_mm": dc.R_NMS_MM, "fixed_operating_points": FIXED, "fp_targets": list(ev.FP_TARGETS)},
        "inference": {a: json.loads((c.RESULTS / "pred" / a / "run.json").read_text()) for a in ("A", "B")},
        "bootstrap": boot,
        "fixed_op_fp_per_patient": fixed_fp_patient,
        "redundant_points_per_scan": {a: float(np.mean([r[f"{a}_redundant"] for r in rows])) for a in ("A", "B")},
        "points_per_scan@0.05": {a: float(np.mean([r[f"{a}_points"] for r in rows])) for a in ("A", "B")},
        "A_only": {"mean_dice@0.5": float(np.mean([r["A_dice@0.5"] for r in rows])),
                   "overlap_rule_reference": {"cpm": ref["cpm"], "sens@1.0": ref["sens@1.0"],
                                              **{f"sens@1[{b}]": ref[f"sens@1[{b}]"] for b in ev.BIN_NAMES}},
                   "merged_components@0.05_total": int(sum(r["A_merged@0.05"] for r in rows)),
                   "scans_with_merged_component": int(sum(r["A_merged@0.05"] > 0 for r in rows)),
                   "components@0.05_total": int(sum(r["A_components@0.05"] for r in rows))},
        "gt_anchor_check": gt_check,
        "example_selection": sel,
        "prereg_deviations": "see PREREG.md D1-D6",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (c.RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(c.RESULTS / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = c.E01 / "report"
    rep.mkdir(exist_ok=True)
    fig_froc(data, rep / "froc_comparison.png")
    fig_bins(boot, rep / "sensitivity_by_size_bin.png")
    fig_examples(sel, gts, rep / "example_slices.png")
    write_table(summary, rep / "results_table.md")
    print(json.dumps({k: summary[k] for k in ("n_scans", "n_patients", "n_lesions", "lesions_per_bin_1mm_grid")}))


def write_table(s, path):
    from jannus.evaluation.detection.report import fmt_ci, md_table

    A, B, D = s["bootstrap"]["arms"]["A"], s["bootstrap"]["arms"]["B"], s["bootstrap"]["B_minus_A"]

    def f(e):
        return fmt_ci(e["point"], e["ci95"][0], e["ci95"][1])

    keys = [("CPM (primary)", "cpm")] + [(f"Sensitivity @ {t} FP/scan", f"sens@{t}") for t in ev.FP_TARGETS]
    keys += [(f"Sensitivity @ 1 FP/scan, ESD {b} mm", f"sens@1[{b}]") for b in ev.BIN_NAMES]
    rows = []
    for name, k in keys:
        t = k.replace("sens@", "") if k.startswith("sens@") and "[" not in k else None
        flag = lambda arm: "" if t is None or arm.get(f"reached@{t}", 1.0) else " (unreached)"  # noqa: E731
        rows.append([name, f(A[k]) + flag(A), f(B[k]) + flag(B), f(D[k])])
    for name, k in (("Fixed op: sensitivity", "fixed.sensitivity"), ("Fixed op: FP/scan", "fixed.fp_per_scan")):
        rows.append([name, f(A[k]), f(B[k]), "-"])
    txt = (f"# E01 fold-0 results ({s['n_scans']} scans, {s['n_patients']} patients, {s['n_lesions']} lesions)\n\n"
           "Point estimates with 95% patient-bootstrap CIs (2,000 resamples, paired). Fixed operating points: "
           "A p >= 0.5, B peak >= 0.3.\n\n"
           + md_table(["Metric", "A: segmentation", "B: heatmap", "B - A"], rows) + "\n")
    path.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
