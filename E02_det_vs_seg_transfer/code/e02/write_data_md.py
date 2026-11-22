"""Write report\\E02_data.md and the PREREG §11 figures from the evaluation outputs.

    run.cmd -m e02.write_data_md

Reads results\\eval\\summary.json, froc_curves.npz and cases\\*.pkl (per-image records), the
training logs, the C2 crop build record and the GPU checks. Data only, no interpretation
(that is report\\E02_analysis.md). Every FROC value it tabulates is checked against the
evaluation's own sensitivity at the four targets before anything is written.
Section 12 (post-hoc descriptives) is computed here from the per-image records; nothing in
the design was changed because of it.
"""
from __future__ import annotations

import glob
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

from e02 import common as c  # first: environment
from e02 import params as P

EVAL = c.RESULTS / "eval"
REP = c.REPORT
ARMS = ("A", "B", "C1", "C2")
SEG = ("A", "C1", "C2", "C4")
BINS = P.BIN_NAMES
LINES = ("A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3")
FIXED = {"A": "p >= 0.5", "B": "peak >= 0.3", "C1": "s >= 0.4", "C2": "s >= 0.4", "C4": "s >= 0.4 (C1's)"}
FIXED_T = {"A": P.TAU["A"], "B": P.TAU["B"], "C1": P.TAU["C1"], "C2": P.TAU["C2"]}
GRID = (1, 2, 5, 10, 14.95, 20, 29.9, 40, 59.81, 89.71)
# validated reference palette (dataviz references/palette.md, light mode), fixed order per arm
COL = {"A": "#2a78d6", "B": "#eb6834", "C1": "#1baf7a", "C2": "#eda100", "C4": "#e87ba4"}
LS = {"A": "-", "B": "--", "C1": "-.", "C2": ":", "C4": "-"}
MK = {"A": "o", "B": "s", "C1": "^", "C2": "D", "C4": "v"}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d9d8d4"


# ------------------------------------------------------------------ formatting ---------------------
def f3(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.3f}"


def sg(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if 0 < abs(x) < 0.0005:                  # would print as +0.000 / -0.000
        return f"{x:+.1e}"
    return f"{x:+.3f}"


def ci(e: dict | None, signed: bool = False) -> str:
    if not e:
        return "n/a"
    fn = sg if signed else f3
    lo, hi = e["ci95"][0], e["ci95"][1]
    return f"{fn(e['point'])} [{fn(lo)}, {fn(hi)}]"


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ FROC ---------------------------
def froc(z, arm: str, b: str):
    return z[f"{arm}|{b}|thresholds"], z[f"{arm}|{b}|sensitivity"], z[f"{arm}|{b}|fp_per_scan"]


def sens_at(fp: np.ndarray, se: np.ndarray, t: float) -> tuple[float, bool]:
    """Step FROC, no interpolation: best sensitivity with FP/image <= t; curve max if t is beyond it."""
    ok = fp <= t
    reached = bool(fp[-1] >= t)
    return float(se[ok].max()) if ok.any() else 0.0, reached


def thr_at(th: np.ndarray, fp: np.ndarray, t: float) -> float:
    ok = np.flatnonzero(fp <= t)
    return float(th[ok[-1]])


# ------------------------------------------------------------------ inputs -------------------------
def training_record() -> list[dict]:
    rows = []
    for arm, ds, tr in (("A", c.DATASET, "E02SegTrainer"), ("B", c.DATASET, "E02HeatmapTrainer"),
                        ("C2", "Dataset922_E02C2Crops", "E02C2Trainer")):
        d = c.NN_RES / ds / f"{tr}__{c.PLANS}__{c.CONFIG}" / f"fold_{c.FOLD}"
        ep: dict[int, dict] = {}
        cur = None
        for log in sorted(d.glob("training_log_*.txt")):
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r": Epoch (\d+)\s*$", line)
                if m:
                    cur = int(m.group(1))
                    ep[cur] = {}
                    continue
                for key in ("train_loss", "Epoch time"):
                    m = re.search(rf": {key}:? (-?[\d.]+)", line)
                    if m and cur is not None:
                        ep[cur][key] = float(m.group(1))
        done = sorted(e for e, v in ep.items() if "Epoch time" in v)
        times = [ep[e]["Epoch time"] for e in done]
        tl = lambda e: ep.get(e, {}).get("train_loss")
        rows.append({"arm": arm, "trainer": tr, "epochs": len(done), "median_s": float(np.median(times)),
                     "hours": sum(times) / 3600, "tl": (tl(0), tl(99), tl(249)),
                     "sha": c.sha256(d / "checkpoint_final.pth")[:16]})
    return rows


def load_records() -> list[dict]:
    recs = []
    for f in sorted(glob.glob(str(EVAL / "cases" / "*.pkl"))):
        with open(f, "rb") as fh:
            recs.append(pickle.load(fh))
    return recs


def gpu_split() -> dict:
    out = {}
    for arm, d in (("A", c.RESULTS / "pred" / "A"), ("B", c.RESULTS / "pred" / "B"),
                   ("C2", c.RESULTS / "two_stage" / "c2_pred")):
        cnt: dict[str, int] = {}
        for line in (d / "gpu_by_case.jsonl").read_text(encoding="utf-8").splitlines():
            g = json.loads(line)["gpu"].replace("NVIDIA GeForce ", "")
            cnt[g] = cnt.get(g, 0) + 1
        out[arm] = cnt
    return out


def stage_seconds() -> dict:
    out = {}
    for line in (EVAL / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("event") == "stage_done":
            out[e["stage"]] = out.get(e["stage"], 0) + e["seconds"]
    return out


# ------------------------------------------------------------------ post hoc (section 12) ----------
def posthoc(recs: list[dict]) -> dict:
    bins = np.concatenate([r["bins"] for r in recs])
    sa = np.concatenate([r["det"]["one"]["A"]["gt_score"] for r in recs])
    sb = np.concatenate([r["det"]["one"]["B"]["gt_score"] for r in recs])
    out = {"by_bin": {}}
    for name in BINS:
        m = bins == name
        a, b = sa[m], sb[m]
        fa, fb = a >= P.TAU["A"], b >= P.TAU["B"]
        out["by_bin"][name] = {
            "n": int(m.sum()), "no_A": int((~np.isfinite(a)).sum()), "no_B": int((~np.isfinite(b)).sum()),
            "A_med": float(np.median(np.where(np.isfinite(a), a, 0))),
            "B_med": float(np.median(np.where(np.isfinite(b), b, 0))),
            "both": int((fa & fb).sum()), "only_A": int((fa & ~fb).sum()), "only_B": int((~fa & fb).sum()),
            "neither": int((~fa & ~fb).sum())}
    # B points per detected cell (one-hit: hits + redundant over hits)
    hits = sum(int(np.isfinite(r["det"]["one"]["B"]["gt_score"]).sum()) for r in recs)
    red = sum(r["det"]["one"]["B"]["n_redundant"] for r in recs)
    out["B_points_per_detected_cell"] = (hits + red) / hits
    # FPs at the fixed operating points, by image
    out["fp"] = {}
    for arm in ARMS:
        per = np.array([int((np.asarray(r["det"]["one"][arm]["fp_score"]) >= FIXED_T[arm]).sum()) for r in recs])
        top = np.sort(per)[::-1]
        k = max(1, int(round(0.05 * len(per))))
        out["fp"][arm] = {"total": int(per.sum()), "zero": int((per == 0).sum()),
                          "share_top5pct": float(top[:k].sum() / max(1, per.sum())), "max": int(per.max())}
    # per line: GT cells per image, A components and merged components per image
    out["by_line"] = {}
    for ln in LINES:
        rr = [r for r in recs if r["line"] == ln]
        out["by_line"][ln] = {"images": len(rr), "gt_per_image": float(np.mean([r["n_gt"] for r in rr])),
                              "a_comp": float(np.mean([r["a_components"] for r in rr])),
                              "a_merged": float(np.mean([r["a_merged"] for r in rr]))}
    tot_c = sum(r["a_components"] for r in recs)
    out["a_components_total"] = tot_c
    out["a_merged_total"] = sum(r["a_merged"] for r in recs)
    return out


# ------------------------------------------------------------------ figures ------------------------
def _style(ax):
    ax.grid(True, color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_froc(z, path: Path):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, b in zip(axs.ravel(), ("all",) + BINS):
        for arm in ARMS:
            _, se, fp = froc(z, arm, b)
            u, idx = np.unique(fp, return_index=True)          # one point per FP level (curve is monotone)
            last = np.r_[idx[1:] - 1, len(fp) - 1]
            ax.step(np.maximum(u, 0.05), se[last], where="post", color=COL[arm], ls=LS[arm], lw=2, label=arm)
            ax.plot(max(u[-1], 0.05), se[-1], marker=MK[arm], color=COL[arm], ms=8, mec="white", mew=1.5)
        for t in P.FP_TARGETS:
            ax.axvline(t, color=INK2, lw=0.8, ls=(0, (1, 3)))
        ax.set_xscale("log")
        ax.set_xlim(0.1, 120)
        ax.set_ylim(0, 1)
        ax.set_title("all cells" if b == "all" else f"{b} cells", color=INK, fontsize=10, loc="left")
        _style(ax)
    for ax in axs[1]:
        ax.set_xlabel("false positives per image (log scale)", color=INK2, fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("sensitivity (one-hit)", color=INK2, fontsize=9)
    axs[0, 0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("E02 test set: FROC by arm (dotted verticals: the four CPM targets; marker: end of curve)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_bins(s: dict, path: Path):
    import matplotlib.pyplot as plt

    arms = s["bootstrap"]["arms"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for i, arm in enumerate(ARMS):
        xs = np.arange(3) + (i - 1.5) * w
        e = [arms[arm][f"sens@1FP[{b}]"] for b in BINS]
        y = np.array([x["point"] for x in e])
        lo = y - np.array([x["ci95"][0] for x in e])
        hi = np.array([x["ci95"][1] for x in e]) - y
        ax.bar(xs, y, w * 0.9, color=COL[arm], label=arm, edgecolor="white", lw=1)
        ax.errorbar(xs, y, yerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.set_xticks(range(3), [f"{b}\n(n = {s['descriptives']['gt_by_bin'][b]:,})" for b in BINS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('sensitivity at "1 FP" (29.90 FP/image)', color=INK2, fontsize=9)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left")
    ax.set_title("Sensitivity by size bin, 95% well-bootstrap CIs (A never reaches 29.90 FP/image: its maximum)",
                 color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_lines(s: dict, path: Path):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    ref = {"H1": (-0.05, 0.05), "H5": (-0.02,)}
    for ax, h in zip(axs.ravel(), ("H1", "H2", "H3", "H4", "H5", "H6")):
        v = s["verdicts"][h]
        ys = np.arange(len(LINES))
        vals = [v["per_line"].get(ln, float("nan")) for ln in LINES]
        for y, x in zip(ys, vals):
            if x is not None and np.isfinite(x):
                ax.plot(x, y, "o", color=COL["B"] if h in ("H1", "H3") else COL["C1"] if h in ("H4", "H5") else
                        COL["A"] if h == "H2" else COL["C4"], ms=8, mec="white", mew=1.5)
            else:
                ax.text(0, y, "  not taking part", va="center", fontsize=7, color=INK2)
        p, (lo, hi) = v["point"], v["ci95"]
        ax.errorbar(p, -1.2, xerr=[[p - lo], [hi - p]], fmt="D", color=INK, ms=6, capsize=3, lw=1.2)
        for r in ref.get(h, (0.0,)):
            ax.axvline(r, color=INK2, lw=1, ls="--")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(list(ys) + [-1.2], list(LINES) + ["all (95% CI)"], fontsize=8)
        ax.invert_yaxis()
        word = "replicated" if v["replicated"] else "not replicated"
        ax.set_title(f"{h}: {v['contrast']}, {v['metric']}\n{word}", color=INK, fontsize=9, loc="left")
        _style(ax)
    fig.suptitle("Per-cell-line differences for each hypothesis (dashed: the hypothesis' threshold)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_examples(s: dict, path: Path):
    import matplotlib.pyplot as plt
    import tifffile
    from PIL import Image
    from scipy import ndimage

    from e02 import decode as dc
    from e02.predict import load_prob

    ex = s["examples_by_gt"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (why, case) in zip(axs, ex.items()):
        img = np.asarray(Image.open(c.NN_RAW / c.DATASET / "imagesTs" / f"{case}_0000.png")).astype(float)
        inst = tifffile.imread(c.INSTANCES / "Ts" / f"{case}.tif")
        edge = (ndimage.grey_dilation(inst, size=3) != ndimage.grey_erosion(inst, size=3)) & (inst > 0)
        pa, pb = load_prob("test", "A", case), load_prob("test", "B", case)
        pts, sc = dc.points_from_heatmap(pb)
        pts = pts[sc >= P.TAU["B"]]
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        ov = np.zeros(img.shape + (4,))
        ov[pa >= P.MASK_P] = [0.165, 0.471, 0.839, 0.30]            # A mask, blue tint
        ov[edge] = [1, 1, 1, 0.9]                                    # GT outlines, white
        ax.imshow(ov)
        ax.plot(pts[:, 1], pts[:, 0], "o", ms=2.2, color=COL["B"], mec="none")
        ax.set_title(f"{why.replace('_', ' ')}: {case}\n{int(inst.max() and len(np.unique(inst)) - 1)} GT cells, "
                     f"{len(pts)} B points >= 0.3", fontsize=8, color=INK, loc="left")
        ax.axis("off")
    fig.suptitle("Examples chosen by GT only (PREREG §11). White: GT outlines; blue tint: A mask p >= 0.5; "
                 "orange dots: B points with peak >= 0.3", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main -----------------------------
def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    s = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))
    z = np.load(EVAL / "froc_curves.npz")
    bs, arms, con, v, d = s["bootstrap"], s["bootstrap"]["arms"], s["bootstrap"]["contrasts"], s["verdicts"], s["descriptives"]
    verify = json.loads((c.RESULTS / "dataset_verify.json").read_text())
    c2b = json.loads((c.RESULTS / "two_stage" / "c2_build.json").read_text())
    gd = json.loads((c.RESULTS / "gpu_diff" / "gpu_diff.json").read_text())
    bench = json.loads((c.RESULTS / "two_stage" / "c2_bench.json").read_text())

    # check: our FROC reading reproduces the evaluation's sensitivities at the 4 targets
    for arm in ARMS:
        _, se, fp = froc(z, arm, "all")
        for k, t in enumerate(P.FP_TARGETS, 1):
            got, reached = sens_at(fp, se, t)
            assert abs(got - arms[arm][f"sens@t{k}"]["point"]) < 1e-9, (arm, k, got)
            assert reached == bool(arms[arm][f"reached@t{k}"]), (arm, k)
    recs = load_records()
    assert len(recs) == d["n_images"] == 1460
    ph = posthoc(recs)
    tr = training_record()
    gs = gpu_split()
    secs = stage_seconds()
    REP.mkdir(exist_ok=True)

    L: list[str] = []
    w = L.append
    w("# E02 test-set results: data record\n")
    w("Data only, with no interpretation (see `E02_analysis.md`). Generated by `e02.write_data_md` from "
      "`results/eval/summary.json`, `results/eval/froc_curves.npz`, the per-image records in "
      "`results/eval/cases/`, the training logs and the build records.\n")
    w(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; evaluation finished "
      f"{s['meta']['utc']}. Point estimates are followed by 95% well-level cluster-bootstrap CIs "
      f"({bs['n_resamples']:,} paired resamples, seed {bs['seed']}, {bs['n_wells']} wells). "
      "\"(unreached)\" marks an FP target beyond the end of an arm's FROC: the value is the curve's maximum, "
      "flagged and not extrapolated. Matching is the pre-registered one-hit rule unless stated otherwise. "
      "C4's detection equals C1's by construction, so C4 appears only in the segmentation rows.\n")
    w(f"PREREG SHA-256 at the test run: `{s['meta']['prereg_sha256'][:16]}…`.\n")

    # 1 test set
    w("## 1. Test set\n")
    ts = verify["Ts"]
    w(table(["Item", "Value"], [
        ["Images / wells / fields of view", f"{d['n_images']:,} / {ts['wells']} / {ts['fovs']}"],
        ["GT cells (instances)", f"{d['n_gt']:,} ({ts['n_lost']} polygons lost to the overlap rule)"],
        ["Cells per size bin (small / medium / large)",
         " / ".join(f"{d['gt_by_bin'][b]:,}" for b in BINS) + "  (ESD < 13.63 px, 13.63–29.45, ≥ 29.45)"],
        ["Cells emptied by the gap rule (left out of object Dice)", d["objects_left_out_of_object_dice"]],
        ["Images by confluence tertile (low / mid / high)",
         " / ".join(str(d["images_by_tertile"][t]) for t in ("low", "mid", "high"))],
        ["Wells shared with training", verify["shared_wells"]],
    ]))
    w("")
    w(table(["Cell line", "Images", "Wells", "GT cells", "small", "medium", "large", "Cells per image"],
            [[ln, d["images_by_line"][ln], 2 if ln == "SKOV3" else 1,
              f"{sum(d['gt_by_line_bin'][ln].values()):,}"] +
             [f"{d['gt_by_line_bin'][ln][b]:,}" for b in BINS] + [f"{ph['by_line'][ln]['gt_per_image']:.1f}"]
             for ln in LINES]))
    w("")

    # 2 settings
    w("## 2. Fixed evaluation settings (PREREG §5–§8)\n")
    w(table(["Setting", "Value"], [
        ["Hit rule", f"one-hit: a point hits at most one cell (the one it lies on, else the nearest within "
                     f"{P.HIT_TOL_PX:.2f} px); extra points on a cell are redundant, not FPs"],
        ["Candidate floor", f"{P.CANDIDATE_FLOOR} (every arm)"],
        ["A points", "1 per 8-connected component of p ≥ 0.05, at its maximum"],
        ["B points", f"local maxima, NMS half-width {P.NMS_HALFWIDTH_PX} px, value ≥ 0.05"],
        ["C1 / C2 / C4 points", "B's points, rescored (§7.1)"],
        ["Fixed operating points", ", ".join(f"{a} {t}" for a, t in FIXED.items())],
        ["CPM FP targets (per image)", ", ".join(f"{t:.2f}" for t in P.FP_TARGETS) + ' ("1 FP" = 29.90)'],
        ["Inference", "nnU-Net sliding window, step 0.5, Gaussian, mirroring TTA, fp32, checkpoint_final"],
    ]))
    w("")

    # 3 hypotheses
    w("## 3. Hypotheses (PREREG §2)\n")
    rows = []
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        x = v[h]
        nb = f"{x['lines_in_direction']} of {x['lines_taking_part']}"
        rows.append([h + (" (primary)" if h == "H4" else ""), x["contrast"], x["metric"],
                     ci({"point": x["point"], "ci95": x["ci95"]}, True), "yes" if x["condition_A"] else "no",
                     nb, "yes" if x["condition_B"] else "no",
                     ("**replicated**" if x["replicated"] else "not replicated") +
                     (f" ({x['h1_description']})" if x.get("h1_description") else "")])
    w(table(["Hypothesis", "Contrast", "Metric", "Point [95% CI]", "Cond. A", "Lines in direction",
             "Cond. B", "Outcome"], rows))
    w("\nCondition A: H1 CI includes 0 and lies within ±0.05; H2–H4, H6 CI > 0; H5 CI lower bound > −0.02. "
      "Condition B: the per-line point estimate points the hypothesised way in ≥ 7 of 8 lines (H2, H3: lines "
      "with ≥ 200 test cells in the bin take part; with fewer than 8, at most one may go the other way). "
      "H1's description is by where its CI lies.\n")
    w("### 3.1 Per-cell-line differences\n")
    w(table(["Line"] + [f"{h}: {v[h]['contrast']} {v[h]['metric']}" for h in ("H1", "H2", "H3", "H4", "H5", "H6")],
            [[ln] + [(sg(v[h]["per_line"].get(ln)) if v[h]["per_line"].get(ln) is not None and
                      np.isfinite(v[h]["per_line"].get(ln)) else "not taking part")
                     for h in ("H1", "H2", "H3", "H4", "H5", "H6")] for ln in LINES]))
    w("\nFigure: `report/per_line_differences.png`.\n")

    # 4 detection endpoints
    w("## 4. Detection endpoints by arm\n")
    met = [("cpm", "CPM (all cells)")] + [(f"sens@t{k}", f"Sensitivity @ {t:.2f} FP/image")
                                         for k, t in enumerate(P.FP_TARGETS, 1)]
    met += [(f"cpm[{b}]", f"CPM, {b}") for b in BINS] + [(f"sens@1FP[{b}]", f'Sensitivity @ "1 FP", {b}') for b in BINS]
    rows = []
    for key, name in met:
        r = [name]
        for arm in ARMS:
            e = arms[arm][key]
            tag = ""
            m = re.match(r"sens@t(\d)", key)
            if m and not arms[arm][f"reached@t{m.group(1)}"]:
                tag = " (unreached)"
            if key.startswith("sens@1FP") and 2 in s["unreached_targets"].get(arm, []):
                tag = " (unreached)"
            r.append(ci(e) + tag)
        rows.append(r)
    w(table(["Metric"] + list(ARMS), rows))
    w("\nUnreached targets (1 = 14.95 … 4 = 89.71 FP/image): " +
      "; ".join(f"{a}: {', '.join(map(str, t))}" for a, t in s["unreached_targets"].items()) + ".\n")

    w("### 4.1 Fixed operating points\n")
    rows = []
    for key, name in (("fixed.sensitivity", "Sensitivity, all"), ("fixed.sensitivity[small]", "Sensitivity, small"),
                      ("fixed.sensitivity[medium]", "Sensitivity, medium"),
                      ("fixed.sensitivity[large]", "Sensitivity, large"), ("fixed.fp_per_image", "FP per image")):
        rows.append([name] + [ci(arms[a][key]) for a in ARMS])
    w(table(["Metric"] + [f"{a} ({FIXED[a]})" for a in ARMS], rows))
    w("")

    w("### 4.2 Thresholds at the FP targets (all cells)\n")
    rows = []
    for k, t in enumerate(P.FP_TARGETS, 1):
        r = [f"{t:.2f}"]
        for arm in ARMS:
            th, se, fp = froc(z, arm, "all")
            val, reached = sens_at(fp, se, t)
            r += [f"{thr_at(th, fp, t):.4f}" if reached else f"{th[-1]:.4f} (floor)", f3(val),
                  "yes" if reached else "**no**"]
        rows.append(r)
    w(table(["FP/image"] + [f"{a} {x}" for a in ARMS for x in ("threshold", "sens", "reached")], rows))
    w("\nEnd of each curve (FP/image, sensitivity): " + "; ".join(
        f"{a} {froc(z, a, 'all')[2][-1]:.2f}, {froc(z, a, 'all')[1][-1]:.3f}" for a in ARMS) + ".\n")

    w("### 4.3 FROC values on a grid of FP rates\n")
    w("Step FROC, no interpolation. `*` = beyond the end of the curve (its maximum shown).\n")
    head = ["FP/image"] + [f"{a} {b}" for a in ARMS for b in ("all",) + BINS]
    rows = []
    for g in GRID:
        r = [f"{g:g}"]
        for arm in ARMS:
            for b in ("all",) + BINS:
                _, se, fp = froc(z, arm, b)
                val, reached = sens_at(fp, se, g)
                r.append(f3(val) + ("" if reached else "*"))
        rows.append(r)
    w(table(head, rows))
    w("\nFigures: `report/froc_comparison.png`, `report/sensitivity_by_size_bin.png`.\n")

    # 5 segmentation
    w("## 5. Segmentation endpoints\n")
    rows = []
    for key, name in (("dice_image", "Per-image Dice (mean)"), ("dice_object", "Per-object Dice, all cells")) + \
            tuple((f"dice_object[{b}]", f"Per-object Dice, {b}") for b in BINS):
        rows.append([name] + [ci(arms[a][key]) for a in SEG] + ["n/a"])
    w(table(["Metric"] + [f"{a}" for a in SEG] + ["B"], rows))
    w("\nMasks: A p ≥ 0.5; C1, C2, C4 as PREREG §7.1. Ground truth: the gapped binary label.\n")

    # 6 contrasts
    w("## 6. Paired contrasts\n")
    keys = ["cpm", "sens@t1", "sens@t2", "cpm[small]", "cpm[medium]", "cpm[large]", "sens@1FP[small]",
            "sens@1FP[medium]", "sens@1FP[large]", "fixed.sensitivity", "fixed.fp_per_image",
            "dice_image", "dice_object", "dice_object[small]", "dice_object[medium]", "dice_object[large]"]
    cols = [k for k in ("B-A", "C1-B", "C1-A", "C2-B", "C2-A", "C4-A", "C4-C1")]
    w(table(["Metric"] + cols, [[k] + [ci(con[cc].get(k), True) if k in con[cc] else "–" for cc in cols]
                                for k in keys]))
    w("\n(A − B is B − A with the sign flipped; it is in `summary.json` for H2.)\n")

    # 7 multi-hit
    w("## 7. Sensitivity analysis: E01's multi-hit rule (PREREG §7, point estimates)\n")
    mh = s["multi_hit"]
    mk = ["cpm", "sens@t1", "sens@t2", "cpm[small]", "cpm[medium]", "cpm[large]", "sens@1FP[large]", "fixed.sensitivity",
          "fixed.fp_per_image"]
    w(table(["Metric"] + list(ARMS) + ["B − A", "C1 − B"],
            [[k] + [f3(mh["arms"][a].get(k)) for a in ARMS] +
             [sg(mh["contrasts"]["B-A"].get(k)), sg(mh["contrasts"].get("C1-B", {}).get(k))] for k in mk]))
    w("")

    # 8 tertiles and lines
    w("## 8. By confluence tertile and by cell line (point estimates)\n")
    tk = [("cpm", "CPM"), ("dice_image", "image Dice"), ("dice_object", "object Dice")]
    rows = []
    for t in ("low", "mid", "high"):
        pt = s["per_tertile"][t]
        rows.append([f"{t} ({d['images_by_tertile'][t]} images)"] +
                    [f3(pt[a].get(k)) for k, _ in tk[:1] for a in ARMS] +
                    [f3(pt[a].get(k)) for k, _ in tk[1:] for a in SEG])
    head = ["Tertile"] + [f"CPM {a}" for a in ARMS] + [f"{n} {a}" for _, n in tk[1:] for a in SEG]
    w(table(head, rows))
    w("\nTertile edges (cells per image, from training images): < 143, 143–323, ≥ 323.\n")
    rows = []
    for ln in LINES:
        pl = s["per_line"][ln]
        rows.append([ln] + [f3(pl[a]["cpm"]) for a in ARMS] + [f3(pl[a]["dice_image"]) for a in SEG] +
                    [f3(pl[a]["dice_object"]) for a in SEG])
    w(table(["Line"] + [f"CPM {a}" for a in ARMS] + [f"image Dice {a}" for a in SEG] +
            [f"object Dice {a}" for a in SEG], rows))
    w("")

    # 9 descriptives
    w("## 9. Descriptives (PREREG §10)\n")
    pp, rp = d["points_per_image"], d["redundant_per_image"]
    w(table(["Quantity", "A", "B", "C1", "C2"], [
        ["Candidate points per image (≥ 0.05)"] + [f"{pp['one'][a]:.1f}" for a in ARMS],
        ["Redundant points per image, one-hit"] + [f"{rp['one'][a]:.2f}" for a in ARMS],
        ["Redundant points per image, multi-hit"] + [f"{rp['multi'][a]:.2f}" for a in ARMS],
    ]))
    w("")
    am = d["A_merged"]
    w(table(["A's components at p ≥ 0.05", "Value"], [
        ["Total components / per image", f"{ph['a_components_total']:,} / {ph['a_components_total'] / d['n_images']:.1f}"],
        ["Merged (overlap ≥ 2 GT cells): total / per image", f"{am['total']:,} / {am['per_image']:.2f}"],
        ["Merged per image by tertile (low / mid / high)",
         " / ".join(f"{am['by_tertile'][t]:.2f}" for t in ("low", "mid", "high"))],
        ["Points with score exactly 1.0 (saturated)", d["A_saturated_points"]],
    ]))
    w("")
    w(table(["Line", "GT cells per image", "A components per image", "A merged components per image"],
            [[ln, f"{x['gt_per_image']:.1f}", f"{x['a_comp']:.1f}", f"{x['a_merged']:.1f}"]
             for ln, x in ph["by_line"].items()]))
    w("")

    # 10 training and inference record
    w("## 10. Training and inference record\n")
    w(table(["Arm", "Trainer", "Epochs", "Median s/epoch", "Train hours", "Train loss ep 0 / 99 / 249",
             "checkpoint_final SHA-256 (first 16)"],
            [[r["arm"], r["trainer"], r["epochs"], f"{r['median_s']:.1f}", f"{r['hours']:.2f}",
              " / ".join(f"{x:.4f}" if x is not None else "–" for x in r["tl"]), f"`{r['sha']}`"] for r in tr]))
    w("\nAll arms: 250 epochs, fold `all`, seed 1600, bf16 autocast without GradScaler, RTX 3070 Ti, one at a "
      "time. Losses are not comparable across arms (A, C2: Dice + CE; B: focal). nnU-Net's validation numbers "
      "under fold `all` are on training images and were not used (PREREG §9).\n")
    w(table(["C2 crop dataset (PREREG §7.1)", "Pre-registered text", "Built"], [
        ["Positive crops", "7,454 (2 per training image)", f"{c2b['by_kind']['pos']:,} "
         "(2 per image, or every cell if an image has fewer than 2)"],
        ["Negative attempts", "3,727 (0.5 per positive)", f"{c2b['negatives_planned']:,} (floor of 0.5 × "
         f"{c2b['positives_planned']:,})"],
        ["Negative crops built: random / bright", "–", f"{c2b['by_kind']['neg_rand']:,} / {c2b['by_kind']['neg_bright']:,}"],
        ["Negative attempts skipped (distance rule): random / bright", "–",
         f"{c2b['by_kind']['neg_rand_skipped']:,} / {c2b['by_kind']['neg_bright_skipped']:,}"],
        ["Total crops", "–", f"{c2b['crops']:,}"],
        ["Manifest SHA-256 (first 16)", "–", f"`{c2b['manifest_sha256'][:16]}`"],
    ]))
    w("")
    w(table(["Test-run stage", "Wall-clock", "Images per GPU (3070 Ti / 5060 Ti)"], [
        ["A prediction", f"{secs['pred_A'] / 60:.1f} min", f"{gs['A'].get('RTX 3070 Ti', 0)} / {gs['A'].get('RTX 5060 Ti', 0)}"],
        ["B prediction", f"{secs['pred_B'] / 60:.1f} min", f"{gs['B'].get('RTX 3070 Ti', 0)} / {gs['B'].get('RTX 5060 Ti', 0)}"],
        ["C2 stage 2 (crops at every B peak)", f"{secs['pred_C2'] / 3600:.2f} h",
         f"{gs['C2'].get('RTX 3070 Ti', 0)} / {gs['C2'].get('RTX 5060 Ti', 0)}"],
        ["Per-image scoring", f"{secs['cases'] / 60:.1f} min", "CPU, 6 processes"],
        ["Bootstrap", f"{secs['bootstrap'] / 60:.1f} min", "CPU, 6 processes"],
    ]))
    w("\nThe test set was predicted once, split image by image across the two GPUs; which GPU produced each "
      "image is recorded in `gpu_by_case.jsonl` next to the predictions. C2's crops ran through `CropPredictor` "
      "(nnU-Net's own preprocessing, forward pass and export, without its per-call overhead), which is "
      f"bit-identical to nnU-Net's single-array path ({bench['batched']['1']['vs_reference']['stage2_identical']}"
      f"/{bench['batched']['1']['vs_reference']['n']} crops, max |Δp| "
      f"{bench['batched']['1']['vs_reference']['max_abs_diff']}; `results/two_stage/c2_bench.json`).\n")
    w("**Cross-GPU check** (`results/gpu_diff/gpu_diff.json`, 24 training images, before the test run). Same "
      f"torch {gd['env']['torch'][0]}, cuDNN {gd['env']['cudnn'][0]}, TF32 convolutions on both:\n")
    w(table(["Arm", "Max |Δp|", "Pixels across the mask threshold", "Points identical", "Other"], [
        ["A", f"{gd['A']['max_abs_diff']:.4f}", f"{gd['A']['px_flip_mask']} of {gd['A']['px']:,}",
         f"{gd['A']['points_same_set']} of {gd['A']['points_1']}", f"max score Δ {gd['A']['max_score_diff']:.4f}"],
        ["B", f"{gd['B']['max_abs_diff']:.4f}", f"{gd['B']['px_flip_mask']} of {gd['B']['px']:,} (at 0.3)",
         f"{gd['B']['points_same_set']:,} of {gd['B']['points_1']:,}", f"max score Δ {gd['B']['max_score_diff']:.4f}"],
        ["C2", f"{gd['C2']['max_abs_diff']:.4f}", f"{gd['C2']['px_flip_mask']:,} over {gd['C2']['crops']:,} crops",
         "same crops (fixed input)", f"assigned differs {gd['C2']['assigned_differs']}; max a₂ Δ "
                                    f"{gd['C2']['max_a2_diff']:.4f}; regions differing by ≥ 1 px "
                                    f"{gd['C2']['regions_differ']:,}"],
    ]))
    w("")

    # 11 examples
    ex = s["examples_by_gt"]
    w("## 11. Example images (chosen by GT only, PREREG §11)\n")
    w(table(["Rule", "Image"], [[k.replace("_", " "), f"`{x}`"] for k, x in ex.items()]))
    w("\nFigure: `report/example_images.png`.\n")

    # 12 post hoc
    w("## 12. Post-hoc descriptives (not pre-registered; nothing was changed because of them)\n")
    w("Per-cell detection score = max score of the points that hit the cell (one-hit; 0 if none). Agreement at "
      "the fixed operating points (A p ≥ 0.5, B ≥ 0.3).\n")
    w(table(["Bin", "Cells", "No A point at 0.05", "No B point at 0.05", "A median score", "B median score",
             "Both detect", "Only A", "Only B", "Neither"],
            [[b, f"{x['n']:,}", f"{x['no_A']:,}", f"{x['no_B']:,}", f3(x["A_med"]), f3(x["B_med"]),
              f"{x['both']:,}", f"{x['only_A']:,}", f"{x['only_B']:,}", f"{x['neither']:,}"]
             for b, x in ph["by_bin"].items()]))
    w(f"\nB points per detected cell (one-hit, hits + redundant over hits): {ph['B_points_per_detected_cell']:.2f}.\n")
    w(table(["Arm (fixed op)", "Total FP", "Images with 0 FP", "Share of FPs in the top 5% of images", "Max FP in one image"],
            [[f"{a} ({FIXED[a]})", f"{x['total']:,}", x["zero"], f"{x['share_top5pct']:.2f}", x["max"]]
             for a, x in ph["fp"].items()]))
    w("")

    # 13 files
    w("## 13. Files\n")
    w("- `results/eval/summary.json`: all endpoints, CIs, verdicts, per line, per tertile, multi-hit, descriptives\n"
      "- `results/eval/froc_curves.npz`: full FROC curves; `results/eval/cases/*.pkl`: per-image records\n"
      "- `results/eval/run_log.jsonl`: the test run, stage by stage, with code and model hashes\n"
      "- `results/pred/{A,B}`, `results/two_stage/c2_pred`: predictions, with `model.json` and `gpu_by_case.jsonl`\n"
      "- `report/froc_comparison.png`, `report/sensitivity_by_size_bin.png`, `report/per_line_differences.png`, "
      "`report/example_images.png`\n"
      "- `PREREG.md`: design and deviations")

    (REP / "E02_data.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    fig_froc(z, REP / "froc_comparison.png")
    fig_bins(s, REP / "sensitivity_by_size_bin.png")
    fig_lines(s, REP / "per_line_differences.png")
    fig_examples(s, REP / "example_images.png")
    (c.RESULTS / "eval" / "posthoc.json").write_text(json.dumps(ph, indent=1), encoding="utf-8")
    print("wrote", REP / "E02_data.md")


if __name__ == "__main__":
    main()
