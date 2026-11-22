"""Post-hoc DESCRIPTIVE exploration of fold-0 results (not pre-registered; nothing is tuned).
Writes results/exploratory.json. Run after e01.final."""
import json
import numpy as np
import torch
from e01 import common as c
from e01 import evaluate as ev, decode as dc
from jannus.evaluation.detection.matching import extract_components

def train_summary(folder):
    ck = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    lg = ck["logging"]
    tr, va = np.asarray(lg["train_losses"], float), np.asarray(lg["val_losses"], float)
    ep = np.asarray(lg["epoch_end_timestamps"]) - np.asarray(lg["epoch_start_timestamps"])
    pick = lambda a: {str(i): round(float(a[i]), 4) for i in (0, 49, 99, 149, 199, 249)}
    return {"epochs": len(tr), "train_loss": pick(tr), "val_loss_fold0": pick(va),
            "val_loss_min_epoch": int(np.nanargmin(va)), "val_loss_min": round(float(np.nanmin(va)), 4),
            "median_epoch_s": round(float(np.median(ep)), 1), "total_train_h": round(float(ep.sum()) / 3600, 2)}

g = c.guard()
ids = g.pool_ids(folds=[0])
R = c.NN_RES / c.DATASET
out = {"note": "post-hoc descriptive only; not pre-registered; not used to change anything",
       "training": {"A": train_summary(R / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres" / "fold_0"),
                    "B": train_summary(R / "E01HeatmapTrainer__nnUNetPlans__3d_fullres" / "fold_0")}}
rows = []
for sid in ids:
    m, sp = ev.load_label(c.NN_RAW / c.DATASET / "labelsTr" / f"{sid}.nii.gz")
    gt = extract_components(m, sp, connectivity=26)
    pa = ev.load_nnunet_prob(c.RESULTS / "pred" / "A" / f"{sid}.npz", m.shape)
    pb = ev.load_nnunet_prob(c.RESULTS / "pred" / "B" / f"{sid}.npz", m.shape)
    ma = ev.match_points(*dc.points_from_prob(pa)[:2], gt)
    mb = ev.match_points(*dc.points_from_heatmap(pb, sp), gt)
    for k in range(gt.n):
        les = gt.labels == k + 1
        rows.append({"bin": ev.gt_bins(gt)[k], "esd": float(gt.esd_mm[k]),
                     "a": float(ma.gt_score[k]), "b": float(mb.gt_score[k]),
                     "b_max_in_lesion": float(pb[les].max()), "a_max_in_lesion": float(pa[les].max()),
                     "b_hits": len(mb.gt_hit_by[k])})
def q(v):
    v = np.asarray(v, float); v = np.where(np.isfinite(v), v, 0.0)
    return {"median": round(float(np.median(v)), 3), "q25": round(float(np.percentile(v, 25)), 3),
            "q75": round(float(np.percentile(v, 75)), 3)}
per_bin = {}
for b in ev.BIN_NAMES:
    S = [r for r in rows if r["bin"] == b]
    per_bin[b] = {"n": len(S),
                  "A_detection_score": q([r["a"] for r in S]), "B_detection_score": q([r["b"] for r in S]),
                  "B_max_heatmap_inside_lesion": q([r["b_max_in_lesion"] for r in S]),
                  "A_max_prob_inside_lesion": q([r["a_max_in_lesion"] for r in S]),
                  "no_B_point_at_0.05": sum(not np.isfinite(r["b"]) for r in S),
                  "no_A_point_at_0.05": sum(not np.isfinite(r["a"]) for r in S),
                  "B_below_0.3": sum(r["b"] < 0.3 for r in S), "A_below_0.5": sum(r["a"] < 0.5 for r in S),
                  "B_points_per_detected_lesion_mean": round(float(np.mean([r["b_hits"] for r in S if r["b_hits"]])), 2)}
big = [r for r in rows if r["bin"] == ">10"]
esd_sorted = sorted(big, key=lambda r: r["esd"])
half = len(esd_sorted) // 2
per_bin[">10"]["B_detection_score_by_esd_half"] = {
    "smaller_half_esd_range": [round(esd_sorted[0]["esd"], 1), round(esd_sorted[half - 1]["esd"], 1)],
    "smaller_half_median_B": q([r["b"] for r in esd_sorted[:half]])["median"],
    "larger_half_esd_range": [round(esd_sorted[half]["esd"], 1), round(esd_sorted[-1]["esd"], 1)],
    "larger_half_median_B": q([r["b"] for r in esd_sorted[half:]])["median"]}
out["lesion_scores_by_bin"] = per_bin
both = lambda f: sum(f(r) for r in rows)
out["complementarity_at_fixed_ops"] = {
    "both_detect": both(lambda r: r["a"] >= 0.5 and r["b"] >= 0.3),
    "only_A": both(lambda r: r["a"] >= 0.5 and r["b"] < 0.3),
    "only_B": both(lambda r: r["a"] < 0.5 and r["b"] >= 0.3),
    "neither": both(lambda r: r["a"] < 0.5 and r["b"] < 0.3),
    "only_A_by_bin": {b: both(lambda r, b=b: r["bin"] == b and r["a"] >= 0.5 and r["b"] < 0.3) for b in ev.BIN_NAMES},
    "only_B_by_bin": {b: both(lambda r, b=b: r["bin"] == b and r["a"] < 0.5 and r["b"] >= 0.3) for b in ev.BIN_NAMES}}
(c.RESULTS / "exploratory.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
