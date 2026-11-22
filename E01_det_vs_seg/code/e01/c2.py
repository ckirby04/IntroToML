"""Addendum C, variant C2 (PREREG_C_two_stage.md, Amendment C-1): crop dataset and stage-2 rules.

    run.cmd -m e01.c2 --build          write Dataset902_E01C2Crops from pool folds 1-4 (C-1.1)
    run.cmd -m e01.c2 --preprocess     nnU-Net fingerprint / plan / preprocess (3d_fullres) for 902

Nothing here trains a model or reads fold-0 data. Training is `e01.launch_train --arm C2`.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
from scipy import ndimage

from e01 import common as c  # first: environment

DATASET_ID = 902
DATASET = "Dataset902_E01C2Crops"
CROP = 96                         # voxels per axis
SEED = 20261113
NEG_PER_POS = 0.5
NEG_MIN_DIST_MM = 10.0
BRIGHT_FRAC = 0.005               # brightest 0.5% of brain voxels
NEG_RETRIES = 100
JITTER_MM = (4.0, 10.0)           # j = round(clip(ESD/4, 4, 10) mm), per axis in voxels
TRAIN_FOLDS = (1, 2, 3, 4)
_S26 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------- crop geometry (pure) ---------
def crop_start(center: Sequence[int], size: int = CROP) -> np.ndarray:
    """Start index of a size^3 crop whose centre voxel is `center` (centre = start + size // 2)."""
    return np.asarray(center, int) - size // 2


def extract_crop(vol: np.ndarray, center: Sequence[int], size: int = CROP, fill=0) -> np.ndarray:
    """size^3 crop centred on `center`; parts outside vol are `fill` (zero padding)."""
    st = crop_start(center, size)
    out = np.full((size,) * 3, fill, dtype=vol.dtype)
    lo = np.maximum(st, 0)
    hi = np.minimum(st + size, vol.shape)
    if np.any(hi <= lo):
        return out
    src = tuple(slice(l, h) for l, h in zip(lo, hi))
    dst = tuple(slice(l - s, h - s) for l, h, s in zip(lo, hi, st))
    out[dst] = vol[src]
    return out


def paste_or(full: np.ndarray, crop: np.ndarray, start: Sequence[int]) -> None:
    """full |= crop placed at start; parts in the padding are dropped."""
    st = np.asarray(start, int)
    lo = np.maximum(st, 0)
    hi = np.minimum(st + np.asarray(crop.shape), full.shape)
    if np.any(hi <= lo):
        return
    dst = tuple(slice(l, h) for l, h in zip(lo, hi))
    src = tuple(slice(l - s, h - s) for l, h, s in zip(lo, hi, st))
    full[dst] |= crop[src]


def crop_affine(affine: np.ndarray, start: Sequence[int]) -> np.ndarray:
    a = affine.copy()
    a[:3, 3] = affine[:3, :3] @ np.asarray(start, float) + affine[:3, 3]
    return a


def jitter_halfwidth(esd: float, spacing: Sequence[float]) -> np.ndarray:
    j_mm = float(np.clip(esd / 4.0, *JITTER_MM))
    return np.asarray([int(round(j_mm / s)) for s in spacing])


# ---------------------------------------------------------------- stage 2 per crop (pure) ------
def stage2_from_crop(p2: np.ndarray, spacing: Sequence[float], tau_mask: float = 0.5):
    """§2 rules on one stage-2 crop map; the peak is the crop centre. Returns (a2, region, assigned)."""
    from e01.decode import CANDIDATE_FLOOR
    from e01.two_stage import assign_peaks

    q = np.full(3, p2.shape[0] // 2)
    lab, n = ndimage.label(p2 >= CANDIDATE_FLOOR, structure=_S26)
    if n == 0:
        return 0.0, np.zeros(p2.shape, bool), False
    cmax = np.asarray(ndimage.maximum(p2, lab, index=np.arange(1, n + 1)), float)
    k = int(assign_peaks(q[None], lab, cmax, spacing)[0])
    if k == 0:
        return 0.0, np.zeros(p2.shape, bool), False
    return float(cmax[k - 1]), (lab == k) & (p2 >= tau_mask), True


# ---------------------------------------------------------------- crop sampling ----------------
def sample_scan_crops(img: np.ndarray, lab: np.ndarray, spacing, n_rand: int, n_bright: int, rng):
    """Positive and negative crop centres for one scan (C-1.1). Returns list of dicts."""
    from e01 import heatmap as hm

    out = []
    for i, les in enumerate(hm.find_lesions(lab, spacing)):
        j = jitter_halfwidth(les.esd_mm, spacing)
        off = np.array([rng.integers(-jj, jj + 1) for jj in j])
        out.append({"kind": "pos", "idx": i, "anchor": tuple(les.anchor), "offset": tuple(int(o) for o in off),
                    "center": tuple(int(v) for v in np.asarray(les.anchor) + off), "esd_mm": round(les.esd_mm, 3)})
    if n_rand + n_bright == 0:
        return out
    if lab.any():
        dist = ndimage.distance_transform_edt(~(lab > 0), sampling=spacing)
    else:
        dist = np.full(lab.shape, np.inf)
    brain = np.flatnonzero(img != 0)
    thr = np.quantile(img.ravel()[brain], 1.0 - BRIGHT_FRAC)
    bright = brain[img.ravel()[brain] >= thr]
    k = 0
    for kind, pool, n in (("neg_rand", brain, n_rand), ("neg_bright", bright, n_bright)):
        for _ in range(n):
            for _try in range(NEG_RETRIES):
                v = int(pool[rng.integers(len(pool))])
                if dist.ravel()[v] >= NEG_MIN_DIST_MM:
                    cen = np.unravel_index(v, img.shape)
                    out.append({"kind": kind, "idx": k, "anchor": None, "offset": (0, 0, 0),
                                "center": tuple(int(x) for x in cen), "esd_mm": None, "tries": _try + 1})
                    break
            else:
                out.append({"kind": kind + "_skipped", "idx": k, "center": None})
            k += 1
    return out


def negative_plan(lesion_counts: Sequence[int], n_pos: int, seed: int = SEED):
    """Per-scan (n_rand, n_bright): scans drawn proportional to lesion count, types half/half."""
    rng = np.random.default_rng(seed)
    n_neg = int(np.floor(NEG_PER_POS * n_pos))
    cnt = np.asarray(lesion_counts, float)
    draws = rng.choice(len(cnt), size=n_neg, p=cnt / cnt.sum())
    types = rng.permutation(np.r_[np.zeros(n_neg // 2, int), np.ones(n_neg - n_neg // 2, int)])
    plan = np.zeros((len(cnt), 2), int)
    np.add.at(plan, (draws, types), 1)
    return plan


# ---------------------------------------------------------------- dataset build ----------------
def _paths(sid):
    raw = c.NN_RAW / c.DATASET
    return raw / "imagesTr" / f"{sid}_0000.nii.gz", raw / "labelsTr" / f"{sid}.nii.gz"


def _count_lesions(sid):
    import cc3d
    from e01.evaluate import load_label

    m, _ = load_label(_paths(sid)[1])
    return int(cc3d.connected_components(np.ascontiguousarray(m), connectivity=26, return_N=True)[1])


def _build_scan(args):
    import nibabel as nib
    from e01.evaluate import load_label

    i, sid, n_rand, n_bright = args
    ip, lp = _paths(sid)
    im = nib.load(str(ip))
    img = np.asarray(im.dataobj, dtype=np.float32)
    lab, sp = load_label(lp)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
    rows = []
    out = c.NN_RAW / DATASET
    for r in sample_scan_crops(img, lab, sp, n_rand, n_bright, rng):
        tag = {"pos": "p", "neg_rand": "r", "neg_bright": "b"}.get(r["kind"])
        row = {"scan": sid, **{k: r.get(k) for k in ("kind", "idx", "center", "anchor", "offset", "esd_mm", "tries")}}
        if tag is None:
            row["crop_id"] = None
            rows.append(row)
            continue
        cid = f"C2_{sid}_{tag}{r['idx']:03d}"
        st = crop_start(r["center"])
        aff = crop_affine(im.affine, st)
        ci = extract_crop(img, r["center"])
        cl = extract_crop(lab.astype(np.uint8), r["center"])
        nib.save(nib.Nifti1Image(ci, aff), str(out / "imagesTr" / f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(cl, aff), str(out / "labelsTr" / f"{cid}.nii.gz"))
        row.update({"crop_id": cid, "start": tuple(int(v) for v in st), "label_vox": int(cl.sum())})
        rows.append(row)
    return rows


def build(workers: int = 6) -> dict:
    g = c.guard()
    ids = sorted(g.pool_ids(folds=list(TRAIN_FOLDS)))
    assert all(g.fold_of(s) in TRAIN_FOLDS for s in ids) and not any(g.fold_of(s) == 0 for s in ids)
    out = c.NN_RAW / DATASET
    assert c._inside(out, c.E01)
    if (out / "dataset.json").exists():
        raise SystemExit(f"{out} already built (dataset.json exists); delete it deliberately to rebuild")
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        counts = list(ex.map(_count_lesions, ids, chunksize=8))
    n_pos = int(sum(counts))
    plan = negative_plan(counts, n_pos)
    print(f"{len(ids)} scans, {n_pos} lesions, negatives {plan.sum(0).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    jobs = [(i, s, int(plan[i, 0]), int(plan[i, 1])) for i, s in enumerate(ids)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_build_scan, jobs, chunksize=2)):
            rows.extend(r)
            if k % 25 == 0:
                print(f"  {k + 1}/{len(ids)} scans, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    made = [r for r in rows if r["crop_id"]]
    n_pos_made = sum(r["kind"] == "pos" for r in made)
    assert n_pos_made == n_pos, (n_pos_made, n_pos)
    dj = {"channel_names": {"0": "noNorm"}, "labels": {"background": 0, "metastasis": 1},
          "numTraining": len(made), "file_ending": ".nii.gz",
          "e01": {"variant": "C2 stage-2 crops (PREREG_C_two_stage.md Amendment C-1)",
                  "source": c.DATASET, "source_folds": list(TRAIN_FOLDS), "crop": CROP, "seed": SEED,
                  "split_manifest_sha256": c.SPLIT_V1_SHA256}}
    (out / "dataset.json").write_text(json.dumps(dj, indent=1))
    od = c.RESULTS / "two_stage"
    od.mkdir(parents=True, exist_ok=True)
    keys = ["crop_id", "scan", "kind", "idx", "center", "start", "anchor", "offset", "esd_mm", "tries", "label_vox"]
    with open(od / "c2_crop_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    summ = {"scans": len(ids), "lesions": n_pos, "crops": len(made),
            "by_kind": {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})},
            "negatives_with_lesion_voxels": sum(1 for r in made if r["kind"] != "pos" and r["label_vox"] > 0),
            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (od / "c2_crop_build.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    return summ


def preprocess() -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess as pp)

    c.assert_paths()
    extract_fingerprints([DATASET_ID], num_processes=6, check_dataset_integrity=True)
    plan_experiments([DATASET_ID])
    pp([DATASET_ID], configurations=(c.CONFIG,), num_processes=(6,))
    c.assert_paths()
    plans = json.loads((c.NN_PRE / DATASET / f"{c.PLANS}.json").read_text())
    cf = plans["configurations"][c.CONFIG]
    print("C2 3d_fullres plan:", {k: cf.get(k) for k in ("patch_size", "batch_size", "spacing",
                                                          "normalization_schemes", "use_mask_for_norm")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.preprocess:
        preprocess()
