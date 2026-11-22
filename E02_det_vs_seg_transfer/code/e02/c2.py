"""C2 crop dataset (PREREG §7.1): Dataset922_E02C2Crops from TRAINING images only.

    run.cmd -m e02.c2 --build [--workers 8]     crops + manifest + dataset.json
    run.cmd -m e02.c2 --preprocess              nnU-Net fingerprint/plan/preprocess (2d)

- Crop: 384 x 384 px of image and gapped label; outside the image, the image is padded with the
  source image's median intensity and the label with 0.
- Positives: 2 per training image, each centred on the anchor of a cell drawn uniformly (without
  replacement) from that image's cells, plus a uniform integer offset per axis in [-j, j],
  j = round(clip(ESD / 4, 16.34, 40.85) px).
- Negatives: 0.5 per positive. Images drawn in proportion to their cell counts; half the centres
  uniform over the image, half uniform over its brightest 0.5% of pixels. A centre must be
  >= 40.85 px from every cell pixel; up to 100 draws, then skipped (reported).
- Seed 20261117. Manifest: results\\two_stage\\c2_crop_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from e02 import common as c
from e02 import params as P
from e02.heatmap import find_cells
from e02.two_stage import crop_start, extract_crop

DATASET_ID = 922
DATASET = "Dataset922_E02C2Crops"
MANIFEST = c.RESULTS / "two_stage" / "c2_crop_manifest.csv"


def jitter_halfwidth(esd: float) -> int:
    return int(round(float(np.clip(esd / 4.0, *P.C2_JITTER_PX))))


def negative_plan(cell_counts, n_pos: int, seed: int = P.C2_SEED) -> np.ndarray:
    """Per image (n_rand, n_bright): images drawn in proportion to cell count, types half/half."""
    rng = np.random.default_rng(seed)
    n_neg = int(np.floor(P.C2_NEG_PER_POS * n_pos))
    cnt = np.asarray(cell_counts, float)
    draws = rng.choice(len(cnt), size=n_neg, p=cnt / cnt.sum())
    types = rng.permutation(np.r_[np.zeros(n_neg // 2, int), np.ones(n_neg - n_neg // 2, int)])
    plan = np.zeros((len(cnt), 2), int)
    np.add.at(plan, (draws, types), 1)
    return plan


def sample_image_crops(img: np.ndarray, inst: np.ndarray, n_rand: int, n_bright: int, rng) -> list[dict]:
    out = []
    cells = find_cells(inst)
    n_pos = min(P.C2_POS_PER_IMAGE, len(cells))
    for k, ci in enumerate(sorted(rng.choice(len(cells), size=n_pos, replace=False)) if n_pos else []):
        cell = cells[int(ci)]
        j = jitter_halfwidth(cell.esd)
        off = rng.integers(-j, j + 1, size=2)
        out.append({"kind": "pos", "idx": k, "cell_id": cell.id, "anchor": cell.anchor,
                    "offset": tuple(int(o) for o in off),
                    "center": tuple(int(v) for v in np.asarray(cell.anchor) + off),
                    "esd_px": round(cell.esd, 3)})
    if n_rand + n_bright == 0:
        return out
    dist = ndimage.distance_transform_edt(inst == 0) if (inst > 0).any() else np.full(inst.shape, np.inf)
    allpix = np.arange(img.size)
    thr = np.quantile(img.ravel(), 1.0 - P.C2_BRIGHT_FRAC)
    bright = allpix[img.ravel() >= thr]
    k = 0
    for kind, pool, n in (("neg_rand", allpix, n_rand), ("neg_bright", bright, n_bright)):
        for _ in range(n):
            for t in range(P.C2_NEG_RETRIES):
                v = int(pool[rng.integers(len(pool))])
                if dist.ravel()[v] >= P.C2_NEG_MIN_DIST_PX:
                    cen = np.unravel_index(v, img.shape)
                    out.append({"kind": kind, "idx": k, "center": tuple(int(x) for x in cen), "tries": t + 1})
                    break
            else:
                out.append({"kind": kind + "_skipped", "idx": k, "center": None})
            k += 1
    return out


def _load(case: str):
    import tifffile
    from PIL import Image

    raw = c.NN_RAW / c.DATASET
    img = np.asarray(Image.open(raw / "imagesTr" / f"{case}_0000.png"))
    lab = np.asarray(Image.open(raw / "labelsTr" / f"{case}.png"))
    inst = tifffile.imread(c.INSTANCES / "Tr" / f"{case}.tif")
    assert img.ndim == 2 and img.shape == lab.shape == inst.shape, case
    return img, lab, inst


def _build_image(args):
    from PIL import Image

    i, case, n_rand, n_bright = args
    img, lab, inst = _load(case)
    rng = np.random.default_rng(np.random.SeedSequence([P.C2_SEED, i]))
    fill = int(np.median(img))
    out = c.NN_RAW / DATASET
    rows = []
    for r in sample_image_crops(img, inst, n_rand, n_bright, rng):
        row = {"case": case, **{k: r.get(k) for k in ("kind", "idx", "cell_id", "center", "anchor", "offset",
                                                      "esd_px", "tries")}}
        tag = {"pos": "p", "neg_rand": "r", "neg_bright": "b"}.get(r["kind"])
        if tag is None:
            rows.append({**row, "crop_id": None})
            continue
        cid = f"C2_{case}_{tag}{r['idx']:02d}"
        ci = extract_crop(img, r["center"], fill=fill)
        cl = extract_crop(lab, r["center"], fill=0)
        Image.fromarray(ci).save(out / "imagesTr" / f"{cid}_0000.png")
        Image.fromarray(cl).save(out / "labelsTr" / f"{cid}.png")
        rows.append({**row, "crop_id": cid, "start": tuple(int(v) for v in crop_start(r["center"])),
                     "label_px": int((cl > 0).sum()), "fill": fill})
    return rows


def build(workers: int = 8) -> dict:
    tr = c.image_rows("Tr")
    assert tr and all(r["split"] == "Tr" for r in tr)
    cases = [r["case"] for r in tr]
    counts = [int(r["n_instances"]) for r in tr]
    n_pos = sum(min(P.C2_POS_PER_IMAGE, n) for n in counts)
    plan = negative_plan(counts, n_pos)
    out = c.NN_RAW / DATASET
    for d in ("imagesTr", "labelsTr"):
        (out / d).mkdir(parents=True, exist_ok=True)
    assert not any((out / "imagesTr").iterdir()), f"{out} is not empty"
    jobs = [(i, cs, int(plan[i, 0]), int(plan[i, 1])) for i, cs in enumerate(cases)]
    with Pool(workers) as pool:
        rows = [r for part in pool.map(_build_image, jobs, chunksize=8) for r in part]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    keys = ["crop_id", "case", "kind", "idx", "cell_id", "center", "anchor", "offset", "esd_px", "tries",
            "start", "label_px", "fill"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    n_crops = sum(1 for r in rows if r.get("crop_id"))
    dj = {"channel_names": {"0": "phase"}, "labels": {"background": 0, "cell": 1}, "numTraining": n_crops,
          "file_ending": ".png",
          "e02": {"what": "C2 crops, PREREG §7.1", "source": c.DATASET, "source_split": "Tr only",
                  "seed": P.C2_SEED, "crop": P.C2_CROP, "manifest": str(MANIFEST)}}
    (out / "dataset.json").write_text(json.dumps(dj, indent=1))
    summary = {"crops": n_crops, "by_kind": kinds, "positives_planned": n_pos,
               "negatives_planned": int(plan.sum()), "manifest_sha256": c.sha256(MANIFEST)}
    (c.RESULTS / "two_stage" / "c2_build.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


def preprocess() -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess as pp)

    c.assert_paths()
    extract_fingerprints([DATASET_ID], num_processes=6, check_dataset_integrity=True)
    plan_experiments([DATASET_ID])
    pp([DATASET_ID], configurations=(c.CONFIG,), num_processes=(6,))
    c.assert_paths()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.build:
        build(a.workers)
    if a.preprocess:
        preprocess()
