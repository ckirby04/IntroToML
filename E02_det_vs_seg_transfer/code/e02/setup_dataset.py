"""Build Dataset921_E02LiveCell (nnU-Net 2D) plus instance maps from the LIVECell download.

Run:  env\\run.cmd -m e02.setup_dataset [--workers 8] [--limit N] [--verify-only]

Rules (PLAN.md §3, §3.1; owner decisions of 2026-11-16):
- Training = official train + val; test = official test. Images are read from images.zip;
  nothing in data\\raw is written.
- Doubly annotated images (same file name, different annotation sets): in training, keep the
  set with the higher COCO image id; in test, leave the image out (test GT is never a pick).
- Rasterisation: a pixel belongs to a polygon if its centre lies inside it (COCO coordinates,
  pixel (r, c) spans [c, c+1) x [r, r+1)). Annotations are drawn in ascending annotation id.
- Instance map (uint16): a pixel covered by >= 2 polygons belongs to the **smallest** of them
  (owner decision 2026-11-16, replacing "to neither": that rule erased small round cells lying
  on flat cells). Ties in area -> the higher annotation id. Instances are numbered 1..N in
  ascending annotation id; polygons left with no pixel are dropped and counted.
- Binary label for A / C2 / Dice ("gapped"): where two instances touch (8-neighbourhood), the
  pixel of the **larger** one is set to background (equal area -> the lower id gives way). This
  leaves a 1-px gap, keeps small cells whole, and no two instances stay 8-connected in the label.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from e02 import common as c

ZIP = c.RAW / "images.zip"
ANN = c.RAW / "annotations"
ZIP_DIRS = {"train": "images/livecell_train_val_images", "val": "images/livecell_train_val_images",
            "test": "images/livecell_test_images"}
H, W = 520, 704
OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
META_KEYS = ["case", "split", "source_split", "image_id", "cell_line", "well", "fov", "time", "crop",
             "n_polygons", "n_instances", "n_lost", "overlap_px", "label_px",
             "n_empty_after_gap", "n_fragmented", "n_cc_multi_instance"]


# ---------------------------------------------------------------- pure functions (unit-tested)
def parse_name(file_name: str) -> dict:
    """'A172_Phase_C7_1_00d00h00m_1.tif' -> cell line, well, field of view, time, crop."""
    p = Path(file_name).stem.split("_")
    assert len(p) == 6 and p[1] == "Phase", file_name
    return {"cell_line": p[0], "well": f"{p[0]}_{p[2]}", "fov": f"{p[0]}_{p[2]}_{p[3]}",
            "time": p[4], "crop": p[5]}


def rasterize(polygons: list[list[float]], shape=(H, W)) -> tuple[np.ndarray, dict]:
    """Instance map (uint16, overlaps -> 0, ids 1..N in input order) and counts."""
    from skimage.draw import polygon as draw_polygon

    cnt = np.zeros(shape, np.uint8)
    raw = np.zeros(shape, np.int32)
    pix = []
    for poly in polygons:
        xy = np.asarray(poly, np.float64).reshape(-1, 2)
        pix.append(draw_polygon(xy[:, 1] - 0.5, xy[:, 0] - 0.5, shape=shape))
    # largest first, so on overlap the smaller polygon is drawn last and keeps the pixels
    for k in np.argsort([-len(rr) for rr, _ in pix], kind="stable"):
        rr, cc = pix[k]
        cnt[rr, cc] = np.minimum(cnt[rr, cc].astype(np.int16) + 1, 255).astype(np.uint8)
        raw[rr, cc] = k + 1
    overlap = cnt > 1
    present = np.unique(raw[raw > 0])
    lut = np.zeros(len(polygons) + 1, np.uint16)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.uint16)
    inst = lut[raw]
    return inst, {"n_polygons": len(polygons), "n_instances": int(len(present)),
                  "n_lost": int(len(polygons) - len(present)), "overlap_px": int(overlap.sum())}


def gap_rank(inst: np.ndarray) -> np.ndarray:
    """rank[id]: lower = keeps its pixels. Smaller area first; equal area -> higher id first.
    rank[0] is the largest value, so background never makes a pixel drop."""
    n = int(inst.max())
    area = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    ids = np.arange(1, n + 1)
    order = np.lexsort((-ids, area))                 # primary: area asc; secondary: id desc
    rank = np.empty(n + 1, np.int64)
    rank[ids[order]] = np.arange(n)
    rank[0] = n + 1
    return rank


def gapped_label(inst: np.ndarray) -> np.ndarray:
    """Binary label: where two instances touch (8-nbhd), the larger one loses its border pixel
    (owner decision 2026-11-16, consistent with 'smaller wins' on overlap)."""
    rank = gap_rank(inst)
    r = rank[inst]
    pad = np.pad(r, 1, constant_values=rank[0])
    drop = np.zeros(inst.shape, bool)
    for dr, dc in OFFSETS:
        nb = pad[1 + dr: 1 + dr + inst.shape[0], 1 + dc: 1 + dc + inst.shape[1]]
        drop |= nb < r
    return ((inst > 0) & ~drop).astype(np.uint8)


def label_checks(inst: np.ndarray, lab: np.ndarray) -> dict:
    """Instances emptied or split by the gap; label components spanning > 1 instance (must be 0)."""
    from scipy import ndimage

    cc, _ = ndimage.label(lab, structure=np.ones((3, 3), bool))
    m = lab > 0
    pairs = np.unique(np.stack([cc[m], inst[m].astype(np.int64)]), axis=1)
    inst_per_cc = np.bincount(pairs[0])
    cc_per_inst = np.bincount(pairs[1], minlength=int(inst.max()) + 1)
    n = int(inst.max())
    return {"label_px": int(m.sum()),
            "n_empty_after_gap": int(n - (cc_per_inst[1:] > 0).sum()),
            "n_fragmented": int((cc_per_inst[1:] > 1).sum()),
            "n_cc_multi_instance": int((inst_per_cc[1:] > 1).sum())}


# ---------------------------------------------------------------- selection
def load_split(name: str) -> tuple[dict, dict]:
    import orjson

    d = orjson.loads((ANN / f"livecell_coco_{name}.json").read_bytes())
    imgs = {i["id"]: i for i in d["images"]}
    polys = defaultdict(list)
    for a in sorted(d["annotations"], key=lambda a: a["id"]):
        assert len(a["segmentation"]) == 1 and not a.get("iscrowd", 0)
        polys[a["image_id"]].append(a["segmentation"][0])
    return imgs, polys


def select() -> tuple[list[dict], list[dict]]:
    """Return (jobs, exclusions). One job per output case."""
    by_file = {"trainval": defaultdict(list), "test": defaultdict(list)}
    store = {}
    for split in ("train", "val", "test"):
        imgs, polys = load_split(split)
        grp = "test" if split == "test" else "trainval"
        for iid, im in imgs.items():
            assert (im["width"], im["height"]) == (W, H)
            by_file[grp][im["file_name"]].append((iid, split))
            store[(split, iid)] = polys[iid]
    jobs, excl = [], []
    for grp, files in by_file.items():
        for fn, entries in sorted(files.items()):
            entries = sorted(entries)
            if len(entries) > 1:
                excl.append({"file_name": fn, "group": grp, "entries": [list(e) for e in entries],
                             "action": "dropped (test)" if grp == "test" else f"kept image_id {entries[-1][0]}"})
                if grp == "test":
                    continue
            iid, split = entries[-1]
            jobs.append({"file_name": fn, "split": "Tr" if grp == "trainval" else "Ts", "source_split": split,
                         "image_id": iid, "polygons": store[(split, iid)]})
    return jobs, excl


# ---------------------------------------------------------------- build
def _work(job: dict) -> dict:
    import tifffile
    from PIL import Image

    case = Path(job["file_name"]).stem
    with zipfile.ZipFile(ZIP) as z:
        img = tifffile.imread(io.BytesIO(z.read(f"{ZIP_DIRS[job['source_split']]}/{job['file_name']}")))
    assert img.shape == (H, W) and img.dtype == np.uint8, (case, img.shape, img.dtype)
    inst, cnt = rasterize(job["polygons"])
    lab = gapped_label(inst)
    chk = label_checks(inst, lab)
    root = c.NN_RAW / c.DATASET
    s = job["split"]
    Image.fromarray(img).save(root / f"images{s}" / f"{case}_0000.png")
    Image.fromarray(lab).save(root / f"labels{s}" / f"{case}.png")
    tifffile.imwrite(c.INSTANCES / s / f"{case}.tif", inst, compression="zlib")
    return {"case": case, "split": s, "source_split": job["source_split"], "image_id": job["image_id"],
            **parse_name(job["file_name"]), **cnt, **chk}


def build(workers: int, limit: int | None) -> None:
    t0 = time.time()
    root = c.NN_RAW / c.DATASET
    for d in (root / "imagesTr", root / "labelsTr", root / "imagesTs", root / "labelsTs",
              c.INSTANCES / "Tr", c.INSTANCES / "Ts", c.META, c.SPLITS, c.RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    jobs, excl = select()
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} cases ({sum(j['split'] == 'Tr' for j in jobs)} Tr), {len(excl)} duplicate file names",
          flush=True)
    with Pool(workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=8), 1):
            rows.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["split"], r["case"]))
    with open(c.META / "images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_KEYS)
        w.writeheader()
        w.writerows(rows)
    (c.META / "duplicates.json").write_text(json.dumps(excl, indent=1))
    n_tr = sum(r["split"] == "Tr" for r in rows)
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "phase"}, "labels": {"background": 0, "cell": 1},
        "numTraining": n_tr, "file_ending": ".png",
        "e02": {"source": "LIVECell 2021 (CC BY-NC 4.0), official train+val -> Tr, test -> Ts",
                "raw_sha256": (c.DATA / "raw_sha256.txt").read_text().split(),
                "label": "gapped binary (setup_dataset docstring)", "instances": str(c.INSTANCES)}}, indent=1))
    print(f"built {len(rows)} cases in {time.time() - t0:.0f}s", flush=True)


def verify() -> dict:
    root = c.NN_RAW / c.DATASET
    rows = list(csv.DictReader(open(c.META / "images.csv")))
    out = {}
    for s in ("Tr", "Ts"):
        cases = {r["case"] for r in rows if r["split"] == s}
        imgs = {p.name[: -len("_0000.png")] for p in (root / f"images{s}").glob("*_0000.png")}
        labs = {p.stem for p in (root / f"labels{s}").glob("*.png")}
        insts = {p.stem for p in (c.INSTANCES / s).glob("*.tif")}
        assert cases == imgs == labs == insts, f"{s}: file sets differ"
        sel = [r for r in rows if r["split"] == s]
        out[s] = {"images": len(cases), "wells": len({r["well"] for r in sel}),
                  "fovs": len({r["fov"] for r in sel}),
                  **{k: sum(int(r[k]) for r in sel) for k in
                     ("n_polygons", "n_instances", "n_lost", "overlap_px", "n_empty_after_gap",
                      "n_fragmented", "n_cc_multi_instance")}}
        assert out[s]["n_cc_multi_instance"] == 0, "gap rule failed to separate instances"
    tr = {r["case"] for r in rows if r["split"] == "Tr"}
    te = {r["case"] for r in rows if r["split"] == "Ts"}
    wtr = {r["well"] for r in rows if r["split"] == "Tr"}
    wte = {r["well"] for r in rows if r["split"] == "Ts"}
    assert not (tr & te), "case in both Tr and Ts"
    out["shared_wells"] = len(wtr & wte)
    out["test_wells_by_line"] = {w: sum(r["well"] == w for r in rows if r["split"] == "Ts") for w in sorted(wte)}
    (c.RESULTS / "dataset_verify.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N cases (smoke test)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        build(a.workers, a.limit)
    if not a.limit:
        verify()
