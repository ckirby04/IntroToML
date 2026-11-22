"""How much does the same image's prediction differ between the RTX 3070 Ti and the RTX 5060 Ti?

    run.cmd -m e02.gpu_diff --gpu 5060ti          predict (run first: its B peaks define the C2 crops)
    run.cmd -m e02.gpu_diff --gpu 3070ti          predict (memory-capped: may share the GPU with training)
    run.cmd -m e02.gpu_diff --compare             write results\\gpu_diff\\gpu_diff.json

TRAINING images only (never the test set). Same paths as run_eval: A and B through
predict_from_files_sequential with checkpoint_final; C2 through CropPredictor(batch=1) on crops
around the 5060 Ti's B peaks (identical inputs on both GPUs) with the C2 benchmark checkpoint copy
(results\\tmp\\c2_bench_checkpoint.pth; any epoch: this checks arithmetic, not quality).
Compares probability maps and everything downstream that a difference could change: A candidate
points and scores, A masks at MASK_P, B peaks and scores, C2 (a2, region, assigned).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from e02 import common as c  # first: environment

N_IMAGES = 24
OUT = c.RESULTS / "gpu_diff"
C2_CK = c.RESULTS / "tmp" / "c2_bench_checkpoint.pth"
MEM_CAP_GB = 1.8          # the 3070 Ti has ~2.2 GB free while C2 trains


def cases() -> list[str]:
    tr = sorted(r["case"] for r in c.image_rows("Tr"))
    return tr[len(tr) // (2 * N_IMAGES):: len(tr) // N_IMAGES][:N_IMAGES]


def predict(gpu: str) -> None:
    c.select_gpu(c.GPUS[gpu])
    import torch
    from PIL import Image

    from e02 import c2
    from e02 import c2_predict as c2p
    from e02 import decode as dc
    from e02 import two_stage as ts
    from e02.predict import build_predictor

    name = c.verify_gpu(c.GPUS[gpu])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(MEM_CAP_GB * 1e9 / total)
    ids = cases()
    img_dir = c.NN_RAW / c.DATASET / "imagesTr"
    od = OUT / gpu
    t0 = time.time()
    meta = {"gpu": name, "cases": ids, "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
            "tf32_conv": torch.backends.cudnn.allow_tf32, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    for arm in ("A", "B"):
        pr, m = build_predictor(arm)
        meta[arm] = {"sha256": m["checkpoint_sha256"], "epoch": m["epoch"]}
        (od / arm).mkdir(parents=True, exist_ok=True)
        tmp = od / arm / "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for s in ids:
            if (od / arm / f"{s}.npz").exists():
                continue
            pr.predict_from_files_sequential([[str(img_dir / f"{s}_0000.png")]], [str(tmp / s)],
                                             save_probabilities=True, overwrite=True)
            os.replace(tmp / f"{s}.npz", od / arm / f"{s}.npz")
        shutil.rmtree(tmp, ignore_errors=True)
        del pr
        torch.cuda.empty_cache()
        print(arm, f"{time.time() - t0:.0f}s", flush=True)
    pr, m = build_predictor("C2", dataset=c2.DATASET, checkpoint=C2_CK)
    meta["C2"] = {"sha256": m["checkpoint_sha256"], "epoch": m["epoch"]}
    cp = c2p.CropPredictor(pr, batch=1)
    (od / "C2").mkdir(parents=True, exist_ok=True)
    for s in ids:
        img = np.asarray(Image.open(img_dir / f"{s}_0000.png"))
        hb = np.load(OUT / "5060ti" / "B" / f"{s}.npz")["probabilities"][1, 0].astype(np.float32)
        pts, _ = dc.points_from_heatmap(hb)
        fill = int(np.median(img))
        probs = cp([ts.extract_crop(img, q, fill=fill).astype(np.float32) for q in pts])
        np.savez_compressed(od / "C2" / f"{s}.npz", probs=np.stack(probs).astype(np.float16) if probs
                            else np.zeros((0, 1, 1), np.float16), points=pts,
                            stage2=np.array([[a2, ok] for a2, _, ok in map(ts.stage2_from_crop, probs)]).reshape(-1, 2),
                            regions=np.packbits(np.stack([r for _, r, _ in map(ts.stage2_from_crop, probs)])) if probs
                            else np.zeros(0, np.uint8))
    meta["seconds"] = round(time.time() - t0, 1)
    meta["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (od / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


def compare() -> None:
    from e02 import decode as dc
    from e02 import params as P

    ids = cases()
    g1, g2 = OUT / "5060ti", OUT / "3070ti"
    m1, m2 = (json.loads((g / "meta.json").read_text()) for g in (g1, g2))
    for arm in ("A", "B", "C2"):
        assert m1[arm]["sha256"] == m2[arm]["sha256"], arm
    res = {"cases": len(ids), "gpus": [m1["gpu"], m2["gpu"]], "env": {k: (m1[k], m2[k]) for k in
           ("torch", "cudnn", "tf32_conv", "tf32_matmul")}}
    for arm in ("A", "B"):
        r = {"max_abs_diff": 0.0, "mean_abs_diff": [], "identical_images": 0, "px": 0, "px_flip_floor": 0,
             "px_flip_mask": 0, "points_1": 0, "points_2": 0, "points_same_set": 0, "max_score_diff": 0.0,
             "images_points_identical": 0}
        for s in ids:
            p, q = (np.load(g / arm / f"{s}.npz")["probabilities"][1, 0].astype(np.float32) for g in (g1, g2))
            d = np.abs(p - q)
            r["max_abs_diff"] = max(r["max_abs_diff"], float(d.max()))
            r["mean_abs_diff"].append(float(d.mean()))
            r["identical_images"] += int(d.max() == 0)
            r["px"] += p.size
            r["px_flip_floor"] += int(((p >= P.CANDIDATE_FLOOR) != (q >= P.CANDIDATE_FLOOR)).sum())
            thr = P.MASK_P if arm == "A" else P.TAU["B"]
            r["px_flip_mask"] += int(((p >= thr) != (q >= thr)).sum())
            if arm == "A":
                (a, sa, *_), (b, sb, *_) = dc.points_from_prob(p), dc.points_from_prob(q)
            else:
                (a, sa), (b, sb) = dc.points_from_heatmap(p), dc.points_from_heatmap(q)
            r["points_1"] += len(a)
            r["points_2"] += len(b)
            sa_, sb_ = {tuple(x) for x in a}, {tuple(x) for x in b}
            r["points_same_set"] += len(sa_ & sb_)
            same = len(a) == len(b) and np.array_equal(a, b)
            r["images_points_identical"] += int(same)
            if same and len(a):
                r["max_score_diff"] = max(r["max_score_diff"], float(np.abs(sa - sb).max()))
        r["mean_abs_diff"] = float(np.mean(r["mean_abs_diff"]))
        r["mask_threshold"] = P.MASK_P if arm == "A" else P.TAU["B"]
        res[arm] = r
    r = {"crops": 0, "max_abs_diff": 0.0, "px_flip_mask": 0, "assigned_differs": 0, "max_a2_diff": 0.0,
         "regions_differ": 0}
    for s in ids:
        z1, z2 = np.load(g1 / "C2" / f"{s}.npz"), np.load(g2 / "C2" / f"{s}.npz")
        assert np.array_equal(z1["points"], z2["points"])
        n = len(z1["points"])
        r["crops"] += n
        if not n:
            continue
        p, q = z1["probs"].astype(np.float32), z2["probs"].astype(np.float32)
        r["max_abs_diff"] = max(r["max_abs_diff"], float(np.abs(p - q).max()))
        r["px_flip_mask"] += int(((p >= P.MASK_P) != (q >= P.MASK_P)).sum())
        s1, s2 = z1["stage2"], z2["stage2"]
        r["assigned_differs"] += int((s1[:, 1] != s2[:, 1]).sum())
        r["max_a2_diff"] = max(r["max_a2_diff"], float(np.abs(s1[:, 0] - s2[:, 0]).max()))
        u1, u2 = (np.unpackbits(z["regions"]).reshape(n, -1) for z in (z1, z2))
        r["regions_differ"] += int((u1 != u2).any(1).sum())
    r["note"] = "probs stored as fp16; region/a2/assigned computed from fp32 maps"
    res["C2"] = r
    (OUT / "gpu_diff.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=sorted(c.GPUS))
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    assert bool(a.gpu) != a.compare
    if a.compare:
        compare()
    else:
        assert a.gpu == "5060ti" or (OUT / "5060ti" / "meta.json").exists(), "run --gpu 5060ti first"
        predict(a.gpu)


if __name__ == "__main__":
    main()
