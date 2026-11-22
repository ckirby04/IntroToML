"""Equivalence and speed of batched C2 crop prediction against nnU-Net's single-array path.

    run.cmd -m e02.c2_bench [--checkpoint PATH] [--gpu 5060ti] [--n-ref 96] [--batches 8,16,32]

TRAINING images only: the 2 pre-flight images and the B smoke model's peaks on them
(results\\pred_smoke\\B). The checkpoint defaults to a copy of C2's checkpoint_latest (any epoch:
this checks computation, not quality). Compares, crop by crop, the reference path twice (to see the
GPU's own run-to-run noise) and the batched path at each batch size: max |p| difference, pixels on
the other side of 0.05 and 0.5, and whether stage 2's (a2, region, assigned) are identical.
Writes results\\two_stage\\c2_bench.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from e02 import common as c  # first: environment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="5060ti")
    ap.add_argument("--n-ref", type=int, default=96)
    ap.add_argument("--batches", default="8,16,32")
    a = ap.parse_args()
    c.select_gpu(c.GPUS[a.gpu])
    import torch
    from PIL import Image

    from e02 import c2
    from e02 import c2_predict as c2p
    from e02 import decode as dc
    from e02 import two_stage as ts
    from e02.predict import build_predictor, case_list, load_prob

    gpu = c.verify_gpu(c.GPUS[a.gpu])
    ck = a.checkpoint
    if ck is None:                               # copy: the trainer replaces checkpoint_latest each epoch
        src = (c.NN_RES / c2.DATASET / f"E02C2Trainer__{c.PLANS}__{c.CONFIG}" / f"fold_{c.FOLD}"
               / "checkpoint_latest.pth")
        tmpd = c.RESULTS / "tmp"
        tmpd.mkdir(exist_ok=True)
        ck = tmpd / "c2_bench_checkpoint.pth"
        shutil.copyfile(src, ck)
    pr, meta = build_predictor("C2", dataset=c2.DATASET, checkpoint=ck)

    ids, folder = case_list("smoke")
    crops = []
    for case in ids:
        img = np.asarray(Image.open(c.NN_RAW / c.DATASET / folder / f"{case}_0000.png"))
        pts, _ = dc.points_from_heatmap(load_prob("smoke", "B", case))
        fill = int(np.median(img))
        crops += [ts.extract_crop(img, q, fill=fill).astype(np.float32) for q in pts]
    rng = np.random.default_rng(0)
    sel = sorted(rng.choice(len(crops), size=min(a.n_ref, len(crops)), replace=False))
    ref_crops = [crops[i] for i in sel]

    def timed(fn):
        torch.cuda.synchronize()
        t = time.time()
        r = fn()
        torch.cuda.synchronize()
        return r, time.time() - t

    def s2(p):
        return [ts.stage2_from_crop(x) for x in p]

    def compare(p, q):
        sp, sq = s2(p), s2(q)
        return {"max_abs_diff": float(max(np.abs(x - y).max() for x, y in zip(p, q))),
                "px_flip_0.05": int(sum(((x >= 0.05) != (y >= 0.05)).sum() for x, y in zip(p, q))),
                "px_flip_0.5": int(sum(((x >= 0.5) != (y >= 0.5)).sum() for x, y in zip(p, q))),
                "stage2_identical": int(sum(u[0] == v[0] and u[2] == v[2] and np.array_equal(u[1], v[1])
                                            for u, v in zip(sp, sq))),
                "n": len(p)}

    c2p.predict_crop_single(pr, ref_crops[0])                     # warm-up
    ref1, t_ref = timed(lambda: [c2p.predict_crop_single(pr, x) for x in ref_crops])
    ref2, _ = timed(lambda: [c2p.predict_crop_single(pr, x) for x in ref_crops])
    out = {"gpu": gpu, "checkpoint_epoch": meta["epoch"], "checkpoint_sha256": meta["checkpoint_sha256"],
           "n_crops_total": len(crops), "reference": {"sec_per_crop": t_ref / len(ref_crops),
                                                     "repeat_vs_itself": compare(ref1, ref2)},
           "batched": {}}
    print(json.dumps(out, indent=1), flush=True)
    for bs in [int(x) for x in a.batches.split(",")]:
        bp = c2p.CropPredictor(pr, batch=bs)
        bp(ref_crops[:bs])                                         # warm-up (cudnn)
        got, _ = timed(lambda: bp(ref_crops))
        _, t_all = timed(lambda: bp(crops))
        out["batched"][str(bs)] = {"sec_per_crop": t_all / len(crops), "speedup": (t_ref / len(ref_crops)) / (t_all / len(crops)),
                                   "vs_reference": compare(got, ref1), "slow_path_crops": bp.n_slow,
                                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(bs, json.dumps(out["batched"][str(bs)]), flush=True)
        torch.cuda.reset_peak_memory_stats()
    dst = c.RESULTS / "two_stage" / "c2_bench.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
