"""C2 stage-2 inference on crops around B's peaks (PREREG_C_two_stage.md Amendment C-1.3).

    run.cmd -m e01.c2_predict --cases smoke [--smoke-model]   pre-flight: 2 training-fold scans (results\\pred_smoke\\B)
    run.cmd -m e01.c2_predict --cases test      fold 0 (results\\pred\\B), once C2 is trained

For every B peak q (PREREG §5) a 96^3 crop centred on q is taken from the image (zero padding),
predicted by the C2 nnU-Net (checkpoint_final, fold "all"; step 0.5, Gaussian, mirroring, fp32,
through nnU-Net's own preprocessing), and reduced by the §2 rules to (a2, region, assigned).
Output per scan: results\\two_stage\\c2_pred[_smoke]\\<case>.npz with
    points (n,3), b (n,), a2 (n,), assigned (n,), starts (n,3), regions (n, packed 96^3 bits).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from e01 import common as c  # first: environment

SMOKE_CASES = ("BMS_Mets_013", "BMS_Mets_019")      # training folds; B maps in results\pred_smoke\B


def out_dir(cases: str):
    return c.RESULTS / "two_stage" / ("c2_pred" if cases == "test" else "c2_pred_smoke")


def load_c2(path):
    """Per-scan C2 outputs; regions unpacked to (n, 96, 96, 96) bool."""
    from e01.c2 import CROP

    z = np.load(path)
    n = len(z["b"])
    reg = np.unpackbits(z["regions"], axis=1, count=CROP ** 3).astype(bool).reshape(n, CROP, CROP, CROP) \
        if n else np.zeros((0, CROP, CROP, CROP), bool)
    return {"points": z["points"], "b": z["b"], "a2": z["a2"], "assigned": z["assigned"],
            "starts": z["starts"], "regions": reg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", choices=["test", "smoke"], required=True)
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti")
    ap.add_argument("--smoke-model", action="store_true",
                    help="use the 2-epoch E01C2_smoke checkpoint (training-fold pre-flight only)")
    a = ap.parse_args()
    c.select_gpu(c.GPUS[a.gpu])
    import nibabel as nib
    import torch  # noqa: F401

    from e01 import c2
    from e01 import decode as dc
    from e01 import evaluate as ev
    from e01.predict import build_predictor

    gpu = c.verify_gpu(c.GPUS[a.gpu])
    c.assert_paths()
    g = c.guard()
    if a.cases == "test":
        ids, bdir = g.pool_ids(folds=[0]), c.RESULTS / "pred" / "B"
    else:
        ids, bdir = list(SMOKE_CASES), c.RESULTS / "pred_smoke" / "B"
        assert all(g.fold_of(s) in c2.TRAIN_FOLDS for s in ids)
    od = out_dir(a.cases)
    od.mkdir(parents=True, exist_ok=True)
    assert not (a.smoke_model and a.cases == "test"), "the smoke model is for training-fold pre-flight only"
    pr, meta = build_predictor("C2", dataset=c2.DATASET, fold="all", smoke=a.smoke_model)
    t0, n_peaks = time.time(), 0
    for sid in ids:
        dst = od / f"{sid}.npz"
        if dst.exists():
            continue
        im = nib.load(str(c.NN_RAW / c.DATASET / "imagesTr" / f"{sid}_0000.nii.gz"))
        img = np.asarray(im.dataobj, dtype=np.float32)
        sp = tuple(float(z) for z in im.header.get_zooms()[:3])
        pb = ev.load_nnunet_prob(bdir / f"{sid}.npz", img.shape)
        pts, b = dc.points_from_heatmap(pb, sp)
        a2 = np.zeros(len(pts))
        assigned = np.zeros(len(pts), bool)
        regions = []
        for i, q in enumerate(pts):
            crop = c2.extract_crop(img, q)
            _, prob = pr.predict_single_npy_array(crop.transpose(2, 1, 0)[None],
                                                  {"spacing": [sp[2], sp[1], sp[0]]},
                                                  None, None, True)
            p2 = np.ascontiguousarray(prob[1].transpose(2, 1, 0), dtype=np.float32)
            assert p2.shape == crop.shape and np.isfinite(p2).all(), (sid, i, p2.shape)
            a2[i], reg, assigned[i] = c2.stage2_from_crop(p2, sp)
            regions.append(np.packbits(reg.ravel()))
        starts = np.asarray([c2.crop_start(q) for q in pts], int).reshape(-1, 3)
        np.savez_compressed(dst, points=pts, b=b, a2=a2, assigned=assigned, starts=starts,
                            regions=np.asarray(regions, np.uint8).reshape(len(pts), -1))
        n_peaks += len(pts)
        print(f"{sid}: {len(pts)} peaks, {assigned.sum()} assigned, {time.time() - t0:.0f}s", flush=True)
    missing = [s for s in ids if not (od / f"{s}.npz").exists()]
    meta.update({"cases": a.cases, "n_cases": len(ids), "peaks_this_run": n_peaks, "gpu": gpu,
                 "crop": c2.CROP, "seconds": round(time.time() - t0, 1), "missing": missing,
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (od / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "checkpoint"}, indent=1), flush=True)
    assert not missing, missing


if __name__ == "__main__":
    main()
