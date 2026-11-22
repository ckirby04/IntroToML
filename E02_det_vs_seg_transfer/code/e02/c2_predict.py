"""C2 stage-2 inference on crops around B's peaks (PREREG §7.1).

    run.cmd -m e02.c2_predict --cases smoke [--smoke-model]   2 training images (B maps in results\\pred_smoke\\B)
    run.cmd -m e02.c2_predict --cases test                    every Ts image, once C2 is trained

For every B peak q (PREREG §6) a 384 x 384 crop centred on q is taken from the image (padded with
the image's median intensity, as the training crops), predicted by C2 (checkpoint_final, step
0.5, Gaussian, mirroring, fp32, through nnU-Net's own preprocessing) and reduced by the C1 rules
to (a2, region, assigned). Regions are stored as their bounding box only.

Output per image: results\\two_stage\\c2_pred[_smoke]\\<case>.npz with points (n,2), b, a2,
assigned, starts (n,2) (crop starts), rbox (n,4) (y0,y1,x0,x1 inside the crop), rbits (packed
bits of every region box, concatenated), roff (n+1,) bit offsets.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from e02 import common as c  # first: environment



def out_dir(cases: str):
    from e02.predict import OUT_SUFFIX

    return c.RESULTS / "two_stage" / f"c2_pred{OUT_SUFFIX[cases]}"


def pack_regions(regions: list[np.ndarray]):
    boxes, bits, offs = [], [], [0]
    for r in regions:
        if r.any():
            ys, xs = np.flatnonzero(r.any(1)), np.flatnonzero(r.any(0))
            b = (int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1)
            sub = r[b[0]:b[1], b[2]:b[3]].ravel()
        else:
            b, sub = (0, 0, 0, 0), np.zeros(0, bool)
        boxes.append(b)
        bits.append(sub)
        offs.append(offs[-1] + sub.size)
    allbits = np.concatenate(bits) if bits else np.zeros(0, bool)
    return np.asarray(boxes, np.int32).reshape(-1, 4), np.packbits(allbits), np.asarray(offs, np.int64)


PROPS = {"spacing": [999.0, 1.0, 1.0]}


def predict_crop_single(pr, crop: np.ndarray) -> np.ndarray:
    """Reference path: nnU-Net's predict_single_npy_array on one crop -> class-1 probability."""
    _, prob = pr.predict_single_npy_array(crop[None, None].astype(np.float32), dict(PROPS), None, None, True)
    return np.ascontiguousarray(prob[1, 0], dtype=np.float32)


class CropPredictor:
    """Same computation as predict_single_npy_array, without its per-call overhead.

    nnU-Net's single-array path does, per crop: preprocess (run_case_npy) -> pad to the patch ->
    one sliding-window tile (the crop IS one patch) -> mirrored forward passes averaged
    (_internal_maybe_mirror_and_predict, fp32 here) -> x gaussian, summed into an fp16 buffer,
    divided by the fp16 gaussian -> export (softmax, revert cropping). About 2/3 of its time is
    per-call overhead (a producer thread, buffer allocation, CUDA cache flushes, reloading weights).

    Here preprocessing and export are nnU-Net's own functions, called per crop unchanged; the
    forward passes and the same gaussian / fp16 operations follow in the same order and dtypes. A crop
    whose preprocessed shape is not exactly one patch (nnU-Net cropped a zero border) takes the
    reference path.

    batch = 1 (the default, and what run_eval uses) is BIT-IDENTICAL to the reference path:
    results\\two_stage\\c2_bench.json, 96/96 crops, max |diff| 0.0, 3.2x faster. Larger batches are no
    faster (the GPU is saturated) and are NOT identical: cuDNN picks other TF32 algorithms per batch
    size (c2_bench_batches_8_16_32.json: |diff| up to 0.01). Do not use batch > 1 for results.
    """

    def __init__(self, pr, batch: int = 1):
        import torch
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        assert len(pr.list_of_parameters) == 1, "one checkpoint (fold all)"
        self.pr, self.batch = pr, batch
        self.cm, self.pm, self.dj = pr.configuration_manager, pr.plans_manager, pr.dataset_json
        self.lm = pr.label_manager
        self.ps = tuple(self.cm.patch_size)
        assert len(self.ps) == 2
        self.pre = self.cm.preprocessor_class(verbose=False)
        net = pr.network._orig_mod if hasattr(pr.network, "_orig_mod") else pr.network
        net.load_state_dict(pr.list_of_parameters[0])
        pr.network = pr.network.to(pr.device)
        pr.network.eval()
        self.gauss = compute_gaussian(self.ps, sigma_scale=1. / 8, value_scaling_factor=10, device=pr.device) \
            if pr.use_gaussian else None
        self.n_fast = self.n_slow = 0
        self._torch = torch

    def _forward(self, tiles: np.ndarray):
        """tiles (B, c, y, x) float32 -> logits (B, C, y, x) fp16, as nnU-Net's single tile."""
        torch = self._torch
        pr = self.pr
        with torch.inference_mode(), torch.autocast(pr.device.type, enabled=True):
            x = torch.from_numpy(tiles).to(pr.device)
            pred = pr._internal_maybe_mirror_and_predict(x)
            logits = torch.zeros(pred.shape, dtype=torch.half, device=pr.device)
            n = torch.zeros(self.ps, dtype=torch.half, device=pr.device)
            if self.gauss is not None:
                pred *= self.gauss
                n += self.gauss
            else:
                n += 1
            logits += pred
            torch.div(logits, n, out=logits)
            if torch.any(torch.isinf(logits)):
                raise RuntimeError("inf in predicted logits")
        return logits.cpu()

    def __call__(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

        out: list[np.ndarray | None] = [None] * len(crops)
        fast: list[tuple[int, np.ndarray, dict]] = []
        for i, crop in enumerate(crops):
            data, _, props = self.pre.run_case_npy(crop[None, None].astype(np.float32), None, dict(PROPS),
                                                   self.pm, self.cm, self.dj)
            if tuple(data.shape[1:]) == (1, *self.ps):
                fast.append((i, data[:, 0], props))
            else:                                   # rare: nnU-Net cropped a zero border
                out[i] = predict_crop_single(self.pr, crop)
                self.n_slow += 1
        for b0 in range(0, len(fast), self.batch):
            chunk = fast[b0:b0 + self.batch]
            logits = self._forward(np.stack([d for _, d, _ in chunk]).astype(np.float32))
            for j, (i, _, props) in enumerate(chunk):
                _, prob = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits[j][:, None], self.pm, self.cm, self.lm, props, return_probabilities=True)
                out[i] = np.ascontiguousarray(prob[1, 0], dtype=np.float32)
            self.n_fast += len(chunk)
        return out


def predict_case(pr, img: np.ndarray, pb: np.ndarray, batched: "CropPredictor | None" = None) -> dict:
    """Stage 2 for one image: every B peak's crop through C2 (PREREG §7.1). Arrays for np.savez."""
    from e02 import decode as dc
    from e02 import two_stage as ts

    fill = int(np.median(img))
    assert pb.shape == img.shape, (pb.shape, img.shape)
    pts, b = dc.points_from_heatmap(pb)
    crops = [ts.extract_crop(img, q, fill=fill).astype(np.float32) for q in pts]
    probs = batched(crops) if batched is not None else [predict_crop_single(pr, cr) for cr in crops]
    a2 = np.zeros(len(pts))
    assigned = np.zeros(len(pts), bool)
    regions = []
    for i, p2 in enumerate(probs):
        assert p2.shape == crops[i].shape and np.isfinite(p2).all(), (i, p2.shape)
        a2[i], reg, assigned[i] = ts.stage2_from_crop(p2)
        regions.append(reg)
    starts = np.asarray([ts.crop_start(q) for q in pts], int).reshape(-1, 2)
    rbox, rbits, roff = pack_regions(regions)
    return dict(points=pts, b=b, a2=a2, assigned=assigned, starts=starts, rbox=rbox, rbits=rbits, roff=roff)


def save_atomic(dst, arrays: dict) -> None:
    """np.savez_compressed to a temp file, then rename: a killed run never leaves a partial file."""
    import os

    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dst)


def load_c2(path, size: int | None = None) -> dict:
    """Per-image C2 outputs; regions unpacked to (n, size, size) bool."""
    from e02 import params as P

    size = size or P.C2_CROP
    z = np.load(path)
    n = len(z["b"])
    bits = np.unpackbits(z["rbits"], count=int(z["roff"][-1])).astype(bool) if n else np.zeros(0, bool)
    reg = np.zeros((n, size, size), bool)
    for i, (y0, y1, x0, x1) in enumerate(z["rbox"]):
        if y1 > y0:
            reg[i, y0:y1, x0:x1] = bits[z["roff"][i]:z["roff"][i + 1]].reshape(y1 - y0, x1 - x0)
    return {"points": z["points"], "b": z["b"], "a2": z["a2"], "assigned": z["assigned"],
            "starts": z["starts"], "regions": reg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", choices=["test", "smoke"], required=True)
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti")
    ap.add_argument("--smoke-model", action="store_true", help="2-epoch C2 checkpoint (pre-flight only)")
    a = ap.parse_args()
    assert not (a.smoke_model and a.cases == "test")
    c.select_gpu(c.GPUS[a.gpu])
    import torch  # noqa: F401
    from PIL import Image

    from e02 import c2
    from e02 import decode as dc
    from e02 import params as P
    from e02 import two_stage as ts
    from e02.predict import build_predictor, case_list, load_prob

    gpu = c.verify_gpu(c.GPUS[a.gpu])
    c.assert_paths()
    if a.cases == "test":
        c.assert_prereg()
    ids, folder = case_list(a.cases)
    od = out_dir(a.cases)
    od.mkdir(parents=True, exist_ok=True)
    pr, meta = build_predictor("C2", dataset=c2.DATASET, smoke=a.smoke_model)
    t0, n_peaks = time.time(), 0
    for case in ids:
        dst = od / f"{case}.npz"
        if dst.exists():
            continue
        img = np.asarray(Image.open(c.NN_RAW / c.DATASET / folder / f"{case}_0000.png"))
        out = predict_case(pr, img, load_prob(a.cases, "B", case))
        save_atomic(dst, out)
        n_peaks += len(out["points"])
        print(f"{case}: {len(out['points'])} peaks, {out['assigned'].sum()} assigned, "
              f"{time.time() - t0:.0f}s", flush=True)
    missing = [s for s in ids if not (od / f"{s}.npz").exists()]
    meta.update({"cases": a.cases, "n_cases": len(ids), "peaks_this_run": n_peaks, "gpu": gpu,
                 "crop": P.C2_CROP, "seconds": round(time.time() - t0, 1), "missing": missing,
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (od / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "checkpoint"}, indent=1), flush=True)
    assert not missing, missing


if __name__ == "__main__":
    main()
