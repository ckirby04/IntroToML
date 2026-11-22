"""Fold-0 inference for both arms through ONE path (PREREG §4).

    run.cmd -m e01.predict --arm A|B [--cases test|smoke] [--gpu 3070ti]

- nnU-Net's own nnUNetPredictor: sliding window, tile step 0.5, Gaussian tile weighting,
  mirroring TTA on all axes allowed by the checkpoint. checkpoint_final.pth only.
- Networks are built by the arm's trainer class directly (E01HeatmapTrainer is not in
  site-packages, so nnU-Net's folder-based loader cannot find it; manual_initialization is
  used for BOTH arms so the path is the same).
- Model B (1 logit channel) is wrapped to output cat([0, logit]): nnU-Net's 2-class softmax
  then gives exactly sigmoid(logit) as class 1, so the export pipeline (TTA/tile averaging of
  logits, resampling to the image grid, .npz "probabilities") is identical for both arms.
- Precision: fp32 for both arms (autocast disabled around every forward pass). A trained in
  fp16 AMP and B in bf16 (PREREG D6); fp32 inference is the common, overflow-free choice.
- Outputs: results\\pred\\{A,B}\\<case>.npz (+ .pkl, .nii.gz) and run.json. Inputs are the
  read-only hard-linked images in data\\nnUNet_raw\\Dataset901_E01T1c\\imagesTr.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from e01 import common as c  # first: environment

SMOKE_CASES = 2          # training-fold cases, for a pre-flight check only


def build_predictor(arm: str, dataset: str | None = None, fold=None, smoke: bool = False):
    """dataset/fold default to E01 (Dataset901, fold 0); C2 uses Dataset902, fold "all"."""
    import torch
    from torch import nn
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    from e01 import trainers

    cls = trainers.get(arm, smoke)
    dataset = c.DATASET if dataset is None else dataset
    fold = c.FOLD if fold is None else fold
    mdir = c.NN_RES / dataset / f"{cls.__name__}__{c.PLANS}__{c.CONFIG}"
    ck_path = mdir / f"fold_{fold}" / "checkpoint_final.pth"
    assert c._inside(ck_path, c.NN_RES) and ck_path.exists(), ck_path
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["trainer_name"] == cls.__name__, (ck["trainer_name"], cls.__name__)
    assert ck["current_epoch"] == (trainers.SMOKE_EPOCHS if smoke else 250), ck["current_epoch"]
    plans = PlansManager(load_json(str(mdir / "plans.json")))
    dj = load_json(str(mdir / "dataset.json"))
    cm = plans.get_configuration(ck["init_args"]["configuration"])
    lm = plans.get_label_manager(dj)
    net = cls.build_network_architecture(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                         cm.network_arch_init_kwargs_req_import,
                                         determine_num_input_channels(plans, cm, dj),
                                         lm.num_segmentation_heads, enable_deep_supervision=False)
    net.load_state_dict(ck["network_weights"])
    for k, v in ck["network_weights"].items():
        assert torch.isfinite(v).all(), f"non-finite weights in {k}"

    if arm == "B":
        class TwoClassFromLogit(nn.Module):
            """1 logit -> [0, logit]; softmax(...)[1] == sigmoid(logit)."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                lg = self.inner(x)
                assert lg.shape[1] == 1
                return torch.cat([torch.zeros_like(lg), lg], 1)

            def load_state_dict(self, sd, *a, **kw):          # predictor reloads raw weights
                return self.inner.load_state_dict(sd, *a, **kw)

        net = TwoClassFromLogit(net)

    class FP32Predictor(nnUNetPredictor):
        def _internal_maybe_mirror_and_predict(self, x):
            with torch.autocast("cuda", enabled=False):
                return super()._internal_maybe_mirror_and_predict(x.float())

    pr = FP32Predictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                       perform_everything_on_device=True, device=torch.device("cuda"),
                       verbose=False, allow_tqdm=False)
    pr.manual_initialization(net, plans, cm, [ck["network_weights"]], dj, cls.__name__,
                             ck.get("inference_allowed_mirroring_axes"))
    meta = {"arm": arm, "trainer": cls.__name__, "checkpoint": str(ck_path),
            "checkpoint_sha256": c.sha256(ck_path), "epoch": int(ck["current_epoch"]),
            "mirror_axes": list(ck.get("inference_allowed_mirroring_axes") or []),
            "tile_step_size": 0.5, "use_gaussian": True, "use_mirroring": True, "precision": "fp32",
            "b_wrapper": "cat([0, logit]) -> softmax[1] = sigmoid(logit)" if arm == "B" else None}
    return pr, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--cases", choices=["test", "smoke"], default="test",
                    help="test = all fold-0 scans; smoke = 2 training-fold scans (pre-flight)")
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti")
    a = ap.parse_args()
    c.select_gpu(c.GPUS[a.gpu])
    import torch  # noqa: F401

    gpu = c.verify_gpu(c.GPUS[a.gpu])
    c.assert_paths()
    g = c.guard()
    ids = g.pool_ids(folds=[0]) if a.cases == "test" else sorted(g.pool_ids(folds=[1]))[:SMOKE_CASES]
    if a.cases == "smoke":
        assert all(g.fold_of(s) != 0 for s in ids)
    img = c.NN_RAW / c.DATASET / "imagesTr"
    out = c.RESULTS / ("pred" if a.cases == "test" else "pred_smoke") / a.arm
    out.mkdir(parents=True, exist_ok=True)
    pr, meta = build_predictor(a.arm)
    inputs = [[str(img / f"{s}_0000.nii.gz")] for s in ids]
    for f in inputs:
        assert Path(f[0]).exists(), f
    t0 = time.time()
    pr.predict_from_files_sequential(inputs, [str(out / s) for s in ids], save_probabilities=True,
                                     overwrite=False)
    meta.update({"cases": a.cases, "n_cases": len(ids), "gpu": gpu, "seconds": round(time.time() - t0, 1),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    missing = [s for s in ids if not (out / f"{s}.npz").exists()]
    meta["missing"] = missing
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "checkpoint"}, indent=1), flush=True)
    assert not missing, missing


if __name__ == "__main__":
    main()
