"""Whole-image inference for A and B through ONE path (PREREG §6).

    run.cmd -m e02.predict --arm A|B [--cases test|smoke] [--smoke-model] [--gpu 3070ti]

- nnU-Net's nnUNetPredictor: tile step 0.5, Gaussian weighting, mirroring TTA on the axes the
  checkpoint allows, checkpoint_final only, fp32 (autocast disabled around every forward pass).
- Networks are built by the arm's trainer class (E02 trainers are not in site-packages).
- Model B's single logit is wrapped as cat([0, logit]), so nnU-Net's 2-class softmax gives
  exactly sigmoid(logit) as class 1 and the export path is identical for both arms.
- Outputs: results\\pred[_smoke]\\{A,B}\\<case>.npz ("probabilities", (2, 1, y, x)) and run.json.
- test = every Ts image (PREREG §9 step 5, run ONCE); smoke = 2 training images (pre-flight).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from e02 import common as c  # first: environment

SMOKE_CASES = 2


def build_predictor(arm: str, dataset: str | None = None, smoke: bool = False, checkpoint=None):
    """checkpoint: benchmarks/equivalence checks only (any epoch); evaluation always uses
    checkpoint_final of the arm's trainer folder."""
    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from torch import nn

    from e02 import trainers

    cls = trainers.get(arm, smoke)
    dataset = c.DATASET if dataset is None else dataset
    mdir = c.NN_RES / dataset / f"{cls.__name__}__{c.PLANS}__{c.CONFIG}"
    ck_path = mdir / f"fold_{c.FOLD}" / "checkpoint_final.pth" if checkpoint is None else Path(checkpoint)
    assert checkpoint is not None or c._inside(ck_path, c.NN_RES), ck_path
    assert ck_path.exists(), ck_path
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["trainer_name"] == cls.__name__, (ck["trainer_name"], cls.__name__)
    if checkpoint is None:
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

            def load_state_dict(self, sd, *a, **kw):          # the predictor reloads raw weights
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


def case_list(cases: str) -> tuple[list[str], str]:
    """(case ids, nnU-Net images folder name)."""
    if cases == "test":
        return [r["case"] for r in c.image_rows("Ts")], "imagesTs"
    tr = sorted(r["case"] for r in c.image_rows("Tr"))
    n = PREFLIGHT_CASES if cases == "preflight" else SMOKE_CASES
    assert cases in ("smoke", "preflight"), cases
    return tr[:: max(1, len(tr) // n)][:n], "imagesTr"


PREFLIGHT_CASES = 8          # training images, run with the FINAL models before the test run
OUT_SUFFIX = {"test": "", "smoke": "_smoke", "preflight": "_preflight"}


def out_dir(cases: str, arm: str):
    return c.RESULTS / f"pred{OUT_SUFFIX[cases]}" / arm


def load_prob(cases: str, arm: str, case: str):
    """Class-1 probability (y, x) float32."""
    import numpy as np

    p = np.load(out_dir(cases, arm) / f"{case}.npz")["probabilities"]
    assert p.shape[0] == 2 and p.shape[1] == 1, p.shape
    return np.ascontiguousarray(p[1, 0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--cases", choices=["test", "smoke"], default="test")
    ap.add_argument("--smoke-model", action="store_true", help="2-epoch checkpoint (pre-flight only)")
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti")
    a = ap.parse_args()
    assert not (a.smoke_model and a.cases == "test"), "the smoke model is for the training pre-flight only"
    c.select_gpu(c.GPUS[a.gpu])
    import torch  # noqa: F401

    gpu = c.verify_gpu(c.GPUS[a.gpu])
    c.assert_paths()
    if a.cases == "test":
        c.assert_prereg()
    ids, folder = case_list(a.cases)
    img = c.NN_RAW / c.DATASET / folder
    out = out_dir(a.cases, a.arm)
    out.mkdir(parents=True, exist_ok=True)
    pr, meta = build_predictor(a.arm, smoke=a.smoke_model)
    inputs = [[str(img / f"{s}_0000.png")] for s in ids]
    missing_in = [f[0] for f in inputs if not os.path.exists(f[0])]
    assert not missing_in, missing_in[:3]
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
