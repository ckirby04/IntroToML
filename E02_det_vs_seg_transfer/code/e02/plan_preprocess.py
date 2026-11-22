"""nnU-Net fingerprint / plan / preprocess (2d) for Dataset921 into E02 paths only (PREREG §4).

    run.cmd -m e02.plan_preprocess [--show-only]

nnU-Net reads imagesTr/labelsTr only; imagesTs is never fingerprinted or preprocessed. The 2d
plan is recorded in results\\plans_2d.json and not edited.
"""
from __future__ import annotations

import argparse
import json

from e02 import common as c  # noqa: F401  (sets nnUNet_* before nnunetv2 is imported)

KEYS = ("patch_size", "batch_size", "spacing", "median_image_size_in_voxels", "normalization_schemes",
        "use_mask_for_norm", "architecture")


def run() -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess)

    extract_fingerprints([c.DATASET_ID], num_processes=6, check_dataset_integrity=True)
    plan_experiments([c.DATASET_ID])
    preprocess([c.DATASET_ID], configurations=(c.CONFIG,), num_processes=(6,))


def show() -> dict:
    plans = json.loads((c.NN_PRE / c.DATASET / f"{c.PLANS}.json").read_text())
    cfg = plans["configurations"][c.CONFIG]
    out = {k: cfg.get(k) for k in KEYS}
    out["transpose_forward"] = plans.get("transpose_forward")
    (c.RESULTS / "plans_2d.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("patch_size", "batch_size", "spacing", "normalization_schemes",
                                          "transpose_forward")}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-only", action="store_true")
    a = ap.parse_args()
    print("paths:", c.assert_paths())
    if not a.show_only:
        run()
    print("paths after:", c.assert_paths())
    show()
