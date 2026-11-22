"""Phase 1 step 4: nnU-Net fingerprint/plan/preprocess for Dataset901 into E01 paths only,
then install E01's splits_final.json and compare the 3d_fullres plan with Dataset170.

    run.cmd -m e01.plan_preprocess [--compare-only]
"""
from __future__ import annotations

import argparse
import json
import shutil

from e01 import common as c  # noqa: F401  (sets nnUNet_* before nnunetv2 is imported)

D170_PLANS = r"G:\BrainMetShare\1.60_artifacts\nnUNet_preprocessed\Dataset170_JannusT1c\nnUNetPlans.json"
KEYS = ("patch_size", "batch_size", "spacing", "median_image_size_in_voxels", "normalization_schemes",
        "use_mask_for_norm", "architecture")


def run() -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess)

    extract_fingerprints([c.DATASET_ID], num_processes=6, check_dataset_integrity=True)
    plan_experiments([c.DATASET_ID])
    preprocess([c.DATASET_ID], configurations=(c.CONFIG,), num_processes=(6,))


def install_splits() -> None:
    dst = c.NN_PRE / c.DATASET / "splits_final.json"
    src = c.SPLITS / "splits_final.json"
    if dst.exists():
        assert dst.read_bytes() == src.read_bytes(), f"{dst} differs from {src}"
    else:
        shutil.copyfile(src, dst)
    print("splits_final.json installed, sha256", c.sha256(dst))


def compare() -> dict:
    mine = json.loads((c.NN_PRE / c.DATASET / "nnUNetPlans.json").read_text())["configurations"][c.CONFIG]
    ref = json.loads(open(D170_PLANS).read())["configurations"][c.CONFIG]
    diff = {k: (mine.get(k), ref.get(k)) for k in KEYS if mine.get(k) != ref.get(k)}
    print("3d_fullres:", {k: mine.get(k) for k in ("patch_size", "batch_size", "spacing")})
    print("differences vs Dataset170:", diff or "none")
    assert mine["patch_size"] == [160, 128, 112] and mine["batch_size"] == 2, "plan differs from Dataset170"
    return diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    print("paths:", c.assert_paths())
    if not a.compare_only:
        run()
    print("paths after:", c.assert_paths())
    install_splits()
    compare()
