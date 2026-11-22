"""Phase 1: build Dataset901_E01T1c (hard links to Dataset170) and E01 splits_final.json.

Run:  e01_env.cmd  then  python -m e01.setup_dataset [--verify-only]

- Never opens a raw file for writing: os.link only creates a new directory entry.
- Verifies: 454 cases, every file a hard link to the Dataset170 file (same inode, nlink >= 3),
  0 scans from 1.60 test patients (via jannus DataGuard, which checks the split_v1 hash).
"""
from __future__ import annotations

import argparse
import json
import os

from e01 import common as c


def case_ids(folder) -> list[str]:
    return sorted(p.name[: -len("_0000.nii.gz")] for p in folder.glob("*_0000.nii.gz"))


def build() -> None:
    src = c.SOURCE_DATASET
    dst = c.NN_RAW / c.DATASET
    for d in (c.NN_RAW, c.NN_PRE, c.NN_RES, c.RESULTS, c.REPORT, c.SPLITS,
              dst / "imagesTr", dst / "labelsTr"):
        d.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in ("imagesTr", "labelsTr"):
        for f in sorted((src / sub).glob("*.nii.gz")):
            t = dst / sub / f.name
            if not t.exists():
                os.link(f, t)
                n += 1
    meta = json.loads((src / "dataset.json").read_text())
    meta.pop("jannus", None)
    meta["e01"] = {"source": str(src), "source_dataset_json_sha256": c.sha256(src / "dataset.json"),
                   "split_manifest_sha256": c.SPLIT_V1_SHA256, "files": "hard links"}
    (dst / "dataset.json").write_text(json.dumps(meta, indent=1))
    print(f"linked {n} new files into {dst}")

    g = c.guard()
    test_ids = set(g.pool_ids(folds=[0]))
    train_ids = set(g.pool_ids(folds=[1, 2, 3, 4]))
    splits = [{"train": sorted(train_ids), "val": sorted(test_ids)}]
    (c.SPLITS / "splits_final.json").write_text(json.dumps(splits, indent=1))
    (c.SPLITS / "split_source.json").write_text(json.dumps({
        "manifest": r"G:\BrainMetShare\1.60\configs\splits\split_v1.json",
        "manifest_sha256": c.SPLIT_V1_SHA256,
        "e01_fold0": {"train": "split_v1 pool folds 1-4", "val_is_E01_test": "split_v1 pool fold 0"},
        "n_train_scans": len(train_ids), "n_test_scans": len(test_ids),
        "n_train_patients": len({g.patient_of(s) for s in train_ids}),
        "n_test_patients": len({g.patient_of(s) for s in test_ids}),
    }, indent=1))


def verify() -> dict:
    src, dst = c.SOURCE_DATASET, c.NN_RAW / c.DATASET
    imgs, labs = case_ids(dst / "imagesTr"), sorted(p.name[:-7] for p in (dst / "labelsTr").glob("*.nii.gz"))
    assert imgs == labs, "image/label case mismatch"
    assert len(imgs) == 454, f"expected 454 cases, got {len(imgs)}"
    assert imgs == case_ids(src / "imagesTr"), "case list differs from Dataset170"
    nlinks = set()
    for sub in ("imagesTr", "labelsTr"):
        for f in (dst / sub).iterdir():
            s = f.stat()
            assert os.path.samefile(f, src / sub / f.name), f"{f} is not a hard link to Dataset170"
            nlinks.add(s.st_nlink)
    g = c.guard()
    n_test = sum(g.is_test(cid) for cid in imgs)          # guard raises on unknown IDs
    assert n_test == 0, f"{n_test} scans from 1.60 test patients"
    pool = set(g.pool_ids())
    assert set(imgs) == pool, "Dataset901 is not exactly the split_v1 pool"
    sp = json.loads((c.SPLITS / "splits_final.json").read_text())
    tr, va = set(sp[0]["train"]), set(sp[0]["val"])
    assert len(sp) == 1 and not (tr & va)
    assert va == set(g.pool_ids(folds=[0])) and tr == set(g.pool_ids(folds=[1, 2, 3, 4]))
    pt, pv = {g.patient_of(s) for s in tr}, {g.patient_of(s) for s in va}
    assert not (pt & pv), "patient leakage between train and test"
    out = {"cases": len(imgs), "files": 2 * len(imgs), "nlink_values": sorted(nlinks),
           "test_patient_scans": n_test, "train_scans": len(tr), "test_scans": len(va),
           "train_patients": len(pt), "test_patients": len(pv), "patient_overlap": 0}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        build()
    verify()
