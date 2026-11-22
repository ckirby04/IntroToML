"""E02 process environment. Import this FIRST, before torch or nnunetv2.

- Points nnUNet_raw / nnUNet_preprocessed / nnUNet_results at E02\\data\\ only.
- PYTHONDONTWRITEBYTECODE so importing jannus (1.60\\src, editable) writes no __pycache__.
- GPU chosen by exact nvidia-smi name; seeds; Windows EcoQoS opt-out; distutils shim.
These patterns are copied from E01's common.py (not imported), so nothing here can repoint
nnU-Net at E01 or 1.60.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from pathlib import Path  # noqa: E402

E02 = Path(__file__).resolve().parents[2]
DATA = E02 / "data"
RAW = DATA / "raw"                      # LIVECell download (read-only after download)
NN_RAW = DATA / "nnUNet_raw"
NN_PRE = DATA / "nnUNet_preprocessed"
NN_RES = DATA / "nnUNet_results"
INSTANCES = DATA / "instances"          # uint16 instance maps, per split (Tr / Ts)
META = DATA / "meta"                    # per-image metadata
RESULTS = E02 / "results"
REPORT = E02 / "report"
SPLITS = E02 / "splits"
PREREG = E02 / "PREREG.md"
PREREG_SHA_FILE = RESULTS / "prereg_sha256.txt"

DATASET_ID = 921
DATASET = "Dataset921_E02LiveCell"
PLANS = "nnUNetPlans"
CONFIG = "2d"
FOLD = "all"                            # PREREG §4: every training image; test is imagesTs
SEED = 1600                             # PREREG §4, as E01
GPU_NAME = "NVIDIA GeForce RTX 3070 Ti"
GPUS = {"3070ti": "NVIDIA GeForce RTX 3070 Ti", "5060ti": "NVIDIA GeForce RTX 5060 Ti"}

for _k, _v in (("nnUNet_raw", NN_RAW), ("nnUNet_preprocessed", NN_PRE), ("nnUNet_results", NN_RES)):
    os.environ[_k] = str(_v)
os.environ["nnUNet_n_proc_DA"] = "6"      # PREREG Deviation D1: 12 + 6 default workers exhaust commit memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

FORBIDDEN_IMPORTS = ("p3_common", "pilot_common")

# E01 reference values, from E01's training folds 1-4 only (3,825 lesions, 357 scans), read from
# E01 results\two_stage\c2_crop_manifest.csv (esd_mm of the 'pos' rows) and gt_anchor_check.json.
E01_TRAIN_MEDIAN_ESD_MM = 4.243
E01_TRAIN_BIN_SHARES = (0.2546, 0.6188, 0.1265)     # < 3, 3-10, > 10 mm
E01_TRAIN_LESIONS_PER_SCAN = 3825 / 357             # 10.714
# Share of E01 training lesions with another lesion's anchor inside the NMS box (Chebyshev
# distance <= 3 voxels, E01's half-width on the ~1 mm grid), from the manifest's 'anchor' column.
E01_TRAIN_NMS_COLLISION_SHARE = 0.0094


def _disable_power_throttling() -> None:
    if os.name != "nt":
        return
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    s = _State(1, 0x1, 0)
    k32.SetProcessInformation(ctypes.c_void_p(k32.GetCurrentProcess()), 4, ctypes.byref(s), ctypes.sizeof(s))


_disable_power_throttling()

try:  # nnU-Net 2.6.4 imports distutils; the 1.60 venv does not process distutils-precedence.pth
    import _distutils_hack

    _distutils_hack.add_shim()
except ImportError:  # pragma: no cover
    pass


def sha256(p: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _inside(p: str | Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_paths() -> dict:
    """Env vars point inside E02; if nnunetv2 is importable, its resolved roots do too."""
    for m in FORBIDDEN_IMPORTS:
        assert m not in sys.modules, f"{m} imported: it would repoint nnUNet_* at 1.60"
    got = {}
    for k, v in (("nnUNet_raw", NN_RAW), ("nnUNet_preprocessed", NN_PRE), ("nnUNet_results", NN_RES)):
        assert Path(os.environ[k]).resolve() == v.resolve(), f"{k} is not E02's"
        got[k] = os.environ[k]
    if "nnunetv2" in sys.modules or "nnunetv2.paths" in sys.modules:
        from nnunetv2 import paths

        got = {"nnUNet_raw": paths.nnUNet_raw, "nnUNet_preprocessed": paths.nnUNet_preprocessed,
               "nnUNet_results": paths.nnUNet_results}
    for k, v in got.items():
        assert v is not None and _inside(v, E02), f"{k}={v!r} is not inside {E02}"
        assert "1.60" not in Path(v).resolve().parts, f"{k}={v!r} points into 1.60"
    return got


def assert_prereg() -> str:
    """PREREG.md must match the last hash in results\\prereg_sha256.txt (approved version)."""
    last = PREREG_SHA_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].split()[0]
    got = sha256(PREREG)
    assert got == last, f"PREREG.md sha256 {got} != recorded {last}: amend and re-record first"
    return got


def gpu_index_by_name(name: str) -> int:
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        idx, n = (x.strip() for x in line.split(",", 1))
        if n == name:
            return int(idx)
    raise SystemExit(f"GPU {name!r} not found:\n{out}")


def select_gpu(name: str = GPU_NAME) -> None:
    """Call before torch initialises CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index_by_name(name))


def verify_gpu(name: str = GPU_NAME) -> str:
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() == 1, f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}"
    got = torch.cuda.get_device_name(0)
    assert got == name, f"visible GPU is {got!r}, expected {name!r}"
    return got


def seed_everything(seed: int = SEED) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_rows(split: str | None = None) -> list[dict]:
    """data\\meta\\images.csv rows (case, split, cell_line, well, ...)."""
    import csv

    with open(META / "images.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]
