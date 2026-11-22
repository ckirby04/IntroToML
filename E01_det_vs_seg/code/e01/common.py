"""E01 process environment. Import this FIRST, before torch or nnunetv2.

- Points nnUNet_raw / nnUNet_preprocessed / nnUNet_results at E01\\data\\ (never 1.60).
- PYTHONDONTWRITEBYTECODE so importing jannus (1.60\\src, editable) writes no __pycache__.
- GPU chosen by exact nvidia-smi name; seeds; Windows EcoQoS opt-out; distutils shim.
  These operational patterns are copied from 1.60 (p3_common / pilot_common), not imported:
  importing those modules would repoint nnUNet_* at 1.60_artifacts.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from pathlib import Path  # noqa: E402

E01 = Path(__file__).resolve().parents[2]
DATA = E01 / "data"
NN_RAW = DATA / "nnUNet_raw"
NN_PRE = DATA / "nnUNet_preprocessed"
NN_RES = DATA / "nnUNet_results"
RESULTS = E01 / "results"
REPORT = E01 / "report"
SPLITS = E01 / "splits"

DATASET_ID = 901
DATASET = "Dataset901_E01T1c"
PLANS = "nnUNetPlans"
CONFIG = "3d_fullres"
FOLD = 0                       # fold 0 = E01 test set (split_v1 pool fold 0)
SEED = 1600                    # same seed for both arms (1.60 convention: 1600 + fold)
GPU_NAME = "NVIDIA GeForce RTX 3070 Ti"          # default E01 GPU
GPUS = {"3070ti": "NVIDIA GeForce RTX 3070 Ti", "5060ti": "NVIDIA GeForce RTX 5060 Ti"}
SPLIT_V1_SHA256 = "740ff72c2201a83f0a9e06902f0592feaabac230e526aa64a391df1587314f1c"
SOURCE_DATASET = Path(r"G:\BrainMetShare\1.60_artifacts\nnUNet_raw\Dataset170_JannusT1c")

for _k, _v in (("nnUNet_raw", NN_RAW), ("nnUNet_preprocessed", NN_PRE), ("nnUNet_results", NN_RES)):
    os.environ[_k] = str(_v)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

FORBIDDEN_IMPORTS = ("p3_common", "pilot_common")


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


def _inside(p: str | Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_paths() -> dict:
    """Assert nnU-Net resolves every root inside E01 and no 1.60 launcher module is loaded."""
    for m in FORBIDDEN_IMPORTS:
        assert m not in sys.modules, f"{m} imported: it would repoint nnUNet_* at 1.60"
    from nnunetv2 import paths

    got = {"nnUNet_raw": paths.nnUNet_raw, "nnUNet_preprocessed": paths.nnUNet_preprocessed,
           "nnUNet_results": paths.nnUNet_results}
    for k, v in got.items():
        assert v is not None and _inside(v, E01), f"{k}={v!r} is not inside {E01}"
        assert "1.60" not in Path(v).resolve().parts, f"{k}={v!r} points into 1.60"
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


def guard():
    """1.60 DataGuard (read-only import). Verifies the split_v1 manifest hash on load."""
    from jannus.splits.guard import DataGuard

    return DataGuard()


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
