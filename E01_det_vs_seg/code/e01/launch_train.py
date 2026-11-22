"""Train one E01 arm on fold 0 (train = split_v1 pool folds 1-4). Training only: inference on
fold 0 is a separate Phase 4 step (e01.predict).

    run.cmd -m e01.launch_train --arm A [--smoke] [--continue]
    run.cmd -m e01.launch_train --arm C2 [--smoke]     (Dataset902 crops, fold "all"; PREREG_C C-1.2)

- Selects the GPU by exact name and asserts it after CUDA initialises.
- Asserts that every nnUNet_* path lies inside E01 before and after the trainer is built, and
  that the output folder does too.
- Asserts that splits_final.json in nnUNet_preprocessed equals E01\\splits\\splits_final.json.
- Instantiates the trainer class directly; nothing is registered in site-packages.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from e01 import common as c  # first: sets nnUNet_* and process environment
from e01 import pause as pz


def log_event(fdir: Path, ev: dict) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    ev = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
    with open(fdir / "e01_sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C2"], required=True)
    ap.add_argument("--smoke", action="store_true", help=f"2-epoch run into a separately named folder")
    ap.add_argument("--continue", dest="cont", action="store_true", help="resume from checkpoint_latest")
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti",
                    help="selected by exact nvidia-smi name and asserted after CUDA init")
    a = ap.parse_args()
    gpu_name = c.GPUS[a.gpu]

    c.select_gpu(gpu_name)
    import torch

    c.seed_everything(c.SEED)
    gpu = c.verify_gpu(gpu_name)
    paths = c.assert_paths()

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from torch.backends import cudnn

    from e01 import trainers

    if a.arm == "C2":
        import csv

        from e01 import c2

        pre, fold = c.NN_PRE / c2.DATASET, "all"
        dj = load_json(str(c.NN_RAW / c2.DATASET / "dataset.json"))
        assert dj["e01"]["source_folds"] == list(c2.TRAIN_FOLDS), "C2 crops must come from folds 1-4 only"
        g = c.guard()
        with open(c.RESULTS / "two_stage" / "c2_crop_manifest.csv", encoding="utf-8") as f:
            scans = {r["scan"] for r in csv.DictReader(f)}
        assert scans and all(g.fold_of(s) in c2.TRAIN_FOLDS for s in scans), "C2 crop from a non-training fold"
    else:
        pre, fold = c.NN_PRE / c.DATASET, c.FOLD
        assert (pre / "splits_final.json").read_bytes() == (c.SPLITS / "splits_final.json").read_bytes(), \
            "nnU-Net splits_final.json differs from E01 splits"
    cls = trainers.get(a.arm, a.smoke)
    trainer = cls(plans=load_json(str(pre / f"{c.PLANS}.json")), configuration=c.CONFIG, fold=fold,
                  dataset_json=load_json(str(pre / "dataset.json")), device=torch.device("cuda"))
    fdir = Path(trainer.output_folder)
    assert c._inside(fdir, c.NN_RES), f"output folder {fdir} not inside E01"
    c.assert_paths()

    if (fdir / "checkpoint_final.pth").exists():
        print(f"already trained: {fdir / 'checkpoint_final.pth'}; nothing to do", flush=True)
        return
    if pz.pause_requested():
        print(f"PAUSE flag present ({pz.PAUSE_FLAG}); not starting", flush=True)
        raise SystemExit(pz.EXIT_PAUSED)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    pz.install_pause(nnUNetTrainer)           # all E01 trainer classes inherit on_epoch_end
    resume = a.cont or (fdir / "checkpoint_latest.pth").exists()
    maybe_load_checkpoint(trainer, resume, False, None)
    cudnn.deterministic = False
    cudnn.benchmark = True
    log_event(fdir, {"event": "start", "arm": a.arm, "trainer": cls.__name__, "smoke": a.smoke,
                     "resume": resume, "epoch": int(trainer.current_epoch), "gpu": gpu, "seed": c.SEED,
                     "num_epochs": int(trainer.num_epochs), "paths": paths, "pause": True,
                     "fold": fold, "dataset": pre.name,
                     "splits_sha256": c.sha256(pre / "splits_final.json") if fold != "all" else None,
                     "plans_sha256": c.sha256(pre / f"{c.PLANS}.json")})
    print(f"E01 arm {a.arm} ({cls.__name__}) on {gpu}; seed {c.SEED}; resume={resume} at epoch "
          f"{trainer.current_epoch}; out {fdir}", flush=True)
    try:
        trainer.run_training()
    except pz.TrainingPaused as p:
        log_event(fdir, {"event": "paused", "epochs_completed": int(p.args[0])})
        print(f"PAUSED after {p.args[0]} completed epochs; resume continues from there", flush=True)
        raise SystemExit(pz.EXIT_PAUSED)
    log_event(fdir, {"event": "training_done", "epochs_completed": int(trainer.current_epoch),
                     "checkpoint_final_sha256": c.sha256(fdir / "checkpoint_final.pth")})
    print("training done:", fdir / "checkpoint_final.pth", flush=True)


if __name__ == "__main__":
    main()
