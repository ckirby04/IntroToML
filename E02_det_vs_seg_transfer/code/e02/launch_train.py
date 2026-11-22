"""Train one E02 arm (PREREG §4), fold "all" of its dataset. Training only; inference is e02.predict.

    run.cmd -m e02.launch_train --arm A|B|C2 [--smoke] [--continue] [--gpu 3070ti]

- Asserts PREREG.md matches its recorded SHA-256 (full runs and smoke runs alike).
- Selects the GPU by exact name and asserts it after CUDA initialises.
- Asserts every nnUNet_* path, and the output folder, lies inside E02.
- A and B train on Dataset921 (training images only; the test images are imagesTs and are never
  preprocessed). C2 trains on Dataset922, whose manifest must list training images only.
- Instantiates the trainer class directly; nothing is registered in site-packages.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from e02 import common as c  # first: sets nnUNet_* and process environment
from e02 import pause as pz


def log_event(fdir: Path, ev: dict) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    ev = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
    with open(fdir / "e02_sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def check_training_only(pre: Path, arm: str) -> None:
    import csv

    tr = {r["case"] for r in c.image_rows("Tr")}
    ts = {r["case"] for r in c.image_rows("Ts")}
    if arm == "C2":
        from e02 import c2

        with open(c2.MANIFEST, encoding="utf-8") as f:
            src = {r["case"] for r in csv.DictReader(f)}
        assert src and src <= tr and not (src & ts), "C2 crop from a non-training image"
    else:
        ids = {p.name.split(".")[0].removesuffix("_seg") for p in pre.joinpath(f"{c.PLANS}_{c.CONFIG}").iterdir()}
        assert ids and ids <= tr and not (ids & ts), "preprocessed data contains non-training images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C2"], required=True)
    ap.add_argument("--smoke", action="store_true", help="2-epoch run into a separately named folder")
    ap.add_argument("--continue", dest="cont", action="store_true", help="resume from checkpoint_latest")
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default="3070ti")
    a = ap.parse_args()
    gpu_name = c.GPUS[a.gpu]

    prereg = c.assert_prereg()
    c.select_gpu(gpu_name)
    import torch

    c.seed_everything(c.SEED)
    gpu = c.verify_gpu(gpu_name)
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from torch.backends import cudnn

    from e02 import trainers

    paths = c.assert_paths()
    if a.arm == "C2":
        from e02 import c2

        pre = c.NN_PRE / c2.DATASET
    else:
        pre = c.NN_PRE / c.DATASET
    check_training_only(pre, a.arm)
    cls = trainers.get(a.arm, a.smoke)
    trainer = cls(plans=load_json(str(pre / f"{c.PLANS}.json")), configuration=c.CONFIG, fold=c.FOLD,
                  dataset_json=load_json(str(pre / "dataset.json")), device=torch.device("cuda"))
    fdir = Path(trainer.output_folder)
    assert c._inside(fdir, c.NN_RES), f"output folder {fdir} not inside E02"
    c.assert_paths()

    if (fdir / "checkpoint_final.pth").exists():
        print(f"already trained: {fdir / 'checkpoint_final.pth'}; nothing to do", flush=True)
        return
    if pz.pause_requested():
        print(f"PAUSE flag present ({pz.PAUSE_FLAG}); not starting", flush=True)
        raise SystemExit(pz.EXIT_PAUSED)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    pz.install_pause(nnUNetTrainer)           # all E02 trainer classes inherit on_epoch_end
    resume = a.cont or (fdir / "checkpoint_latest.pth").exists()
    maybe_load_checkpoint(trainer, resume, False, None)
    cudnn.deterministic = False
    cudnn.benchmark = True
    log_event(fdir, {"event": "start", "arm": a.arm, "trainer": cls.__name__, "smoke": a.smoke,
                     "resume": resume, "epoch": int(trainer.current_epoch), "gpu": gpu, "seed": c.SEED,
                     "num_epochs": int(trainer.num_epochs), "paths": paths, "pause": True,
                     "fold": c.FOLD, "dataset": pre.name, "prereg_sha256": prereg,
                     "plans_sha256": c.sha256(pre / f"{c.PLANS}.json")})
    print(f"E02 arm {a.arm} ({cls.__name__}) on {gpu}; seed {c.SEED}; resume={resume} at epoch "
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
