"""Diagnostic for the Model B NaN (2026-11-11 run, diverged epoch 3 -> NaN epoch 4).

Runs E01HeatmapTrainer's real training loop (same plans, patch, batch, optimizer, AMP, grad
clipping, TRAINING folds only via nnU-Net's training dataloader; no validation, no fold-0 data)
for up to --iters iterations and logs per iteration: loss, per-scale loss, max |logit| per scale,
n_pos per scale, grad norm, GradScaler scale, whether the step was skipped, and parameter
finiteness. Stops at the first non-finite parameter. Output: results\\diag_nan\\ (never nnUNet_results).

    run.cmd -m e01.diag_nan [--iters 1250] [--variant base|fix]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from e01 import common as c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1250)
    ap.add_argument("--variant", default="base")
    a = ap.parse_args()

    c.select_gpu(c.GPUS["3070ti"])
    import numpy as np
    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from torch.backends import cudnn

    c.seed_everything(c.SEED)
    c.verify_gpu(c.GPUS["3070ti"])
    c.assert_paths()
    from e01 import trainer_heatmap as th

    cls = th.E01HeatmapTrainer
    pre = c.NN_PRE / c.DATASET
    real = Path(c.NN_RES) / c.DATASET / "E01HeatmapTrainer__nnUNetPlans__3d_fullres"
    existed = real.exists()
    t = cls(plans=load_json(str(pre / f"{c.PLANS}.json")), configuration=c.CONFIG, fold=c.FOLD,
            dataset_json=load_json(str(pre / "dataset.json")), device=torch.device("cuda"))
    import os
    if os.path.exists(t.log_file):
        os.remove(t.log_file)
    if not existed and real.exists():
        shutil.rmtree(real)
    out = c.RESULTS / "diag_nan" / a.variant
    out.mkdir(parents=True, exist_ok=True)
    t.output_folder = t.output_folder_base = str(out)
    t.log_file = str(out / "training_log_diag.txt")
    cudnn.benchmark = True
    t.on_train_start()
    g = c.guard()
    loss_fn = th.HeatmapFocalLoss()
    log = open(out / "iters.jsonl", "w", encoding="utf-8")
    t.network.train()
    ipe = t.num_iterations_per_epoch
    t0 = time.time()
    for it in range(a.iters):
        if it % ipe == 0:
            t.lr_scheduler.step(it // ipe)
        batch = next(t.dataloader_train)
        assert all(g.fold_of(k) != 0 for k in batch["keys"]), "fold-0 case in training batch"
        data = batch["data"].to(t.device, non_blocking=True)
        target = [x.to(t.device, non_blocking=True) for x in batch["target"]]
        t.optimizer.zero_grad(set_to_none=True)
        with t._autocast() if hasattr(t, "_autocast") else torch.autocast("cuda", enabled=True):
            output = t.network(data)
            l = t.loss(output, target)
        rec = {"it": it, "loss": float(l.detach()),
               "out_dtype": str(output[0].dtype),
               "max_abs_logit": [float(o.detach().float().abs().max()) for o in output],
               "nonfinite_logit": [int((~torch.isfinite(o.detach())).sum()) for o in output],
               "n_pos": [int((x >= 1 - 1e-6).sum()) for x in target],
               "scale_loss": [float(loss_fn(o.detach(), x)) for o, x in zip(output, target)]}
        if t.grad_scaler is not None:                       # original fp16 path ("base" run)
            scale_before = float(t.grad_scaler.get_scale())
            t.grad_scaler.scale(l).backward()
            t.grad_scaler.unscale_(t.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(t.network.parameters(), 12)
            t.grad_scaler.step(t.optimizer)
            t.grad_scaler.update()
            scale_after = float(t.grad_scaler.get_scale())
        else:                                               # fixed trainer: bf16, no scaler
            scale_before = scale_after = 1.0
            l.backward()
            gn = torch.nn.utils.clip_grad_norm_(t.network.parameters(), 12)
            t.optimizer.step()
        bad = [n for n, p in t.network.named_parameters() if not torch.isfinite(p).all()]
        rec.update({"grad_norm": float(gn), "scale": scale_before, "skipped": scale_after < scale_before,
                    "n_bad_params": len(bad), "first_bad": bad[:3], "lr": t.optimizer.param_groups[0]["lr"],
                    "keys": list(batch["keys"]), "sec": round(time.time() - t0, 1)})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 25 == 0 or bad:
            print(f"it {it} loss {rec['loss']:.3f} maxlogit {max(rec['max_abs_logit']):.1f} "
                  f"gn {rec['grad_norm']:.2f} scale {scale_before:.0f} bad {len(bad)}", flush=True)
        if bad:
            print("NON-FINITE PARAMETERS at it", it, bad[:5], flush=True)
            break
    try:
        t.dataloader_train._finish()
        t.dataloader_val._finish()
    except Exception:
        pass
    log.close()


if __name__ == "__main__":
    main()
