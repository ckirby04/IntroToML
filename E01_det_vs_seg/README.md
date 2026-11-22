# E01_det_vs_seg: lesion detection vs. segmentation on one nnU-Net backbone

A class ML sub-experiment of JANNUS / BrainMetScan. It is **not** part of the 1.X release line
and produces **no 1.60 artifacts**. The design is in `PREREG.md`.

## Layout
```
env\e01_env.cmd, env\run.cmd    session env (nnUNet_* -> data\), python launcher
data\nnUNet_raw\Dataset901_E01T1c   hard links to 1.60_artifacts\nnUNet_raw\Dataset170_JannusT1c
data\nnUNet_preprocessed\           E01 plans/preprocessing (+ splits_final.json copy)
data\nnUNet_results\                 E01 checkpoints
splits\                              splits_final.json (fold 0) + split_source.json
configs\                             run configuration
code\e01\                            all E01 code;  code\tests\  unit tests
results\                             logs, predictions, metrics, summary.json
report\                              figures and tables
```

## How to run
`env\run.cmd -m e01.<module> [args]`. This uses `G:\BrainMetShare\1.60_artifacts\venv-1.60`
read-only.

## Addendum C: two-stage "detect (B), then segment" (exploratory; `PREREG_C_two_stage.md`)
| Step | Command |
|---|---|
| C1 cascade (B peaks, A's map as stage 2; no training) | `env\run.cmd -m e01.two_stage` |
| C2 crop dataset (folds 1-4 only) / nnU-Net preprocessing | `env\run.cmd -m e01.c2 --build`, then `--preprocess` |
| C2 training (stock nnU-Net, Dataset902, fold `all`) | `env\run.cmd -m e01.launch_train --arm C2 [--smoke]` (only on the owner's go-ahead) |
| C2 stage-2 inference on crops at B peaks | `env\run.cmd -m e01.c2_predict --cases smoke [--smoke-model]` / `--cases test` |
| C2 evaluation (A, B, C1, C2 on shared bootstrap draws) | `env\run.cmd -m e01.two_stage --c2` |
| C3 size-gated router (Amendment C-2, post hoc; no training) | `env\run.cmd -m e01.two_stage --c3` |
| C4 C1 scores + C2 outlines (Amendment C-3, post hoc; no training) | `env\run.cmd -m e01.two_stage --c4` |

The addendum's SHA-256 history is in `results\prereg_C_sha256.txt` (last line = current); `e01.two_stage`
refuses to run if the file has changed since.

## Training control (sequential queue: A, then B, on the 3070 Ti)
| Command | Effect |
|---|---|
| `env\status.cmd` | PAUSE flag, running E01 processes, queue events, last epoch per arm, GPUs |
| `env\pause.cmd` | Sets `results\PAUSE`. The current epoch finishes (~1.5 min), a checkpoint is saved, the job exits with code 3, and the queue stops. Nothing is lost. |
| `env\resume.cmd` | Clears PAUSE and starts the queue detached. Finished jobs are skipped; an interrupted job resumes from `checkpoint_latest.pth`. Refuses to start if an E01 job is already running. |

Never run two nnU-Net trainings at once on this machine. Together they exhaust commit memory
(PREREG D3).

## Rules (binding for every agent and subagent)
1. **Locks from `1.60\CLAUDE.md` apply here too:**
   - Never list or read `Raw Data\Yale-Brain-Mets-Longitudinal`, `Superset`, `Dataset001`,
     `Dataset002`, or `BMS_SYNTH_` / `BMS_PSEUDO_` / `Yale_` / `YG_` / `PRETREAT_` paths.
   - Never touch the 85 1.60 test patients. Dataset901 contains none of them, and
     `setup_dataset.py --verify-only` checks this with the 1.60 DataGuard.
2. **Write nothing outside this folder.** `1.60\` and `1.60_artifacts\` are read-only. In
   particular, never open a hard-linked raw file for writing.
3. **Never import `p3_common` or `pilot_common`.** They repoint `nnUNet_*` at 1.60.
   `e01.common` sets the paths and `assert_paths()` checks them.
4. **No E01 trainer goes in site-packages.** `launch_train.py` instantiates trainer classes
   directly.
5. **Fold 0 is the test set.** Nothing is tuned by looking at fold-0 outputs (`PREREG.md`).
6. **No data leaves this machine** (1.60 D20). E01 weights are research-only (D19).
