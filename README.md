# Detect or segment? Two pre-registered nnU-Net experiments

These are class experiments for Intro to ML. They ask one question: if you want to **find** small
objects in images, should you train a network as a **detector** (predict a heatmap peak at each
object) or as a **segmenter** (predict a mask and read off one point per blob)?

Both arms use the same nnU-Net v2 backbone, plans, data, augmentation, optimizer, schedule and
seed. They are scored with the same point-matching rule. Each experiment was pre-registered before
its test set was looked at.

| Experiment | Data | Status |
|---|---|---|
| [`E01_det_vs_seg`](E01_det_vs_seg/) | Brain metastases on 3D T1c MRI (BrainMetShare + UCSF). 97 test scans, 66 patients, 866 lesions | Done |
| [`E02_det_vs_seg_transfer`](E02_det_vs_seg_transfer/) | LIVECell 2D phase-contrast microscopy. 1,460 test images, 444,600 cells, 8 cell lines | Closed 2026-11-18 |

## Arms

- **A: segmenter.** Stock nnU-Net (Dice + CE). Its prediction becomes one point per connected
  component.
- **B: heatmap detector.** The same network trained on Gaussian peaks with a CenterNet focal loss.
  Its prediction becomes local maxima after NMS.
- **C1 to C4: two-stage "detect, then segment".** B proposes points, then A's map (C1) or a
  trained crop segmenter (C2) scores and outlines them. C3 and C4 combine the two. These arms are
  exploratory in E01 and confirmatory in E02.

The primary metric is **CPM**: mean sensitivity at 0.5, 1, 2 and 3 false positives per image.
CIs come from a cluster bootstrap over patients (E01) or wells (E02).

## Results

**E01 (MRI).** A and B **tie overall**: CPM 0.768 vs 0.773, B − A +0.005 [−0.036, 0.055]. They do
fail in different places:

- On large lesions (> 10 mm), A is much better: sensitivity at 1 FP/scan is 0.978 vs 0.728.
- On small lesions (< 3 mm), B is better: CPM +0.100 [0.048, 0.181].

Their errors overlap only partly, so the cascade **C1 beats both**: CPM 0.802, +0.029 over B. It
also keeps A's Dice. This result is exploratory, because C1 was designed after the fold-0 results
were known.

**E02 (cells).** The cascade did **not** transfer as a general rule. A segments cells well (Dice
0.906), but touching cells merge into one component (about 32 components per image against 304
cells), so its detection CPM collapses to 0.093. B reaches 0.943. That left C1 almost nothing to
fix: +0.003 [0.000, 0.004] over B, the primary hypothesis, not replicated.

**Takeaway.** Whether to detect or segment depends on how crowded the objects are. Reading points
off a segmentation is only safe when objects rarely touch.

## Where to read

Each experiment folder has its own `README.md` (layout, commands, rules). The most useful files:

| File | What it is |
|---|---|
| `*/report/E0x_analysis.md` | Interpretation, limitations, follow-ups. Start here |
| `*/report/E0x_data.md`, `*/report/*.md` | Every number, generated from `results/` |
| `*/report/*.png` | FROC curves, per-size sensitivity, example images |
| `*/PREREG*.md` | The pre-registered design, with dated amendments and deviations |
| `E02_det_vs_seg_transfer/PLAN.md` | E02's design history |
| `*/results/` | Summary JSON/CSV, logs, PREREG hash history |
| `*/code/e0x/`, `*/code/tests/` | All experiment code and unit tests |

## What is not in this repo

About 45 GB is left out by `.gitignore`:

- every `data/` folder (images, labels, nnU-Net raw/preprocessed data, trained checkpoints)
- prediction arrays (`.npz`, `.pkl`, `.nii.gz`)

The MRI data is not redistributed here. LIVECell is public.

## Running the code

The code was written for one Windows workstation (RTX 3070 Ti + 5060 Ti) and will not run as-is
anywhere else:

- The `env/*.cmd` launchers point at absolute `G:\` paths.
- They also point at an external Python environment (nnU-Net v2.6.4, torch 2.10 + CUDA 12.8).
- The `data/` folders are not included.

Within that setup, every step is `env\run.cmd -m e0x.<module> [args]`, and the per-experiment
READMEs list each command. The unit tests run with
`env\run.cmd -m pytest -q -p no:cacheprovider code`.
