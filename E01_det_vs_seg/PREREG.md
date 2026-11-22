# E01 pre-registration: detection vs. segmentation on one backbone

**Status:** APPROVED by the owner on 2026-11-09 at Checkpoint 1, with Amendment 1 (see end),
before any E01 training run. Any further change is a dated amendment at the end of this file,
with the reason, and it must state whether any fold-0 output had been seen.

## 1. Question

With the backbone, data, augmentation, optimiser, schedule and seed held fixed, does training
nnU-Net as a **centroid-heatmap detector** (Model B) find more metastases at a given false-positive
rate than training it as a **voxel-wise segmenter** (Model A), overall and by lesion size?

## 2. Data and split

- **Data:** `Dataset901_E01T1c`, made of hard links to 1.60 `Dataset170_JannusT1c`. It is the
  split_v1 development pool: 454 scans from 334 patients, with 0 scans from the 85 locked 1.60
  test patients. The input is one channel, `t1c`, on the 1.60 Phase 1 grid (about 1.0 mm
  isotropic, z-scored inside the brain mask).
- **Split:** split_v1 (manifest SHA-256 `740ff72c…f1c`), used as it stands:
  - **Train:** pool folds 1–4, 268 patients and 357 scans.
  - **Test:** pool fold 0, 66 patients and 97 scans.
  - These are patient-disjoint, which `setup_dataset.py` checks.
- **There is no validation set and no model selection.** Nothing is tuned on fold 0. Every
  parameter below is fixed in this file before training.

## 3. Models

The shared settings are nnU-Net v2 2.6.4, `3d_fullres`, plans from the default planner (patch
160×128×112, batch 2), `nnUNetTrainer_250epochs` (250 epochs), SGD with poly LR, default
augmentation, deep supervision on, seed 1600, RTX 3070 Ti, and **`checkpoint_final` only**.
`checkpoint_best` is never used.

| | Model A: segmentation | Model B: heatmap detection |
|---|---|---|
| Trainer | stock `nnUNetTrainer_250epochs` | `E01HeatmapTrainer(nnUNetTrainer_250epochs)` |
| Output | 2-class softmax | 1-channel sigmoid heatmap |
| Target | binary mask | max over lesions of 3D Gaussians, see §3.1 |
| Loss | Dice + CE (nnU-Net default) | penalty-reduced focal loss (CenterNet): α = 2, β = 4, normalised by the number of peak voxels, wrapped with nnU-Net's deep-supervision weights |
| Output-layer init | nnU-Net default | bias −4.6 (prior 0.01); weights default |

Only the final 1×1×1 output convolutions, the target and the loss differ between the arms.

### 3.1 Model B target (fixed)

The target is built on the fly from the **augmented** patch mask:
- Take the 26-connected components (cc3d).
- For each component, the peak voxel is its **interior anchor** (Amendment 1): the component
  voxel with the largest Euclidean distance transform.
  - The EDT is computed in mm, using the plan spacing, on the component mask padded with
    background. A lesion cut by the patch border is therefore treated as its visible part.
  - Ties go to the voxel nearest the component centroid in mm, then to the first in C order.
  - The anchor always lies inside the lesion, even for ring-enhancing or irregular lesions whose
    centroid falls outside the mask.
- The target is exactly 1.0 at the anchor.
- Around the anchor, place a Gaussian with σ in **mm**, converted to voxels per axis with the
  plan spacing:
  **σ_L = clip(ESD_L / 4, 1.0 mm, 4.0 mm)**, where ESD_L = (6V/π)^(1/3) and V is the component
  volume in mm³ within the patch.
- Truncate each Gaussian at 3σ. Where lesions overlap, take the elementwise max.
- A lesion cut by the patch border gets the anchor of its visible part. This is a known
  approximation.
- Deep-supervision targets follow §7 of the Phase 3 test (to be appended): either downsampling
  of the full-resolution heatmap, or heatmaps generated per scale with σ in mm.

## 4. Inference (identical for both arms)

- nnU-Net sliding window: tile step 0.5, Gaussian tile weighting, mirroring TTA on (nnU-Net
  default).
- Outputs go back to the Dataset901 image grid. Model A's output is the class-1 probability;
  Model B's is the sigmoid heatmap.

## 5. From output maps to scored point predictions (fixed)

Each arm turns its map into a list of **points with scores**:

- **Model A:**
  - Take the 26-connected components of `p ≥ 0.05` (the candidate threshold).
  - Each component gives one point, at its voxel of maximum probability (ties go to the first
    in C order).
  - The score is that maximum probability.
- **Model B:**
  - A local maximum is a voxel equal to the maximum over a box neighbourhood of half-width
    **r_NMS = 3.0 mm** per axis (in voxels: `round(3.0 / spacing)`), with value ≥ 0.05.
  - Plateaus are reduced to one voxel, the first in C order.
  - The score is the peak value.

The 0.05 floor only bounds the length of the FROC. It is the same for both arms and is not an
operating point.

## 6. Matching rule (identical for both arms)

- **GT lesions:** 26-connected components of the Dataset901 label on the same grid, with
  spacing from the NIfTI header.
- **Hit rule:** a predicted point hits GT lesion L if its Euclidean distance in mm to the
  nearest voxel of L is **≤ r_tol = 2.0 mm**. A point inside L has distance 0. One point may hit
  several lesions.
- **Lesion detection score:** the max score of the points that hit it, or −∞ if none does.
- **False positive:** a point that hits no lesion. Extra points on an already-detected lesion
  are **not** FPs. They are counted and reported separately as "redundant" per arm, with
  `MatchResult.n_redundant`.
- **Implementation:** a point-rule `MatchResult` (rule `"point"`) fed to
  `jannus.evaluation.detection.metrics`. The legacy `jannus.evaluation.metrics` is not used.

## 7. Endpoints

**Units:**
- Case = scan (97).
- FP/scan is the primary FP rate. FP/patient (66) is reported as a secondary rate.
- Size bins use the GT lesion's ESD on the evaluation grid: **< 3 mm, 3–10 mm, > 10 mm**. A
  lesion of exactly 3 or 10 mm goes in the upper bin.
- These bin counts are taken from components of the **1 mm resampled** labels. They will differ
  from the native-grid counts in the split_v1 tables and in the proposal (175 / 597 / 93 for
  fold 0). The E01 counts are reported as computed (Amendment 1).

**Primary:**
1. **CPM:** mean sensitivity at 0.5, 1, 2 and 3 FP/scan (`jannus` `cpm`, a step FROC with no
   interpolation), over all lesions.

**Key secondary:**
2. Sensitivity at 1 FP/scan in each size bin.

**Other:**
- FROC curves, all lesions and per bin.
- Sensitivity at 0.5, 1, 2 and 3 FP/scan.
- Redundant points per scan.
- Model A only:
  - Dice per scan at p ≥ 0.5, averaged over scans; a scan where both prediction and label are
    empty scores 1.
  - The 1.60 overlap rule (`match_by_overlap`) as a reference.

**Pre-registered descriptives (not endpoints; Amendment 1):**
- **Model A merged detections:** the number of 26-connected components at p ≥ 0.05 that overlap
  two or more distinct GT lesions by at least one voxel each. Reported in total and per scan.
- **GT anchor check:** the number of fold-0 GT lesions whose raw centroid is more than 2.0 mm
  from every voxel of the lesion. This uses GT only, no predictions. It is reported with the
  same count for the training folds.

**Fixed operating points, set a priori and not tuned:** A at p ≥ 0.5, B at peak ≥ 0.3. Report
sensitivity, FP/scan and per-bin sensitivity at each.

**Unreached targets:** if an arm's FROC never reaches a target FP rate, its maximum sensitivity
is used and **flagged**, never extrapolated (`FROC.sensitivity_at_fp` → `reached`).

## 8. Uncertainty

- Patient-level cluster bootstrap over the 66 test patients (`jannus` `bootstrap`): 2,000
  resamples, seed 20261109, 95% percentile CIs.
- A − B differences come from **paired** resamples, with the same patient draw for both arms.
- No p-values. A difference whose CI excludes 0 is described as such, and nothing more.

## 9. Figures (selection rule fixed, by GT only)

- FROC comparison: all lesions, plus one panel per size bin.
- Per-bin sensitivity at 1 FP/scan with CIs.
- Example slices. These three test scans are chosen by GT properties only, before any
  prediction is viewed:
  - the scan with the most < 3 mm lesions;
  - the scan whose lesion count is the median;
  - the scan with the largest single lesion.
  
  Each example shows the axial slice through the largest lesion's centroid, with GT, A's mask
  at p ≥ 0.5, and B's points at peak ≥ 0.3 overlaid.

## 10. Known limitations

- **Fold 0 is not fully blind.** The owner has seen 1.60 pilot results whose 454-scan OOF pool
  includes fold 0 (3D nnU-Net, t1c and t1c+flair arms, aggregated). PilotC fold 0, a model like
  A, was part of that evaluation. During training, nnU-Net logs pseudo-Dice for Model A on fold
  0 (its "validation" fold) and Model B's loss on fold 0. These logs are not used for any
  decision, and no checkpoint is selected from them.
- **One split and one seed per arm.** Training variance is not estimated, so the bootstrap CIs
  cover patient sampling only.
- **Grid.** Evaluation is on the 1.0 mm Phase 1 grid, not the native grid (unlike 1.60 D15).
  ESD bins use resampled labels, so lesion counts per bin differ from the split_v1 tables, which
  are native.
- **Partial lesions.** Model B's targets use the centroids of lesions partly visible in a patch
  (§3.1).
- **cudnn.benchmark is on**, as in 1.60, so runs are not bit-reproducible.
- **Dice is computed for Model A only.**

## Deviations

Append-only and dated. These record departures from the plan or from process, as distinct
from changes to the design, which are amendments.

- **D1 (2026-11-09): fold-0 pseudo-Dice was printed.** While checking the 2-epoch Model A smoke
  run (`E01A_smoke`), the main session's log filter printed nnU-Net's per-epoch pseudo-Dice on
  fold 0 for those 2 epochs.
  - Nothing was decided on it, and the smoke model is not used.
  - From then on, every log view filters out `Pseudo dice` / `EMA pseudo Dice` lines.
- **D2 (2026-11-09): extra line in the 1.60 RETRAIN_LOG.** One outcome bullet was appended after
  the owner-approved E01 inventory entry ("SOURCES.md regenerated…"). It records that a trial
  regeneration was byte-identical and states the resulting row changes. It was disclosed to the
  owner at Checkpoint 2.
- **D3 (2026-11-09): Model A crashed at epoch 34 and was resumed from epoch 30.**
  - **Cause:** the owner-approved 2-epoch Model B smoke run on the 5060 Ti ran alongside A. With
    both nnU-Net runs' augmentation workers (nnU-Net default per run) on a 32 GB RAM / 55.8 GB
    commit machine, Windows ran out of commit memory (WinError 1455, "paging file too small").
    A's workers died at the start of epoch 34, at 17:40.
  - **Resume:** A was resumed at 17:46 from `checkpoint_latest.pth`, which nnU-Net writes every
    10 epochs. The resume ran with the same trainer, plans, splits, seed and GPU.
  - **Effect:** epochs 30–33 are retrained. nnU-Net's resume restores network, optimizer, LR
    epoch and grad-scaler state, but not the data-sampling RNG, so A's run is not one
    uninterrupted stream. No result had been seen. Fold-0 pseudo-Dice lines were not viewed.
  - The 1.60 schedule already avoided overlapping nnU-Net runs for this reason. E01 did not, and
    that was the main session's error.
- **D4 (2026-11-09): pause support added; A restarted at the epoch-60 boundary; B queued after A.**
  - **Owner request:** add pause support and run both arms sequentially.
  - **Pause mechanism** (`e01.pause`, copied from the 1.60 pattern): an atomic
    `checkpoint_latest.pth` after every epoch, and a `results\PAUSE` flag that stops the run
    cleanly with exit code 3.
  - **Queue** (`e01.jobqueue`): runs A and then B on the 3070 Ti, one at a time, so two nnU-Net
    runs never overlap (D3).
  - **Restart of A:** the running A process, without pause support, was stopped right after
    nnU-Net's own epoch-59 save. The "Epoch 60" log line confirms that save had finished, and
    the checkpoint was backed up first. A resumes from there under the queue with pause enabled,
    so no epochs are lost. As in D3, the data-sampling RNG restarts at the resume.
  - **B** keeps the identical plan, seed, epochs and GPU (3070 Ti) as A, so there is no
    GPU deviation.
  - **Training is otherwise unchanged:** the per-epoch save replaces nnU-Net's 10-epoch save
    and affects nothing else.
  - **Live test:** A was paused once deliberately after 63 completed epochs (exit 3, queue
    stopped) and resumed at epoch 63. No epoch was lost. The data-sampling RNG restarted once
    more.
- **D5 (2026-11-12): owner pauses of A; more fold-0 pseudo-Dice lines seen.**
  - A was paused at the owner's request after epochs 65, 134 and 199, and resumed each time at
    the next epoch. No epoch was lost; the data-sampling RNG restarted at each resume (as D3).
  - A finished all 250 epochs on 2026-11-11 (exit 0, `checkpoint_final.pth`).
  - While confirming resumes, the agent's log check matched only lowercase "pseudo", so
    nnU-Net's "Pseudo dice" line for fold 0 was seen once (epoch 65 of A: 0.8274). As in D1,
    nothing about A was tuned from it, and A has no free settings left. The status display
    now filters it case-insensitively.
- **D6 (2026-11-12): Model B's first full run diverged to NaN; B's training numerics changed
  and B retrained from scratch.** This changes §3 for Model B.
  - **What happened:** the first full B run (started 2026-11-11 after A finished) had train loss
    61.6, 3.0, 2.4, 18.3 at epochs 0–3, then NaN from epoch 4 to 113, when the owner paused it.
    290 of 292 weight tensors were NaN and the fp16 GradScaler scale had fallen to 0. The run is
    kept, unused, in `data\failed_runs\B_nan_20261111\`.
  - **Diagnosis (training folds only):** `e01.diag_nan` re-ran B's real training loop with
    per-step logging (`results\diag_nan\base\`). Every batch was asserted to come from folds
    1–4, and no validation was run. With the default ("weights default") output-layer init,
    initial logits reached ±15, so the first-step loss was 670–6,800 and fp16 gradients
    overflowed (13 skipped steps). Around steps 780–910, loss spikes at the second
    deep-supervision scale (150–600, grad norm up to 12,000) were followed by fp16 forward
    overflow. The GradScaler scale then fell to 0, and the weights became NaN at step 1,080. The
    fold-0 val losses printed by the failed run were not used.
  - **Changes, B only:**
    1. Output 1×1×1 convs: weights N(0, 0.01) plus the existing bias −4.6, which is the full
       focal-loss prior init. This replaces §3's "weights default".
    2. Mixed precision is bfloat16 autocast with no GradScaler, instead of nnU-Net's fp16 plus
       GradScaler. bf16 has fp32's exponent range.
    3. A NaN guard: a non-finite epoch train loss or weight stops the run before that epoch's
       checkpoint is saved. The job exits non-zero and the queue stops.
  - **Unchanged:** the loss (α, β, normalisation, DS weights), the targets, the plans,
    optimizer, LR, clipping, epochs, seed and GPU. A is unchanged; it was trained in fp16.
  - **Check (training folds only, `results\diag_nan\fix\`):** 2,500 steps (10 epochs) with no
    non-finite loss. The epoch-mean loss fell from 3.93 to 1.37, there were no spikes after
    epoch 0, and the speed was unchanged at 0.37 s/step. All 31 unit tests pass.
  - **Consequence for the comparison:** A and B now also differ in output-layer weight init and
    in AMP dtype. This is reported as a limitation. Inference precision for both arms is set
    in `e01.predict` and is identical for both.
- **D7 (2026-11-13): inference details not fixed by §4; one observation after results.**
  - **Precision:** §4 does not specify it. Both arms ran fp32 (autocast off), chosen before any
    fold-0 output existed. B ran through nnU-Net's predictor as `cat([0, logit])`, so
    softmax[1] = sigmoid(logit). The pre-flight check used 2 training-fold scans
    (`results\pred_smoke\`).
  - **Model selection:** B was retrained after D6 and completed 250 epochs (owner pauses after
    epochs 76 and 151, resumed without loss). As pre-registered, `checkpoint_final` was used
    for both arms. B's fold-0 val loss rose late in training (about 1.0 to 1.4); this was not
    acted on.
  - **Observed after results (not acted on):** 458 of A's 843 candidate points have a score of
    exactly 1.0 (saturated probabilities), 19 of them FPs. A's FROC therefore cannot rank
    within that group: its first step jumps to 0.20 FP/scan. A's curve also ends at about 1.5
    FP/scan at the 0.05 floor, so A's sensitivity at 2 and 3 FP/scan is the flagged maximum
    (unreached), and this enters A's CPM. Both points are reported as limitations, with no
    re-analysis.

## Amendments

### Amendment 1: 2026-11-09, at owner approval (Checkpoint 1)

No E01 model had been trained and no fold-0 output existed when this amendment was made.

1. **§3.1:** Model B's Gaussian now sits at the lesion's interior anchor, the maximum-EDT voxel
   (ties go to the voxel nearest the centroid), instead of the voxel nearest the centroid.
   - **Reason:** the centroid of a ring-enhancing or irregular lesion can fall outside the
     mask, and beyond the 2.0 mm match tolerance.
   - **Required unit test:** a hollow-shell lesion.
2. **§7:** two descriptives added, Model A merged components at p ≥ 0.05 and the GT
   raw-centroid check. Neither is an endpoint.
3. **§7:** a note that size-bin counts come from the 1 mm grid and differ from the
   proposal/split_v1 tables.
