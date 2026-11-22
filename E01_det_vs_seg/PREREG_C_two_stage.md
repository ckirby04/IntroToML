# E01 addendum C: two-stage "detect, then segment" model (exploratory)

**Status:** written 2026-11-13 at the owner's request. Every parameter of variant C1 below was
fixed **before any C1 output was computed**. The SHA-256 of this file is recorded in
`results/prereg_C_sha256.txt` before `e01.two_stage` is run. Changes are dated amendments at
the end of the file.

**This result is exploratory, not confirmatory.** The aggregate E01 fold-0 results for A and B
(`report/E01_data.md`, `report/E01_analysis.md`) had been seen when this design was written, and
the two-stage idea itself was motivated by them (E01_analysis §8). No unused labelled data
exists: the 85 locked 1.60 test patients stay locked. The C1 rules below were derived from
the E01 design and from Model B's training target. Nothing was tuned on fold-0 outputs.
C1 will be evaluated once.

## 1. Question

Does a two-stage model work better than either base model? Stage 1 uses Model B's heatmap peaks
to *determine* where lesions are. Stage 2 *segments* each detection and uses the segmentation to
confirm and re-score it. The two questions are:

1. **Detection:** does it find lesions better than **Model B** alone? (lesion sensitivity, FROC)
2. **Segmentation:** does it segment better than **Model A** alone? (Dice)

## 2. Variant C1: cascade with no new training

C1 uses only the saved fold-0 maps from E01 (`results/pred/A`, `results/pred/B`), on the
Dataset901 grid. It runs no new inference and trains nothing.

**Stage 1 (detect):** B's points exactly as in PREREG §5: local maxima of the heatmap, NMS
half-width 3.0 mm, value ≥ 0.05. Call each peak q, with peak value b(q).

**Stage 2 (segment and confirm):** Model A's class-1 probability map p is the segmenter. It is
read only at B's detections.
- **Support components:** the 26-connected components K of p ≥ 0.05 (the same candidate set as
  PREREG §5).
- **Assignment:** peak q is assigned to the component K that contains q. If no component
  contains q, it is assigned to the component nearest q in mm, provided that distance is
  ≤ 2.0 mm (the E01 match tolerance). Ties go to the higher max p, then to the lower component
  label. If no component is within 2.0 mm, q is unassigned.
- **Stage-2 confidence:** a(q) = max p over q's component, or 0 if q is unassigned.
- **Two-stage score:** s(q) = (b(q) + a(q)) / 2.
- **Points:** every B peak stays a point, at its own location, with score s(q). C1 therefore
  has exactly B's candidate points; only their ranking changes. Its maximum sensitivity equals
  B's at the 0.05 floor.

**Fixed operating point:** s ≥ τ_C = **0.4** = (0.3 + 0.5) / 2. This is the score of a peak
that sits exactly at both base models' pre-registered fixed operating points (B peak 0.3,
A p 0.5).

**C1 mask (for Dice):** the union, over peaks with s(q) ≥ 0.4, of each peak's region:
- If q is assigned to a component K and K has voxels with p ≥ 0.5, the region is those voxels:
  A's own fixed-op mask, restricted to that component.
- Otherwise the region comes from the heatmap. It is the 26-connected component containing q of
  {heatmap ≥ e⁻² · b(q)}, clipped to a ball of radius 8.0 mm around q.
  - **Reason:** B's target Gaussian falls to e⁻² of its peak at 2σ. With σ = ESD/4 (PREREG
    §3.1), that is a radius of ESD/2, which is the lesion's equivalent radius.
  - 8.0 mm = 2σ_max (σ ≤ 4 mm).

A's p ≥ 0.5 regions with no B peak at s ≥ 0.4 are **not** in C1's mask. Detection comes from
stage 1 only.

## 3. Endpoints (fold 0, 97 scans, 66 patients, 866 lesions)

All matching, size bins, FP targets and FROC/CPM rules are E01's (PREREG §6–§7). All
uncertainty comes from the paired patient bootstrap, with the same draws as E01 (2,000
resamples, seed 20261109). A, B and C1 are resampled together.

**Detection, C1 vs B:**
1. **Primary:** CPM, all lesions. Reported as C1 − B with a 95% CI.
2. **Key secondary:** sensitivity at 1 FP/scan per ESD bin (< 3, 3–10, > 10 mm), C1 − B.
3. **Other:** sensitivity at 0.5, 1, 2 and 3 FP/scan; CPM per bin; FROC curves; and, at the
   fixed operating point, sensitivity, FP/scan and per-bin sensitivity.
4. A's values are shown alongside for context. C1 − A is reported but is not an endpoint.

**Segmentation, C1 vs A:**
1. **Primary:** mean per-scan Dice of C1's mask against GT, against A's mean Dice at p ≥ 0.5
   (as in PREREG §7). Reported as C1 − A with a paired 95% CI. A scan where both mask and GT
   are empty scores 1.
2. **Other:** per-lesion Dice by ESD bin, defined as follows.
   - For GT lesion L, P_L is the union of the 26-connected components of the mask that overlap
     L, and Dice_L = 2|P_L ∩ L| / (|P_L| + |L|). Dice_L is 0 if no component overlaps L.
   - The average is taken over all lesions in each bin.
   - This mixes detection and delineation, which is stated as a limitation, so it is reported
     for both A and C1.
   - Base B has no mask, so B's Dice is "n/a".

No p-values. A CI that excludes 0 is described as such and nothing more.

## 4. Variant C2: trained local stage-2 segmenter (planned; details fixed later)

C2 is a dedicated nnU-Net trained on crops around lesions from folds 1–4. Its positives are GT
interior anchors with jitter; its negatives are random brain crops. At test time it runs on
crops centred on B's fold-0 peaks. Its crop size, sampling, training settings, score rule and
fixed operating point will be set in a dated amendment to this file **before C2 is trained**.
C2 will then be evaluated with the endpoints of §3 unchanged.

## 5. Known limitations

- **Exploratory.** Fold-0 aggregates were seen first. See the Status section.
- **Stage 2 is not independent of stage 1's data.** C1's stage 2 is Model A, trained on the same
  folds as B. C1 is therefore a model combination, not a newly trained cascade. C2 addresses
  this.
- **C1 cannot add candidates.** It re-ranks B's points and cannot recover a lesion B never
  peaks on. Masks for B-only detections come from the heatmap shape, so they are approximate:
  for lesions < 4 mm, σ is clipped at 1 mm, so the mask is about 4 mm across.
- **The score average is a fixed, untuned rule.** Other combinations (product, max, a learned
  combination) were not tried and will not be tried on fold 0.
- E01's limitations (PREREG §10, D1–D7) also apply.

## Amendments

### Amendment C-1: 2026-11-13, variant C2 fixed before any C2 data, training or inference

**What had been seen when this was written:** C1's fold-0 results (`report/two_stage_C1_table.md`),
in addition to A and B's.
- C1 improved CPM over B, mostly on lesions > 10 mm. It did not improve Dice over A: its mask
  is A's mask, and no heatmap fallback mask reached s ≥ 0.4.
- C2 exists to test whether a **dedicated** stage-2 segmenter, rather than A, changes this.
- Nothing below was chosen by comparing alternatives on fold 0. §2's assignment, score and
  mask rules carry over unchanged, so C2 differs from C1 only in its stage-2 map.

**C-1.1 Crop dataset (`Dataset902_E01C2Crops`, written inside `data/`):**
- **Source:** Dataset901 images and labels of **pool folds 1–4 only** (357 scans, 3,825
  lesions on the 1 mm grid). Every crop's source scan is asserted to be outside fold 0 with the
  1.60 DataGuard.
- **Crop:** 96 × 96 × 96 voxels (≈ 96 mm), image and binary label. The label holds **all**
  lesion voxels in the crop. Parts outside the image are padded with 0 in the image and the
  label; 0 is the background value of the z-scored images.
- **Positive crops:** one per GT lesion (26-connected), so 3,825.
  - The centre is the lesion's interior anchor, the max-EDT voxel used by B's target
    (`e01.heatmap`, PREREG §3.1).
  - A random integer offset is added per axis, uniform in [−j, j] voxels, with
    j = round(clip(ESD/4, 4, 10) mm). This mimics B's off-centre and split peaks on large
    lesions.
- **Negative crops:** 0.5 per positive, so 1,912. Scans are chosen in proportion to their
  lesion count.
  - Half are centred on a random brain voxel (image ≠ 0).
  - Half are centred on a random voxel among the scan's brightest 0.5% of brain voxels, as
    proxies for vessels and other enhancing mimics.
  - Every negative centre must be ≥ 10 mm from every lesion voxel; a draw is retried up to 100
    times, then skipped. A negative crop may still contain lesion voxels near its edge, which
    are labelled.
- **Sampling seed:** 20261113. A manifest of crop IDs, source scans, centres and offsets is
  written to `results/two_stage/c2_crop_manifest.csv`.

**C-1.2 Training:**
- **Model:** nnU-Net v2 2.6.4, default planner (`nnUNetPlans`) on Dataset902, `3d_fullres`,
  stock `nnUNetTrainer_250epochs` (Dice + CE, fp16 AMP, default augmentation, deep supervision).
- **Data:** fold `all` (trains on every crop) with seed 1600, on the RTX 3070 Ti, through the
  E01 queue/pause path. `checkpoint_final` only.
- **Normalisation:** the channel is named `noNorm` (no re-normalisation). The crops are
  already z-scored inside the brain on the full scan, so per-crop z-scoring would change the
  intensities seen at test time.
- **Plans:** patch and batch sizes are whatever the default planner chooses, recorded and not
  edited.
- **Pre-flight (training folds only):** a 2-epoch smoke run, then stage-2 inference on the two
  training-fold scans in `results/pred_smoke/` (BMS_Mets_013, BMS_Mets_019). Those use the B
  maps already there.

**C-1.3 Fold-0 inference (stage 2):**
- **Crops:** for every B peak q (PREREG §5; 1,827 peaks on fold 0), a 96³ crop centred on q is
  taken from the fold-0 image, with zero padding as in C-1.1.
- **Prediction:** nnU-Net's predictor, step 0.5, Gaussian weighting, mirroring TTA and fp32,
  as in E01 §4/D7. The stage-2 map p₂ is the class-1 probability on that crop.
- **Per peak, the §2 rules apply with A's map replaced by p₂:**
  - the support components are the 26-connected components of p₂ ≥ 0.05 within the crop;
  - assignment is "contains q, else nearest within 2.0 mm";
  - a₂(q) = max p₂ over the assigned component, or 0;
  - s(q) = (b(q) + a₂(q)) / 2, with the fixed operating point s ≥ 0.4;
  - the region is the assigned component ∩ {p₂ ≥ 0.5}, or else the heatmap fallback of §2.
- **C2 mask:** the union of the regions of peaks with s ≥ 0.4, pasted back onto the scan (parts
  in the padding are dropped). When crops overlap, the regions are unioned. Every crop is
  evaluated independently.

**C-1.4 Endpoints:** §3 applies unchanged to C2.
- **Detection:** C2 − B. The primary is CPM; the key secondary is sensitivity at 1 FP/scan per
  bin.
- **Segmentation:** C2 − A. The primary is mean per-scan Dice; per-lesion Dice by bin is
  reported as "other".
- **Also reported, not endpoints:** C2 − C1 for every metric. A, B, C1 and C2 share the
  bootstrap draws.

**C-1.5 Added limitations:**
- **No realistic false-positive crops.** Negatives are random or bright-voxel crops, not B's
  real false positives, because B has no out-of-fold predictions on folds 1–4.
- **Stage 2 sees only 96 mm of context.** A lesion larger than the crop around an off-centre
  peak is truncated in the mask.
- **Mask quality depends on peak count.** A large lesion with several B peaks gets several
  crops, unioned.
- **One seed;** training variance is not estimated.

### Amendment C-2: 2026-11-15, variant C3 (symbolic size-gated router), fixed before any C3 output

**Status: post hoc and exploratory.** The rule was written *after* the fold-0 C1 and C2 results
were seen (C2 is weaker on lesions > 10 mm, stronger on smaller ones). Its fold-0 numbers are
therefore optimistic, and they are not evidence that the rule generalises. No new training, no
new inference: C3 uses the saved A and B maps and the saved C2 stage-2 outputs. A learned
meta-learner was considered and not done, because A, B and C2 have no out-of-fold predictions on
folds 1–4 and fold 0 is the test set.

**C-2.1 Rule, per B peak q (PREREG §5 peaks; same points as C1 and C2):**
- **Gate.** Let k be q's A component from §2 (A map ≥ 0.05, "contains q, else nearest within
  2.0 mm"). Let R_A = component k ∩ {A ≥ 0.5}, and ESD_A = (6·V/π)^(1/3), with V the volume of
  R_A in mm³. q is **large** if k exists and ESD_A > 10 mm. The 10 mm cut is E01's existing
  large-lesion bin edge; it is not tuned.
- **Large q → A route (C1's rule):** s(q) = (b(q) + a(q)) / 2, region = R_A.
- **Otherwise → C2 route (C-1.3):** s(q) = (b(q) + a₂(q)) / 2, region = C2's crop region, or
  the §2 heatmap fallback if that is empty.
- **Operating point:** s ≥ 0.4, unchanged. **C3 mask:** the union of the regions of kept peaks.

**C-2.2 Endpoints and reporting:** §3 applies unchanged. C3 − B (detection) and C3 − A (Dice)
are reported, along with C3 − C1 and C3 − C2 as context, on the same 2,000 bootstrap draws as A,
B, C1 and C2. The C1 and C2 output files are not rewritten.

### Amendment C-3: 2026-11-15, variant C4 (C1 scores, C2 outlines), fixed before any C4 output

**Status: post hoc and exploratory**, for the same reasons as C-2, and more so. The rule comes
from the C3 diagnosis on fold 0: on the 165 A-route peaks, C1 and C2 scores were identical. C1's
lead on lesions > 10 mm came from C2 scoring false positives higher (18.5% of C2's FPs at
s ≥ 0.6, against 7.8% for C1). Its fold-0 numbers are optimistic. No new training, no new
inference.

**C-3.1 Rule, per B peak q (same points as C1–C3):**
- **Score:** C1's s(q) = (b(q) + a(q)) / 2 (§2), for every peak. The operating point is
  s ≥ 0.4, unchanged.
- **Region of a kept peak:** C2's crop region (C-1.3). If that is empty, use the §2 heatmap
  fallback, as in C2.
- **C4 mask:** the union of the regions of kept peaks.
- **Consequence:** C4's detection metrics equal C1's by construction. Only its mask differs.

**C-3.2 Endpoints and reporting:** as C-2.2, with C4 in place of C3. C4 − B (detection), C4 − A
(Dice), and C4 − C1 and C4 − C2 as context, on the same 2,000 bootstrap draws. Earlier variants'
output files are not rewritten.

## Deviations

Append-only and dated.

- **DC1 (2026-11-14): C2 trains on the RTX 5060 Ti, not the 3070 Ti (C-1.2). Owner decision.**
  No fold-0 data had been used.
  - **Cause:** the default planner chose batch 6 at patch 96³ for Dataset902. On the 8 GB 3070
    Ti that needs 8.12 GiB peak, so it spilled into shared system memory.
  - **Effect seen in the 2-epoch smoke run** (`results/logs/C2_smoke.log`): about 1,600–1,760 s
    per epoch, so about 5 days for 250 epochs.
  - **Diagnosis** (training-fold crops and random tensors only): the data loader takes
    0.04–0.07 s per batch, so it is not the bottleneck. A training step takes 3.5–5 s at batch
    6 on the 3070 Ti, 0.25 s at batch 4 (6.1 GiB) and 0.39 s at batch 6 on the 16 GB 5060 Ti
    (9.1 GiB).
  - **Choice:** the plan is kept exactly as the planner made it, and only the GPU changes. The
    alternative, editing the batch size to 4 on the 3070 Ti, was declined because it changes a
    training hyperparameter.
  - **Unchanged:** the trainer, plans, seed, epochs, fold `all` and `checkpoint_final`. The
    queue still runs one nnU-Net training at a time (PREREG D3).
  - **Smoke model:** it was trained on the 3070 Ti and is used only for the training-fold
    pre-flight.
  - Fold-0 stage-2 inference is not GPU-specific in C-1.3. It runs on the 3070 Ti as E01's did.
