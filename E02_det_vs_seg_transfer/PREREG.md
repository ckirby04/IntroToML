# E02 pre-registration: does E01's detect-vs-segment result transfer to cell microscopy?

**Status:** APPROVED by the owner on 2026-11-16 at Checkpoint 1, before any E02 training run.
Written the same day from `PLAN.md` (draft v2 and the owner decisions of 2026-11-16).
- The SHA-256 of this file is in `results\prereg_sha256.txt` (last line = current).
- Any later change is a dated amendment at the end of this file, with the reason, and it must
  state whether any test-set output had been seen.

**What existed when this was written:**
- No E02 model had been trained or run, and no E02 prediction of any kind existed.
- The dataset was built, and the parameters in §5 were computed from training images only
  (`results\train_stats.json`).
- From the test set, only ground-truth counts had been computed: images, wells, cells and cells
  per size bin, as E01 reported its fold-0 bin counts. No test image was viewed.
- E01's results (`..\E01_det_vs_seg\report\`) are known. They are the source of the hypotheses.

## 1. Question

E01 compared, on one nnU-Net backbone and on brain-metastasis MRI, a segmenter (A) with a
heatmap detector (B), and post hoc two-stage combinations (C1–C4). E02 asks whether E01's
findings hold in a different domain: phase-contrast cell microscopy (LIVECell), with every rule,
including the two-stage ones, fixed before any E02 model is trained.

E02 is the first **confirmatory** test of the two-stage design, which E01 could test only
exploratorily.

## 2. Hypotheses

All comparisons are paired, on the E02 test set (§2.1), with 95% CIs from §8.

| ID | Hypothesis (direction from E01) | Comparison | Condition A: CI |
|---|---|---|---|
| H1 | Overall detection is level | CPM, all cells, B − A | CI includes 0 *and* lies within ±0.05 |
| H2 | The segmenter wins on large objects | sensitivity at "1 FP", large bin, A − B | CI > 0 |
| H3 | The detector wins on small objects | CPM, small bin, B − A | CI > 0 |
| H4 | **Primary.** Two-stage beats the detector | CPM, all cells, C1 − B | CI > 0 |
| H5 | Two-stage keeps the segmenter's quality | mean per-image Dice at the fixed operating points, C1 − A | CI lower bound > −0.02 |
| H6 | Crop outlines help | mean per-object Dice, all cells, C4 − A | CI > 0 |

**Condition B, per cell line (owner decision D11).** The test set has only 9 wells, so the
well-clustered CI is crude. A hypothesis counts as **replicated** only if condition A holds
*and* the point estimate, computed on each cell line's test images alone, points the hypothesised
way in at least 7 of the 8 cell lines:
- H2–H4, H6: difference > 0 in the line. H5: difference > −0.02. H1: difference within ±0.05.
- A line takes part in a size-bin hypothesis (H2, H3) only if it has at least **200** test GT
  cells in that bin. If fewer than 8 lines take part, the rule becomes "at most one taking-part
  line goes the other way", and at least 5 lines must take part; otherwise condition B is
  "not assessable" and the hypothesis is reported as **not replicated**.
- The FP targets of §7 stay the same for every line.

**Outcome words:** "replicated" (A and B hold), "not replicated" (otherwise). For H1, "not
replicated" is split in the report into "B better", "A better" or "inconclusive" by where the CI
lies, as description only.

- H4 is primary. H1–H3 test whether E01's base result transfers; H5–H6 are secondary.
- No p-values, and no correction for multiplicity: there is one primary hypothesis.

### 2.1 Data

- **Dataset:** LIVECell (Edlund et al., *Nature Methods* 2021), CC BY-NC 4.0, downloaded
  2026-11-16 from the official bucket. SHA-256 of the four files: `data\raw_sha256.txt`.
- **Built dataset:** `Dataset921_E02LiveCell` (`e02.setup_dataset`), 704 × 520 px grayscale
  PNG, one channel. The build is verified in `results\dataset_verify.json`.
- **Train:** the official train + val sets (D5): 3,727 images, 22 wells, 1,194,095 cells. There
  is no validation set and no model selection, as in E01.
- **Test:** the official test set: 1,460 images, 9 wells (one per cell line, two for SKOV3),
  444,600 cells. No well is in both sets.
- **Doubly annotated images (D10):** 148 file names carry two different annotation sets. In
  training, the set with the higher COCO image id is kept. The 52 such test images are left out.
- **One class:** every cell of every line is "cell". Cell line and well are metadata.
- **Nothing is tuned on the test set.** Test images are read only by the single prediction run
  (§9 step 5) and by GT-only checks.

## 3. Ground truth (fixed in the dataset build)

Rasterisation, from the COCO polygons (`e02.setup_dataset`, with unit tests):
- A pixel belongs to a polygon if its centre lies inside it.
- **Overlaps** go to the **smaller** polygon (equal area: the higher annotation id).
- Polygons left with no pixel are dropped (87 in training, 34 in test).
- **Instance map:** uint16, one id per cell, in `data\instances\`.
- **Gapped binary label:** the union of instances, with a 1-px background gap wherever two
  instances touch (8-neighbourhood). The **larger** instance gives up its border pixel (equal
  area: the lower id). In training no cell is emptied; in test 2 are.

**Objects for detection** are the instances of the instance map. A cell split into pieces by
the overlap rule is still one object.

**Size** of an object: ESD = √(4A/π), A = its pixel count in the instance map. Bins, from
training-set quantiles at E01's training bin shares (§5):
- **small** ESD < 13.63 px; **medium** 13.63–29.45 px; **large** ≥ 29.45 px.
- A cell exactly on an edge goes in the upper bin.
- Test GT counts: 107,660 / 264,573 / 72,367.

**Ground truth for Dice** is the gapped binary label, for every arm.

## 4. Models

**Shared:**
- nnU-Net v2 2.6.4 from the 1.60 venv, used read-only. No trainer is installed into
  site-packages; E02's trainers live in `code\e02`.
- `2d` configuration, plans from the default planner, recorded and not edited.
- Trainer base `nnUNetTrainer_250epochs`: 250 epochs, SGD with poly LR, default augmentation,
  deep supervision on.
- Fold `all` (every training image; nnU-Net's own validation then runs on training images, so
  no test image is ever seen in training). Seed 1600.
- `checkpoint_final` only.
- **bf16 autocast with no GradScaler** for every trained arm, and a NaN guard (a non-finite
  epoch loss or weight stops the run before that epoch's checkpoint is saved). This is E01 D6's
  B setting, applied to all arms so precision is not a confound.
- RTX 3070 Ti, through the E01-style queue/pause path. **Never two nnU-Net trainings at once.**

| | A: segmenter | B: heatmap detector |
|---|---|---|
| Trainer | `nnUNetTrainer_250epochs` with bf16 (§4) | E01's `E01HeatmapTrainer`, ported to 2D |
| Target | gapped binary label (§3) | Gaussians at instance anchors (§4.1) |
| Output | 2-class softmax | 1-channel sigmoid |
| Loss | Dice + CE | penalty-reduced focal loss (CenterNet), α = 2, β = 4, normalised by the number of peak pixels, in nnU-Net's deep-supervision wrapper with weights 1/2^i, normalised |
| Output-layer init | nnU-Net default | weights N(0, 0.01), bias −4.6 |

### 4.1 Model B target (fixed)

- The target is built on the fly from the **augmented instance map**. The instance map goes
  through nnU-Net's augmentation as a second label channel with nearest-neighbour
  interpolation. A unit test shows that instance ids survive augmentation.
- For each instance present in the augmented patch, the peak is its **interior anchor**: the
  instance pixel with the largest Euclidean distance transform, computed on the instance mask
  padded with background. Ties go to the pixel nearest the instance centroid, then to the first
  in C order. A cell cut by the patch border gets the anchor of its visible part.
- Around each anchor, a Gaussian with **σ = clip(ESD / 4, 4.09 px, 16.34 px)**, ESD from the
  visible area, truncated at 3σ, exactly 1.0 at the anchor. Overlapping Gaussians combine by
  elementwise max.
- Deep-supervision targets are rendered per scale as in E01 (`HeatmapDSTransform`): anchors and
  σ in full-resolution pixels, drawn on each downsampled grid.

## 5. Parameters translated from E01 (fixed from training data only)

k = median training ESD (17.33 px) / E01's median training ESD (4.243 mm) = **4.085 px per mm**.

| Parameter | E01 | E02 | Rule |
|---|---|---|---|
| B's σ clip | [1, 4] mm | [4.09, 16.34] px | [max(1, k), 4k] |
| B's NMS half-width | 3.0 mm | **4 px** | largest h at which ≤ 0.94% of training cells have another anchor within Chebyshev distance h (E01's training share at 3 mm). Gives 0.48%; size scaling (12 px) would give 52% (owner decision) |
| Hit tolerance | 2.0 mm | 8.17 px | 2k |
| C1 fallback-mask radius | 8.0 mm | 32.7 px | 8k |
| C2 crop | 96 mm | 384 × 384 px | 96k = 392, rounded to a multiple of 32, at most 512 |
| C2 jitter bounds | 4–10 mm | 16.3–40.9 px | 4k–10k |
| Size bins | < 3, 3–10, > 10 mm | < 13.63, 13.63–29.45, ≥ 29.45 px | training ESD quantiles at E01's training shares 25.5% / 61.9% / 12.6% |
| CPM FP targets | 0.5, 1, 2, 3 per scan | **14.95, 29.90, 59.81, 89.71 per image** | × (320.4 training cells per image / 10.71 E01 training lesions per scan) |
| Candidate floor | 0.05 | 0.05 | unchanged (scores are probabilities) |
| Fixed operating points | A 0.5, B 0.3, C 0.4 | unchanged | |
| Connectivity | 26 (3D) | 8 (2D) | |

"1 FP" means the second target, 29.90 FP per image.

## 6. Inference and scored points

**Inference (every arm):** nnU-Net's predictor on whole test images, tile step 0.5, Gaussian
weighting, mirroring TTA, fp32 (E01 D7). A's map is the class-1 probability; B's is the sigmoid
heatmap.

**Points:**
- **A:** the 8-connected components of p ≥ 0.05. One point per component, at its pixel of
  maximum probability (ties: first in C order). Score = that maximum.
- **B:** local maxima: a pixel equal to the maximum over a box of half-width 4 px, with value
  ≥ 0.05. A plateau gives one point, its first pixel in C order. Score = the peak value.

## 7. Matching and endpoints

**Hit rule (one-hit, D8):**
- A point hits **at most one** object: the instance whose pixel it lies on; otherwise the
  instance with the nearest pixel (Euclidean, pixel centres) within **8.17 px**. Ties go to the
  lower instance id.
- An object's score is the max score of the points that hit it, or −∞.
- A point that hits no object is a **false positive**. Extra points on an object are
  "redundant", counted and reported, and not FPs.
- **Sensitivity analysis only:** E01's original rule, where a point hits every object within
  8.17 px of it.
- **Implementation:** E02's own 2D matcher builds `jannus.evaluation.detection.MatchResult`
  objects with rule `"point"` (E01's route). `jannus` FROC, CPM and bootstrap work on records
  only and never see image arrays, so they apply to 2D unchanged. Case = image, so `jannus`'s
  "FP per scan" is FP per image. Its "patient id" is set to the well.

**Detection endpoints:**
- **CPM:** mean sensitivity at the four FP targets of §5 (`jannus` `cpm`, step FROC, no
  interpolation), all cells and per size bin.
- Sensitivity at each target, all cells and per bin.
- FROC curves, all cells and per bin.
- At the fixed operating points (A p ≥ 0.5, B peak ≥ 0.3, C s ≥ 0.4): sensitivity, FP per image
  and per-bin sensitivity.
- **Unreached targets:** if an arm's FROC ends below a target, its maximum sensitivity is used
  and **flagged**, never extrapolated.

**Segmentation endpoints:**
- **Per-image Dice** of the arm's mask at its fixed operating point (A: p ≥ 0.5; C1, C4: §7.1)
  against the gapped label, averaged over images. An image where both are empty scores 1.
- **Per-object Dice:** for GT instance L (its pixels in the gapped label), P_L is the union of
  the 8-connected components of the mask that overlap L; Dice_L = 2|P_L ∩ L| / (|P_L| + |L|),
  0 if nothing overlaps. Averaged over all cells and per bin. The 2 test cells emptied by the
  gap rule are left out. B has no mask, so its Dice is "n/a".

### 7.1 Two-stage arms (no new training except C2)

**C1 (A and B maps only):**
- **Stage 1:** B's points (§6). Call each peak q, with value b(q).
- **Support components:** the 8-connected components of A's p ≥ 0.05.
- **Assignment:** q goes to the component containing it; otherwise to the nearest component
  within 8.17 px (ties: higher max p, then lower label); otherwise it is unassigned.
- **Score:** s(q) = (b(q) + a(q)) / 2, with a(q) = max p over q's component, or 0 if unassigned.
  Every B peak stays a point at its own location: C1 re-ranks B's points.
- **Mask:** the union, over peaks with s ≥ 0.4, of each peak's region:
  - its component ∩ {p ≥ 0.5}, if that is non-empty;
  - otherwise the 8-connected component containing q of {heatmap ≥ e⁻² · b(q)}, clipped to a
    disc of radius 32.7 px around q.
  - Several peaks may share one component; their regions are unioned.

**C2 (third trained model, a crop segmenter):**
- **Crop dataset** (`Dataset922_E02C2Crops`, training images only, sampling seed 20261117):
  - **Crop:** 384 × 384 px of image and gapped label. Outside the image, the image is padded
    with the source image's median intensity and the label with 0.
  - **Positive crops:** 2 per training image (7,454), each centred on the anchor of a cell drawn
    uniformly from that image's cells, plus a random integer offset per axis, uniform in
    [−j, j], j = round(clip(ESD / 4, 16.3, 40.9) px).
  - **Negative crops:** 0.5 per positive (3,727 attempts). Images are drawn in proportion to
    their cell counts. Half the centres are uniform over the image; half are uniform over the
    image's brightest 0.5% of pixels. A centre must be ≥ 40.9 px (10k) from every cell pixel;
    each draw is retried up to 100 times, then skipped. The number skipped is reported (in dense
    images most will be).
  - A manifest of crop ids, source images, centres and offsets is written to
    `results\two_stage\c2_crop_manifest.csv`.
- **Training:** as §4 (shared settings, Dice + CE, bf16), default planner on the crop dataset,
  default nnU-Net normalisation, fold `all`, seed 1600, `checkpoint_final`.
- **Test-time stage 2:** for every B peak q, a 384 × 384 crop centred on q (padded as above) is
  predicted (as §6). On that crop, the C1 rules apply with A's map replaced by C2's map p₂:
  support components of p₂ ≥ 0.05, the same assignment, a₂(q), s₂(q) = (b(q) + a₂(q)) / 2, and
  region = the assigned component ∩ {p₂ ≥ 0.5}, else the heatmap fallback. The C2 mask is the
  union of kept peaks' regions pasted back (padding dropped). Crops are evaluated independently.

**C4:** C1's score s(q) for every peak, with C2's region (else the fallback) for kept peaks. C4's
detection therefore equals C1's by construction; only its mask differs.

C3 is not run (its detection equalled C2's by construction in E01).

**Also reported, not hypotheses:** C2 − B and C2 − A on all endpoints; C4 − C1 Dice; and A, B, C1,
C2, C4 side by side.

## 8. Uncertainty

- **Cluster bootstrap over test wells** (D9): 2,000 resamples, seed 20261116, 95% percentile
  CIs, through `jannus` `patient_case_weights` with well as the cluster. If a well cannot be
  parsed, the image is the cluster, and this is reported (none was unparsed at the build).
- **Paired:** every arm and every endpoint, Dice included, uses the same resample weights.
- With 9 wells, almost one per cell line, these CIs are crude. Condition B of §2 is there for
  this reason.
- No p-values. A CI that excludes 0 is described as such and nothing more.

## 9. Steps and checkpoints

1. **Checkpoint 1:** the owner approves this file; its SHA-256 is recorded.
2. Port E01's code to 2D (copied, not edited). All unit tests pass, including new tests for the
   gap label, one-hit matching, one B peak per isolated instance, the k-scaled parameters and
   instance ids through augmentation.
3. 2-epoch smoke runs of A and B on training images only, and a prediction pre-flight on two
   training images.
4. Train A, then B, then build the C2 crops and train C2, one at a time.
5. Predict the test set **once** with A, B and C2 (stage 2), then evaluate A, B, C1, C2 and C4 in
   one run.
6. Write `report\E02_data.md` (numbers) and `report\E02_analysis.md` (interpretation, side by
   side with E01).

**Log hygiene:** nnU-Net's pseudo-Dice lines refer to training images under fold `all`. They are
not used for any decision.

## 10. Descriptives (not endpoints)

- **By cell line** (all endpoints, point estimates; also used for §2 condition B).
- **By confluence tertile:** cells per image, with edges from training images: < 143, 143–323,
  ≥ 323 (an image on an edge goes in the upper tertile).
- **A's merged detections:** 8-connected components of A's p ≥ 0.05 that overlap ≥ 2 distinct
  GT instances by at least one pixel each, in total, per image and by confluence tertile.
- **A's saturated scores:** the number of A points with score exactly 1.0 (E01 D7).
- **Redundant points** per image, per arm.
- **The E01 multi-hit rule** as a sensitivity analysis (§7).

## 11. Figures (selection by GT only, fixed now)

- FROC curves, all cells and per bin; per-bin sensitivity at "1 FP" with CIs; the per-line
  differences for each hypothesis.
- Three example test images, chosen by GT before any prediction is viewed: the image with the
  most small cells, the image with the median cell count (ties: first by case name) and the
  image containing the largest cell. Each shows GT outlines, A's mask at p ≥ 0.5 and B's points
  at peak ≥ 0.3.

## 12. Known limitations

- **One dataset, one seed per arm** (D7: a second seed, if run, is exploratory and reported
  separately). Training variance is not estimated.
- **Only 9 test wells, one per cell line except SKOV3.** The cluster bootstrap is close to a
  bootstrap over cell lines; §2 condition B adds a per-line consistency check.
- **Narrow size range.** Cells vary 4.1× in ESD (p95/p5) against 8.5× for E01's lesions, so
  "large" means "large for a cell".
- **Size tracks cell line.** BV2 has almost no large cells; SKOV3, Huh7 and A172 are 59–71% large.
  A size effect cannot be fully separated from a cell-line effect.
- **Touching cells change the task.** The gap label and one-hit matching are departures that
  E01 did not need. Part of any A − B difference may come from separating touching cells.
- **Per-object Dice mixes detection and delineation,** and a merged A component inflates P_L for
  every cell it covers.
- **The C2 crop covers 74% of the image height,** so C2 sees nearly whole images. Its negative
  crops are random or bright-pixel crops, and most negative draws will fail the distance rule in
  dense images.
- **Annotations are not exhaustive:** some cells and debris have no annotation. Every arm's FP
  count rises equally.
- **The pixel translation (§5)** is one reasonable rule. It was fixed from training data only.
- **cudnn.benchmark is on,** as in E01, so runs are not bit-reproducible.

## Deviations

Append-only and dated.

- **D1 (2026-11-16): augmentation workers fixed at 6 (process setting, no design change).**
  No test image had been read; only training-image smoke runs existed.
  - **Cause:** in the 2-epoch smoke run of B, nnU-Net's default of 12 training plus 6
    validation augmentation workers (about 2.7 GB committed each) exhausted Windows commit memory
    (46.9 GB): `_ArrayMemoryError` in a worker, then "background workers are no longer alive".
    A's smoke run had just fitted.
  - **Check (training images only, single process):** one batch of B's pipeline takes about
    1.0 s and the same memory as A's (0.3 s).
  - **Change:** `nnUNet_n_proc_DA = 6` (6 training + 3 validation workers) for every arm, set in
    `code\e02\common.py` and `env\e02_env.cmd`. It changes only how many processes produce
    batches, not the augmentation, sampling or any training setting.

- **D2 (2026-11-18): C2 crop counts in §7.1 corrected (record-keeping; no design change).**
  Test-set outputs had been seen when this was noted.
  - **Cause:** §7.1 states 7,454 positive crops (2 per training image) and 3,727 negative
    attempts. The rule "2 positives per training image" gives fewer for training images with
    fewer than 2 cells.
  - **Record:** the build (`results/two_stage/c2_build.json`, 2026-11-17, before C2 training)
    produced 7,449 positives and 3,724 negative attempts (0.5 × positives, rounded down).
  - **Change:** none to the rule, seed, manifest or trained model; only the counts quoted in
    §7.1 are superseded by these.

## Amendments

Append-only and dated.
