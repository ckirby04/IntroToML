# E02 plan (DRAFT v2): does E01's detect-vs-segment result transfer to cell microscopy?

**Status:** historical. The experiment is CLOSED (2026-11-18); see `README.md` and `report\`.
This file stays as written below: DRAFT v2 for the owner, 2026-11-16.
- v1 used iSAID aerial images. The owner chose cell microscopy (LIVECell) instead and agreed
  to the rest of the v1 setup: bf16 for both arms, C2/C4 included, one class.
- On 2026-11-16 the owner also accepted the touching-cell rules (D8) and the well-level
  bootstrap (D9). At the dataset build the same day, the owner made four further decisions:
  duplicate handling (D10), the replication rule (D11), the overlap and gap rules (§3.1) and the
  NMS rule (§5). The build and the training statistics are done (§11, §12).
- LIVECell is downloaded and the dataset is built; nothing is trained yet.
- **Superseded for the design by `PREREG.md`** (draft written 2026-11-16 for Checkpoint 1). Where
  the two differ, PREREG governs. Its SHA-256 goes into `results\prereg_sha256.txt` at approval,
  before any E02 training.

## 1. Why

E01 (`..\E01_det_vs_seg`) found three things on brain-metastasis MRI:
1. **No overall difference** between a segmenter (A) and a heatmap detector (B) on one nnU-Net
   backbone (CPM 0.768 vs 0.773).
2. **A size trade-off:** A wins on large objects (> 10 mm: 0.978 vs 0.728 sensitivity at
   1 FP/scan). B wins on small ones once more FPs are allowed (< 3 mm CPM +0.100).
3. **Combining them helps:** C1 (B's peaks, scored by averaging with A's probability) beats B on
   CPM (+0.029). C4 (C1's scores, a crop model's outlines) adds per-lesion Dice. **But C1–C4
   were designed after fold-0 results were seen, so they are exploratory.**

E02 asks whether these results hold in a **different domain**. Every rule, including C1 and C4,
is fixed **before** any E02 model is trained, which makes E02 the first confirmatory test of
the two-stage design.

## 2. Hypotheses (directional, from E01)

| ID | Hypothesis | Comparison | Counts as replicated if |
|---|---|---|---|
| H1 | Overall detection is level | CPM, B − A | 95% CI includes 0 *and* lies within ±0.05 |
| H2 | Segmenter wins on large objects | sens @ "1 FP", large bin, A − B | CI > 0 |
| H3 | Detector wins on small objects | CPM, small bin, B − A | CI > 0 |
| H4 | **Primary.** Two-stage beats the detector | CPM, all objects, C1 − B | CI > 0 |
| H5 | Two-stage keeps the segmenter's quality | per-image Dice, C1 − A | CI lower bound > −0.02 |
| H6 | Crop outlines help | per-object Dice, C4 − A | CI > 0 |

**Extra replication condition (D11):** the test set has only 9 wells, which makes a
well-clustered CI unreliable. A hypothesis counts as replicated only if the condition above
holds **and** the point estimate points the same way in at least 7 of the 8 cell lines. For H1,
the per-line difference must lie within ±0.05 in at least 7 of 8 lines.

- H4 is primary, because it is the claim E01 could not make confirmatorily.
- H1–H3 test whether E01's base result transfers. H5–H6 are secondary.
- No p-values; reporting language as in E01.

## 3. Dataset: LIVECell

**What it is** ([site](https://sartorius-research.github.io/LIVECell), Edlund et al., *Nature
Methods* 2021):
- 5,239 phase-contrast images with expert-checked instance annotations of 8 cell lines.
- About 1.6 million cells.
- Annotations in COCO format, with the official train / val / test splits used in the paper.
- Published under CC BY-NC 4.0. The LICENSE file in the download is checked.

**Why it fits better than iSAID:**
- The images are **single-channel grayscale**, like E01's single-channel MRI, so "did not
  transfer" has fewer possible causes.
- All images have the same size, so FP per image works like E01's FP per scan.
- Every image is labelled, and the test labels are public.

**Split:**
- **Train:** official train + val. E01 had no validation set either.
- **Test:** the official LIVECell-wide test set.
- **No tuning:** nothing is decided on test-set outputs.

**Storage:** the owner downloads the data (official S3 bucket, AWS CLI) into
`E02_det_vs_seg_transfer\data\raw\`. It stays on this machine.

**Single class:** all cells, from all 8 lines, form one "cell" class, like E01's single lesion
class. Cell line is kept as metadata for descriptive breakdowns.

### 3.1 The new problem: touching cells

Metastases rarely touch. Cells often do, and in confluent images most do. Under E01's rules
that breaks three things:
1. **A's points.** A gives one point per connected component, so a cluster of touching cells
   becomes one point, and every other cell in the cluster is missed.
2. **GT objects.** E01 defined GT lesions as connected components of the label. Here they must
   be the annotated **instances**.
3. **Matching.** E01 let one point hit several lesions. Among packed cells, one point near a
   border would count for two cells.

**Fixed rules for E02:**
- **A's label (and C2's):** the binary union of instance masks, with a **1-px background gap**
  wherever two instances touch or overlap (standard practice in cell segmentation). A can then
  separate touching cells, and its components stay meaningful. The rule applies identically to
  every image and is computed from GT only. Where two cells touch, the **larger** one gives up
  its border pixel. This was changed on 2026-11-16 from "the higher id gives way", which emptied
  5,203 small training cells.
- **B's target:** built from the **instance map**, one Gaussian per instance at its interior
  anchor (max-EDT pixel of the instance), not from connected components.
- **GT for detection:** the annotated instances. Overlapping pixels go to the **smaller**
  instance. This was changed on 2026-11-16 at the dataset build, before any model existed,
  replacing "to neither". In a 240-image training sample, "neither" erased 269 cells (small
  round cells lying on flat ones) and split up to 27% of cells (SKOV3); "smaller wins" erased 71
  and roughly halved the splitting.
- **GT for Dice:** the gapped binary label above, for every arm.
- **Hit rule (change from E01):** a point hits **at most one** object: the instance that contains
  it, otherwise the nearest instance within the hit tolerance. As a sensitivity analysis only,
  E01's original rule (a point may hit several objects) is also reported.
- **Descriptive:** E01's "merged detections" count (A components overlapping ≥ 2 GT objects),
  overall and by confluence tertile (cells per image).

## 4. Models

These are the E01 arms, kept as close as possible.

**Shared settings:**
- nnU-Net v2, from the same venv as E01, used read-only.
- `2d` configuration from the default planner. Input: 1 grayscale channel.
- 250 epochs, SGD with poly LR, default augmentation, deep supervision.
- One seed, `checkpoint_final` only.
- RTX 3070 Ti, one training at a time.
- **bf16 for A and B**, which removes E01's precision confound.

| Arm | What | Training |
|---|---|---|
| A | stock nnU-Net segmenter, Dice + CE, gapped label (§3.1) | yes |
| B | heatmap detector, CenterNet focal loss, E01's trainer ported to 2D; output init N(0, 0.01), bias −4.6 | yes |
| C1 | B's peaks; s = (b + a) / 2, with a = max of A's probability over the assigned component; A's mask | no |
| C2 | crop segmenter at B's peaks (E01 Amendment C-1 rules, 2D crops) | yes (3rd model) |
| C4 | C1's scores, C2's outlines (E01 Amendment C-3) | no |

C3 is dropped: in E01 its detection was identical to C2 by construction.

## 5. Translating E01's millimetre rules to pixels (fixed from training data only)

E02 maps E01's geometric parameters to pixels with **one scale factor k**, computed on the
**training set only** before any model is trained:

> k = median ESD of E02 training cells (px) / median ESD of E01 training lesions (mm)

| E01 parameter | E01 value | E02 value |
|---|---|---|
| B's Gaussian σ | clip(ESD / 4, 1 mm, 4 mm) | clip(ESD / 4, max(1, k) px, 4k px) |
| B's NMS half-width | 3.0 mm | **changed 2026-11-16:** the largest h at which the share of cells with a neighbour's anchor inside the NMS box is ≤ E01's (0.94%). Size scaling (round(3k) = 12 px) would put a neighbour inside the box for 52% of cells |
| Hit tolerance | 2.0 mm | 2.0 k px |
| C1 fallback-mask radius | 8.0 mm | 8.0 k px |
| C2 crop | 96 mm cube | (96 k px)², rounded to a multiple of 32, at most the image size |
| Size bins | < 3, 3–10, > 10 mm | training-set ESD quantiles that match E01's **training** bin shares (25.5% / 61.9% / 12.6%) |
| Candidate floor, fixed operating points | A 0.05 / 0.5, B 0.05 / 0.3, C 0.4 | unchanged (scores are probabilities) |

- **ESD in 2D** is the equivalent-circle diameter, √(4A/π), with A the area in px.
- Connectivity is 8 in 2D (26 in E01's 3D).

**Check at dataset build (before PREREG):** NMS half-width and hit tolerance must be smaller than
the median distance between neighbouring cell centres in training images. Otherwise NMS merges
neighbouring cells' peaks. If they are not, the scaling rule cannot be applied as written, and
the owner decides the rule before PREREG is frozen (no test data involved).

**FP rate.** The unit is FP per image (all images are the same size). The CPM targets are set at
the same fraction of object density as E01's:
- E01's training folds have 10.71 lesions per scan (3,825 / 357).
- In E02 the targets become (0.5, 1, 2, 3) × (mean training cells per image / 10.71) FP per image.
- v2 of this plan used E01's fold-0 values (8.9 lesions per scan, 26 / 63 / 11% bin shares).
  These were replaced with E01's training values on 2026-11-16, before any E02 statistic was
  computed, so that no test-set number feeds an E02 rule.
- "1 FP" in H2 means the second of these targets.

## 6. Evaluation (as E01, unless stated)

- **Matching:** as §3.1. Extra hits on a detected object count as "redundant", not as FPs.
  Object score = max score of the points that hit it.
- **Metrics:**
  - CPM over the four FP targets;
  - sensitivity per size bin at "1 FP";
  - FROC curves;
  - sensitivity and FP rate at the fixed operating points;
  - per-image Dice at the fixed operating points;
  - per-object Dice by bin (E01 addendum §3).
- **Unreached FP targets** are flagged, never extrapolated.
- **Uncertainty:**
  - 2,000 paired bootstrap resamples, with the same draws for every arm and a fixed seed.
  - The cluster unit is the **well** (cell line + well, parsed from LIVECell file names), since
    images from one well are correlated, like scans from one patient.
  - If wells cannot be parsed, the cluster is the image, and this is reported.
- **Descriptive breakdowns:** by cell line and by confluence tertile. These are not endpoints.
- **Library:** E01's `jannus.evaluation.detection` (FROC, CPM, bootstrap), with E02's own 2D
  matching (§3.1) passed in as `MatchResult`, which is how E01 passes its point rule.
  **To check before PREREG:** how `jannus` handles 2D arrays and per-image FP rates.

## 7. Code

E02 gets a new folder, `experiments\E02_det_vs_seg_transfer\`, with E01's layout (`code\e02`,
`env`, `data`, `results`, `report`, `splits`). E01's code is **copied, not edited**.

| E01 module | E02 change |
|---|---|
| `setup_dataset` | new: LIVECell COCO → nnU-Net `Dataset9xx`; instance map, gapped binary label, well/cell-line metadata; no tiling |
| `heatmap` | 2D; targets from the instance map, not components |
| `decode`, `evaluate` | 2D, 8-connectivity, px parameters (§5), one-hit matching (§3.1) |
| `trainer_heatmap`, `trainers`, `loss` | 2D output convs; bf16 for A and B; B's dataloader carries the instance map |
| `predict` | 2D grayscale TIFF I/O |
| `two_stage`, `c2`, `c2_predict` | 2D crops; C1/C2/C4 only |
| `pause`, `jobqueue`, `control`, `env\*.cmd` | reuse as is |
| `tests` | port all; add tests for touching cells (gap label, one-hit matching, one peak per instance) and for k-scaling |

**The one non-trivial port: B's target under augmentation.** In E01, B's target was rebuilt from
the augmented binary mask. In E02 it must be rebuilt from the augmented **instance map**, so the
instance map has to go through nnU-Net's augmentation as a second label channel, with
nearest-neighbour interpolation. This needs a unit test showing that the instance IDs survive
augmentation.

## 8. Steps and checkpoints

1. The owner settles §10 and downloads LIVECell.
2. Build the dataset. Compute k, size bins, FP targets and the neighbour-distance check, from
   **training data only**, and write them into `PREREG.md`. **Checkpoint 1:** the owner approves
   PREREG, and its SHA-256 is recorded.
3. Port the code; unit tests pass; 2-epoch smoke runs of A and B on training images only.
4. Train A, then B, queued and never together. Then build the C2 crops and train C2.
5. Predict on the test set once. Evaluate A, B, C1, C2 and C4 in one run.
6. Write `report\E02_data.md` (the numbers) and `report\E02_analysis.md` (the interpretation,
   side by side with E01).

**Compute (rough; the smoke runs confirm it):**
- 2D training on 704 × 520 images is far cheaper than E01's 3D runs. Expect hours per arm, not
  days.
- The dataset is a few GB.

## 9. Known limitations (from the start)

- **One dataset and one seed per arm**, so training variance is not estimated (as in E01).
- **Size range.** Cells vary less in size than metastases do, so the size bins are defined by
  quantiles, and "large" means "large for a cell". If H2/H3 fail, part of the reason may be the
  narrower range rather than the domain. The training-set ESD spread is reported next to E01's.
- **Touching cells change the task.** The gap label and the one-hit matching rule (§3.1) are
  departures from E01 that E01 did not need. Part of any A/B difference may come from how well
  each arm separates touching cells. The merged-detection and confluence breakdowns are there
  to show this.
- **The pixel translation (§5)** is one reasonable rule, not the only one. Fixing it from
  training data at least prevents tuning it on test results.
- **A's saturated probabilities** (E01 D7: many scores of exactly 1.0) will likely recur. They are
  kept as they are, for comparability, and reported.

## 10. Open decisions for the owner

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Dataset | **LIVECell** (decided 2026-11-16); iSAID possible later as a replication |
| D2 | Classes | all cells, one class (agreed) |
| D3 | bf16 for both A and B | yes (agreed) |
| D4 | FP unit and CPM targets | FP/image, scaled by density (§5) (agreed) |
| D5 | Train on official train + val | yes (no validation set, as in E01) |
| D6 | Include C2/C4 (third training run) | yes, as secondary (agreed) |
| D7 | Second seed for A and B if compute allows | optional |
| D8 | Touching-cell rules (§3.1: gap label, instance GT, one-hit matching) | **accepted** (2026-11-16) |
| D9 | Bootstrap cluster = well | **accepted**, with image as the fallback (2026-11-16) |

| D10 | Doubly annotated images (148 file names) | **decided 2026-11-16:** training keeps the annotation set with the higher image id; the 52 doubly annotated test images are left out |
| D11 | Replication rule, given only 9 test wells | **decided 2026-11-16:** keep the well-cluster bootstrap, and also require the effect to point the same way in ≥ 7 of 8 cell lines |

Still open: D5 and D7.

## 12. Numbers fixed from training data (2026-11-16, for PREREG)

Source: `results\train_stats.json` (`e02.train_stats`, Tr only) and `results\dataset_verify.json`.

**Dataset (`Dataset921_E02LiveCell`):**

| | Train (official train + val) | Test (official test) |
|---|---|---|
| Images | 3,727 | 1,460 (52 doubly annotated left out) |
| Wells / fields of view | 22 / 44 | 9 / 18 (none shared with train) |
| Cells (instances) | 1,194,095 | 444,600 |
| Polygons with no pixel after rasterisation | 87 | 34 |
| Cells emptied by the gap rule | 0 | 2 |
| Cells split into > 1 piece by gap/overlap | 47,168 (4.0%) | 21,935 (4.9%) |
| Label components spanning > 1 cell | 0 | 0 |

**Size, density and spacing (training):**
- Median cell ESD is 17.3 px (5–95%: 9.8–39.8 px).
- The size spread (p95 / p5) is **4.1×, against 8.5× for E01's lesions**, so the size range is
  much narrower (§9).
- There are 320.4 cells per image, against 10.7 lesions per scan in E01.
- The median distance between neighbouring anchors is 13.9 px (5th percentile 8.2 px).

**Parameters:**

| Parameter | Value |
|---|---|
| k | 4.085 px per E01 mm |
| B's σ clip | [4.09, 16.34] px |
| NMS half-width | **4 px** (collision share 0.48%; 5 px would give 1.26%, above E01's 0.94%) |
| Hit tolerance | 8.17 px (4.99% of cells have a neighbour this close; the one-hit rule handles it) |
| C1 fallback-mask radius | 32.7 px |
| C2 crop | 384 × 384 px (96k = 392, rounded to a multiple of 32) |
| Size bin edges (ESD) | < 13.63 px, 13.63–29.45 px, ≥ 29.45 px |
| Training bin counts | 303,810 / 738,899 / 151,386 (25.4 / 61.9 / 12.7%) |
| Test bin counts (GT only) | 107,660 / 264,573 / 72,367 |
| CPM FP targets per image | 14.95, 29.90, 59.81, 89.71 ("1 FP" = 29.90) |

**The size bins largely track cell line:**
- BV2 is 64% small and has no large cells.
- SKOV3, Huh7 and A172 are 59–71% large.

So a size effect and a cell-line effect cannot be fully separated. The per-line breakdown (D11)
shows how much of any size effect is within-line.

**Notes for the owner:**
- The NMS rule picks 4 px on the full training set. The 400-image sample used for the question
  suggested 5 px, but the rule as decided ("match E01's 0.94%") gives 4.
- The C2 crop is 384 px, about 74% of the image height, so C2 sees nearly whole images. This is
  where the size scaling leads; it is reported, not changed.

## 11. Facts found at download and dataset build (2026-11-16)

- **Download:** 4 files, sizes as listed by the bucket. The zip CRC check passed. SHA-256 hashes
  are in `data\raw_sha256.txt`. The annotation files carry the CC BY-NC 4.0 licence.
- **Doubly annotated images:** 148 file names have two different annotation sets: 65 within
  train, 1 within val, 30 shared between train and val, and 52 within test. All are Huh7 or
  SKOV3. D10 applies.
- **Wells:** train+val has 22 wells and test has 9, with no well in both, so there is no
  well-level leakage. The test set has 18 fields of view and about 150–200 images per well
  (time-lapse × 4 crops). This is why D11 was added.
- **Annotations are not exhaustive:** a visual check shows some cells and debris without an
  annotation. This is label noise that both arms see equally, and it raises the FP count of
  every arm.
