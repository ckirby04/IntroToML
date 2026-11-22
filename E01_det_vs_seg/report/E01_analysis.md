# E01 analysis: lesion detection vs. segmentation on one nnU-Net backbone

This file interprets the results. Every number here comes from `E01_data.md`, which is the
record of record; section numbers in brackets (for example [§3]) point to it. The design and
all departures from it are in `PREREG.md` (Amendment 1, Deviations D1–D7). §9 summarises the
exploratory two-stage addendum (`PREREG_C_two_stage.md`); its numbers come from its own tables,
not `E01_data.md`.

## 1. Summary

- **Question.** Does training a 3D nnU-Net as a lesion *detector* (Model B: one Gaussian heatmap
  peak per metastasis, CenterNet focal loss) find brain metastases better than training it as a
  *segmenter* (Model A: stock nnU-Net Dice + CE) and reading detections off the mask?
- **Primary endpoint: no difference shown.** CPM is 0.768 for A and 0.773 for B; the paired
  difference B − A is +0.005, 95% CI [−0.036, 0.055] [§3]. Both CIs overlap almost entirely.
  Both models find about three quarters of all lesions at 1 FP/scan.
- **But the two models fail in different places.** The overall tie hides a trade-off by lesion
  size:
  - **Large lesions (> 10 mm): A is clearly better.** Sensitivity at 1 FP/scan is 0.978 for A
    and 0.728 for B, a difference of −0.250 [−0.345, −0.123] [§3]. This is the clearest result
    of the experiment.
  - **Small lesions (< 3 mm): B is better once more false positives are allowed.** At 1 FP/scan
    the two are level (0.520 vs 0.533; difference +0.013 [−0.032, 0.152]). Over the full CPM
    range, B's small-lesion CPM is higher by +0.100 [0.048, 0.181] [§3]. That bin-level CPM is
    a pre-registered "other" metric, not the key secondary, so it should be reported as
    supporting evidence rather than a headline.
  - **Mid-size lesions (3–10 mm), 63% of all lesions: no difference** (CPM difference −0.002
    [−0.040, 0.044]).
- **At higher FP rates B keeps going and A stops.** A's FROC ends at about 1.5 FP/scan, because
  A produces few candidates above the 0.05 floor (8.7 per scan vs 18.8 for B) [§6, §7]. At 3
  FP/scan B reaches 0.876, compared with A's maximum of 0.801: a difference of +0.075 [0.027,
  0.117], with A's value flagged as unreached [§3].
- **The two models are complementary.** At the fixed operating points, 74 lesions are found
  only by B (44 of them < 3 mm) and 29 only by A (14 of them > 10 mm) [§9].
- **Combining them helps (exploratory addendum, §9 of this file).** A two-stage
  "B detects, then segment" design beats B on detection without losing A's segmentation
  quality:
  - **C1** (B's peaks scored with A's map; no training): CPM 0.802, +0.029 [0.005, 0.045] over
    B. It recovers most of A's large-lesion sensitivity (0.957 vs B's 0.728 at 1 FP/scan), and
    its per-scan Dice matches A's (0.764 vs 0.769).
  - **C2** (a trained crop segmenter at each B peak) gains less on detection (+0.017) and scores
    false positives too high. It outlines small lesions better but stays level on per-scan Dice.
  - **C4** (C1's scores with C2's outlines) keeps C1's detection and adds a modest outline gain
    (per-lesion Dice +0.027 [0.016, 0.038] over A). It is the best variant, but its rule was
    chosen after seeing fold 0, so its numbers are optimistic.
  - All the two-stage arms were designed after A's and B's fold-0 results were known, so they
    are exploratory, not confirmatory.

In one sentence: **turning segmentation into point detection did not raise overall detection
performance, but it moved where the model succeeds — better on small lesions, worse on large
ones — and the two models' errors overlap only partly, so combining them (B detects, then the
segmenter scores and outlines) does better than either alone, in exploratory follow-up.**

## 2. What was compared, and how fairly

Both arms share the nnU-Net v2 `3d_fullres` plans, data, split, augmentation, optimizer, 250
epochs, seed and GPU. Both were evaluated through one inference path (sliding window, mirroring
TTA, fp32) and one matching rule: a predicted point counts as a hit if it lies within 2.0 mm of
any voxel of a ground-truth lesion [§2]. Model A is converted to points (one per connected
component at p ≥ 0.05, at its maximum), so both arms are scored as detectors on identical terms.
All thresholds, the NMS radius, the Gaussian width and the matching rule were fixed in
`PREREG.md` before training, and nothing was tuned on fold 0.

Two differences crept in beyond the intended ones (target, loss, output layer). Both came from
the fix after Model B's first run diverged to NaN (D6): B's output layer starts from small
weights, and B trained in bfloat16 while A trained in fp16. Neither is plausibly the cause of a
size-specific pattern, but they are differences between the arms and are listed as limitations
(§6).

## 3. Results by lesion size

| ESD bin | Lesions | A sens @ 1 FP/scan | B sens @ 1 FP/scan | B − A [95% CI] |
|---|---|---|---|---|
| < 3 mm | 225 | 0.520 | 0.533 | +0.013 [−0.032, 0.152] |
| 3–10 mm | 549 | 0.845 | 0.811 | −0.035 [−0.071, 0.026] |
| > 10 mm | 92 | 0.978 | 0.728 | **−0.250 [−0.345, −0.123]** |

(From [§3]; figure: `report/sensitivity_by_size_bin.png`.)

**Small lesions.** These are the clinically hard case, and both models miss about half of them
at 1 FP/scan. The models differ in *how* they miss. A produces no candidate at all (nothing at p
≥ 0.05) for 97 of the 225 small lesions; B misses only 33 that way [§9]. B instead tends to put
a weak peak on a small lesion (median score 0.47), so those lesions are recovered as the
threshold drops. That is why B's small-lesion FROC climbs past 1 FP/scan to 0.773 at 3 FP/scan,
while A's flattens at 0.569 [§6]. At the fixed operating points the gap is larger: B finds 0.711
of small lesions at 1.86 FP/scan, and A finds 0.524 at 1.04 FP/scan [§4].

**Large lesions.** A essentially never misses a large lesion: 2 of 92 have no A component, and
the median A score is 1.0. B gives large lesions *lower* confidence than mid-size ones once they
pass about 17 mm: the median B score is 0.833 for ESD 10–17 mm and 0.613 for 17–40 mm [§9]. B
also puts 1.71 points on each large lesion it finds, against about 1.0 for smaller lesions [§9].
Its peaks on big lesions are therefore split and weaker, so at a strict threshold many of them
fall below the line. In `report/example_slices.png`, B has no peak at or above 0.3 on either of
the two large lesions shown, while A's mask traces both.

**Mid-size lesions.** These make up most of the data, and the two models are statistically
indistinguishable on them. The overall CPM tie largely reflects this.

## 4. Operating points and the shape of the FROC curves

- **A's curve is short and starts late.** A's curve starts at 0.2 FP/scan, not 0. 458 of A's 843
  candidates have a probability of exactly 1.0 (saturated), 19 of them false positives. The
  ranking can't separate them, so the first step of the curve already includes those 19 FPs
  [§5, §6, D7]. A's curve then ends at about 1.5 FP/scan, its last threshold being the 0.05
  floor [§5]. Its sensitivity at 2 and 3 FP/scan is its maximum (0.801), flagged as unreached.
- **This affects the primary endpoint.** CPM averages sensitivity at 0.5, 1, 2 and 3 FP/scan.
  For A, half of that average is the capped value 0.801. So A's CPM partly measures how many
  candidates A emits above 0.05, not only how well it ranks them. The floor was the same for
  both arms and was fixed in advance, so this is a legitimate result of the pre-registered
  design. It also means the CPM "tie" should not be read as "equally good everywhere on the
  curve".
- **At strict operating points (≤ 1 FP/scan), A is slightly ahead overall** (0.775 vs 0.730 at
  1 FP/scan; difference −0.045 [−0.077, 0.026]). The CI includes zero. A's edge there comes
  mostly from large lesions, with a smaller edge on mid-size ones (0.845 vs 0.811) whose
  CI also includes zero.
- **The fixed operating points are not equivalent.** A at p ≥ 0.5 sits at 1.04 FP/scan, and B
  at peak ≥ 0.3 at 1.86 FP/scan [§4]. B's higher fixed-op sensitivity (0.829 vs 0.777) is bought
  with almost twice the false positives, so the two should not be compared head to head. The
  FROC comparison at matched FP rates is the fair one.
- **B's false positives are concentrated.** Over half of B's fixed-op FPs (53%) come from 5
  scans, and 2 scans alone give 64 of 180 [§9]. One of those scans has 116 annotated lesions.
  Some "false positives" in such dense scans may be real but unannotated lesions; this was not
  checked, and annotations are taken as ground truth.

## 5. Why might B behave this way? (hypotheses, not tested)

These explanations fit the data but were not tested in E01:

1. **Per-lesion vs per-voxel weighting.** B's focal loss is normalised by the number of lesion
   peaks, so every lesion counts equally regardless of size. A's Dice + CE is dominated by voxels,
   so a 2 mm lesion barely moves the loss next to a 30 mm one. This fits B's better small-lesion
   recall and A's near-perfect large-lesion recall.
2. **A capped Gaussian on a large lesion is an ill-posed target.** B's target width is σ =
   clip(ESD/4, 1, 4) mm, so every lesion above 16 mm ESD gets the same 4 mm Gaussian at one
   interior voxel (the maximum-distance-from-edge point). On a large, often ring-enhancing or
   necrotic lesion, many interior voxels look alike. The network may then spread its confidence
   over several of them, giving split, weaker peaks. That is what the data show: 1.71 points
   per large lesion, and lower scores above 17 mm. A wider σ for large lesions, or a
   size-aware target, would test this.
3. **Saturated segmentation probabilities.** A's softmax pushes most true detections to exactly
   1.0, which makes it an excellent detector of obvious lesions but a poor *ranker*: it has no
   graded confidence to trade sensitivity for false positives. B's heatmap scores are spread
   out (medians 0.47–0.77 by bin), which is why its FROC extends smoothly to 4 FP/scan.

## 6. Limitations and threats to validity

- **One split, one seed per arm.** The CIs cover patient sampling only, not training
  variability. A second seed could move a model's numbers by an amount comparable to the
  differences found for mid-size lesions. The large-lesion gap (25 points) is the result most
  likely to survive reseeding, given its size, but seed variance was not measured here.
- **Arms differ in more than the intended components** (D6). B uses small-weight output
  initialisation and bfloat16 training; A uses nnU-Net defaults and fp16. Both changes were
  needed to make B train at all and were made from training-fold diagnostics only, but they
  are confounds.
- **Fold 0 is not fully blind** (PREREG §10). The owner had seen 1.60 pilot results that
  included fold 0, and nnU-Net logged fold-0 validation numbers during training. The agent saw
  one fold-0 pseudo-Dice line for A (D1, D5). None of this was used to choose anything; all
  choices were fixed in `PREREG.md` or made from training folds.
- **Late rise in B's validation loss.** B's fold-0 validation loss bottomed out at epoch 226
  (0.84) and ended at 1.39 [§8], while its training loss kept falling. This hints at some
  overfitting in the last epochs. The pre-registered rule (use `checkpoint_final`, never select
  on fold 0) was kept, so B is evaluated as trained; an earlier checkpoint might have scored
  differently.
- **The candidate floor shapes the primary endpoint** (§4). CPM depends on A's capped values at
  2 and 3 FP/scan.
- **Multiple comparisons.** Many per-bin and per-FP metrics are reported with no correction.
  Only CPM is primary, and only sensitivity at 1 FP/scan per bin is the key secondary. The
  large-lesion difference is robust (it holds for the key secondary, for CPM by bin and at the
  fixed operating points). The small-lesion advantage of B is clear in CPM but not at 1 FP/scan,
  and should be stated with that caveat.
- **Ground truth.** Annotations are treated as complete. 17 fold-0 lesions have a centroid more
  than 2 mm outside the lesion; B's targets use an interior anchor to avoid that, and the
  matching rule works on lesion voxels, so neither arm is penalised by it [§1]. Dense scans may
  contain unannotated lesions (§4).
- **Small, unbalanced subgroups.** Only 16 of 97 scans are from Stanford BrainMetShare; the rest
  are UCSF [§1, §9]. Both models are less sensitive on BMS at the fixed operating points, but the
  BMS sample is too small to analyse separately.
- **Grid and connectivity.** Evaluation is on the 1 mm resampled grid with 26-connected
  components. Lesion counts per bin (225 / 549 / 92) differ from the native-grid counts in the
  split tables (175 / 597 / 93).
- **Research only.** The data licences prohibit commercial use; these models and numbers are
  for this class project and do not support any clinical claim.

## 7. Conclusions

1. On the pre-registered primary endpoint, heatmap detection (B) and segmentation-derived
   detection (A) perform the same (CPM 0.773 vs 0.768).
2. The choice of training objective changes *which* lesions are found. Segmentation is clearly
   better for large metastases. Heatmap detection recovers more small metastases once roughly
   1.5 or more false positives per scan are acceptable, and its FROC extends to higher
   sensitivity overall (0.876 at 3 FP/scan).
3. Segmentation probabilities saturate, so Model A is a weak ranker: its FROC is short and its
   operating point can barely be moved.
4. The errors overlap only partly (74 lesions found only by B, 29 only by A), so a combination
   could beat either model alone.

## 8. Suggested follow-up (not done in E01)

- **Hybrid detector:** take A's components for lesions it segments confidently, and add B's
  peaks elsewhere. The complementarity counts above suggest this would beat both, but it would
  need its own pre-registered evaluation on data not used here.
- **Size-aware heatmap targets:** let σ grow with lesion size beyond 4 mm, or add a size
  regression head as in CenterNet, to test hypothesis 2.
- **Better-calibrated segmentation scores:** use mean probability over a component, or
  temperature scaling learned on training folds, to give A a usable ranking.
- **More seeds**, to measure training variance next to patient-sampling variance.
- **Review of dense-scan "false positives"** by a reader, to estimate how many are unannotated
  lesions.

## 9. Addendum C: two-stage "detect, then segment" (exploratory)

Added after the A/B results above were known, under `PREREG_C_two_stage.md`: C1 in the original
addendum, C2 in Amendment C-1 (with deviation DC1), C3 in Amendment C-2 and C4 in Amendment C-3.
All variants are evaluated on the same fold 0, with the same matching rule and the same 2,000
paired patient-bootstrap draws for A, B and C1–C4. Because the arms were designed with A's and
B's fold-0 results in view, these are **exploratory** findings, not confirmatory ones, and they
are not the independent test that §8's "hybrid detector" called for. **C3 and C4 are further
post hoc:** their rules were written after C1's and C2's fold-0 results were seen, so their
numbers are optimistic.

Numbers come from `report/two_stage_C{1,2,3,4}_table.md` and
`results/two_stage/summary_C{1,2,3,4}.json`; the comparison chart is
`report/two_stage_C1_to_C4.png`. The coverage/quality split of per-lesion Dice is a post-hoc
analysis (`results/two_stage/lesion_dice_decomp_C2.json`).

- **C1** (no training): B's heatmap peaks are the candidates. Each is scored as (B peak + max p
  of the A component it falls in) / 2, and A's mask (p ≥ 0.5) supplies the segmentation.
- **C2**: the same B peaks, with stage 2 a stock nnU-Net trained on 96³ crops from folds 1–4
  (lesion-centred and negative crops). It segments a crop around each peak.
- **C3** (no training): a size gate. If a peak's A component (p ≥ 0.5) has ESD > 10 mm, the peak
  takes C1's score and A's outline; otherwise it takes C2's score and crop outline.
- **C4** (no training): C1's score for every peak; each kept peak gets C2's crop outline.

**Broad-strokes findings**

1. **A and B tie overall but split by size** (§1, §3). Their CPM is 0.768 vs 0.773. B finds
   more sub-3 mm metastases (CPM 0.614 vs 0.514), and A is far better on lesions above 10 mm
   (0.973 vs 0.785).
2. **Combining them works best for detection.** The training-free cascade C1 gives the best
   detection: CPM 0.802, +0.029 [0.005, 0.045] over B. It recovers most of A's large-lesion
   sensitivity (0.957 at 1 FP/scan vs B's 0.728). At its fixed point it finds 0.782 of lesions
   at 1.08 FP/scan. It costs essentially no segmentation quality against A (per-scan Dice
   −0.005 [−0.018, 0.000]).
3. **The trained crop segmenter C2 did not beat the free cascade on detection.** Its CPM gain
   over B is smaller: +0.017 [0.003, 0.028]. It falls back on large lesions (0.739 at
   1 FP/scan). At the threshold fixed in advance (s ≥ 0.4) it runs at 3.0 FP/scan. Its per-scan
   Dice ties A's (0.770 vs 0.769; +0.001 [−0.016, 0.016]).
4. **Where C2 helps is small lesions.** Its per-lesion Dice is higher than A's: +0.079 overall
   and +0.151 for lesions under 3 mm. A missed lesion scores Dice 0, though, and the post-hoc
   split shows the gain is mostly coverage. C2's mask touches 0.889 of lesions against A's 0.786
   (+0.103 [0.066, 0.133]); for lesions under 3 mm, the figures are 0.756 vs 0.529. On lesions
   both models touch, C2's outlines are only modestly better: Dice +0.031 [0.020, 0.042], and
   +0.056 under 3 mm. Nothing changes for lesions above 10 mm.
5. **Routing by lesion size (C3) changes nothing.** Its results are effectively identical to
   C2's: the same detection numbers, and Dice within 0.001. On all 165 peaks routed to A, C1 and
   C2 gave identical scores, because both models are fully confident on large lesions. C2's
   large-lesion deficit therefore does not come from how it scores large lesions. It comes from
   its false positives scoring higher: 18.5% of C2's false positives reach s ≥ 0.6, against 7.8%
   for C1. That raises the cut-off at 1 FP/scan, which then drops real large lesions with
   middling scores (median 0.68).
6. **C1 scores with C2 outlines (C4) separates C2's two gains.** C4's detection equals C1's by
   construction: CPM 0.802, +0.029 [0.005, 0.045] over B, at 1.08 FP/scan. It keeps C2's better
   outlining but not its extra coverage. Per-lesion Dice is 0.577, between C1's 0.552 and C2's
   0.629, at +0.027 [0.016, 0.038] over A (+0.032 under 3 mm). C4 keeps 855 peaks against C2's
   1,136. The extra peaks C2 keeps carry both its extra small-lesion finds and its extra false
   alarms (3.0 vs 1.08 FP/scan), so a fixed rule cannot keep one without the other. Per-scan
   Dice is about 0.77 for every variant (C4 − A −0.001 [−0.015, 0.010]).

**Bottom line.** If one system had to be chosen, C4 is the best all-round variant: C1's detection
with slightly better outlines than A, and no training beyond C2's stage-2 model. Without C2's
model, the cascade C1 is nearly as good and needs no extra training at all. B, and C2 on top of
it, are worth it only where catching tiny metastases matters more than false positives. C3's and
C4's advantages were found by looking at fold 0, so they need confirming on new data, or with
out-of-fold predictions on folds 1–4. That is also what a learned meta-learner would need, and
none was trained for that reason. All of this rests on one split and one seed per model. The
data licences prohibit commercial use, and none of it supports a clinical claim.
