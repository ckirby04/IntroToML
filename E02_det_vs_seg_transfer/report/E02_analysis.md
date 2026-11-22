# E02 analysis: does E01's detect-vs-segment result transfer to cell microscopy?

This file interprets the results. Every E02 number comes from `E02_data.md`, the record of
record; bracketed section numbers (for example [§3]) point to it. E01 numbers come from
`..\E01_det_vs_seg\report\E01_data.md` and `E01_analysis.md`. The design is `PREREG.md`
(approved 2026-11-16, Deviation D1); the test set was predicted once, on 2026-11-18, and
nothing was tuned on it.

## 1. Summary

- **Question.** E01 found, on brain-metastasis MRI, that a heatmap detector (B) and a segmenter
  read out as points (A) tie overall but split by object size, and that a two-stage cascade
  (B's peaks scored and outlined with A's map, C1) beats both. E02 re-ran the same design, with
  every rule fixed in advance, on LIVECell phase-contrast cell microscopy: 1,460 test images,
  444,600 cells, 8 cell lines, 9 wells [§1].
- **The detector wins by a wide margin, everywhere.** CPM is 0.943 for B and 0.093 for A; B − A
  is +0.849 [0.801, 0.890], and B is ahead in all 8 cell lines and all three size bins [§3, §4].
  E01's tie (H1) and E01's large-object advantage for the segmenter (H2) are **not replicated**;
  both go the other way. B's advantage on small objects (H3) **is replicated**, but only as part
  of B being better on everything.
- **The primary hypothesis is not replicated.** The cascade C1 adds +0.003 [0.000, 0.004] CPM
  over B (H4), in 5 of 8 lines [§3]. §4 shows that, given where B's curve ends, the most C1
  could have added was about +0.007.
- **The cascade keeps the segmenter's mask quality** (H5 replicated: per-image Dice C1 − A
  +7.2e-06), **but crop outlines do not help** (H6 not replicated: per-object Dice C4 − A
  −0.015 [−0.026, −0.006], negative in all 8 lines) [§3].
- **Why A fails as a detector:** not because it segments badly. Its per-image Dice is 0.906
  [§5]. It fails because the pre-registered read-out, one point per connected component of
  p ≥ 0.05, cannot count cells that touch. A produces 32 components per image against 304
  cells per image, and 48% of its components cover two or more cells [§9]. Its FROC ends at
  3.4 FP/image with a sensitivity of 0.093 [§4.2].

In one sentence: **E01's finding that a segmenter reads out as a competitive detector does not
transfer to dense cell images, because that read-out breaks when objects touch; the heatmap
detector transfers well, and the two-stage cascade, E02's primary hypothesis, has almost no
room left to improve on it.**

## 2. Hypotheses at a glance, side by side with E01

| | Hypothesis (direction from E01) | E01 (MRI, fold 0) | E02 (LIVECell, test) | E02 outcome |
|---|---|---|---|---|
| H1 | Overall detection level, B − A CPM | +0.005 [−0.036, 0.055] | +0.849 [0.801, 0.890] | not replicated: **B better** |
| H2 | Segmenter wins on large, A − B sens @ 1 FP | +0.250 [0.123, 0.345] | −0.894 [−0.940, −0.790] | not replicated: reversed |
| H3 | Detector wins on small, B − A CPM small | +0.100 [0.048, 0.181] | +0.776 [0.614, 0.812] | **replicated** (8 of 8 lines) |
| H4 | **Primary.** Cascade beats detector, C1 − B CPM | +0.029 [0.005, 0.045]* | +0.003 [0.000, 0.004] | not replicated (5 of 8 lines) |
| H5 | Cascade keeps segmenter's Dice, C1 − A | −0.005 [−0.018, 0.000]* | +7.2e-06 [−5.8e-06, 2.2e-05] | **replicated** (8 of 8) |
| H6 | Crop outlines help, C4 − A object Dice | +0.027 [0.016, 0.038]* | −0.015 [−0.026, −0.006] | not replicated (0 of 8) |

\* E01's two-stage numbers are exploratory (designed after fold-0 results were known); E02 is
their first confirmatory test. E02 values are from [§3]; the per-line values are in [§3.1] and
`report/per_line_differences.png`.

## 3. Why the segmenter fails as a detector here

A's mask is good: per-image Dice 0.906 [0.884, 0.928] against the gapped label [§5], and
`report/example_images.png` shows it covering nearly every cell. What fails is turning that
mask into points.

- **Touching cells merge into one component.** The rule fixed in PREREG §6, which is E01's,
  takes one point per 8-connected component of p ≥ 0.05. A gives 47,138 components over 1,460
  images, 32.3 per image, against 304.5 GT cells per image. 22,774 of those components (48%)
  overlap two or more cells [§9]. In E01, lesions almost never touched: 5 merged components
  out of 843.
- **Under one-hit matching, a merged component can find at most one cell.** So A's sensitivity
  is capped near (components ÷ cells), whatever its scores. The FROC confirms it: A's curve ends
  at 3.44 FP/image with sensitivity 0.093, never reaches the first CPM target (14.95), and its
  CPM equals that maximum [§4, §4.2]. 102,466 of 107,660 small cells and 64,840 of 72,367 large
  ones have no A point at all [§12].
- **E01's own, looser rule does not rescue it.** Under multi-hit matching, a point hits every
  cell within 8.17 px, and A's CPM rises only to 0.134 [§7]. A point at a blob's maximum reaches
  only its immediate neighbours.
- **The cap tracks density.** A's CPM is 0.307 in the least confluent tertile, 0.162 in the
  middle and 0.035 in the most confluent [§8]. By line, it is highest for Huh7 (0.369, 57 cells
  per image) and lowest for SHSY5Y (0.038) and MCF7 (0.050), at 432 and 527 cells per image
  [§8, §9].
- **No score saturation this time.** E01's A had 458 points scored exactly 1.0; E02's A has
  none [§9]. A's problem here is how many candidates it produces, not how it ranks them.

PREREG §12 listed this risk ("Touching cells change the task ... Part of any A − B difference
may come from separating touching cells"). The result shows it is not part of the difference;
it is nearly all of it. So H1–H3 do not test "segmentation training vs detection training" in
the abstract. They test whether **semantic segmentation plus connected components** can count
objects, against a detector that counts directly. For touching objects, it cannot. An
instance-aware read-out of A (for example, seeded watershed on A's map) or an instance
segmenter would be the fairer comparison. Neither was pre-registered or run (§8).

## 4. The detector, and why the cascade had no room

**B transfers well.** CPM 0.943 [0.923, 0.963]. At "1 FP" (29.90 FP/image) it finds 0.836 of
small, 0.982 of medium and 0.998 of large cells [§4]. Its weakest line is BV2, small dense
microglia, with CPM 0.891 [§8]. Unlike E01's B, which weakened on large lesions, E02's B is
*best* on large cells. A plausible reason, not tested: cells span a narrow size range (4.1× in
ESD against 8.5× for E01's lesions; PREREG §12), so the capped Gaussian (σ ≤ 16.34 px) stays
reasonable even for the largest cells. E01's ill-posed-target explanation predicts exactly this.

**B's curve stops at 29.93 FP/image.** That is just past the second CPM target. The third and
fourth targets (59.81, 89.71) are unreached, and B's sensitivity there is its maximum, 0.949
[§4.2]. Every B candidate above the 0.05 floor is already used by then.

**Why H4 could barely move.** C1 does not add points; it re-scores B's points (PREREG §7.1).
C1's curve therefore ends at the same place as B's: the same maximum, 0.949, at the same FP
count [§4.2], so C1 can never exceed 0.949 at any target. B is already at 0.949 at the second
target and, by construction, at the two unreached ones. Only sensitivity at 14.95 FP/image had
room, where B has 0.923. The largest possible CPM gain was therefore (0.949 − 0.923) ÷ 4 ≈
**0.0065**. C1
gained +0.003, about half of that ceiling. Its gain at the first target, +0.010 [0.000, 0.016],
is real in point terms [§6]. But H4's condition A needs a CI above 0, and the lower bound is
exactly 0.000. Three lines (A172, SKOV3, SkBr3) gain exactly nothing [§3.1]. In E01, C1's gain
came from A's strength on large lesions, which B lacked. Here A has no such strength: within a
merged blob, a(q) is the blob's maximum, so it adds almost no information about which peak is
a cell.

**C2 behaves like C1.** C2 − B CPM is +0.003 [0.000, 0.005] [§6]. The trained crop segmenter
adds nothing a free re-scoring does not.

**The fixed operating points move, they do not improve.** At s ≥ 0.4, C1 finds 0.946 of cells
at 19.4 FP/image, against B's 0.870 at 7.2 FP/image at peak ≥ 0.3 [§4.1]. That is a different
point on essentially the same curve (`report/froc_comparison.png`), so the fixed points should
not be compared head to head.

## 5. Segmentation: H5 and H6

- **H5 holds, almost by construction.** C1's mask is A's component ∩ {p ≥ 0.5} for every kept
  peak (PREREG §7.1). Nearly every A foreground component holds at least one kept B peak, so
  C1's mask is A's mask to within a few pixels: image Dice 0.906 for both, a difference of
  +7.2e-06 [§5, §6]. The replication is real but uninformative. It says the cascade does not
  lose A's mask, not that it adds anything.
- **H6 fails: C2's outlines are slightly worse, not better.** C4 − A per-object Dice is −0.015
  [−0.026, −0.006], negative in every line, worst in BV2 (−0.039) and SkBr3 (−0.020) [§3.1].
  C2 − A is −0.013 [−0.022, −0.005] [§6]. In E01, C2's per-lesion Dice gain came mostly from
  *coverage*: it touched lesions A's mask missed. Here A's mask already covers nearly all cells,
  so there is nothing to gain from coverage. C2 was trained on the same gapped label as A, so its
  crop regions merge touching cells too.
- **Per-object Dice is low for every arm** (A 0.132, C4 0.116) [§5], for the reason PREREG §12
  gave: a merged component enters P_L for every cell it covers. The metric mostly measures
  merging here, and should not be read as outline quality.

## 6. How much to trust this

- **The A − B gap is not fragile.** It is +0.85 CPM with a CI of [0.80, 0.89]. It holds in
  every line, every size bin and every confluence tertile [§3.1, §4, §8], under both hit rules
  [§7], and at the fixed operating points [§4.1]. No seed or well effect plausibly closes a gap
  of that size.
- **The cluster bootstrap is crude.** It has 9 wells, almost one per line (PREREG §8, §12).
  Intervals with a bound of exactly 0.000 (H4, C2 − B) reflect resamples in which the two arms
  tie in the dominant wells; they are not precise estimates. Condition B, the per-line count,
  was added for this reason, and it agrees with condition A on every hypothesis [§3].
- **B's duplicate points are free under the pre-registered rule.** B emits 578 candidates per
  image against 304 cells: 1.90 points per detected cell, 259 redundant points per image [§9,
  §12]. Under one-hit matching, as in E01, redundant points are counted but are not false
  positives. That is part of why B's curve climbs so high. A one-to-one matching, in which every
  extra point is an FP, would lower B's numbers. How much was not measured. The NMS half-width
  (4 px) was fixed from training anchors before training (PREREG §5), so the duplicates are a
  property of the design, not something tuned.
- **B's false positives cluster.** At peak ≥ 0.3, half of B's 10,528 FPs come from 5% of the
  images, and one image has 872 [§12]. LIVECell annotations are not exhaustive (PREREG §12), so
  some of these may be unannotated cells or debris. They were not reviewed.
- **One seed per arm**, and training variance is not estimated (PREREG §12). That matters for
  the small effects (H4: +0.003; H6: −0.015), not for the large ones.
- **Two GPUs predicted the test set.** A cross-GPU check on training images found differences
  of at most 0.0008 in A's and B's probabilities and 0.016 in C2's crop maps. 18 of 12,429 B
  peaks and 0 of 12,429 C2 assignments differed [§10]. Every effect reported here is orders of
  magnitude larger. Which GPU predicted each image is recorded.
- **Two record-keeping notes.**
  - C2's crop dataset has 7,449 positives and 3,724 negative attempts. PREREG §7.1's text says
    7,454 and 3,727. The rule ("2 per image") was implemented as written, and images with fewer
    than 2 cells give fewer [§10]; the PREREG's arithmetic assumed every image had 2. This
    should be recorded as a deviation (§9).
  - The E01-style "1 FP" for A is its curve maximum, flagged as unreached [§4].

## 7. Conclusions

1. **The heatmap detector transfers; the segmenter-as-detector does not.** On LIVECell,
   B finds 94–95% of cells at under 30 FP/image, and A, read out as E01 read it, finds 9%.
2. **The size trade-off of E01 does not transfer.** E01's segmenter won on large lesions; here
   it loses on every size. The cause is object contact, not object size.
3. **The cascade (primary hypothesis H4) is not supported here.** It is not harmful (H5), but it
   cannot add candidates, and B left it less than 0.007 CPM of headroom.
4. **The crop segmenter does not improve outlines** (H6), because in E01 its gain was coverage,
   and A already covers the cells here.
5. **The generalisable lesson:** a detector built on a semantic mask is only as good as its
   object-splitting step. E01 never tested that step, because metastases rarely touch. E02
   shows it decides the outcome when objects do.

## 8. Suggested follow-up (not done in E02)

- **An instance-aware segmenter arm.** Read A's map out with seeds (for example, local maxima
  of the distance transform of p ≥ 0.5, then watershed), or train an instance segmenter. Only
  this would test "segmentation vs detection" fairly on touching objects. It would need its own
  pre-registration, because A's test outputs have now been seen.
- **One-to-one matching** as a sensitivity analysis for B (and every arm), to measure how much
  B's CPM relies on redundant points being free.
- **A cascade that can add points**, for example components of A with no B peak, to give a
  two-stage design headroom above B's curve.
- **Review of B's top false-positive images** by a reader, to estimate how many are unannotated
  cells.
- **More wells or a second seed**, for the small effects (H4, H6).

## 9. Deviation D2 (added 2026-11-18)

The C2 crop counts quoted in PREREG §7.1 did not match the build. The owner approved this
entry, which is now in PREREG.md under "Deviations" (summary):

> **D2 (2026-11-18): C2 crop counts in §7.1 corrected (record-keeping; no design change).**
> Test-set outputs had been seen when this was noted. §7.1 states 7,454 positive crops and
> 3,727 negative attempts. The rule "2 positives per training image" gives fewer for the
> training images with fewer than 2 cells; the build (`results/two_stage/c2_build.json`,
> 2026-11-17, before C2 training) produced 7,449 positives and 3,724 negative attempts
> (0.5 × positives, rounded down). The rule, seed and manifest are unchanged.

PREREG's new SHA-256 is recorded in `results/prereg_sha256.txt` (line `deviation-D2`).
