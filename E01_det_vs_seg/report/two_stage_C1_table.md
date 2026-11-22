# Two-stage C1 (B detects, A segments): fold 0 (97 scans, 66 patients, 866 lesions)

**EXPLORATORY** - see `PREREG_C_two_stage.md` (SHA-256 `d71143e5f73f55cf...`). 95% patient-bootstrap CIs, 2,000 paired resamples. Fixed operating points: A p >= 0.5, B peak >= 0.3, C1 s >= 0.4.

## Detection: C1 vs B

| Metric | B: heatmap | C1: two-stage | C1 - B | A (context) |
|---|---|---|---|---|
| CPM (primary) | 0.773 [0.710, 0.846] | 0.802 [0.737, 0.864] | 0.029 [0.005, 0.045] | 0.768 [0.705, 0.828] |
| Sensitivity @ 0.5 FP/scan | 0.647 [0.567, 0.746] | 0.696 [0.598, 0.784] | 0.050 [0.008, 0.077] | 0.694 [0.594, 0.783] |
| Sensitivity @ 1.0 FP/scan | 0.730 [0.663, 0.835] | 0.774 [0.710, 0.846] | 0.044 [-0.016, 0.078] | 0.775 [0.698, 0.837] |
| Sensitivity @ 2.0 FP/scan | 0.837 [0.771, 0.898] | 0.850 [0.790, 0.906] | 0.013 [-0.007, 0.041] | 0.801 [0.750, 0.852] (unreached) |
| Sensitivity @ 3.0 FP/scan | 0.876 [0.823, 0.922] | 0.887 [0.837, 0.926] | 0.010 [-0.002, 0.027] | 0.801 [0.750, 0.852] (unreached) |
| Sensitivity @ 1 FP/scan, ESD <3 mm | 0.533 [0.426, 0.741] | 0.520 [0.411, 0.669] | -0.013 [-0.137, 0.057] | 0.520 [0.391, 0.643] |
| Sensitivity @ 1 FP/scan, ESD 3-10 mm | 0.811 [0.749, 0.887] | 0.847 [0.799, 0.901] | 0.036 [-0.010, 0.074] | 0.845 [0.784, 0.892] |
| Sensitivity @ 1 FP/scan, ESD >10 mm | 0.728 [0.633, 0.847] | 0.957 [0.826, 0.990] | 0.228 [0.101, 0.308] | 0.978 [0.934, 1.000] |
| CPM, ESD <3 mm | 0.614 [0.502, 0.750] | 0.612 [0.497, 0.732] | -0.002 [-0.040, 0.018] | 0.514 [0.399, 0.628] |
| CPM, ESD 3-10 mm | 0.835 [0.780, 0.896] | 0.862 [0.814, 0.911] | 0.027 [0.004, 0.048] | 0.837 [0.792, 0.882] |
| CPM, ESD >10 mm | 0.785 [0.697, 0.878] | 0.902 [0.840, 0.966] | 0.117 [0.062, 0.174] | 0.973 [0.929, 1.000] |
| Fixed op: sensitivity | 0.829 [0.778, 0.873] | 0.782 [0.728, 0.834] | - | 0.777 [0.721, 0.831] |
| Fixed op: FP/scan | 1.856 [0.922, 2.989] | 1.082 [0.686, 1.533] | - | 1.041 [0.670, 1.463] |

## Segmentation: C1 vs A

| Metric | A: segmentation | C1: two-stage | C1 - A |
|---|---|---|---|
| Mean per-scan Dice (primary) | 0.769 [0.724, 0.806] | 0.764 [0.718, 0.803] | -0.005 [-0.018, 0.000] |
| Mean per-lesion Dice, all | 0.550 [0.490, 0.614] | 0.552 [0.492, 0.614] | 0.002 [-0.002, 0.005] |
| Mean per-lesion Dice, ESD <3 mm | 0.283 [0.212, 0.352] | 0.285 [0.206, 0.358] | 0.001 [-0.007, 0.009] |
| Mean per-lesion Dice, ESD 3-10 mm | 0.612 [0.559, 0.665] | 0.615 [0.565, 0.666] | 0.003 [0.000, 0.007] |
| Mean per-lesion Dice, ESD >10 mm | 0.836 [0.797, 0.868] | 0.828 [0.787, 0.863] | -0.007 [-0.025, 0.000] |

B has no mask (Dice n/a). C1 keeps 855 points at s >= 0.4, of which 0 use the heatmap fallback mask; 901 of 1827 B points are assigned to an A component.
