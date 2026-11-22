# E01 fold-0 results (97 scans, 66 patients, 866 lesions)

Point estimates with 95% patient-bootstrap CIs (2,000 resamples, paired). Fixed operating points: A p >= 0.5, B peak >= 0.3.

| Metric | A: segmentation | B: heatmap | B - A |
|---|---|---|---|
| CPM (primary) | 0.768 [0.705, 0.828] | 0.773 [0.710, 0.846] | 0.005 [-0.036, 0.055] |
| Sensitivity @ 0.5 FP/scan | 0.694 [0.594, 0.783] | 0.647 [0.567, 0.746] | -0.047 [-0.100, 0.015] |
| Sensitivity @ 1.0 FP/scan | 0.775 [0.698, 0.837] | 0.730 [0.663, 0.835] | -0.045 [-0.077, 0.026] |
| Sensitivity @ 2.0 FP/scan | 0.801 [0.750, 0.852] (unreached) | 0.837 [0.771, 0.898] | 0.036 [-0.028, 0.085] |
| Sensitivity @ 3.0 FP/scan | 0.801 [0.750, 0.852] (unreached) | 0.876 [0.823, 0.922] | 0.075 [0.027, 0.117] |
| Sensitivity @ 1 FP/scan, ESD <3 mm | 0.520 [0.391, 0.643] | 0.533 [0.426, 0.741] | 0.013 [-0.032, 0.152] |
| Sensitivity @ 1 FP/scan, ESD 3-10 mm | 0.845 [0.784, 0.892] | 0.811 [0.749, 0.887] | -0.035 [-0.071, 0.026] |
| Sensitivity @ 1 FP/scan, ESD >10 mm | 0.978 [0.934, 1.000] | 0.728 [0.633, 0.847] | -0.250 [-0.345, -0.123] |
| Fixed op: sensitivity | 0.777 [0.721, 0.831] | 0.829 [0.778, 0.873] | - |
| Fixed op: FP/scan | 1.041 [0.670, 1.463] | 1.856 [0.922, 2.989] | - |
