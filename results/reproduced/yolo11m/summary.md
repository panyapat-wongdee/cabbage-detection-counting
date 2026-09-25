# yolo11m evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9873 | 0.8248 | cabbage_detection.ap_101 |
| val | 46 | 0.9835 | 0.7379 | cabbage_detection.ap_101 |
| test | 92 | 0.9723 | 0.7156 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12981 | 12275 | 706 | 122 | +0.0471 | 0.0544 | 0.0098 | 0.9674 |
| val | 46 | 1805 | 1881 | 1768 | 113 | 37 | +0.0421 | 0.0601 | 0.0205 | 0.9593 |
| test | 92 | 3419 | 3603 | 3324 | 279 | 95 | +0.0538 | 0.0774 | 0.0278 | 0.9467 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
