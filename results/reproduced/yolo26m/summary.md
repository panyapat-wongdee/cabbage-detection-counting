# yolo26m evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9845 | 0.8035 | cabbage_detection.ap_101 |
| val | 46 | 0.9799 | 0.7433 | cabbage_detection.ap_101 |
| test | 92 | 0.9687 | 0.7289 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12735 | 12073 | 662 | 324 | +0.0273 | 0.0520 | 0.0261 | 0.9608 |
| val | 46 | 1805 | 1851 | 1748 | 103 | 57 | +0.0255 | 0.0556 | 0.0316 | 0.9562 |
| test | 92 | 3419 | 3497 | 3270 | 227 | 149 | +0.0228 | 0.0649 | 0.0436 | 0.9456 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
