# fcos evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9879 | 0.7949 | cabbage_detection.ap_101 |
| val | 46 | 0.9820 | 0.7205 | cabbage_detection.ap_101 |
| test | 92 | 0.9732 | 0.7133 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12778 | 12236 | 542 | 161 | +0.0307 | 0.0424 | 0.0130 | 0.9721 |
| val | 46 | 1805 | 1870 | 1768 | 102 | 37 | +0.0360 | 0.0545 | 0.0205 | 0.9622 |
| test | 92 | 3419 | 3544 | 3329 | 215 | 90 | +0.0366 | 0.0607 | 0.0263 | 0.9562 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
