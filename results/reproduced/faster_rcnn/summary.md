# faster_rcnn evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9880 | 0.7431 | cabbage_detection.ap_101 |
| val | 46 | 0.9766 | 0.6895 | cabbage_detection.ap_101 |
| test | 92 | 0.9739 | 0.6811 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12737 | 12251 | 486 | 146 | +0.0274 | 0.0382 | 0.0118 | 0.9749 |
| val | 46 | 1805 | 1853 | 1763 | 90 | 42 | +0.0266 | 0.0486 | 0.0233 | 0.9639 |
| test | 92 | 3419 | 3535 | 3307 | 228 | 112 | +0.0339 | 0.0645 | 0.0328 | 0.9511 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
