# yolo12n evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9836 | 0.7783 | cabbage_detection.ap_101 |
| val | 46 | 0.9801 | 0.7135 | cabbage_detection.ap_101 |
| test | 92 | 0.9679 | 0.7047 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12828 | 12154 | 674 | 243 | +0.0348 | 0.0525 | 0.0196 | 0.9636 |
| val | 46 | 1805 | 1860 | 1758 | 102 | 47 | +0.0305 | 0.0548 | 0.0260 | 0.9593 |
| test | 92 | 3419 | 3544 | 3305 | 239 | 114 | +0.0366 | 0.0674 | 0.0333 | 0.9493 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
