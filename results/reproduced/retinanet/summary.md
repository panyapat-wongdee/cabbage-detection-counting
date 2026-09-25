# retinanet evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9736 | 0.7279 | cabbage_detection.ap_101 |
| val | 46 | 0.9695 | 0.6822 | cabbage_detection.ap_101 |
| test | 92 | 0.9607 | 0.6779 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 13388 | 11985 | 1403 | 412 | +0.0799 | 0.1048 | 0.0332 | 0.9296 |
| val | 46 | 1805 | 1949 | 1734 | 215 | 71 | +0.0798 | 0.1103 | 0.0393 | 0.9238 |
| test | 92 | 3419 | 3701 | 3264 | 437 | 155 | +0.0825 | 0.1181 | 0.0453 | 0.9169 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
