# retinanet-detector512 evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9652 | 0.6851 | cabbage_detection.ap_101 |
| val | 46 | 0.9628 | 0.6435 | cabbage_detection.ap_101 |
| test | 92 | 0.9547 | 0.6378 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 13243 | 11774 | 1469 | 623 | +0.0682 | 0.1109 | 0.0503 | 0.9184 |
| val | 46 | 1805 | 1976 | 1733 | 243 | 72 | +0.0947 | 0.1230 | 0.0399 | 0.9167 |
| test | 92 | 3419 | 3736 | 3251 | 485 | 168 | +0.0927 | 0.1298 | 0.0491 | 0.9087 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
