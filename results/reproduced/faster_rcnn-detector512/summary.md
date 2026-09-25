# faster_rcnn-detector512 evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9873 | 0.7145 | cabbage_detection.ap_101 |
| val | 46 | 0.9744 | 0.6563 | cabbage_detection.ap_101 |
| test | 92 | 0.9712 | 0.6495 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12481 | 12158 | 323 | 239 | +0.0068 | 0.0259 | 0.0193 | 0.9774 |
| val | 46 | 1805 | 1823 | 1734 | 89 | 71 | +0.0100 | 0.0488 | 0.0393 | 0.9559 |
| test | 92 | 3419 | 3442 | 3261 | 181 | 158 | +0.0067 | 0.0526 | 0.0462 | 0.9506 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
