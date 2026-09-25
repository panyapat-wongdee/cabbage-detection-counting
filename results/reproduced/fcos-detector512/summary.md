# fcos-detector512 evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9860 | 0.7541 | cabbage_detection.ap_101 |
| val | 46 | 0.9729 | 0.6825 | cabbage_detection.ap_101 |
| test | 92 | 0.9723 | 0.6789 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12642 | 12151 | 491 | 246 | +0.0198 | 0.0388 | 0.0198 | 0.9706 |
| val | 46 | 1805 | 1854 | 1753 | 101 | 52 | +0.0271 | 0.0545 | 0.0288 | 0.9582 |
| test | 92 | 3419 | 3529 | 3298 | 231 | 121 | +0.0322 | 0.0655 | 0.0354 | 0.9493 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
