# yolov8m evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9871 | 0.7729 | cabbage_detection.ap_101 |
| val | 46 | 0.9743 | 0.6797 | cabbage_detection.ap_101 |
| test | 92 | 0.9700 | 0.6676 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 13064 | 12255 | 809 | 142 | +0.0538 | 0.0619 | 0.0115 | 0.9626 |
| val | 46 | 1805 | 1898 | 1766 | 132 | 39 | +0.0515 | 0.0695 | 0.0216 | 0.9538 |
| test | 92 | 3419 | 3653 | 3323 | 330 | 96 | +0.0684 | 0.0903 | 0.0281 | 0.9398 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
