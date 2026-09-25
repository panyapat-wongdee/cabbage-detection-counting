# yolo12m evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9874 | 0.8720 | cabbage_detection.ap_101 |
| val | 46 | 0.9793 | 0.7508 | cabbage_detection.ap_101 |
| test | 92 | 0.9685 | 0.7340 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 13073 | 12293 | 780 | 104 | +0.0545 | 0.0597 | 0.0084 | 0.9653 |
| val | 46 | 1805 | 1904 | 1766 | 138 | 39 | +0.0548 | 0.0725 | 0.0216 | 0.9523 |
| test | 92 | 3419 | 3621 | 3317 | 304 | 102 | +0.0591 | 0.0840 | 0.0298 | 0.9423 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
