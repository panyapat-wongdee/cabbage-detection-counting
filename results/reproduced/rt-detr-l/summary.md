# rt-detr-l evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9853 | 0.7828 | cabbage_detection.ap_101 |
| val | 46 | 0.9794 | 0.7228 | cabbage_detection.ap_101 |
| test | 92 | 0.9721 | 0.7157 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 13696 | 12268 | 1428 | 129 | +0.1048 | 0.1043 | 0.0104 | 0.9403 |
| val | 46 | 1805 | 1989 | 1778 | 211 | 27 | +0.1019 | 0.1061 | 0.0150 | 0.9373 |
| test | 92 | 3419 | 3864 | 3349 | 515 | 70 | +0.1302 | 0.1333 | 0.0205 | 0.9197 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
