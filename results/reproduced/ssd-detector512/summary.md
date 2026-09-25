# ssd-detector512 evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9291 | 0.7547 | cabbage_detection.ap_101 |
| val | 46 | 0.9151 | 0.6128 | cabbage_detection.ap_101 |
| test | 92 | 0.8954 | 0.6131 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 10695 | 10593 | 102 | 1804 | -0.1373 | 0.0095 | 0.1455 | 0.9175 |
| val | 46 | 1805 | 1574 | 1538 | 36 | 267 | -0.1280 | 0.0229 | 0.1479 | 0.9103 |
| test | 92 | 3419 | 3084 | 2991 | 93 | 428 | -0.0980 | 0.0302 | 0.1252 | 0.9199 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
