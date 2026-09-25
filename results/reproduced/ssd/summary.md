# ssd evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9767 | 0.7571 | cabbage_detection.ap_101 |
| val | 46 | 0.9592 | 0.6024 | cabbage_detection.ap_101 |
| test | 92 | 0.9497 | 0.5920 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 11224 | 11067 | 157 | 1330 | -0.0946 | 0.0140 | 0.1073 | 0.9370 |
| val | 46 | 1805 | 1633 | 1571 | 62 | 234 | -0.0953 | 0.0380 | 0.1296 | 0.9139 |
| test | 92 | 3419 | 3223 | 3087 | 136 | 332 | -0.0573 | 0.0422 | 0.0971 | 0.9295 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
