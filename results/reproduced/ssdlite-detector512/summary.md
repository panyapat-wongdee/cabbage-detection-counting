# ssdlite-detector512 evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.6744 | 0.4307 | cabbage_detection.ap_101 |
| val | 46 | 0.6711 | 0.4108 | cabbage_detection.ap_101 |
| test | 92 | 0.7108 | 0.4352 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 5703 | 5658 | 45 | 6739 | -0.5400 | 0.0079 | 0.5436 | 0.6252 |
| val | 46 | 1805 | 837 | 829 | 8 | 976 | -0.5363 | 0.0096 | 0.5407 | 0.6276 |
| test | 92 | 3419 | 1648 | 1622 | 26 | 1797 | -0.5180 | 0.0158 | 0.5256 | 0.6402 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
