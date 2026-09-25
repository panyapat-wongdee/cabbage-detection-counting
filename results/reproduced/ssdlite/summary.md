# ssdlite evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.7814 | 0.4656 | cabbage_detection.ap_101 |
| val | 46 | 0.7867 | 0.4501 | cabbage_detection.ap_101 |
| test | 92 | 0.8304 | 0.4781 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 6731 | 6538 | 193 | 5859 | -0.4570 | 0.0287 | 0.4726 | 0.6836 |
| val | 46 | 1805 | 990 | 964 | 26 | 841 | -0.4515 | 0.0263 | 0.4659 | 0.6898 |
| test | 92 | 3419 | 1892 | 1851 | 41 | 1568 | -0.4466 | 0.0217 | 0.4586 | 0.6970 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
