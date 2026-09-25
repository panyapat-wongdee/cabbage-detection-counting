# yolo11n evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9831 | 0.7565 | cabbage_detection.ap_101 |
| val | 46 | 0.9741 | 0.7041 | cabbage_detection.ap_101 |
| test | 92 | 0.9703 | 0.6956 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12760 | 12120 | 640 | 277 | +0.0293 | 0.0502 | 0.0223 | 0.9635 |
| val | 46 | 1805 | 1866 | 1763 | 103 | 42 | +0.0338 | 0.0552 | 0.0233 | 0.9605 |
| test | 92 | 3419 | 3527 | 3308 | 219 | 111 | +0.0316 | 0.0621 | 0.0325 | 0.9525 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
