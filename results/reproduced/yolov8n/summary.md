# yolov8n evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9813 | 0.6506 | cabbage_detection.ap_101 |
| val | 46 | 0.9732 | 0.6029 | cabbage_detection.ap_101 |
| test | 92 | 0.9670 | 0.5999 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 12871 | 12130 | 741 | 267 | +0.0382 | 0.0576 | 0.0215 | 0.9601 |
| val | 46 | 1805 | 1863 | 1758 | 105 | 47 | +0.0321 | 0.0564 | 0.0260 | 0.9586 |
| test | 92 | 3419 | 3570 | 3295 | 275 | 124 | +0.0442 | 0.0770 | 0.0363 | 0.9429 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
