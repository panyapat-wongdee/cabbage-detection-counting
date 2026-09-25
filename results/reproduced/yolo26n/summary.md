# yolo26n evaluation

## Detection

| split | images | mAP@50 | mAP@50:95 | backend |
|---|---|---|---|---|
| train | 320 | 0.9734 | 0.7163 | cabbage_detection.ap_101 |
| val | 46 | 0.9655 | 0.6848 | cabbage_detection.ap_101 |
| test | 92 | 0.9586 | 0.6780 | cabbage_detection.ap_101 |

## Counting

| split | images | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 320 | 12397 | 11917 | 11370 | 547 | 1027 | -0.0387 | 0.0459 | 0.0828 | 0.9353 |
| val | 46 | 1805 | 1730 | 1649 | 81 | 156 | -0.0416 | 0.0468 | 0.0864 | 0.9330 |
| test | 92 | 3419 | 3280 | 3079 | 201 | 340 | -0.0407 | 0.0613 | 0.0994 | 0.9192 |

Counting metrics are micro-averaged over the split.

## Notes

- counting metrics are micro-averaged over the split; macro holds the per-image mean
- splits are scored separately and are never pooled into one figure
