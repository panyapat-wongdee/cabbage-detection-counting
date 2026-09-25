# Reproduced model comparison (test split, detector512 variant)

Split `test`, 92 images, scored by this repository.

## Detection

| model | framework | scope | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| faster_rcnn* | torchvision | paper_model | 0.9712 | 0.6495 |
| ssd* | torchvision | paper_model | 0.8954 | 0.6131 |
| retinanet* | torchvision | paper_model | 0.9547 | 0.6378 |
| fcos* | torchvision | paper_model | 0.9723 | 0.6789 |
| yolov8n | ultralytics | paper_model | 0.9670 | 0.5999 |
| yolov8m | ultralytics | paper_model | 0.9700 | 0.6676 |
| yolo11n | ultralytics | paper_model | 0.9703 | 0.6956 |
| yolo11m | ultralytics | paper_model | 0.9723 | 0.7156 |
| rt-detr-l | ultralytics | paper_model | 0.9721 | 0.7157 |
| ssdlite* | torchvision | supplementary_unreported | 0.7108 | 0.4352 |
| yolo12n | ultralytics | repository_extension | 0.9679 | 0.7047 |
| yolo12m | ultralytics | repository_extension | 0.9685 | 0.7340 |
| yolo26n | ultralytics | repository_extension | 0.9586 | 0.6780 |
| yolo26m | ultralytics | repository_extension | 0.9687 | 0.7289 |

## Counting

| model | actual | predicted | TP | FP | FN | count error | FDR | FNR | F1 |
|---|---|---|---|---|---|---|---|---|---|
| faster_rcnn* | 3419 | 3442 | 3261 | 181 | 158 | +0.0067 | 0.0526 | 0.0462 | 0.9506 |
| ssd* | 3419 | 3084 | 2991 | 93 | 428 | -0.0980 | 0.0302 | 0.1252 | 0.9199 |
| retinanet* | 3419 | 3736 | 3251 | 485 | 168 | +0.0927 | 0.1298 | 0.0491 | 0.9087 |
| fcos* | 3419 | 3529 | 3298 | 231 | 121 | +0.0322 | 0.0655 | 0.0354 | 0.9493 |
| yolov8n | 3419 | 3570 | 3295 | 275 | 124 | +0.0442 | 0.0770 | 0.0363 | 0.9429 |
| yolov8m | 3419 | 3653 | 3323 | 330 | 96 | +0.0684 | 0.0903 | 0.0281 | 0.9398 |
| yolo11n | 3419 | 3527 | 3308 | 219 | 111 | +0.0316 | 0.0621 | 0.0325 | 0.9525 |
| yolo11m | 3419 | 3603 | 3324 | 279 | 95 | +0.0538 | 0.0774 | 0.0278 | 0.9467 |
| rt-detr-l | 3419 | 3864 | 3349 | 515 | 70 | +0.1302 | 0.1333 | 0.0205 | 0.9197 |
| ssdlite* | 3419 | 1648 | 1622 | 26 | 1797 | -0.5180 | 0.0158 | 0.5256 | 0.6402 |
| yolo12n | 3419 | 3544 | 3305 | 239 | 114 | +0.0366 | 0.0674 | 0.0333 | 0.9493 |
| yolo12m | 3419 | 3621 | 3317 | 304 | 102 | +0.0591 | 0.0840 | 0.0298 | 0.9423 |
| yolo26n | 3419 | 3280 | 3079 | 201 | 340 | -0.0407 | 0.0613 | 0.0994 | 0.9192 |
| yolo26m | 3419 | 3497 | 3270 | 227 | 149 | +0.0228 | 0.0649 | 0.0436 | 0.9456 |

## Evaluation settings

- detection: `backend`: cabbage_detection.ap_101, `ap_iou_threshold`: 0.5, `ap_iou_range`: 0.50:0.95 step 0.05, `export_confidence_threshold`: 0.001
- counting: `iou_threshold`: 0.5, `confidence_threshold`: 0.5, `matching`: one_to_one_greedy_highest_iou, `aggregation`: micro
- code revision: b0ee13d218142bf8b59a40dd4c69349df99b9d81

## Deviations

- * faster_rcnn, ssd, retinanet, fcos, ssdlite: native detector score threshold lowered for full-range AP

## Notes

- every value was produced by this repository; no published number appears here
- one AP backend and one counting threshold pair are enforced across the table
- study_scope records publication scope only; every row is a primary repository model
