from pathlib import Path

import pytest

from cabbage_detection.config import load_config
from cabbage_detection.models.ultralytics_adapter import UltralyticsAdapter


@pytest.mark.integration
@pytest.mark.requires_weights
@pytest.mark.parametrize(
    ("model_name", "config_path", "weight_name"),
    [
        ("yolov8n", "configs/ultralytics/yolov8n.yaml", "yolov8n.pt"),
        ("yolov8m", "configs/ultralytics/yolov8m.yaml", "yolov8m.pt"),
        ("yolo11n", "configs/ultralytics/yolo11n.yaml", "yolo11n.pt"),
        ("yolo11m", "configs/ultralytics/yolo11m.yaml", "yolo11m.pt"),
        ("rt-detr-l", "configs/ultralytics/rt-detr-l.yaml", "rtdetr-l.pt"),
    ],
)
def test_local_weight_smoke(model_name: str, config_path: str, weight_name: str):
    weight = Path(weight_name)
    image = next(Path("prepared/fold1_yolo/train/images").glob("*.png"), None)
    if not weight.is_file() or image is None:
        pytest.skip("local Ultralytics weight and dataset image are not present")
    config = load_config(Path(config_path))
    assert config.model_name == model_name
    torch = pytest.importorskip("torch")
    if config.execution.device != "cpu" and not torch.cuda.is_available():
        pytest.skip("configured CUDA device is not available in this environment")
    if model_name == "rt-detr-l" and weight.name == "rtdetr-l.pt":
        pytest.skip(
            "the base COCO RT-DETR checkpoint is not a fine-tuned one-class "
            "cabbage checkpoint; canonical prediction rejects its non-cabbage "
            "class ids"
        )
    records = UltralyticsAdapter(config).predict(image, weight)
    assert len(records) == 1
    assert records[0].image_id == image.stem
    assert records[0].postprocessing is not None
    expected_nms_owner = "none" if model_name == "rt-detr-l" else "ultralytics"
    assert records[0].postprocessing.nms_owner == expected_nms_owner
