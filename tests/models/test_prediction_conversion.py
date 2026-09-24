import pytest

torch = pytest.importorskip("torch")

from cabbage_detection.models.prediction_conversion import convert_torchvision_predictions


def test_torchvision_conversion_filters_confidence_once_and_records_ownership():
    outputs = [
        {
            "boxes": torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32),
            "scores": torch.tensor([0.5, 0.49]),
            "labels": torch.tensor([1, 2]),
        }
    ]
    records = convert_torchvision_predictions(["image-1"], outputs, 0.5)
    assert [item.score for item in records[0].detections] == [0.5]
    assert records[0].detections[0].class_id == 1
    assert records[0].postprocessing is not None
    assert records[0].postprocessing.nms_owner == "torchvision"


def test_torchvision_conversion_rejects_mismatched_batch_lengths():
    with pytest.raises(ValueError, match="image_ids and outputs"):
        convert_torchvision_predictions(["a", "b"], [{}], 0.5)


def test_torchvision_conversion_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        convert_torchvision_predictions(
            ["a"],
            [{"boxes": [[0, 0, 1, 1]], "scores": [float("nan")], "labels": [1]}],
            0.5,
        )
