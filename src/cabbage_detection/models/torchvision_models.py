"""Model-specific torchvision construction kept behind a lazy import boundary."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig


TORCHVISION_MODEL_NAMES = frozenset({"faster_rcnn", "ssd", "ssdlite", "retinanet", "fcos"})
_USE_CONFIG = object()
SUPPORTED_WEIGHT_IDENTIFIERS = frozenset({"COCO_V1", "none"})


def _torchvision_modules() -> tuple[Any, Any]:
    try:
        import torchvision
        from torchvision.models import detection
    except ImportError as error:
        raise RuntimeError(
            "install the optional torchvision/torch dependencies to construct torchvision models"
        ) from error
    return torchvision, detection


def build_torchvision_model(
    config: ExperimentConfig,
    num_classes: int | object = _USE_CONFIG,
    weights: object | None = _USE_CONFIG,
) -> Any:
    """Build a published torchvision detector or supplementary SSDLite.

    The constructors and head replacements mirror the preserved training
    notebook. Pretrained COCO weights are requested explicitly; downloading
    them is left to torchvision and is never performed during package import.

    Every detector owns a second resize in ``model.transform`` after the
    configured ``image_size`` resize; see ``detector_input_size``.
    """
    model = _build_detector(config, num_classes, weights)
    if config.detector_input == "image_size":
        _use_image_size_as_detector_input(model, config)
    return model


def detector_input_size(model: Any, image_size: tuple[int, int]) -> tuple[int, int]:
    """Return the (height, width) the backbone receives for an ``image_size`` input.

    With ``detector_input: framework_default`` this is 800x800 for Faster
    R-CNN, RetinaNet and FCOS (torchvision's ``min_size=800``), 300x300 for
    SSD and 320x320 for SSDLite (their ``fixed_size``), whatever
    ``image_size`` is.
    """
    import torch

    height, width = image_size
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            images, _ = model.transform([torch.zeros(3, height, width)])
    finally:
        model.train(was_training)
    return tuple(int(value) for value in images.tensors.shape[-2:])


def _use_image_size_as_detector_input(model: Any, config: ExperimentConfig) -> None:
    """Make the detector's own transform keep the configured ``image_size``.

    ``fixed_size`` takes precedence over ``min_size``/``max_size`` inside
    GeneralizedRCNNTransform, so it is the setting that decides the size.
    SSD300's anchor ``steps`` are tied to a 300-pixel input; they are cleared
    so the default boxes follow the actual feature-map grid.
    """
    height, width = config.image_size
    model.transform.min_size = (min(height, width),)
    model.transform.max_size = max(height, width)
    model.transform.fixed_size = (height, width)
    if config.model_name == "ssd":
        model.anchor_generator.steps = None


def _build_detector(
    config: ExperimentConfig,
    num_classes: int | object,
    weights: object | None,
) -> Any:
    if config.framework != "torchvision":
        raise ValueError("torchvision model construction requires framework=torchvision")
    if config.model_name not in TORCHVISION_MODEL_NAMES:
        raise ValueError(f"unsupported torchvision model: {config.model_name}")
    if num_classes is _USE_CONFIG:
        num_classes = config.initialization.num_classes
    if num_classes < 2:
        raise ValueError("num_classes must include background and cabbage")
    if weights is _USE_CONFIG:
        weights = config.initialization.weights
        if weights not in SUPPORTED_WEIGHT_IDENTIFIERS:
            raise ValueError(f"unsupported torchvision weights: {weights}")
        if weights == "none":
            weights = None
    torchvision, detection = _torchvision_modules()
    if config.model_name == "faster_rcnn":
        model = detection.fasterrcnn_resnet50_fpn(weights=weights)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
        return model
    if config.model_name == "retinanet":
        model = detection.retinanet_resnet50_fpn(weights=weights)
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=head.conv[0][0].in_channels,
            num_anchors=head.num_anchors,
            num_classes=num_classes,
        )
        return model
    if config.model_name == "fcos":
        model = detection.fcos_resnet50_fpn(weights=weights)
        from torchvision.models.detection.fcos import FCOSClassificationHead

        head = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(
            in_channels=head.conv[0].in_channels,
            num_anchors=head.num_anchors,
            num_classes=num_classes,
        )
        return model
    size = config.image_size[0]
    if config.model_name == "ssd":
        model = detection.ssd300_vgg16(weights=weights)
        from torchvision.models.detection import _utils
        from torchvision.models.detection.ssd import SSDClassificationHead

        in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=model.anchor_generator.num_anchors_per_location(),
            num_classes=num_classes,
        )
        # No effect on the detector input: SSD300's transform has
        # fixed_size=(300, 300), which GeneralizedRCNNTransform applies before
        # min_size/max_size. Kept so released runs rebuild identically.
        model.transform.min_size = (size,)
        model.transform.max_size = size
        return model

    model = detection.ssdlite320_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
    )
    from functools import partial

    from torch import nn
    from torchvision.models.detection import _utils
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=model.anchor_generator.num_anchors_per_location(),
        num_classes=num_classes,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    # No effect on the detector input: fixed_size=(320, 320) wins, as for SSD.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model
