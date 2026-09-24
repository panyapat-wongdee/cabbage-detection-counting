"""Parameter and FLOP counts for the trained detectors at one input size.

FLOPs are reported as 2 x multiply-accumulates, the convention Ultralytics
uses for its GFLOPs. Every model is measured at the same detector input, the
configured ``image_size`` (512x512 for every shipped profile), so the numbers
compare architectures at one resolution. For the torchvision models that is
not the input the released runs used: their own transform resizes to 800x800
(Faster R-CNN, RetinaNet, FCOS), 300x300 (SSD) or 320x320 (SSDLite). That
input is recorded beside the count as ``run_detector_input``.

Each framework is counted with its own established tool, and every count is
cross-checked with ``torch.utils.flop_counter``:

- torchvision: ``torchinfo.summary`` total mult-adds x 2, on the full
  detector forward with its transform forced to ``image_size``;
- Ultralytics: ``ultralytics.utils.torch_utils.model_info`` (thop), the
  figure ``model.info()`` prints, on the model after ``fuse()``. Ultralytics
  fuses BatchNorm into the convolutions for every predict/val call and
  publishes its official figures fused, and the torchvision counts likewise
  exclude BatchNorm (FrozenBatchNorm2d holds buffers, not parameters, and no
  counter charges it FLOPs). torchinfo is not used here because it misses the
  functional matmuls inside RT-DETR's attention.

Faster R-CNN's box head runs once per RPN proposal, so its count depends on
the image; the seeded input yields the full ``rpn.post_nms_top_n`` proposals
(1000 at test time), which is the upper bound and torchvision's own
convention. The proposal count is recorded. Real test images give fewer
proposals and fewer FLOPs. Post-processing such as NMS is not counted by any
of these tools.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from ..config import load_config


@dataclass(frozen=True)
class ModelComplexity:
    model: str
    # Training run directory name; differs from ``model`` for a variant run
    # such as ``fcos-detector512``.
    run: str
    framework: str
    params: int
    gflops: float
    flops_input: tuple[int, int]
    run_detector_input: tuple[int, int]
    counter: str
    crosscheck_gflops: float
    checkpoint_sha256: str
    # Two-stage detectors only: proposals the box head processed while counted.
    proposals: int | None = None
    # The same counter at ``run_detector_input``: the compute the released run
    # actually spent per image. Equals ``gflops`` when the two inputs agree.
    run_gflops: float | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flop_counter_gflops(module, inputs) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        module(inputs)
    return counter.get_total_flops() / 1e9


def measure_torchvision(run_dir: Path, checkpoint: Path) -> ModelComplexity:
    """Count a torchvision run's detector with its input forced to ``image_size``."""
    try:
        import torch
        from torchinfo import summary
    except ImportError as error:
        raise RuntimeError('install the "analysis" extra (torchinfo) to count torchvision models') from error
    from .torchvision_models import build_torchvision_model, detector_input_size

    config = load_config(Path(run_dir) / "config.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    run_model = build_torchvision_model(config, weights=None)
    run_model.load_state_dict(state)
    run_model.eval()
    run_input = detector_input_size(run_model, config.image_size)

    measured = replace(config, detector_input="image_size")
    model = build_torchvision_model(measured, weights=None)
    model.load_state_dict(state)
    model.eval()
    if detector_input_size(model, config.image_size) != tuple(config.image_size):
        raise RuntimeError(f"{config.model_name}: detector input was not forced to image_size")

    # Two-stage detectors process a content-dependent number of proposals; a
    # seeded image keeps the count reproducible and saturates the proposals.
    generator = torch.Generator().manual_seed(0)
    height, width = config.image_size
    images = [torch.rand(3, height, width, generator=generator)]
    gflops, proposals = _torchinfo_gflops(model, images, summary, config.model_name)
    run_gflops, _ = _torchinfo_gflops(run_model, images, summary, config.model_name)
    return ModelComplexity(
        model=config.model_name,
        run=Path(run_dir).name,
        framework="torchvision",
        params=sum(parameter.numel() for parameter in model.parameters()),
        gflops=gflops,
        flops_input=tuple(config.image_size),
        run_detector_input=run_input,
        counter="torchinfo.summary total_mult_adds x 2",
        crosscheck_gflops=_flop_counter_gflops(model, images),
        checkpoint_sha256=_sha256(checkpoint),
        proposals=proposals,
        run_gflops=run_gflops,
    )


def _torchinfo_gflops(model, images, summary, name: str) -> tuple[float, int | None]:
    """Count one torchvision forward; for two-stage models, insist on the proposal bound."""
    proposals: list[int] = []
    rpn = getattr(model, "rpn", None)
    hook = (
        rpn.register_forward_hook(lambda _module, _inputs, output: proposals.append(len(output[0][0])))
        if rpn is not None
        else None
    )
    try:
        stats = summary(model, input_data=[images], verbose=0, device="cpu")
    finally:
        if hook is not None:
            hook.remove()
    if rpn is not None and proposals[-1] != rpn.post_nms_top_n():
        raise RuntimeError(f"{name}: counted {proposals[-1]} proposals, not the upper bound")
    return 2 * stats.total_mult_adds / 1e9, (proposals[-1] if proposals else None)


def measure_ultralytics(run_dir: Path, checkpoint: Path) -> ModelComplexity:
    """Count an Ultralytics run with the framework's own ``model_info``."""
    import torch
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils.torch_utils import model_info

    config = load_config(Path(run_dir) / "config.yaml")
    constructor = RTDETR if config.model_name == "rt-detr-l" else YOLO
    # Fused as Ultralytics runs it for inference and as its published figures are.
    network = constructor(str(checkpoint)).model.float().eval().fuse()
    height, width = config.image_size
    if height != width:
        raise ValueError("Ultralytics model_info counts a square imgsz only")
    # model_info returns its figures only when it also prints them.
    with contextlib.redirect_stdout(io.StringIO()):
        _, params, _, gflops = model_info(network, imgsz=height, verbose=True)
    return ModelComplexity(
        model=config.model_name,
        run=Path(run_dir).name,
        framework="ultralytics",
        params=int(params),
        gflops=float(gflops),
        flops_input=(height, width),
        run_detector_input=(height, width),
        counter="ultralytics.utils.torch_utils.model_info (thop), fused",
        crosscheck_gflops=_flop_counter_gflops(network, torch.zeros(1, 3, height, width)),
        checkpoint_sha256=_sha256(checkpoint),
        run_gflops=float(gflops),
    )


def measure_run(run_dir: Path, checkpoint: Path | None = None) -> ModelComplexity:
    """Measure one training run, by default from its own ``checkpoints/best.pt``."""
    run_dir = Path(run_dir)
    checkpoint = Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    framework = load_config(run_dir / "config.yaml").framework
    measure = measure_torchvision if framework == "torchvision" else measure_ultralytics
    return measure(run_dir, checkpoint)


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "ultralytics", "torchinfo", "ultralytics-thop"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_complexity(records: Sequence[ModelComplexity], output: Path) -> Path:
    """Write the counts with the method and tool versions that produced them."""
    payload = {
        "flops_definition": "2 x multiply-accumulates; post-processing (NMS) not counted",
        "note": (
            "All models are counted at flops_input. run_detector_input is the input the "
            "released run's detector actually received, which differs for torchvision models."
        ),
        "versions": _versions(),
        "models": [
            {**asdict(record), "flops_input": list(record.flops_input),
             "run_detector_input": list(record.run_detector_input)}
            for record in records
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Count parameters and FLOPs of trained runs.")
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="training run directory; repeat for several models")
    parser.add_argument("--output", type=Path, default=Path("results/reproduced/model_complexity.json"))
    args = parser.parse_args(argv)
    records = [measure_run(run) for run in args.run]
    for record in records:
        print(f"{record.model}: {record.params / 1e6:.2f}M params, {record.gflops:.2f} GFLOPs "
              f"at {record.flops_input[0]}x{record.flops_input[1]} "
              f"(cross-check {record.crosscheck_gflops:.2f})")
    print(write_complexity(records, args.output))
    return 0
