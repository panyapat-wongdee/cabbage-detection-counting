"""README figures, regenerated from evaluation outputs rather than drawn by hand.

``comparison`` reads the generated cross-model comparison JSON and the
measured model complexity. ``detection`` overlays one image's counting
outcome for one or more models, using the same confidence filter and
one-to-one matching rule as the counting metrics, so every box colour is the
match the reported TP/FP/FN totals were computed from.

Both figures use a print-style layout: serif type, inward ticks, framework
encoded by marker shape as well as colour so the figure survives greyscale
printing. The chart is also saved as SVG for print use; PDF is not used
because the publication boundary keeps PDFs out of the tree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import load_config
from ..evaluation.confidence import filter_by_confidence
from ..evaluation.io import read_prediction_records, read_target_records
from ..evaluation.targets import load_evaluation_dataset
from ..metrics.counting import match_pairs

# Categorical slots 1-3 of the dataviz reference palette; validated all-pairs
# on a light surface (CVD and normal-vision separation, 3:1 contrast).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
INK = "#1a1a1a"
GRID = "#d9d9d9"

FRAMEWORK_STYLE = {
    "torchvision": {"color": BLUE, "marker": "o"},
    "ultralytics": {"color": ORANGE, "marker": "^"},
}
MODEL_LABELS = {
    "faster_rcnn": "Faster R-CNN",
    "ssd": "SSD",
    "retinanet": "RetinaNet",
    "fcos": "FCOS",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "yolo11n": "YOLO11n",
    "yolo11m": "YOLO11m",
    "rt-detr-l": "RT-DETR-L",
    "ssdlite": "SSDLite",
    "yolo12n": "YOLO12n",
    "yolo12m": "YOLO12m",
    "yolo26n": "YOLO26n",
    "yolo26m": "YOLO26m",
}
# Footnote marks: † trained in the original work but not compared in the
# paper; ‡ added by this repository after the study.
SCOPE_MARKS = {"paper_model": "", "supplementary_unreported": "†", "repository_extension": "‡"}
# Label anchors in data coordinates (GFLOPs, mAP@50:95), placed by eye so no
# two labels collide; a thin leader joins each label to its marker.
COMPUTE_LABEL_POSITIONS = {
    "yolo12n": (1.25, 0.742, "left"),
    "yolo11n": (1.25, 0.718, "left"),
    "yolo26n": (1.25, 0.664, "left"),
    "yolov8n": (1.25, 0.604, "left"),
    "yolo12m": (8.5, 0.764, "left"),
    "yolo26m": (8.5, 0.744, "left"),
    "yolo11m": (8.5, 0.700, "left"),
    "yolov8m": (12, 0.640, "left"),
    "ssd": (20, 0.606, "left"),
    "rt-detr-l": (95, 0.764, "left"),
    "fcos": (420, 0.735, "left"),
    "faster_rcnn": (420, 0.700, "left"),
    "retinanet": (420, 0.668, "left"),
}
COMPUTE_RANGE = {"x": (1.0, 3000), "y": (0.565, 0.775)}
# Label offsets in points for panel (b), chosen by eye.
COUNTING_LABEL_OFFSETS = {
    "faster_rcnn": (-7, 4, "right"),
    "ssd": (7, 0, "left"),
    "retinanet": (7, -7, "left"),
    "fcos": (-7, 5, "right"),
    "yolov8n": (7, -2, "left"),
    "yolov8m": (-7, -2, "right"),
    "yolo11n": (-7, 4, "right"),
    "yolo11m": (-6, -7, "right"),
    "rt-detr-l": (-7, 3, "right"),
    "yolo12n": (6, 6, "left"),
    "yolo12m": (-7, -4, "right"),
    "yolo26m": (6, 5, "left"),
    "yolo26n": (-7, -3, "right"),
}
LEADER = {"arrowstyle": "-", "color": "#8c8c8c", "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    return plt


def _save(figure, output_path: Path, dpi: int = 300, vector: bool = False) -> Path:
    """Save the raster the README embeds, and optionally an SVG beside it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A photograph compresses far better as JPEG than as PNG.
    options = {"pil_kwargs": {"quality": 88}} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03, **options)
    if vector:
        # A fixed hash salt keeps the SVG byte-identical across runs.
        import matplotlib

        matplotlib.rcParams["svg.hashsalt"] = "cabbage-detection-counting"
        # Keep text as text: no font outlines are embedded in the file.
        matplotlib.rcParams["svg.fonttype"] = "none"
        figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03,
                       metadata={"Date": None})
    return output_path


def _label(record: dict) -> str:
    return MODEL_LABELS.get(record["model"], record["model"]) + SCOPE_MARKS[record["study_scope"]]


def _params_area(params_millions: float) -> float:
    """Marker area in points squared, proportional to the parameter count."""
    return 8 + 2.6 * params_millions


def _scatter(axis, record: dict, x: float, y: float, size: float = 34, alpha: float = 1.0) -> None:
    style = FRAMEWORK_STYLE[record["framework"]]
    # Filled: paper model. Hollow: unreported supplement. Half-toned: extension.
    face = {
        "paper_model": style["color"],
        "supplementary_unreported": "white",
        "repository_extension": _tint(style["color"]),
    }[record["study_scope"]]
    from matplotlib.colors import to_rgba

    axis.scatter(
        x, y, s=size, marker=style["marker"], zorder=3, linewidths=0.9,
        facecolors=[to_rgba(face, alpha)], edgecolors=style["color"],
    )


def _tint(color: str, amount: float = 0.6) -> str:
    """Mix ``color`` with white; ``amount`` is the share of white."""
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    mixed = (round(channel + (255 - channel) * amount) for channel in (red, green, blue))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def plot_model_comparison(comparison_path: Path, complexity_path: Path, output_path: Path) -> Path:
    """(a) mAP@50:95 against GFLOPs, bubble area by parameters; (b) mAP@50:95 against counting F1.

    Panel (a) places each model at the GFLOPs of its own run, counted at the
    input its detector actually received, so it matches the results table.
    Bubbles are translucent so that the medium YOLO models, which lie close
    together, stay visible where they overlap.
    """
    import matplotlib.ticker as ticker

    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    # Keyed by run: one model can have runs at two inputs (fcos, fcos-detector512).
    complexity = {
        record["run"]: record
        for record in json.loads(Path(complexity_path).read_text(encoding="utf-8"))["models"]
    }
    models = comparison["models"]
    missing = sorted({record["source"] for record in models} - set(complexity))
    if missing:
        raise ValueError(f"no complexity record for: {', '.join(missing)}")

    plt = _pyplot()
    figure, (compute, counting) = plt.subplots(1, 2, figsize=(7.4, 3.7))

    (x_low, x_high), (y_low, y_high) = COMPUTE_RANGE["x"], COMPUTE_RANGE["y"]
    compute_outliers = []
    for record in models:
        cost = complexity[record["source"]]
        x, y = cost["run_gflops"], record["detection"]["map50_95"]
        if not (x_low <= x <= x_high and y_low <= y <= y_high):
            compute_outliers.append(f"{_label(record)}: ({x:.2f}, {y:.3f})")
            continue
        _scatter(compute, record, x, y, size=_params_area(cost["params"] / 1e6), alpha=0.6)
        label_x, label_y, align = COMPUTE_LABEL_POSITIONS.get(record["model"], (x * 1.3, y, "left"))
        # Start the leader at the label edge nearer the marker, never across the text.
        leader = {**LEADER, "relpos": (1.0, 0.5) if label_x < x else (0.0, 0.5)}
        compute.annotate(_label(record), (x, y), xytext=(label_x, label_y), textcoords="data",
                         ha=align, va="center", fontsize=7.5, arrowprops=leader)
    compute.set_xscale("log")
    compute.set_xlim(x_low, x_high)
    compute.set_ylim(y_low, y_high)
    compute.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    compute.set_xlabel("GFLOPs at the model's detector input (log scale)")
    compute.set_ylabel("mAP@50:95")
    compute.set_title("(a) Detection accuracy vs. computation", loc="center", pad=7)
    if compute_outliers:
        compute.text(0.02, 0.03, "Outside range:\n" + "\n".join(compute_outliers),
                     transform=compute.transAxes, ha="left", va="bottom", fontsize=6.8,
                     style="italic")
    for params in (5, 20, 40):
        compute.scatter([], [], s=_params_area(params), marker="o", facecolors="none",
                        edgecolors="#7a7a7a", linewidths=0.8, label=f"{params}M")
    compute.legend(title="Parameters", loc="lower right", fontsize=7, title_fontsize=7.5,
                   labelspacing=1.0, borderpad=0.6, handletextpad=0.6)

    # The supplementary SSDLite row lies far below the others on both axes;
    # the panel is zoomed on the nine paper models and SSDLite is annotated.
    x_range, y_range = (0.575, 0.795), (0.915, 0.96)
    outliers = []
    for record in models:
        x, y = record["detection"]["map50_95"], record["counting"]["f1"]
        if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            outliers.append(f"{_label(record)}: ({x:.3f}, {y:.3f})")
            continue
        _scatter(counting, record, x, y)
        dx, dy, align = COUNTING_LABEL_OFFSETS.get(record["model"], (7, 0, "left"))
        counting.annotate(_label(record), (x, y), xytext=(dx, dy), textcoords="offset points",
                          ha=align, va="center", fontsize=7.5)
    counting.set_xlim(*x_range)
    counting.set_ylim(*y_range)
    counting.set_xlabel("mAP@50:95")
    counting.set_ylabel("Counting F1 (IoU 0.5, conf. 0.5)")
    counting.set_title("(b) Detection vs. counting", loc="center", pad=7)
    if outliers:
        counting.text(0.03, 0.97, "Outside range:\n" + "\n".join(outliers), transform=counting.transAxes,
                      ha="left", va="top", fontsize=6.8, style="italic")

    for axis in (compute, counting):
        axis.grid(True, which="major", color=GRID, linewidth=0.5, linestyle=(0, (4, 3)))
        axis.set_axisbelow(True)
        axis.minorticks_on()

    frameworks = [
        plt.Line2D([], [], marker=style["marker"], linestyle="none", markersize=6,
                   markerfacecolor=style["color"], markeredgecolor=style["color"], label=framework)
        for framework, style in FRAMEWORK_STYLE.items()
    ]
    scopes = {record["study_scope"] for record in models}
    if "supplementary_unreported" in scopes:
        frameworks.append(plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                                     markerfacecolor="white", markeredgecolor=BLUE,
                                     label="not compared in the paper (†)"))
    if "repository_extension" in scopes:
        frameworks.append(plt.Line2D([], [], marker="^", linestyle="none", markersize=6,
                                     markerfacecolor=_tint(ORANGE), markeredgecolor=ORANGE,
                                     label="added after the study (‡)"))
    figure.legend(handles=frameworks, loc="lower center", ncol=len(frameworks), fontsize=8,
                  bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.6)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.18, wspace=0.28)
    output = _save(figure, output_path, vector=True)
    plt.close(figure)
    return output


@dataclass(frozen=True)
class OverlayCounts:
    model: str
    true_positives: int
    false_positives: int
    false_negatives: int


def _overlay_panel(axis, run_dir: Path, records_dir: Path, dataset, image_id: str, image) -> OverlayCounts:
    from matplotlib.patches import Rectangle

    config = load_config(Path(run_dir) / "config.yaml")
    predictions = {
        record.image_id: record
        for record in filter_by_confidence(
            read_prediction_records(Path(records_dir) / "predictions.json"),
            config.confidence_threshold,
        )
    }
    targets = {record.image_id: record for record in read_target_records(Path(records_dir) / "targets.json")}
    if image_id not in predictions or image_id not in targets:
        raise ValueError(f"image_id {image_id!r} is not in {records_dir}")
    detections = predictions[image_id].detections
    boxes = targets[image_id].boxes
    pairs = match_pairs(detections, boxes, config.counting_iou_threshold)
    matched_predictions = {prediction for prediction, _ in pairs}
    matched_targets = {target for _, target in pairs}

    # torchvision records live in the resized image_size space; Ultralytics
    # records are already original pixels (see evaluation.targets).
    if config.framework == "torchvision":
        scale_x = image.width / config.image_size[1]
        scale_y = image.height / config.image_size[0]
    else:
        scale_x = scale_y = 1.0

    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])

    def draw(coordinates, color, dashed):
        x1, y1, x2, y2 = coordinates
        geometry = (x1 * scale_x, y1 * scale_y), (x2 - x1) * scale_x, (y2 - y1) * scale_y
        # A thin white halo lifts each box off the photograph.
        axis.add_patch(Rectangle(*geometry, fill=False, edgecolor="white", linewidth=2.4, alpha=0.85))
        axis.add_patch(Rectangle(*geometry, fill=False, edgecolor=color, linewidth=1.3,
                                 linestyle=(0, (3, 1.5)) if dashed else "solid"))

    for index, detection in enumerate(detections):
        draw(detection.box, BLUE if index in matched_predictions else ORANGE, dashed=False)
    for index, box in enumerate(boxes):
        if index not in matched_targets:
            draw(box.coordinates, VIOLET, dashed=True)
    return OverlayCounts(
        model=config.model_name,
        true_positives=len(pairs),
        false_positives=len(detections) - len(pairs),
        false_negatives=len(boxes) - len(pairs),
    )


def plot_counting_overlay(
    runs: Sequence[tuple[Path, Path]],
    dataset_root: Path,
    image_id: str,
    output_path: Path,
) -> list[OverlayCounts]:
    """Draw one image's TP/FP/FN boxes per model, exactly as the counting metric matched them.

    ``runs`` holds ``(run directory, evaluation records directory)`` pairs,
    one panel each, left to right.
    """
    from PIL import Image

    dataset = load_evaluation_dataset(Path(dataset_root))
    with Image.open(dataset.image_path(image_id)) as source:
        image = source.convert("RGB")

    plt = _pyplot()
    figure, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 4.25), squeeze=False)
    counts = []
    for index, (axis, (run_dir, records_dir)) in enumerate(zip(axes[0], runs)):
        result = _overlay_panel(axis, run_dir, records_dir, dataset, image_id, image)
        counts.append(result)
        predicted = result.true_positives + result.false_positives
        annotated = result.true_positives + result.false_negatives
        # Subfigure caption, centred under the image as in a paper figure.
        axis.text(0.5, -0.035, f"({chr(ord('a') + index)}) {MODEL_LABELS.get(result.model, result.model)}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=10)
        axis.text(
            0.5, -0.115,
            f"TP {result.true_positives}   FP {result.false_positives}   FN {result.false_negatives}"
            f"   ·   {predicted} predicted, {annotated} annotated",
            transform=axis.transAxes, ha="center", va="top", fontsize=8.2, color="#404040",
        )
    handles = [
        plt.Line2D([], [], color=BLUE, linewidth=1.8, label="true positive (TP)"),
        plt.Line2D([], [], color=ORANGE, linewidth=1.8, label="false positive (FP)"),
        plt.Line2D([], [], color=VIOLET, linewidth=1.8, linestyle=(0, (3, 1.5)),
                   label="missed cabbage (FN)"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.2,
                  bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=2.2)
    figure.tight_layout(rect=(0, 0.1, 1, 0.94), w_pad=1.2)
    _save(figure, output_path, dpi=200)
    plt.close(figure)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the README figures from evaluation outputs.")
    commands = parser.add_subparsers(dest="command", required=True)

    comparison = commands.add_parser("comparison", help="accuracy vs. computation and counting")
    comparison.add_argument("--comparison", type=Path, default=Path("results/reproduced/model_comparison_test.json"))
    comparison.add_argument("--complexity", type=Path, default=Path("results/reproduced/model_complexity.json"))
    comparison.add_argument("--output", type=Path, default=Path("docs/figures/model_comparison_test.png"))

    detection = commands.add_parser("detection", help="TP/FP/FN overlay for one evaluated image")
    detection.add_argument("--run", type=Path, action="append", required=True,
                           help="training run directory; repeat for one panel per model")
    detection.add_argument("--records", type=Path, action="append", required=True,
                           help="evaluation records/ directory, one per --run, in the same order")
    detection.add_argument("--dataset", type=Path, required=True, help="extracted official dataset root")
    detection.add_argument("--image-id", required=True)
    detection.add_argument("--output", type=Path, default=Path("docs/figures/counting_example.jpg"))

    args = parser.parse_args(argv)
    if args.command == "comparison":
        print(plot_model_comparison(args.comparison, args.complexity, args.output))
        return 0
    if len(args.run) != len(args.records):
        parser.error("give one --records directory per --run")
    counts = plot_counting_overlay(list(zip(args.run, args.records)), args.dataset, args.image_id, args.output)
    print(json.dumps({"output": str(args.output), "panels": [count.__dict__ for count in counts]}))
    return 0
