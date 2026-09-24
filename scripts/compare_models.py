"""Command-line entry point for the cross-model reproduced comparison table."""

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["compare-models", *__import__("sys").argv[1:]]))
