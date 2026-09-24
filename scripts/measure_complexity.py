"""Command-line entry point for parameter and FLOP counts of trained runs."""

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.models.complexity import main


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
