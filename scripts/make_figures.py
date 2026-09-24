"""Command-line entry point for the README figures."""

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.visualization.figures import main


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
