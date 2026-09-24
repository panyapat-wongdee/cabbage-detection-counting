"""Command-line entry point for assembling a run's promotion evidence."""

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["collect-evidence", *__import__("sys").argv[1:]]))
