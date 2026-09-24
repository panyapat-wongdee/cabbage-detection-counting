"""Train, evaluate, and promote every released run, then build the comparison artifacts."""

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.reproduce import main


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
