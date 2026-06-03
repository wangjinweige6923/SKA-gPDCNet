import argparse
import importlib
import os
import sys

from torch_utils_backbone import BACKBONE_CHOICES, normalize_backbone_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate.py through the backbone-aware wrapper without modifying the original file."
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=BACKBONE_CHOICES,
        help="Torchvision backbone used by the new wrapper pipeline.",
    )
    return parser.parse_known_args()


def main():
    args, passthrough = parse_args()
    os.environ["CLASSIFIER_BACKBONE"] = normalize_backbone_name(args.backbone)

    # Alias the original torch_utils import so evaluate.py can stay unchanged.
    sys.modules.pop("torch_utils", None)
    sys.modules["torch_utils"] = importlib.import_module("torch_utils_backbone")

    old_argv = sys.argv
    try:
        sys.argv = ["evaluate.py", *passthrough]
        baseline_evaluate = importlib.import_module("evaluate")
        baseline_evaluate.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
