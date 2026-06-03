import argparse
import importlib
import os
import sys

from torch_utils_backbone import BACKBONE_CHOICES, BACKBONE_MODEL_LABELS, normalize_backbone_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run aggregate_results.py through the backbone-aware wrapper without modifying the original file."
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=BACKBONE_CHOICES,
        help="Torchvision backbone used by the new wrapper pipeline.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Optional display label written into the SCI main table. Defaults to the backbone display name.",
    )
    return parser.parse_known_args()


def main():
    args, passthrough = parse_args()
    canonical_backbone = normalize_backbone_name(args.backbone)
    os.environ["CLASSIFIER_BACKBONE"] = canonical_backbone

    # Alias the original torch_utils import so aggregate_results.py can profile the new checkpoints unchanged.
    sys.modules.pop("torch_utils", None)
    sys.modules["torch_utils"] = importlib.import_module("torch_utils_backbone")

    model_label = args.model_label or BACKBONE_MODEL_LABELS[canonical_backbone]

    old_argv = sys.argv
    try:
        sys.argv = ["aggregate_results.py", *passthrough, "--model-label", model_label]
        aggregate_module = importlib.import_module("aggregate_results")
        aggregate_module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
