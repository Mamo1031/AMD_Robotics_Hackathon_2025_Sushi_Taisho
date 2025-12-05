"""Command-line interface entry points for Sushi-Bot.

These are minimal CLIs for:
- training: ``train``
- inference: ``infer``

The actual logic will be implemented during the AMD Open Robotics Hackathon.
For now, they just parse arguments and raise ``NotImplementedError`` so that
the CLI interface itself can be tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sushi-Bot model.")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("config/training_config.yaml"),
        help="Path to training config YAML file (default: config/training_config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use (e.g., "cuda", "cuda:0", "cpu").',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser


def _build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sushi-Bot inference.")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("config/inference_config.yaml"),
        help="Path to inference config YAML file (default: config/inference_config.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["grasp", "vla"],
        default="grasp",
        help='Inference mode: "grasp" for basic picking, "vla" for VLA-based control.',
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help='Natural language instruction (used when mode="vla").',
    )
    return parser


def train_main() -> None:
    """Entry point for training CLI (``train``)."""
    parser = _build_train_parser()
    args = parser.parse_args()
    print(f"[train] config: {args.config}")
    print(f"[train] device: {args.device}")
    print(f"[train] seed: {args.seed}")
    raise NotImplementedError("Training pipeline will be implemented during the hackathon.")


def inference_main() -> None:
    """Entry point for inference CLI (``infer``)."""
    parser = _build_infer_parser()
    args = parser.parse_args()
    print(f"[infer] config: {args.config}")
    print(f"[infer] mode: {args.mode}")
    if args.mode == "vla":
        print(f"[infer] instruction: {args.instruction}")
    raise NotImplementedError("Inference pipeline will be implemented during the hackathon.")
