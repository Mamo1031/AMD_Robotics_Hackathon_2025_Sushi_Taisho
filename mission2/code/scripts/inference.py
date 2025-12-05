"""Inference script placeholder for Sushi-Bot.

This module provides a simple CLI entry point for running inference on the robot.
The actual inference logic will be implemented during the AMD Open Robotics Hackathon.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run Sushi-Bot inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/inference_config.yaml"),
        help="Path to inference config YAML file.",
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
    return parser.parse_args()


def main() -> None:
    """Entry point for running inference on the robot via CLI.

    NOTE:
        - This function currently only parses CLI arguments and raises
          NotImplementedError.
        - The actual inference pipeline (camera, LeRobot policy, robot control, etc.)
          will be implemented during the hackathon.
    """
    args = parse_args()
    print(f"[inference] Using config: {args.config}")
    print(f"[inference] Mode: {args.mode}")
    if args.mode == "vla":
        print(f"[inference] Instruction: {args.instruction}")

    raise NotImplementedError("Inference pipeline will be implemented during the hackathon.")


if __name__ == "__main__":
    main()
