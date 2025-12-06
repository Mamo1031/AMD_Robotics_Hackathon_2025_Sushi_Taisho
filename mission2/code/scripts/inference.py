"""Inference script placeholder for Sushi-Bot.

This module provides a simple CLI entry point for running inference on the robot.
The actual inference logic will be implemented during the AMD Open Robotics Hackathon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os
import time
import numpy
import torch

from rich.progress import track
from hydra.utils import instantiate
from ml_networks import torch_fix_seed
from typing import Optional, List
from omegaconf import OmegaConf
from mission2.config import ExperimentConfig, EnergyMatchingConfig, StreamingFlowMatchingConfig
from mission2.utils import visualize_attention_video
from mission2.policy import Policy, EMPolicy, StreamingPolicy
from mission2.dataset import joint_transform, joint_detransform
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


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


    for t in range(max_timesteps):
        start_loop_t = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        action_values = act_processed_policy
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)


if __name__ == "__main__":
    main()
