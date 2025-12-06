"""Inference script placeholder for Sushi-Bot.

This module provides a simple CLI entry point for running inference on the robot.
The actual inference logic will be implemented during the AMD Open Robotics Hackathon.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy
import torch
from hydra.utils import instantiate
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    RobotAction,
    make_default_processors,
)
from lerobot.configs.parser import get_cli_overrides
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from ml_networks import torch_fix_seed
from omegaconf import OmegaConf
from rich.progress import track

from mission2 import (
    ExperimentConfig,
    Policy,
    StreamingFlowMatchingConfig,
    StreamingPolicy,
    joint_transform,
    visualize_attention_video,
)


def evaluation(
    model_name: str,
    seed: int,
    device: int,
    dataset: str,
    adjusting_methods: list | None = None,
    load_last: bool = False,
    inference_every: int = 16,
    max_steps: int = 300,
    n_candidates: int = 1,
    robot_config: RobotConfig | None = None,
):
    """
    This script evaluates a pre-trained diffusion policy.
    It renders the environment and saves the frames of the evaluation as a video.
    """
    conf_path = f"models/cfg/{model_name}.yaml"

    config = OmegaConf.load(conf_path)
    config.seed = seed
    config.device = device
    config.datamodule.id = dataset

    data_config = LeRobotDatasetMetadata(config.datamodule.id)
    config.action_dim = data_config.features["action"]["shape"][0]
    config.obs_shape = list(data_config.features["observation.images.main"]["shape"])[::-1]

    torch_fix_seed(config.seed)

    OmegaConf.resolve(config)
    config: ExperimentConfig = instantiate(
        config,
    )

    path = f"models/params/{config.datamodule.id}/{model_name}/seed:{config.seed}/"
    if adjusting_methods is not None:
        path += "_".join(adjusting_methods) + "/"
    log_path = f"reports/{config.datamodule.id}/{model_name}/seed:{config.seed}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    if isinstance(config.policy.framework_cfg, StreamingFlowMatchingConfig):
        model = StreamingPolicy(config.policy)
    else:
        model = Policy(data_config.features["task_index"]["shape"], config.policy)
    model.eval()
    model.freeze()

    inference_every = min(inference_every, config.policy.policy_length)

    if load_last:
        param = torch.load(f"{path}/last.ckpt", map_location="cpu")["state_dict"]
    else:
        param = torch.load(f"{path}/model.ckpt", map_location="cpu")["state_dict"]

    model.load_state_dict(param)
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} parameter"
        assert not torch.isinf(param).any(), f"Inf in {name} parameter"

    # Create a directory to store the video of the evaluation
    output_directory = Path(f"reports/{model_name}/seed:{seed}/")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize processors (following lerobot_record.py pattern)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Create dataset features for build_dataset_frame
    dataset_features = data_config.features

    # Initialize robot (following lerobot_record.py pattern)
    if robot_config is None:
        raise ValueError(
            "Robot config is required for evaluation. "
            "Please provide robot_config argument or set it in the config file."
        )

    robot = make_robot_from_config(robot_config)
    robot.connect()

    # Initialize tracking variables
    attentions = []
    rewards = []
    frames = []

    # Get initial observation from robot
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

    # Extract state and image from observation frame
    if OBS_STATE not in observation_frame:
        raise ValueError(
            f"observation.state not found in observation frame. Available keys: {list(observation_frame.keys())}"
        )

    # Get the first image key (assuming there's at least one camera)
    image_keys = [k for k in observation_frame.keys() if k.startswith(OBS_IMAGES + ".")]
    if not image_keys:
        raise ValueError(
            f"No image keys found in observation frame. Available keys: {list(observation_frame.keys())}"
        )

    # Use the first image key
    image_key = image_keys[0]

    # Prepare initial state and image for the policy
    state_np = observation_frame[OBS_STATE]
    if isinstance(state_np, torch.Tensor):
        state_np = state_np.cpu().numpy()
    state = torch.from_numpy(state_np).float()

    image_np = observation_frame[image_key]
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.cpu().numpy()
    # Convert image from (C, H, W) to (H, W, C) if needed, then back to (C, H, W)
    if len(image_np.shape) == 3 and image_np.shape[0] == 3:
        # Already in (C, H, W) format
        image = torch.from_numpy(image_np).float()
    else:
        # Assume (H, W, C) format
        image = torch.from_numpy(image_np).float()
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)

    # Normalize image to [0, 1] if it's in [0, 255]
    if image.max() > 1.0:
        image = image / 255.0

    # Add batch dimension
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Transform state using joint_transform
    state = joint_transform(
        state,
        data_config.stats["observation.state"]["max"],
        data_config.stats["observation.state"]["min"],
    )

    # Initialize action from current state (as initial action)
    action = state.clone()

    step = 0
    done = False

    fps = 30  # Default FPS for timing control

    for t in track(range(max_steps)):
        start_loop_t = time.perf_counter()

        # Get robot observation (following main() pattern)
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        # Build dataset frame (following main() pattern)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        # Extract state and image from observation frame
        state_np = observation_frame[OBS_STATE]
        if isinstance(state_np, torch.Tensor):
            state_np = state_np.cpu().numpy()
        state = torch.from_numpy(state_np).float()

        # Get the first image key (assuming there's at least one camera)
        image_keys = [k for k in observation_frame.keys() if k.startswith(OBS_IMAGES + ".")]
        if not image_keys:
            raise ValueError(
                f"No image keys found in observation frame. Available keys: {list(observation_frame.keys())}"
            )
        image_key = image_keys[0]

        image_np = observation_frame[image_key]
        if isinstance(image_np, torch.Tensor):
            image_np = image_np.cpu().numpy()
        # Convert image from (C, H, W) to (H, W, C) if needed, then back to (C, H, W)
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            # Already in (C, H, W) format
            image = torch.from_numpy(image_np).float()
        else:
            # Assume (H, W, C) format
            image = torch.from_numpy(image_np).float()
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)

        # Normalize image to [0, 1] if it's in [0, 255]
        if image.max() > 1.0:
            image = image / 255.0

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        state = joint_transform(
            state,
            data_config.stats["observation.state"]["max"],
            data_config.stats["observation.state"]["min"],
        )
        image = image.unsqueeze(0)

        obs_embed = model.encoder(image)
        pos_embed = model.pos_encoder(state)
        attentions.append(obs_embed)
        if step % inference_every == 0:
            action_chunk = model.inference(
                batch_size=n_candidates, obs=obs_embed, pos=pos_embed, initial_action=action
            )
            action_chunk = action_chunk.mean(dim=0)

        action = action_chunk[step % inference_every]

        # Convert action to RobotAction format (following main() pattern)
        # make_robot_action expects a tensor, so we pass the action tensor directly
        act_processed_policy: RobotAction = make_robot_action(action, dataset_features)

        # Applies a pipeline to the action (following main() pattern)
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # Send action to robot (following lerobot_record.py pattern)
        _sent_action = robot.send_action(robot_action_to_send)

        # Keep track of frames (extract from current observation)
        # Get image from observation_frame for visualization
        image_keys = [k for k in observation_frame.keys() if k.startswith(OBS_IMAGES + ".")]
        if image_keys:
            image_key = image_keys[0]
            frame_np = observation_frame[image_key]
            if isinstance(frame_np, torch.Tensor):
                frame_np = frame_np.cpu().numpy()
            # Convert to (H, W, C) format for visualization
            if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                frame_np = frame_np.transpose(1, 2, 0)
            frames.append(frame_np.copy())

        # Note: For real robot evaluation, we don't have rewards or termination signals
        # These would need to be determined by task-specific logic
        reward = 0.0  # Placeholder - should be determined by task-specific logic
        rewards.append(reward)

        # The rollout is considered done when the maximum number of iterations is reached
        # For real robot evaluation, termination would need to be determined by task-specific logic
        step += 1
        if step >= max_steps:
            done = True

        # Timing control (following main() pattern)
        dt_s = time.perf_counter() - start_loop_t
        # precise_sleep(1 / fps - dt_s)
        precise_sleep(1)
        if done:
            break

    # Disconnect robot
    robot.disconnect()

    print("Evaluation completed!")

    rewards = numpy.array(rewards)
    print(f"Total reward: {rewards.sum()}")

    frames = numpy.array(frames)
    print(f"Total frames: {frames.shape}")

    attentions = torch.stack(attentions, dim=0).squeeze(1).cpu().numpy()
    print(f"Total attentions: {attentions.shape}")
    visualize_attention_video(
        frames[None, ...],
        attentions[None, ...],
        save_path=output_directory,
    )


def parse_robot_config_from_cli() -> RobotConfig | None:
    """
    Parse robot config from CLI arguments using the same method as lerobot_record.py.
    
    This function uses get_cli_overrides("robot") to extract --robot.* arguments
    and parses them using draccus, similar to how @parser.wrap() works in lerobot_record.py.
    
    Returns:
        RobotConfig instance if --robot.* arguments are provided, None otherwise.
        
    Example CLI usage:
        --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_robot
    """
    import draccus
    
    try:
        cli_overrides = get_cli_overrides("robot")
        if cli_overrides:
            robot_config = draccus.parse(config_class=RobotConfig, args=cli_overrides)
            logging.info(f"robot_config parsed from CLI: {robot_config}")
            return robot_config
        else:
            return None
    except Exception as e:
        logging.warning(
            f"Failed to parse robot config from CLI using --robot.* arguments: {e}. "
            "You can provide robot config via --robot-config file or individual --robot-* arguments."
        )
        return None


if __name__ == "__main__":
    # Run evaluation mode
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained diffusion policy.")
    parser.add_argument("--model", type=str, help="model name in models/", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--load_last", action="store_true", help="load last model")
    parser.add_argument(
        "--inference_every",
        type=int,
        default=8,
        help="Number of steps between policy inferences",
    )
    parser.add_argument(
        "--max-steps",
        "-n",
        type=int,
        default=300,
        help="number of episodes to evaluate",
    )
    parser.add_argument(
        "--n_candidates",
        "-nc",
        type=int,
        default=1,
        help="number of candidates to sample from the model",
    )
    parser.add_argument(
        "--adjusting_methods",
        "-am",
        type=str,
        nargs="+",
        default=None,
        help="list of adjusting methods",
    )
    
    # Robot config options: either use a config file or specify individual parameters
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="Path to robot config YAML/JSON file. If not provided, use --robot-type, --robot-port, etc.",
    )

    args = parser.parse_args()
    print("=======================")
    print(args.dataset)
    print("=======================")

    
    import draccus
    from lerobot.cameras import CameraConfig
    from pathlib import Path

    try:
        if args.robot_config:
            # Load robot config from file
            print("Loading robot_config from file:", args.robot_config)
            config = OmegaConf.load(args.robot_config)
            robot_config = instantiate(config, _recursive_=True, _convert_="object")
            print(f"robot_config loaded from file: {robot_config}")
        else:
            raise ValueError(
                "Either --robot.* arguments (e.g., --robot.type=so101_follower --robot.port=/dev/ttyACM1), "
                "--robot-config file, or both --robot-type and --robot-port must be provided."
            )
    except Exception as e:
        logging.error(
            f"Failed to create robot config: {e}. "
            "Please provide either --robot.* arguments (like lerobot_record.py), "
            "--robot-config file, or --robot-type and --robot-port."
        )
        raise
    

    print("args.model", args.model)
    print("args.dataset", args.dataset)
    evaluation(
        model_name=args.model,
        seed=args.seed,
        device=args.device,
        dataset=args.dataset,
        adjusting_methods=args.adjusting_methods,
        load_last=args.load_last,
        inference_every=args.inference_every,
        max_steps=args.max_steps,
        n_candidates=args.n_candidates,
        robot_config=robot_config,
    )
