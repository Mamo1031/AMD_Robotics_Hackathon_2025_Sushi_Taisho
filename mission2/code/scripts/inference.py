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
import tempfile
from typing import Literal

import numpy
import torch
from hydra.utils import instantiate
import wandb
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
    task: Literal["Salmon", "Tuna", "Egg"] = "Salmon"
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
    config.obs_shape = list(data_config.features["observation.images.wrist"]["shape"])[::-1]

    torch_fix_seed(config.seed)

    OmegaConf.resolve(config)
    config: ExperimentConfig = instantiate(
        config,
    )
    
    task_index = {"Salmon": 0, "Tuna": 1, "Egg": 2}[task]

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

    # wandbからモデルを読み込むか、ローカルから読み込む
    param = None
    if config.run_path is not None:
        try:
            # wandbからモデルを読み込む
            print(f"Attempting to load model from wandb run_path: {config.run_path}")
            api = wandb.Api()
            run = api.run(config.run_path)
            
            # artifactを検索（最新のmodel artifactを取得）
            artifacts = list(run.logged_artifacts())
            model_artifact = None
            for artifact in artifacts:
                if artifact.type == "model":
                    model_artifact = artifact
                    # 最新のartifactを使用（通常は最後にログされたもの）
                    break
            
            # 見つからない場合は、artifact名で検索
            if model_artifact is None:
                # artifact名のパターンで検索（train.pyで保存した形式: model-{run_id}）
                artifact_name_pattern = f"model-{run.id}"
                try:
                    # entity/project/artifact_name:version の形式で取得
                    entity, project, run_id = config.run_path.split("/")
                    artifact_name = f"{entity}/{project}/{artifact_name_pattern}:latest"
                    model_artifact = api.artifact(artifact_name)
                except Exception:
                    pass
            
            if model_artifact is not None:
                # artifactをダウンロード
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_dir = model_artifact.download(root=tmpdir)
                    artifact_path = Path(artifact_dir)
                    
                    # モデルファイルを探す
                    if load_last:
                        model_file = artifact_path / "last.ckpt"
                    else:
                        model_file = artifact_path / "model.ckpt"
                    
                    if model_file.exists():
                        param = torch.load(str(model_file), map_location="cpu")["state_dict"]
                        print(f"Successfully loaded model from wandb: {model_file}")
                    else:
                        print(f"Warning: Model file not found in artifact: {model_file}")
            else:
                print("Warning: No model artifact found in wandb run")
        except Exception as e:
            print(f"Warning: Failed to load model from wandb: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to local model")
    
    # wandbからの読み込みに失敗した場合、ローカルから読み込む
    if param is None:
        if load_last:
            param_path = f"{path}/last.ckpt"
        else:
            param_path = f"{path}/model.ckpt"
        
        if os.path.exists(param_path):
            param = torch.load(param_path, map_location="cpu")["state_dict"]
            print(f"Loaded model from local path: {param_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found at {param_path}. "
                f"Please ensure the model has been trained or provide a valid run_path in config."
            )

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

    # Get obs_length from config (same as dataset)
    obs_length = config.datamodule.obs_length

    # Initialize image and state sequence buffers (FIFO queue)
    # These will maintain obs_length frames of history
    image_sequence_buffer = []
    state_sequence_buffer = []

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

    # Helper function to process a single observation
    def process_observation(observation_frame):
        """Process a single observation frame and return image and state tensors."""
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

        return image, state

    # Initialize the sequence buffers with the first observation (repeat it obs_length times)
    image, state = process_observation(observation_frame)
    for _ in range(obs_length):
        image_sequence_buffer.append(image.clone())
        state_sequence_buffer.append(state.clone())

    # Transform state sequence using joint_transform (same as dataset)
    # Stack states: (obs_length, state_dim)
    state_sequence = torch.stack(state_sequence_buffer, dim=0)
    state_sequence = joint_transform(
        state_sequence,
        data_config.stats["observation.state"]["max"],
        data_config.stats["observation.state"]["min"],
    )

    # Stack images: (obs_length, C, H, W)
    image_sequence = torch.stack(image_sequence_buffer, dim=0)

    # Add batch dimension to match dataset format: (1, obs_length, C, H, W) and (1, obs_length, state_dim)
    image_sequence = image_sequence.unsqueeze(0)  # (1, obs_length, C, H, W)
    state_sequence = state_sequence.unsqueeze(0)  # (1, obs_length, state_dim)

    # Encode the sequences (same as dataset)
    obs_embed = model.encoder(image_sequence)  # (1, obs_length, embed_dim)
    pos_embed = model.pos_encoder(state_sequence)  # (1, obs_length, embed_dim)

    # Initialize action from current state (as initial action)
    # Use the last state in the sequence: state_sequence is (1, obs_length, state_dim)
    action = state_sequence[0, -1].clone()  # (state_dim,)

    step = 0
    done = False

    fps = 30  # Default FPS for timing control

    goal = torch.tensor([task_index], dtype=torch.long, device=f"{model.device}")
    if model.cfg.goal_conditioned:
        goal_emb = model.goal_encoder(goal).reshape(1, 1, -1)  # (1, 1, embed_dim)
    else:
        goal_emb = None
    for t in track(range(max_steps)):
        start_loop_t = time.perf_counter()

        # Get robot observation (following main() pattern)
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        # Build dataset frame (following main() pattern)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        # Process the new observation
        image, state = process_observation(observation_frame)

        # Update sequence buffers (FIFO queue)
        # Remove the oldest observation and add the new one
        image_sequence_buffer.pop(0)
        image_sequence_buffer.append(image.clone())
        state_sequence_buffer.pop(0)
        state_sequence_buffer.append(state.clone())

        # Stack sequences to match dataset format
        # Stack states: (obs_length, state_dim)
        state_sequence = torch.stack(state_sequence_buffer, dim=0)
        state_sequence = joint_transform(
            state_sequence,
            data_config.stats["observation.state"]["max"],
            data_config.stats["observation.state"]["min"],
        )

        # Stack images: (obs_length, C, H, W)
        image_sequence = torch.stack(image_sequence_buffer, dim=0)

        # Add batch dimension to match dataset format: (1, obs_length, C, H, W) and (1, obs_length, state_dim)
        image_sequence = image_sequence.unsqueeze(0)  # (1, obs_length, C, H, W)
        state_sequence = state_sequence.unsqueeze(0)  # (1, obs_length, state_dim)

        # Encode the sequences (same as dataset)
        obs_embed = model.encoder(image_sequence)  # (1, obs_length, embed_dim)
        pos_embed = model.pos_encoder(state_sequence)  # (1, obs_length, embed_dim)

        attentions.append(obs_embed)
        if step % inference_every == 0:
            action_chunk = model.inference(
                batch_size=n_candidates, obs=obs_embed, pos=pos_embed, goal=goal_emb, initial_action=action
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
