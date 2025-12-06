from __future__ import annotations


import ml_networks as ml
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as vf
import cv2

from .config import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

BRIGHTNESS_MIN = 0.7
BRIGHTNESS_MAX = 1.3
CONTRAST_MIN = 0.4
CONTRAST_MAX = 1.6
GAMMA_MIN = 0.4
GAMMA_MAX = 1.6
GAMMA_MULT = 1.0
HUE_MIN = -0.45
HUE_MAX = 0.45
SATURATION_MIN = 0.4
SATURATION_MAX = 1.9
SHARPNESS_MIN = 0.1
SHARPNESS_MAX = 1.9

ADJUST_METHODS = {
    "brightness": (vf.adjust_brightness, BRIGHTNESS_MIN, BRIGHTNESS_MAX),
    "contrast": (vf.adjust_contrast, CONTRAST_MIN, CONTRAST_MAX),
    "gamma": (vf.adjust_gamma, GAMMA_MIN, GAMMA_MAX),
    "hue": (vf.adjust_hue, HUE_MIN, HUE_MAX),
    "saturation": (vf.adjust_saturation, SATURATION_MIN, SATURATION_MAX),
    "sharpness": (vf.adjust_sharpness, SHARPNESS_MIN, SHARPNESS_MAX),
}

def resize_sequence(sequence: np.ndarray, new_size: int ) -> np.ndarray:
    """
    Resize a sequence of images to the specified size.

    Args:
        sequence (np.ndarray): Image sequence of shape (length, H, W) or (length, H, W, C)
        new_size (tuple[int, int]): Target size (new_H, new_W)

    Returns:
        np.ndarray: Resized image sequence of shape (length, new_H, new_W) or (length, new_H, new_W, C)
    """
    length = sequence.shape[0]
    resized = []

    for i in range(length):
        img = sequence[i]
        # 2D grayscale image
        if img.ndim == 2:
            resized_img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        # 3D color or multi-channel image
        else:
            resized_img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_img)

    return np.stack(resized)

@torch.jit.script
def add_noise(
    input_data: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    max: float = 0.95,
    min: float = -0.95,
    clamp: bool = True,
):
    """データにノイズを加える

    Args:
        input_data (torch.Tensor): 入力データ
        mean (float, optional): ノイズの平均値. Defaults to 0.0.
        std (float, optional): ノイズの標準偏差. Defaults to 0.1.
        max (float, optional): 最大値. Defaults to 0.95.
        min (float, optional): 最小値. Defaults to -0.95.
        clamp (bool, optional): Trueの場合、出力をminとmaxでクリッピングする. Defaults to True.

    Returns:
        torch.Tensor: ノイズを加えた関節データ

    """

    noise = torch.normal(mean, std, size=input_data.shape)

    data = input_data + noise
    if clamp:

        clip_data = torch.clamp(data, min, max)

    else:
        clip_data = data
    return clip_data


def joint_transform(
    data: torch.Tensor,
    max: np.ndarray,
    min: np.ndarray,
    gripper_dim: list = [7, 14],
) -> torch.Tensor:
    """ 関節データを各データ・シーケンス・次元ごとに正規化する

    Args:
        data(np.ndarary):
            (data_num, data_length, data_dim)のデータ
        config (dict): 正規化のための設定
        gripper_dim (list, optional): グリッパーの次元. Defaults to [7, 14].
    """

    if type(data) == torch.Tensor:
        data = data.detach().clone().cpu().numpy()
    *_, dim = data.shape

    for d in range(dim):

        data[..., d] -= min[d]
        data[..., d] /= (max[d] - min[d])
        if d not in gripper_dim:
            data[..., d] *= 2
            data[..., d] -= 1
            data[..., d] *= 0.95
            # assert np.max(data[..., d]) <= 1 and np.min(
            #     data[..., d]) >= -1, f'{d} is not normalized: {np.max(data[..., d])} {np.min(data[..., d])}'

        else:
            # assert np.max(data[..., d]) <= 1 and np.min(
            #     data[..., d]) >= 0, f'{d} is not normalized: {np.max(data[..., d])} {np.min(data[..., d])}'
            pass

    if type(data) == np.ndarray:
        data = torch.from_numpy(data).float()

    return data


def joint_detransform(
    data: torch.Tensor,
    max: np.ndarray,
    min: np.ndarray,
    gripper_dim: list = [7, 14],
):

    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().clone().numpy()

    *_, dim = data.shape

    for d in range(dim):
        if d in gripper_dim:
            data[..., d] *= (max[d] - min[d])
            data[..., d] += min[d]
        else:
            data[..., d] /= 0.95
            data[..., d] += 1
            data[..., d] /= 2
            data[..., d] *= (max[d] - min[d])
            data[..., d] += min[d]

    data = np.clip(data, min, max)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    return data



class CogBotsDataset(LeRobotDataset):
    def __init__(self, cfg: DatasetConfig, *args, train: bool = True):
        delta_timestamps = {
            # Load the previous image and state at -0.1 seconds before current frame,
            # then load current image and state corresponding to 0.0 second.
            # "observation.image": [-0.1, 0.0],
            "observation.images.wrist": [-1/cfg.fps * i for i in range(0, cfg.obs_length)][::-1],
            "observation.state": [-1/cfg.fps * i for i in range(0, cfg.obs_length)][::-1],
            # Load the previous action (-0.1), the next action to be executed (0.0),
            # and 14 future actions with a 0.1 seconds spacing. All these actions will be
            # used to supervise the policy.
            "action": [1/cfg.fps * (i+1) for i in range(cfg.policy_length)],
        }
        print(f"delta_timestamps: {delta_timestamps}")

        super().__init__(cfg.id, delta_timestamps=delta_timestamps)
        self.metadata = LeRobotDatasetMetadata(cfg.id)
        self.cfg = cfg
        self.train = train

    def __getitem__(self, idx: int):
        res = super().__getitem__(idx)
        obs = res["observation.images.wrist"].unsqueeze(0)  # Remove the batch dimension
        action = joint_transform(res["action"], self.metadata.stats["action"]["max"], self.metadata.stats["action"]["min"])
        state = res["observation.state"]
        state = joint_transform(state, self.metadata.stats["observation.state"]["max"], self.metadata.stats["observation.state"]["min"])

        return obs, action, state, res["task_index"]

    def sample_paths(self, n_data: int):
        """Sample n_data random paths from the dataset.
        
        Returns:
            tuple: (rgb, target, pos, goal) tensors
        """
        import random
        indices = random.sample(range(len(self)), min(n_data, len(self)))
        rgb_list = []
        target_list = []
        pos_list = []
        goal_list = []
        
        for idx in indices:
            obs, action, state, goal = self[idx]
            rgb_list.append(obs)
            target_list.append(action)
            pos_list.append(state)
            goal_list.append(goal)
        
        rgb = torch.stack(rgb_list, dim=0)
        target = torch.stack(target_list, dim=0)
        pos = torch.stack(pos_list, dim=0)
        goal = torch.stack(goal_list, dim=0)
        
        return rgb, target, pos, goal

    def sample_img_seq(self, n_data: int, seq_length: int):
        """Sample n_data image sequences of length seq_length from the dataset.
        
        Returns:
            torch.Tensor: Image sequences of shape (n_data, seq_length, C, H, W)
        """
        import random
        indices = random.sample(range(len(self)), min(n_data, len(self)))
        img_seqs = []
        
        for idx in indices:
            obs, _, _, _ = self[idx]
            # obs shape: (1, obs_length, C, H, W) or similar
            # We need to extract a sequence of length seq_length
            if obs.shape[1] >= seq_length:
                seq = obs[0, :seq_length]  # (seq_length, C, H, W)
            else:
                # Pad or repeat if sequence is shorter
                seq = obs[0]
                while seq.shape[0] < seq_length:
                    seq = torch.cat([seq, obs[0]], dim=0)
                seq = seq[:seq_length]
            img_seqs.append(seq)
        
        return torch.stack(img_seqs, dim=0)  # (n_data, seq_length, C, H, W) 



class DatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DatasetConfig,
    ) -> None:
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg

        data_config = LeRobotDatasetMetadata(cfg.id)
        self.train_episodes = None
        self.val_episodes = None
        self.dataset_class = CogBotsDataset
        self.action_mean = (2 * data_config.stats["action"]["mean"] - (data_config.stats["action"]["max"] + data_config.stats["action"]["min"])) \
            / (data_config.stats["action"]["max"] - data_config.stats["action"]["min"])
        self.action_std = (data_config.stats["action"]["std"] * 2) / (data_config.stats["action"]["max"] - data_config.stats["action"]["min"])

    def setup(self, stage: str) -> None:
        # Create train dataset
        train_episodes_list = list(self.cfg.train_episodes) if self.cfg.train_episodes is not None else None
        train_dataset = CogBotsDataset(
            self.cfg,
            train=True,
        )
        # Note: LeRobotDataset supports episodes parameter, but we need to pass it to parent __init__
        # For now, we'll create the dataset and filter later if needed

        # Create val dataset
        val_episodes_list = list(self.cfg.val_episodes) if self.cfg.val_episodes is not None else None
        val_dataset = CogBotsDataset(
            self.cfg,
            train=False,
        )

        # Split dataset if train_episodes/val_episodes are not specified
        if self.cfg.train_episodes is None and self.cfg.val_episodes is None:
            self.train_data, self.val_data = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset)-self.cfg.batch_size, self.cfg.batch_size],
            )
        else:
            # Use separate datasets for train and val
            self.train_data = train_dataset
            self.val_data = val_dataset

        # Set train_episodes and val_episodes for eval.py compatibility
        # These should be the dataset objects themselves (CogBotsDataset instances)
        self.train_episodes = self.train_data
        self.val_episodes = self.val_data




    def train_dataloader(self):
        return ml.determine_loader(
            self.train_data,
            self.cfg.seed,
            self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return ml.determine_loader(
            self.val_data,
            self.cfg.seed,
            self.batch_size,
            shuffle=False)
