import os
import warnings
import torch
from dataclasses import asdict
from typing import Optional, List

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
from .dataset import DatasetModule
from .utils import visualize_attention_video, visualize_joint_prediction
from .policy import Policy
from .config import ExperimentConfig, TrainerConfig
from ml_networks import torch_fix_seed
import logging
class SuppressSeedMessage(logging.Filter):
    def filter(self, record):
        return "Seed set to" not in record.getMessage()

# すべてのロガーにフィルターを追加
for handler in logging.getLogger().handlers:
    handler.addFilter(SuppressSeedMessage())

# もしくは root logger に直接
logging.getLogger().addFilter(SuppressSeedMessage())

logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.ERROR)
wandb_logger = logging.getLogger("wandb")
wandb_logger.setLevel(logging.ERROR)  # または CRITICAL
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.")
warnings.filterwarnings("ignore", ".*Input tensor has dimensions outside of.*")
warnings.filterwarnings("ignore", ".*torch.load.*")
warnings.filterwarnings(
    "ignore", ".*I found a path object that I don't think is part of a bar chart.*"
)
warnings.filterwarnings("ignore", ".*is not.*")
warnings.filterwarnings("ignore", ".*Tight.*")


@torch.no_grad()
def eval(
    model_name: str,
    seed: int,
    device: int,
    n_data: int = 4,
    n_samples: int = 3,
    nfe: int = 50,
    load_last: bool = False,
    dataset: str = "lerobot-rope",
    adjusting_methods: Optional[List] = None,
):

    conf_path = f"models/cfg/{model_name}.yaml"

    config = OmegaConf.load(conf_path)
    config.seed = seed
    config.device = device
    config.datamodule.env = dataset

    data_cfg = OmegaConf.load(f"datasets/{config.datamodule.env}/config.yaml")
    config.action_dim = data_cfg.action_dim
    config.obs_shape = data_cfg.obs_shape
    config.datamodule.train_episodes = data_cfg.train_episodes
    config.datamodule.val_episodes = data_cfg.val_episodes

    torch_fix_seed(config.seed)

    OmegaConf.resolve(config)
    config: ExperimentConfig = instantiate(
        config,
    )

    datamodule = DatasetModule(
        config.datamodule
            )

    path = f"models/params/{config.datamodule.env}/{model_name}/seed:{config.seed}/"
    if adjusting_methods is not None:
        path += "_".join(adjusting_methods) + "/"
    log_path = f"reports/{config.datamodule.env}/{model_name}/seed:{config.seed}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    model = Policy(config.policy)

    if load_last:
        param = torch.load(
                f"{path}/last.ckpt",
                map_location="cpu")["state_dict"]
    else:
        param = torch.load(
                f"{path}/model.ckpt",
                map_location="cpu")["state_dict"]

    model.load_state_dict(param)
    model.eval()

    model.freeze()

    train_rgb, train_target, train_pos, train_goal = datamodule.train_episodes.sample_paths(n_data)

    if not config.policy.pos_only:
        obs_emb = model.encoder(train_rgb)
        goal_emb = model.encoder(train_goal)
    else:
        obs_emb = None
        goal_emb = None
    if not config.policy.obs_only:
        pos_emb = model.pos_encoder(train_pos)
    else:
        pos_emb = None


    val_rgb, val_target, val_pos, val_goal = datamodule.val_episodes.sample_paths(n_data)
    if not config.policy.pos_only:
        val_obs_emb = model.encoder(val_rgb)
        val_goal_emb = model.encoder(val_goal)
    else:
        val_obs_emb = None
        val_goal_emb = None
    if not config.policy.obs_only:
        val_pos_emb = model.pos_encoder(val_pos)
    else:
        val_pos_emb = None


    train_samples = []
    val_samples = []
    print("Visualizing joint prediction...")
    for i in range(n_data):


        train_sample = model.inference(
            batch_size=n_samples,
            obs=obs_emb[i].unsqueeze(0) if obs_emb is not None else None,
            pos=pos_emb[i].unsqueeze(0) if pos_emb is not None else None,
            goal=goal_emb[i].unsqueeze(0) if goal_emb is not None else None,
            nfe=nfe,
        )

        val_sample = model.inference(
            batch_size=n_samples,
            obs=val_obs_emb[i].unsqueeze(0) if val_obs_emb is not None else None,
            pos=val_pos_emb[i].unsqueeze(0) if val_pos_emb is not None else None,
            goal=val_goal_emb[i].unsqueeze(0) if val_goal_emb is not None else None,
            nfe=nfe,
        )
        train_samples.append(train_sample)
        val_samples.append(val_sample)

    train_samples = torch.stack(train_samples, dim=0)
    val_samples = torch.stack(val_samples, dim=0)

    visualize_joint_prediction(
        train_samples.detach().cpu().numpy(),
        train_target.detach().cpu().numpy(),
        as_wandb=False,
        save_path=f"{log_path}/train",
    )
    visualize_joint_prediction(
        val_samples.detach().cpu().numpy(),
        val_target.detach().cpu().numpy(),
        as_wandb=False,
        save_path=f"{log_path}/val",
    )

    if not config.policy.pos_only:
        train_img_seq = datamodule.train_episodes.sample_img_seq(n_data, 50)
        print("train_img_seq", train_img_seq.shape)
        val_img_seq = datamodule.val_episodes.sample_img_seq(n_data, 50)
        print("val_img_seq", val_img_seq.shape)
        
        train_attn = model.encoder(train_img_seq)
        print("train_attn", train_attn.shape)
        val_attn = model.encoder(val_img_seq)
        print("val_attn", val_attn.shape)
        
        print("Visualizing attention maps...")
        visualize_attention_video(
            train_img_seq.detach().cpu().numpy(),
            train_attn.detach().cpu().numpy(),
            f"{log_path}/train",
            fps=10,
        )
        
        visualize_attention_video(
            val_img_seq.detach().cpu().numpy(),
            val_attn.detach().cpu().numpy(),
            f"{log_path}/val",
            fps=10,
        )


    


