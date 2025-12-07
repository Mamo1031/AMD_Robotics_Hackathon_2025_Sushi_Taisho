"""Training script placeholder for Sushi-Bot.

This script will be implemented during the AMD Open Robotics Hackathon.
"""



import argparse
import os
import warnings
import torch
import signal
import sys
from dataclasses import asdict
from typing import Optional, List
from pathlib import Path

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from mission2 import DatasetModule
from mission2 import Policy, StreamingPolicy
from mission2 import ExperimentConfig, TrainerConfig, StreamingFlowMatchingConfig
from mission2 import WandbModelCheckpoint
from ml_networks import torch_fix_seed
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
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


def main(
    model_name: str,
    seed: int,
    device: int,
    epochs: int = 1000,
    dataset: str = "lerobot-rope",
    adjusting_methods: Optional[List] = None,
):

    conf_path = f"models/cfg/{model_name}.yaml"

    config_dict = OmegaConf.load(conf_path)
    config_dict.seed = seed
    config_dict.epochs = epochs
    config_dict.device = device
    config_dict.datamodule.id = dataset

    data_config = LeRobotDatasetMetadata(config_dict.datamodule.id)
    print(data_config)
    config_dict.action_dim = data_config.features["action"]["shape"][0]
    print(data_config.features.keys())
    print(data_config.total_tasks)
    config_dict.obs_shape = list(data_config.features["observation.images.main"]["shape"])[::-1]
    print(config_dict.obs_shape)
    config_dict.datamodule.adjusting_methods = adjusting_methods


    torch_fix_seed(config_dict.seed)


    OmegaConf.resolve(config_dict)
    config: ExperimentConfig = instantiate(
        config_dict,
    )

    callbacks = config.callbacks

    tags = [f"seed_{config.seed}", f"{config.datamodule.id}"]
    if adjusting_methods is not None:
        tags += adjusting_methods

    os.makedirs("wandb", exist_ok=True)
    name = f"{config.datamodule.id.split('/')[-1]}_{model_name}_seed={config.seed}"
    logger = WandbLogger(
        name=name,
        project="lerobot",
        save_dir="./wandb/",
        config=config,
        tags=tags,
    )

    path = f"models/params/{config.datamodule.id}/{model_name}/seed:{config.seed}/"
    if adjusting_methods is not None:
        path += "_".join(adjusting_methods) + "/"
    os.makedirs(path, exist_ok=True)
    if isinstance(config.policy.framework_cfg, StreamingFlowMatchingConfig):
        model = StreamingPolicy(data_config.total_tasks, config.policy)
    else:
        model = Policy(data_config.total_tasks, config.policy)
    config.dictcfg2dict()

    datamodule = DatasetModule(
        config.datamodule
            )
    if model.cfg.use_data_mean_std:
        model.noise_mean = torch.tensor(datamodule.action_mean).reshape(1, 1, -1).expand(-1, config.policy.policy_length, -1)
        model.noise_std = torch.tensor(datamodule.action_std).reshape(1, 1, -1).expand(-1, config.policy.policy_length, -1)
        print(f"Using data mean: {model.noise_mean}, std: {model.noise_std}")
    train(
        model=model,
        datamodule=datamodule,
        trainer_cfg=config.trainer,
        n_epochs=config.epochs,
        path=path,
        callbacks=callbacks,
        logger=logger,
        early_stop=config.policy.loss_cfg.get("early_stop", 0),
        config=config,
        config_dict=config_dict,
        conf_path=conf_path,
    )

def train(
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer_cfg: TrainerConfig,
        n_epochs: int,
        path: str,
        callbacks: list,
        logger: WandbLogger,
        early_stop: int = 0,
        config: Optional[ExperimentConfig] = None,
        config_dict: Optional[OmegaConf] = None,
        conf_path: Optional[str] = None,
        ):
    
    # シグナルハンドラ用の変数を準備
    save_model_ref = [None]  # リストとして保持（後で更新可能にするため）
    
    # シグナルハンドラを設定して、KeyboardInterrupt時にもconfigを保存
    def signal_handler(sig, frame):
        print("\nInterrupt received, saving config and uploading model to wandb...")
        try:
            # コールバック経由でwandbにアップロードを試みる
            if save_model_ref[0] is not None:
                # trainerが存在する場合、コールバックのメソッドを直接呼ぶ
                # ただし、trainerはまだ作成されていない可能性があるため、安全に処理
                pass
            
            # configを保存
            if config is not None and config_dict is not None and conf_path is not None:
                try:
                    if logger.experiment is not None:
                        run_path = logger.experiment.path
                        config.run_path = run_path
                        config_dict.run_path = run_path
                        OmegaConf.save(config_dict, conf_path)
                        print(f"Config saved with run_path: {run_path}")
                except Exception as e:
                    print(f"Warning: Failed to save config on interrupt: {e}")
        except Exception as e:
            print(f"Warning: Error in signal handler: {e}")
        finally:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    save_model = WandbModelCheckpoint(
        config=config,
        config_dict=config_dict,
        conf_path=conf_path,
        dirpath=path,
        filename="model",
        monitor=f"loss/val",
        save_on_train_epoch_end=False,
        save_last=True,
    )
    
    save_model_ref[0] = save_model  # シグナルハンドラからアクセス可能にする

    callbacks.append(save_model)
    if early_stop > 0:
        callbacks.append(
            EarlyStopping(
                monitor=f"loss/val",
                patience=early_stop,
                mode="min",
            )
        )
    trainer = pl.Trainer(
        logger=logger,
        fast_dev_run=False,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=n_epochs,
        **asdict(trainer_cfg),
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    # コールバック内で既にアップロードされているが、念のため再度確認
    # （コールバックが呼ばれなかった場合のフォールバック）
    try:
        if not save_model._wandb_uploaded and trainer.logger is not None:
            save_model._upload_to_wandb(trainer)
    except Exception as e:
        print(f"Warning: Failed to upload to wandb after training: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main.py", epilog="end", add_help=True)
    parser.add_argument("--model", type=str, help="model name in models/", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="lerobot-rope")
    parser.add_argument("--train", "-tr", action="store_true", help="train model")
    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate")
    parser.add_argument("--n_data", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--load_last", action="store_true", help="load last model")
    parser.add_argument(
        "--adjusting_methods",
        "-am",
        type=str,
        nargs="+",
        default=None,
        help="list of adjusting methods",
    )


    args = parser.parse_args()
    if args.train:
        main(
            args.model,
            args.seed,
            args.device,
            args.epochs,
            args.dataset,
            args.adjusting_methods,
        )



    if args.evaluate:
        from mission2 import eval
        eval(
            args.model,
            args.seed,
            args.device,
            args.n_data,
            args.n_samples,
            args.nfe,
            args.load_last,
            args.dataset,
            args.adjusting_methods,
        )

