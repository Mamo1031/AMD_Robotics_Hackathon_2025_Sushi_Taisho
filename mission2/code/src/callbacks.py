from __future__ import annotations
from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from .utils import visualize_attention, visualize_joint_prediction
import wandb
from omegaconf import OmegaConf

class ProgressBarCallback(RichProgressBar):
    """
    Make the progress bar richer.

    References
    ----------
    * https://qiita.com/akihironitta/items/edfd6b29dfb67b17fb00
    """

    def __init__(self) -> None:
        """Rich progress bar with custom theme."""
        theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        super().__init__(theme=theme)


class VisualizePrediction(pl.Callback):
    def __init__(self, save_every_n_epoch: int):
        super().__init__()
        self.save_every_n_epoch = save_every_n_epoch

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.current_epoch % self.save_every_n_epoch != 0:
            return
        media_dict = {}

        pred_actions = outputs["prediction/samples"].cpu().detach().numpy()
        original_actions = outputs["prediction/original"].cpu().detach().numpy()

        media_dict["prediction/actions"] = visualize_joint_prediction(
            pred_actions, original_actions
        )

        if outputs["prediction/spatial"] is not None:
            media_dict["prediction/attention"] = visualize_attention(
                outputs["prediction/obs"].cpu().detach().numpy()[:, -1],
                outputs["prediction/spatial"].cpu().detach().numpy()[:, -1],
            )

        if "prediction/obs_recon" in outputs:
            obs_recon = outputs["prediction/obs_recon"].cpu().detach().numpy()
            obs_recon *= 255.0
            obs_recon = obs_recon.astype("uint8")
            media_dict["prediction/obs_recon"] = [
                wandb.Image(obs_recon[i, 0].transpose(1,2,0)) for i in range(min(4, obs_recon.shape[0]))
            ]

        trainer.logger.experiment.log(media_dict)


class WandbModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpointを拡張して、wandbにモデルをアップロードするカスタムコールバック。
    中断時（KeyboardInterrupt等）でもwandbに保存されるように、on_fit_endとon_exceptionを実装。
    """
    
    def __init__(
        self,
        config: Optional[object] = None,
        config_dict: Optional[OmegaConf] = None,
        conf_path: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.config_dict = config_dict
        self.conf_path = conf_path
        self._wandb_uploaded = False  # 重複アップロードを防ぐフラグ
    
    def _upload_to_wandb(self, trainer: pl.Trainer) -> None:
        """wandbにモデルをアップロードし、configを更新する"""
        if self._wandb_uploaded:
            return
        
        if trainer.logger is None or not hasattr(trainer.logger, 'experiment'):
            return
        
        experiment = trainer.logger.experiment
        if experiment is None:
            return
        
        try:
            # wandb run pathを取得 (entity/project/run_id)
            run_path = experiment.path
            if self.config is not None:
                self.config.run_path = run_path
            
            # モデルファイルをwandb artifactとして保存
            if self.dirpath is None:
                return
            
            model_path = Path(self.dirpath) / "model.ckpt"
            last_path = Path(self.dirpath) / "last.ckpt"
            
            artifact_name = f"model-{experiment.id}"
            artifact = wandb.Artifact(artifact_name, type="model")
            
            files_added = False
            if model_path.exists():
                artifact.add_file(str(model_path), name="model.ckpt")
                files_added = True
            if last_path.exists():
                artifact.add_file(str(last_path), name="last.ckpt")
                files_added = True
            
            if files_added:
                experiment.log_artifact(artifact)
                experiment.log({"run_path": run_path})
                print(f"Model uploaded to wandb: {run_path}")
            
            # configを更新して保存
            if self.conf_path is not None and self.config_dict is not None:
                self.config_dict.run_path = run_path
                OmegaConf.save(self.config_dict, self.conf_path)
                print(f"Config saved with run_path: {run_path}")
            
            self._wandb_uploaded = True
        except Exception as e:
            print(f"Warning: Failed to save model to wandb or update config: {e}")
            import traceback
            traceback.print_exc()
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """学習終了時（正常終了・中断問わず）に呼ばれる"""
        super().on_fit_end(trainer, pl_module)
        self._upload_to_wandb(trainer)
    
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        """例外発生時（KeyboardInterrupt等）に呼ばれる"""
        super().on_exception(trainer, pl_module, exception)
        self._upload_to_wandb(trainer)

