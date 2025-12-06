from __future__ import annotations
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from .utils import visualize_attention, visualize_joint_prediction
import wandb

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

