import numpy as np
import wandb
import os
import torch
import matplotlib.pyplot as plt
from rich.progress import Progress, track
from omegaconf import OmegaConf
from typing import Optional, Dict, Any
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')

##############################################################################


def visualize_attention(
        observation: np.ndarray,
        attention: np.ndarray,
        as_wandb: bool = True,
        save_path: Optional[str] = None,
    ):
    """Attentionの可視化
    Args:
        observation(np.ndarray): 観測データ, shape(batch, C, H, W)
        attention(np.ndarray): Attention, shape(batch, channels*2)

    Returns:
        wandb.Image: 可視化結果
    """

    attention = attention.reshape(
        attention.shape[0], -1, 2
    )

    figures = {}

    for i in range(attention.shape[0]):
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(observation[i].transpose(1, 2, 0))
        ax.scatter(
            (attention[i, :, 0] + 1.) / 2. * observation.shape[-1], 
            (attention[i, :, 1] + 1.) / 2. * observation.shape[-2],
            color="red",  
            alpha=1.0, 
            s=50
        )
        ax.set_xlim(0, observation.shape[-1])
        ax.set_ylim(observation.shape[-2], 0)  # Y軸を上に向ける

        ax.set_title("Attention")

        if as_wandb:
            figures[f"attention_{i}"] = [wandb.Image(fig)]
        else:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/attention_{i}.png")
            else:
                fig.savefig(f"attention_{i}.png")
        fig.clear()
        fig.clf()
        plt.close(fig)
    return figures

def visualize_joint_prediction(
        prediction: np.ndarray, 
        target: np.ndarray,
        as_wandb: bool = True,
        save_path: Optional[str] = None,
    ):
    """関節角度の予測結果の可視化

    Args:
        prediction(np.ndarray): 予測結果, shape(batch, samples, length, dim)
        target(np.ndarray): 目標データ, shape(batch, length, dim)

    Returns:
        figures(dict): 可視化結果
    """
    figures = {}

    for i, pred in enumerate(prediction):
        fig = plt.figure()
        # fig.tight_layout()

        axes = [
            fig.add_subplot(pred.shape[-1], 1, i + 1)
            for i in range(pred.shape[-1])
        ]

        for d, ax in enumerate(axes):
            ax.plot(target[i, :, d], color="black", label="target")
            for j in range(pred.shape[0]):
                ax.plot(pred[j, :, d], color="blue", label="prediction")
            ax.set_ylim(-1.0, 1.0)

        axes[-1].legend(
            borderpad=0.01,
            handletextpad=0.01,
            loc="lower left",
            bbox_to_anchor=(0.99, 0.99),
        )

        if as_wandb:
            figures[f"joint_{i}"] = [wandb.Image(fig)]
        else:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/joint_{i}.png")
            else:
                fig.savefig(f"joint_{i}.png")
        fig.clear()
        fig.clf()
        plt.close(fig)
    return figures

def visualize_attention_video(
        observation: np.ndarray,
        attention: np.ndarray,
        save_path: Optional[str] = None,
        fps: int = 30,
    ):
    """Attentionの動的可視化
    Args:
        observation(np.ndarray): 観測データ, shape(batch, length, rgb, height, width)
        attention(np.ndarray): Attention, shape(batch, length, channels*2)
        save_path(Optional[str]): 保存先パス
        fps(int): フレームレート

    Returns:
        None
    """
    attention = attention.reshape(
        attention.shape[0], attention.shape[1], -1, 2
    )
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for i in track(range(attention.shape[0])):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def update(frame):
            ax.cla()
            ax.clear()
            img = observation[i, frame]
            if img.shape[-1] != 3:
                img = img.transpose(1, 2, 0)
            ax.imshow(img)
            ax.scatter(
                (attention[i, frame, :, 0] + 1.) / 2. * img.shape[-2],
                (attention[i, frame, :, 1] + 1.) / 2. * img.shape[-3],
                color="red",
                alpha=1.0,
                s=50
            )
            ax.set_xlim(0, img.shape[-2])
            ax.set_ylim(img.shape[-3], 0)  # Y軸を上に向ける
            ax.set_title(f"Attention Frame {frame}")

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=observation.shape[1],
            interval=1000/fps,
        )

        if save_path is not None:
            anim.save(f"{save_path}/attention_video_{i}.mp4", fps=fps)
        else:
            anim.save(f"attention_video_{i}.mp4", fps=fps)

        plt.close(fig)


