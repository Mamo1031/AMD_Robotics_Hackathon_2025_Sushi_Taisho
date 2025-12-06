from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence, Tuple, Type, Union, Dict

import ml_networks as ml
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback

@dataclass
class DiffuserConfig:
    num_timesteps: int
    num_inference_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    diff_type: Literal["ddpm", "ddim"] = "ddpm"

@dataclass
class FlowMatchingConfig:
    num_inference_steps: int = 10
    sigma: float = 0.1
    ot_method: Literal["sinkhorn", "exact"] = "exact"
    solver: Literal["dopri5", "rk4", "euler"] = "dopri5"
    sensitivity: str = "adjoint"
    atol: float = 1e-4
    rtol: float = 1e-4

@dataclass
class StableFlowMatchingConfig:
    sigma: float
    lambda_x: float
    eps: float = 0.1
    num_inference_steps: int = 10

    def __post_init__(self):
        self.eps = min(1/self.lambda_x, self.eps)

@dataclass
class StreamingFlowMatchingConfig:
    sigma_0: float = 0.05
    stabilization: float = 2.5
    num_inference_steps: int = 10


@dataclass
class DatasetConfig:
    batch_size: int
    seed: int
    action_noise: bool = False
    adjusting_methods: Optional[Tuple[str, ...]] = None
    id: str = "Mamo1031/test"  # 環境名（tabletop_<env>のenv部分）
    n_iterations: int = 1
    fps: int = 10
    policy_length: int = 32
    obs_length: int = 1
    train_episodes: Optional[Tuple[int, ...]] = None
    val_episodes: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if isinstance(self.adjusting_methods, list) or isinstance(self.adjusting_methods, ListConfig):
            self.adjusting_methods = tuple(self.adjusting_methods)
        if isinstance(self.train_episodes, list) or isinstance(self.train_episodes, ListConfig):
            self.train_episodes = tuple(self.train_episodes)
        if isinstance(self.val_episodes, list) or isinstance(self.val_episodes, ListConfig):
            self.val_episodes = tuple(self.val_episodes)

@dataclass
class PosEmbConfig:
    vector: int
    scale: float = 1000.0


@dataclass
class TransformerConfig:
    """
    Transformer configuration.

    Attributes
    ----------
    d_model : int
        Dimension of model.
    nhead : int
        Number of heads.
    dim_ff : int
        Dimension of feedforward network.
    n_enc_layers : int
        Number of layers.
    n_dec_layers : int
        Number of layers.
    dropout : float
        Dropout rate. Default is 0.1.
    hidden_activation : Literal["ReLU", "GELU"]
        Activation function for hidden layer. Default is "GELU".
    output_activation : str
        Activation function for output layer. Default is "GeLU".

    """

    input_dim: int
    cond_step: int
    cond_dim: int
    horizon: int
    d_model: int
    nhead: int
    dim_feedforward: int
    n_layers: int
    n_enc_layers: int = 0
    dropout: float = 0.1
    hidden_activation: Literal["ReLU", "GELU"] = "GELU"
    only_encoder: bool = False
    is_causal: bool = False

    def __post_init__(self):
        self.hidden_activation = self.hidden_activation.lower()

@dataclass
class DiTConfig:
    """
    Transformer configuration.

    Attributes
    ----------
    d_model : int
        Dimension of model.
    nhead : int
        Number of heads.
    dim_ff : int
        Dimension of feedforward network.
    n_enc_layers : int
        Number of layers.
    n_dec_layers : int
        Number of layers.
    dropout : float
        Dropout rate. Default is 0.1.
    hidden_activation : Literal["ReLU", "GELU"]
        Activation function for hidden layer. Default is "GELU".
    output_activation : str
        Activation function for output layer. Default is "GeLU".

    """

    input_dim: int
    cond_step: int
    horizon: int
    d_model: int
    nhead: int
    mlp_ratio: float
    time_emb: Union[PosEmbConfig, ml.SoftmaxTransConfig]
    cond_cfg: ml.MLPConfig
    n_layers: int
    learn_sigma: bool = True

@dataclass
class PolicyConfig:
    obs_shape: Tuple[int, ...]
    obs_dim: int
    n_obs_steps: int
    action_dim: int
    policy: Union[ml.UNetConfig, TransformerConfig, DiTConfig]
    framework_cfg: Union[DiffuserConfig, FlowMatchingConfig]
    cond_cfg: ml.MLPConfig
    optimizer_cfg: Union[DictConfig, Dict]
    loss_cfg: Union[DictConfig, Dict]
    ema: Union[DictConfig, Dict]
    time_emb: Union[PosEmbConfig, ml.SoftmaxTransConfig] = None
    encoder: Optional[ml.EncoderConfig] = None
    pos_encoder: Optional[ml.MLPConfig] = None
    goal_encoder: Optional[ml.MLPConfig] = None
    policy_length: int = 32
    obs_length: int = 1
    use_uniform: bool = False
    freeze_encoder: bool = False
    use_data_mean_std: bool = False
    pred_obs_action: bool = False

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)
        if self.encoder is None:
            self.pos_only = True
        else:
            self.pos_only = False

        if self.pos_encoder is None:
            self.obs_only = True
        else:
            self.obs_only = False
        if self.goal_encoder is not None:
            self.goal_conditioned = True
        else:
            self.goal_conditioned = False

    def dictcfg2dict(self) -> None:
        """Convert OmegaConf DictConfig to a dictionary."""
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.ema = dict(self.ema)
        if self.encoder is not None:
            self.encoder.dictcfg2dict()


@dataclass
class TrainerConfig:
    accelerator: str
    devices: Tuple[int, ...]
    deterministic: bool
    precision: int
    log_every_n_steps: int
    check_val_every_n_epoch: int
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None
    

    def __post_init__(self):
        self.devices = tuple(self.devices)


@dataclass
class ExperimentConfig:
    epochs: int
    seed: int
    log_every: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    device: int
    activation: str
    datamodule: DatasetConfig
    trainer: TrainerConfig
    policy: PolicyConfig
    callbacks: Sequence[Callback] = field(default_factory=list)

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)

    def tuple_callbacks(self):
        if not isinstance(self.callbacks, tuple):
            self.callbacks = tuple(self.callbacks)

    def dictcfg2dict(self):
        if self.policy is not None:
            self.policy.dictcfg2dict()

