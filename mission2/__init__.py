"""Mission 2 package for Sushi-Bot."""
# Backbone classes
from .code.src.backbone import DiT, TransformerPolicy, SinusoidalPosEmb

# Callbacks
from .code.src.callbacks import RichProgressBar, VisualizePrediction, WandbModelCheckpoint

# Config classes
from .code.src.config import (
    DiffuserConfig,
    FlowMatchingConfig,
    StableFlowMatchingConfig,
    StreamingFlowMatchingConfig,
    DatasetConfig,
    PosEmbConfig,
    TransformerConfig,
    DiTConfig,
    PolicyConfig,
    TrainerConfig,
    ExperimentConfig,
)

# Dataset classes and functions
from .code.src.dataset import (
    joint_transform,
    joint_detransform,
    CogBotsDataset,
    DatasetModule,
)

# Policy classes
from .code.src.policy import Policy, StreamingPolicy, PolicyBase

# Flow matching classes
from .code.src.fm import StableFlowMatcher, StreamingFlowMatcher

# Utility functions
from .code.src.utils import (
    visualize_attention,
    visualize_joint_prediction,
    visualize_attention_video,
)

# CLI functions
from .code.src.cli import train_main, inference_main

# Public API
__all__ = [
    # Backbone classes
    "DiT",
    "TransformerPolicy",
    "SinusoidalPosEmb",
    # Callbacks
    "RichProgressBar",
    "VisualizePrediction",
    "WandbModelCheckpoint",
    # Config classes
    "DiffuserConfig",
    "FlowMatchingConfig",
    "StableFlowMatchingConfig",
    "StreamingFlowMatchingConfig",
    "DatasetConfig",
    "PosEmbConfig",
    "TransformerConfig",
    "DiTConfig",
    "PolicyConfig",
    "TrainerConfig",
    "ExperimentConfig",
    # Dataset classes and functions
    "joint_transform",
    "joint_detransform",
    "CogBotsDataset",
    "DatasetModule",
    # Policy classes
    "Policy",
    "StreamingPolicy",
    "PolicyBase",
    # Flow matching classes
    "StableFlowMatcher",
    "StreamingFlowMatcher",
    # Utility functions
    "visualize_attention",
    "visualize_joint_prediction",
    "visualize_attention_video",
    # CLI functions
    "train_main",
    "inference_main",
]
