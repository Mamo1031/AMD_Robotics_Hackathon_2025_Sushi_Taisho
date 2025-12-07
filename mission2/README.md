# Mission 2: Flow Matching Policy

Mission 2 implements robot control using a Flow Matching-based policy.

## 📋 Overview

Mission 2 is a policy learning and inference system using the Diffusion Transformer (DiT) architecture and Flow Matching framework. It provides the following features:

- **Stable Flow Matching**: Policy learning with stable Flow Matching
- **Streaming Flow Matching**: Flow Matching with streaming support
- **Transformer Policy**: Transformer-based policy architecture
- **WandB Integration**: Experiment tracking and visualization

## 📁 Directory Structure

```
mission2/
├── README.md                    # This file
├── __init__.py                  # Package public API
├── code/
│   ├── config/
│   │   ├── training_config.yaml # Training configuration (placeholder)
│   │   └── inference_config.yaml# Inference configuration (placeholder)
│   ├── scripts/
│   │   ├── train.py            # Training script
│   │   └── inference.py        # Inference and evaluation script
│   └── src/
│       ├── backbone.py         # Backbones such as DiT, TransformerPolicy
│       ├── callbacks.py        # PyTorch Lightning callbacks
│       ├── cli.py              # Command-line interface
│       ├── config.py          # Configuration classes
│       ├── dataset.py         # Dataset-related code
│       ├── eval.py            # Evaluation functions
│       ├── fm.py              # Flow Matching implementation
│       ├── policy.py          # Policy classes
│       └── utils.py           # Utility functions
├── models/
│   ├── cfg/                    # Model configuration files (.yaml)
│   └── params/                 # Trained model parameters
└── wandb/                      # WandB log directory
```

## 🚀 Setup

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Lightning
- LeRobot
- Other dependencies (see `pyproject.toml`)

### Installation

Run the following commands from the project root:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Verify that dependencies are installed
```

## 📊 Dataset

By default, the `lerobot-rope` dataset is used. To use other datasets, specify them with the `--dataset` option.

## 🎓 Model Training

### Basic Usage

```bash
cd mission2/code

# Basic training execution
python scripts/train.py --train \
    --model <model_name> \
    --seed <seed> \
    --device <gpu_id> \
    --epochs <num_epochs> \
    --dataset <dataset_name>
```

### Main Options

- `--model` / `-m`: Model name (loads configuration from `models/cfg/<model_name>.yaml`)
- `--seed`: Random seed (default: 0)
- `--device`: GPU device ID (default: 0)
- `--epochs`: Number of epochs (default: 1000)
- `--dataset`: Dataset name (default: "lerobot-rope")
- `--adjusting_methods` / `-am`: List of adjustment methods (optional)

### Examples

```bash
# Training with seed 4
python scripts/train.py --train \
    --model unet \
    --seed 4 \
    --device 0 \
    --epochs 1000 \
    --dataset lerobot-rope

# Training with adjustment methods specified
python scripts/train.py --train \
    --model unet \
    --seed 4 \
    --device 0 \
    --adjusting_methods method1 method2
```

### Model Configuration Files

Model configurations are placed in `models/cfg/<model_name>.yaml`. This file contains the following settings:

- Policy configuration (PolicyConfig)
- Flow Matching configuration (StableFlowMatchingConfig / StreamingFlowMatchingConfig)
- Dataset configuration (DatasetConfig)
- Trainer configuration (TrainerConfig)

### Output

The following are generated during training:

- **Model checkpoints**: `models/params/<dataset>/<model_name>/seed:<seed>/`
  - `model.ckpt`: Model with best validation loss
  - `last.ckpt`: Model from the last epoch
- **WandB logs**: Saved in the `wandb/` directory and can be visualized in the WandB dashboard

## 🔍 Inference and Evaluation

### Evaluation Mode

```bash
cd mission2/code

# Basic evaluation execution
python scripts/inference.py --evaluate \
    --model <model_name> \
    --seed <seed> \
    --device <gpu_id> \
    --dataset <dataset_name>
```

### Main Options

- `--model` / `-m`: Model name
- `--seed`: Random seed (default: 0)
- `--device`: GPU device ID (default: 0)
- `--dataset`: Dataset name (default: "lerobot-rope")
- `--load_last`: Load the last checkpoint (default: uses `model.ckpt`)
- `--inference_every`: Interval for policy inference execution (default: 8)
- `--max-steps` / `-n`: Maximum number of steps (default: 300)
- `--n_candidates` / `-nc`: Number of candidates to sample (default: 1)
- `--adjusting_methods` / `-am`: List of adjustment methods (optional)

### Robot Configuration

When evaluating on a real robot, specify the robot configuration via command-line arguments:

```bash
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --dataset lerobot-rope \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyUSB0
```

### Examples

```bash
# Basic evaluation
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --dataset lerobot-rope \
    --max-steps 300 \
    --inference_every 16

# Using the last checkpoint
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --load_last

# Sampling multiple candidates
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --n_candidates 5
```

### Output

The following are generated by evaluation execution:

- **Visualization videos**: Saved in `reports/<model_name>/seed:<seed>/`
  - Videos including attention visualization
- **Logs**: Information such as total reward, number of frames, and attention shape is output to the console

## 🔧 Configuration Customization

### Creating Model Configuration Files

To create a new model configuration, place a YAML file in the `models/cfg/` directory. The configuration file structure is based on the `ExperimentConfig` class.

### Changing the Dataset

To use a different dataset:

1. For LeRobot datasets, specify the dataset name with the `--dataset` option
2. For custom datasets, extend the `DatasetModule` class

## 📚 API Reference

### Main Classes and Functions

The following main components can be imported from the Mission 2 package:

```python
from mission2 import (
    # Backbones
    DiT, TransformerPolicy, SinusoidalPosEmb,
    
    # Policies
    Policy, StreamingPolicy, PolicyBase,
    
    # Flow Matching
    StableFlowMatcher, StreamingFlowMatcher,
    
    # Datasets
    CogBotsDataset, DatasetModule,
    joint_transform, joint_detransform,
    
    # Utilities
    visualize_attention,
    visualize_joint_prediction,
    visualize_attention_video,
    
    # CLI
    train_main, inference_main,
)
```

See `mission2/__init__.py` for details.

## 🐛 Troubleshooting

### Common Issues

1. **Model configuration file not found**
   - Verify that `models/cfg/<model_name>.yaml` exists

2. **Checkpoint not found**
   - Verify that checkpoints exist in `models/params/<dataset>/<model_name>/seed:<seed>/`
   - If using the `--load_last` option, verify that `last.ckpt` exists

3. **GPU memory insufficient**
   - Specify a different GPU with `--device`
   - Adjust batch size or model size

4. **Dataset not found**
   - Verify that the LeRobot dataset is properly installed
   - Verify that the dataset name is correct

## 📝 Notes

- It is recommended to run training and evaluation from the `mission2/code/` directory
- Model configuration file paths are specified as relative paths (relative to `models/cfg/`)
- When using WandB, verify that it is properly configured

## 🔗 Related Links

- [Project README](../README.md)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)

