# AMD_Robotics_Hackathon_2025_Sushi_Taisho🍣
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**Title:** AMD_RoboticHackathon2025-Sushi-Taisho

**Team:** Cog Bots
- [Mamoru Ota](https://github.com/Mamo1031) (@Mamo1031)
- [Kentaro Fujii](https://github.com/oakwood-fujiken) (@oakwood-fujiken)
- [Yuta Nomura](https://github.com/nomutin) (@nomutin)
- [Tetsugo To](https://github.com/tetsugo02) (@tetsugo02)



## 🎯 Summary
This project simulates a rotary sushi bar ('Kaiten-sushi') using a motorized toy train track. The SO-101 robot arm is tasked with dynamically tracking and picking up sushi samples moving along the rails.

![Sushi-Bot demo](assets/demo.gif)



## ✨ Features
<!-- TODO: 実装した機能の詳細を追記 -->
- **??:** ??
- **??:** ??



## 📦 Installation
### Prerequisites
<!-- TODO: Prerequisitesを追記 -->
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- ??


### Setup Instructions
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/Mamo1031/AMD_Robotics_Hackathon_2025_Sushi_Taisho.git
cd AMD_Robotics_Hackathon_2025_Sushi_Taisho

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# On Windows: .venv\Scripts\activate

# Additional setup steps
# TODO: 追加のセットアップ手順を追記
```


### Environment Variables
<!-- TODO: 必要な環境変数を追記 -->
```bash
# TODO: Environment variables as needed
```



## 📊 Dataset
<!-- TODO: 実際のデータセット情報を確定させたら更新 -->
- **Description:**  
  - [TODO: 1〜2文で「どんなタスクのデータか」「どういうフォーマットか」を書く]
- **Hugging Face URL:**  
  - [TODO: Hugging FaceにアップロードしたらデータセットのURLを追記]



## 🤖 Model Training
### Model Architecture
<!-- TODO: 使用したモデルの詳細を追記 -->
?? (TODO: モデルの詳細を追記)


### Training Scripts
<!-- TODO: トレーニングスクリプトの使用方法を追記 -->
```bash
# Train the model (optional)
uv run train                # uses config/training_config.yaml by default

# Or specify a custom config
uv run train configs/custom_training.yaml
```


### Trained Models
<!-- TODO: トレーニング済みモデルの情報を追記 -->
- **Model Information:**  
  - [TODO: トレーニング済みモデルの情報を追記]
- **Hugging Face URL:**  
  - [TODO: Hugging FaceにアップロードしたらモデルのURLを追記]



## 🚀 Usage
### Running Inference

<!-- TODO: 推論の実行方法を追記 -->
```bash
# Basic grasping demo (uses config/inference_config.yaml by default)
uv run infer --mode grasp

# VLA demo with natural language instruction (default config)
uv run infer --mode vla --instruction "I want to eat salmon."

# Or specify a custom inference config
uv run infer configs/custom_inference.yaml --mode vla --instruction "I want to eat salmon."
```



## 🎥 Demo Video
<!-- TODO: デモビデオのリンクを追記 -->
Link to demo video: [TODO: デモビデオのリンクを追記]


## 📁 Project Structure
```
AMD_Robotics_Hackathon_2025_Sushi_Taisho/
├── README.md
├── pyproject.toml
├── uv.lock
├── LICENSE
├── assets/
│   └── test.gif
├── mission1/
│   ├── code/
│   │   └── [code and script]
│   └── wandb/
│       └── [latest run directory copied from wandb of your training job]
└── mission2/
    ├── code/
    │   ├── config/
    │   │   ├── training_config.yaml
    │   │   └── inference_config.yaml
    │   ├── scripts/
    │   │   ├── train.py
    │   │   └── inference.py
    │   └── src/
    │       ├── __init__.py
    │       └── cli.py
    └── wandb/
        └── [latest run directory copied from wandb of your training job]
```



## 🔮 Future Improvements
<!-- TODO: 今後の改善点を追記 -->
?? (TODO: 今後の改善点を追記)



## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments
- **AMD and the AMD Open Robotics Hackathon organizers** for providing SO-101 robotics kits, AMD Ryzen™ AI laptops, and AMD Developer Cloud access with AMD Instinct™ MI300X GPUs, as well as the event venue and support infrastructure ([event site](https://amdroboticshackathon.datamonsters.com/)).
- **Data Monsters** for operating and coordinating the hackathon program.
- **Hugging Face and the LeRobot team** for releasing the LeRobot framework and examples that this project builds upon.
- **All staff and fellow participants** at the Tokyo venue.


---


**Note:** This repository was created for the AMD Open Robotics Hackathon 2025. All code, models, and documentation are original work developed specifically for this competition.
