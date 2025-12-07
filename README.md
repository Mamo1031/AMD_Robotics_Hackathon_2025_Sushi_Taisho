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


## 📦 Installation
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

```


### Environment Variables
<!-- TODO: 必要な環境変数を追記 -->
- **HUGGINGFACE_API_TOKEN**
- **WANDB_API_KEY**
```bash
wandb login
huggingface-cli login
```
## 📊 Dataset
<!-- TODO: 実際のデータセット情報を確定させたら更新 -->
- **Description:**  
  - Collected **90** episodes with varied initial positions and settings to capture the dynamics of the environment.
- **Hugging Face URL:**  
  - [https://huggingface.co/datasets/Mamo1031/sushi_dynamic](https://huggingface.co/datasets/Mamo1031/sushi_dynamic)


## 🤖 Model
### Trained Models
<!-- TODO: トレーニング済みモデルの情報を追記 -->
- **Hugging Face URL:**  
- [Mamo1031/sushi-taisho-streaming](https://huggingface.co/Mamo1031/sushi-taisho-streaming)
### Model Details
- [mission2/README.md](mission2/README.md)




## 🎥 Demo Video
- [Sushi-Bot egg](https://drive.google.com/file/d/18KnuXQMYKmlZ_oblMTRiSq74vVhkm9Jy/view?usp=sharing)


## 📁 Project Structure
```
AMD_Robotics_Hackathon_2025_Sushi_Taisho/
├── README.md
├── pyproject.toml
├── uv.lock
├── LICENSE
├── assets/
│   └── demo.gif
├── mission1/
│   ├── code/
│   │   └── [code and script]
│   └── wandb/
│       └── [run logs]
└── mission2/
    ├── code/
    │   ├── config/     # Configuration files
    │   ├── scripts/    # Training and inference scripts
    │   └── src/        # Core implementation modules
    ├── models/         # Model configurations
    └── wandb/          # Experiment logs
```



## 🔮 Future Improvements
  - Enhance the voice recognition accuracy 
  - Add more sushi varieties and complex orders

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments
- **AMD and the AMD Open Robotics Hackathon organizers** for providing SO-101 robotics kits, AMD Ryzen™ AI laptops, and AMD Developer Cloud access with AMD Instinct™ MI300X GPUs, as well as the event venue and support infrastructure ([event site](https://amdroboticshackathon.datamonsters.com/)).
- **Data Monsters** for operating and coordinating the hackathon program.
- **Hugging Face and the LeRobot team** for releasing the LeRobot framework and examples that this project builds upon.
- **All staff and fellow participants** at the Tokyo venue.


---


**Note:** This repository was created for the AMD Open Robotics Hackathon 2025. All code, models, and documentation are original work developed specifically for this competition.