# Mission 2: Flow Matching Policy

Mission 2では、Flow Matchingベースのポリシーを使用したロボット制御を実装しています。

## 📋 概要

Mission 2は、Diffusion Transformer (DiT) アーキテクチャとFlow Matchingフレームワークを使用したポリシー学習・推論システムです。以下の機能を提供します：

- **Stable Flow Matching**: 安定したFlow Matchingによるポリシー学習
- **Streaming Flow Matching**: ストリーミング対応のFlow Matching
- **Transformer Policy**: Transformerベースのポリシーアーキテクチャ
- **WandB統合**: 実験の追跡と可視化

## 📁 ディレクトリ構造

```
mission2/
├── README.md                    # このファイル
├── __init__.py                  # パッケージの公開API
├── code/
│   ├── config/
│   │   ├── training_config.yaml # トレーニング設定（プレースホルダー）
│   │   └── inference_config.yaml# 推論設定（プレースホルダー）
│   ├── scripts/
│   │   ├── train.py            # トレーニングスクリプト
│   │   └── inference.py        # 推論・評価スクリプト
│   └── src/
│       ├── backbone.py         # DiT、TransformerPolicyなどのバックボーン
│       ├── callbacks.py        # PyTorch Lightningコールバック
│       ├── cli.py              # コマンドラインインターフェース
│       ├── config.py          # 設定クラス
│       ├── dataset.py         # データセット関連
│       ├── eval.py            # 評価関数
│       ├── fm.py              # Flow Matching実装
│       ├── policy.py          # ポリシークラス
│       └── utils.py           # ユーティリティ関数
├── models/
│   ├── cfg/                    # モデル設定ファイル（.yaml）
│   └── params/                 # 学習済みモデルのパラメータ
└── wandb/                      # WandBのログディレクトリ
```

## 🚀 セットアップ

### 前提条件

- Python 3.8+
- PyTorch
- PyTorch Lightning
- LeRobot
- その他の依存パッケージ（`pyproject.toml`を参照）

### インストール

プロジェクトルートから以下のコマンドを実行：

```bash
# 仮想環境をアクティベート
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# 依存パッケージがインストールされていることを確認
```

## 📊 データセット

デフォルトでは `lerobot-rope` データセットを使用します。他のデータセットを使用する場合は、`--dataset` オプションで指定できます。

## 🎓 モデルのトレーニング

### 基本的な使用方法

```bash
cd mission2/code

# 基本的なトレーニング実行
python scripts/train.py --train \
    --model <model_name> \
    --seed <seed> \
    --device <gpu_id> \
    --epochs <num_epochs> \
    --dataset <dataset_name>
```

### 主要なオプション

- `--model` / `-m`: モデル名（`models/cfg/<model_name>.yaml` から設定を読み込み）
- `--seed`: ランダムシード（デフォルト: 0）
- `--device`: GPUデバイスID（デフォルト: 0）
- `--epochs`: エポック数（デフォルト: 1000）
- `--dataset`: データセット名（デフォルト: "lerobot-rope"）
- `--adjusting_methods` / `-am`: 調整メソッドのリスト（オプション）

### 実行例

```bash
# シード4でトレーニング
python scripts/train.py --train \
    --model unet \
    --seed 4 \
    --device 0 \
    --epochs 1000 \
    --dataset lerobot-rope

# 調整メソッドを指定してトレーニング
python scripts/train.py --train \
    --model unet \
    --seed 4 \
    --device 0 \
    --adjusting_methods method1 method2
```

### モデル設定ファイル

モデル設定は `models/cfg/<model_name>.yaml` に配置します。このファイルには以下の設定が含まれます：

- ポリシー設定（PolicyConfig）
- Flow Matching設定（StableFlowMatchingConfig / StreamingFlowMatchingConfig）
- データセット設定（DatasetConfig）
- トレーナー設定（TrainerConfig）

### 出力

トレーニング中に以下が生成されます：

- **モデルチェックポイント**: `models/params/<dataset>/<model_name>/seed:<seed>/`
  - `model.ckpt`: 最良のバリデーション損失のモデル
  - `last.ckpt`: 最後のエポックのモデル
- **WandBログ**: `wandb/` ディレクトリに保存され、WandBダッシュボードで可視化可能

## 🔍 推論・評価

### 評価モード

```bash
cd mission2/code

# 基本的な評価実行
python scripts/inference.py --evaluate \
    --model <model_name> \
    --seed <seed> \
    --device <gpu_id> \
    --dataset <dataset_name>
```

### 主要なオプション

- `--model` / `-m`: モデル名
- `--seed`: ランダムシード（デフォルト: 0）
- `--device`: GPUデバイスID（デフォルト: 0）
- `--dataset`: データセット名（デフォルト: "lerobot-rope"）
- `--load_last`: 最後のチェックポイントを読み込む（デフォルト: `model.ckpt`を使用）
- `--inference_every`: ポリシー推論の実行間隔（デフォルト: 8）
- `--max-steps` / `-n`: 最大ステップ数（デフォルト: 300）
- `--n_candidates` / `-nc`: サンプリングする候補数（デフォルト: 1）
- `--adjusting_methods` / `-am`: 調整メソッドのリスト（オプション）

### ロボット設定

実機ロボットで評価する場合、ロボット設定をコマンドライン引数で指定します：

```bash
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --dataset lerobot-rope \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyUSB0
```

### 実行例

```bash
# 基本的な評価
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --dataset lerobot-rope \
    --max-steps 300 \
    --inference_every 16

# 最後のチェックポイントを使用
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --load_last

# 複数の候補をサンプリング
python scripts/inference.py --evaluate \
    --model unet \
    --seed 4 \
    --device 0 \
    --n_candidates 5
```

### 出力

評価実行により以下が生成されます：

- **可視化動画**: `reports/<model_name>/seed:<seed>/` に保存
  - アテンション可視化を含む動画
- **ログ**: コンソールに総報酬、フレーム数、アテンション形状などの情報が出力

## 🔧 設定のカスタマイズ

### モデル設定ファイルの作成

新しいモデル設定を作成するには、`models/cfg/` ディレクトリにYAMLファイルを配置します。設定ファイルの構造は `ExperimentConfig` クラスに基づいています。

### データセットの変更

異なるデータセットを使用するには：

1. LeRobotデータセットの場合は、データセット名を `--dataset` オプションで指定
2. カスタムデータセットの場合は、`DatasetModule` クラスを拡張

## 📚 APIリファレンス

### 主要なクラスと関数

Mission 2パッケージから以下の主要なコンポーネントをインポートできます：

```python
from mission2 import (
    # バックボーン
    DiT, TransformerPolicy, SinusoidalPosEmb,
    
    # ポリシー
    Policy, StreamingPolicy, PolicyBase,
    
    # Flow Matching
    StableFlowMatcher, StreamingFlowMatcher,
    
    # データセット
    CogBotsDataset, DatasetModule,
    joint_transform, joint_detransform,
    
    # ユーティリティ
    visualize_attention,
    visualize_joint_prediction,
    visualize_attention_video,
    
    # CLI
    train_main, inference_main,
)
```

詳細は `mission2/__init__.py` を参照してください。

## 🐛 トラブルシューティング

### よくある問題

1. **モデル設定ファイルが見つからない**
   - `models/cfg/<model_name>.yaml` が存在することを確認

2. **チェックポイントが見つからない**
   - `models/params/<dataset>/<model_name>/seed:<seed>/` にチェックポイントが存在することを確認
   - `--load_last` オプションを使用する場合は `last.ckpt` が存在することを確認

3. **GPUメモリ不足**
   - `--device` で異なるGPUを指定
   - バッチサイズやモデルサイズを調整

4. **データセットが見つからない**
   - LeRobotデータセットが正しくインストールされていることを確認
   - データセット名が正しいことを確認

## 📝 注意事項

- トレーニングと評価は `mission2/code/` ディレクトリから実行することを推奨
- モデル設定ファイルのパスは相対パスで指定（`models/cfg/` からの相対パス）
- WandBを使用する場合、適切に設定されていることを確認

## 🔗 関連リンク

- [プロジェクトREADME](../README.md)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)

