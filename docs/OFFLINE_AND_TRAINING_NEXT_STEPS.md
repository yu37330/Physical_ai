# オフライン推論・学習環境 次の実行手順

## 実装済み

- PARC公式Policy Serverをそのまま維持し、`MyPolicy`だけ差し替え
- 完全ローカルCheckpoint Loader
- Sylvest版Checkpoint IDのハードコード回避
- Transformers 4.40.1用双方向SDPA Attention Patch
- PARC画像の180度回転、224px Resize、90% Center Crop
- 2画像＋公式と同じ8次元Proprioception入力
- Action Head、Proprio Projectorのローカルロード
- 8×7 Action chunk cache
- LIBERO公式と同じGripper後処理
- OpenVLA-OFT Repoの固定Commit Checkout
- 既存Action Head・Proprio Projectorを引き継ぐ学習Patch
- Stage Aは新規LoRAなしでVLA本体を完全Freeze
- Stage BだけLoRA rank 8を追加しVision側LoRAをFreeze
- 単GPUでも分散Barrierを成立させる`torchrun`実行
- Model revision記録付きDownloader
- Cold start、VRAM、First/Warm推論のベンチマーク
- 提出ZIPの容量・SHA256生成

## 明日最初に行うこと

### 1. リポジトリを取得

```bash
git clone -b feat/parc2026-foundation https://github.com/yu37330/Physical_ai.git
cd Physical_ai
```

### 2. A100 40GBを確認

```bash
nvidia-smi
```

### 3. 学習環境を構築

```bash
export PROJECT_ROOT=/content/Physical_ai
bash training/openvla_oft_a100/scripts/bootstrap_colab.sh
```

### 4. Base CheckpointをDriveへ固定保存

```bash
python training/openvla_oft_a100/scripts/download_base_checkpoint.py \
  --output /content/drive/MyDrive/PARC2026/models/openvla_oft_plus_base
```

### 5. 提出用Prismaticコードを準備

```bash
export OPENVLA_OFT_SOURCE=/content/openvla-oft
bash submission/openvla_oft_offline/scripts/prepare_vendor.sh
```

### 6. Offline runtimeを計測

```bash
python submission/openvla_oft_offline/tools/benchmark_runtime.py \
  --model-dir submission/openvla_oft_offline/model_weights/openvla_oft_plus \
  --repeats 20 \
  --output benchmark_results/openvla_offline.json \
  --actions-output benchmark_results/openvla_offline_actions.npy
```

### 7. 公式ForkとのParity

```bash
python submission/openvla_oft_offline/tools/compare_action_chunks.py \
  benchmark_results/official_fork_actions.npy \
  benchmark_results/openvla_offline_actions.npy
```

### 8. Stage A Smoke Fine-tuning

```bash
cd /content/openvla-oft
export DATA_ROOT_DIR=/content/drive/MyDrive/PARC2026/datasets/rlds
export RUN_ROOT_DIR=/content/drive/MyDrive/PARC2026/experiments
export DATASET_NAME=<prepared_dataset_name>
export CHECKPOINT_DIR=/content/drive/MyDrive/PARC2026/models/openvla_oft_plus_base
export USE_LORA=False
export TRAIN_VLA_LORA=False
bash /content/Physical_ai/training/openvla_oft_a100/scripts/train_smoke.sh
```

Stage AはVLA本体へ新しいLoRAを追加せず、事前学習済みAction HeadとProprio Projectorだけを調整する。

Stage Bへ進む場合だけ次を設定する。

```bash
export USE_LORA=True
export TRAIN_VLA_LORA=True
export FREEZE_VISION_LORA=True
```

### 9. 提出ZIPを作る

```bash
python submission/openvla_oft_offline/tools/build_submission_zip.py \
  --output /content/drive/MyDrive/PARC2026/submissions/openvla_oft_plus_v001.zip
```

その後、運営Repoの`validate_submission.py`と`evaluate.py`を必ず実行する。

## Go / No-Go

GO:

- 100 steps以上OOMなし
- Checkpoint保存・再ロード成功
- Offline runtime parity合格
- L4 24GBでPeak VRAM 22GB未満
- 最大推論8秒未満
- ZIP 20GB未満

No-Go:

- 2日以内に上記を満たせない
- MolmoAct2へ切り替える
