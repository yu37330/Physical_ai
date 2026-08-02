# オフライン推論・学習環境 次の実行手順

## 実装済み

- PARC公式Policy Serverのシリアライゼーション仕様へ整合
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
- Stage AのHead＋Proprioだけ学習する設定
- Stage BでVision側LoRAを凍結する設定
- Model revision記録付きDownloader

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

### 6. 公式Fork基準出力を保存

固定入力でAction chunk、Peak VRAM、Cold start、First inference、Warm inference 20回を保存する。

### 7. Offline runtimeとのParity

```bash
python submission/openvla_oft_offline/tools/compare_action_chunks.py \
  official_fork_actions.npy offline_runtime_actions.npy
```

Attention差分を先に検証するため、公式側で前処理済み画像を保存して同じ画像を使う。PIL版前処理と公式TensorFlow版のEnd-to-End差分は別に評価する。

### 8. Stage A Smoke Fine-tuning

```bash
cd /content/openvla-oft
export DATA_ROOT_DIR=/content/drive/MyDrive/PARC2026/datasets/rlds
export RUN_ROOT_DIR=/content/drive/MyDrive/PARC2026/experiments
export DATASET_NAME=<prepared_dataset_name>
export CHECKPOINT_DIR=/content/drive/MyDrive/PARC2026/models/openvla_oft_plus_base
export TRAIN_VLA_LORA=False
bash /content/Physical_ai/training/openvla_oft_a100/scripts/train_smoke.sh
```

Stage Aは既存VLAを固定し、事前学習済みAction HeadとProprio Projectorだけを小さく調整する。Stage Bへ進む場合のみ`TRAIN_VLA_LORA=True`とし、Vision側LoRAはFreezeする。

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
