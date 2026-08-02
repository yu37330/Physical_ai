# オフライン推論・学習環境 次の実行手順

## 実装済み

- PARC公式Policy Serverのシリアライゼーション仕様へ整合
- 完全ローカルCheckpoint Loader
- Sylvest版Checkpoint IDのハードコード回避
- Transformers 4.40.1用双方向SDPA Attention Patch
- 2画像＋8次元Proprioception入力
- Action Head、Proprio Projectorのローカルロード
- 8×7 Action chunk cache
- LIBERO公式と同じGripper後処理
- OpenVLA-OFT Repoの固定Commit Checkout
- 既存Action Head・Proprio Projectorを引き継ぐ学習Patch
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

`model_source_manifest.json`へ解決済みRevisionを保存する。

### 5. 提出用Prismaticコードを準備

```bash
export OPENVLA_OFT_SOURCE=/content/openvla-oft
bash submission/openvla_oft_offline/scripts/prepare_vendor.sh
```

### 6. 公式Fork基準出力を保存

固定した画像、Proprioception、Instructionで以下を保存する。

- Action chunk
- Peak VRAM
- Cold start
- First inference
- Warm inference 20回
- Package versions
- Git commit

### 7. Offline runtimeとのParity

```bash
python submission/openvla_oft_offline/tools/compare_action_chunks.py \
  official_fork_actions.npy offline_runtime_actions.npy
```

最初はAttention差分だけを確認するため、公式側で前処理済み画像を保存して同じ画像を使う。画像前処理を含むEnd-to-End比較は別に実施する。

### 8. Smoke Fine-tuning

```bash
cd /content/openvla-oft
export DATA_ROOT_DIR=/content/drive/MyDrive/PARC2026/datasets/rlds
export RUN_ROOT_DIR=/content/drive/MyDrive/PARC2026/experiments
export DATASET_NAME=<prepared_dataset_name>
export CHECKPOINT_DIR=/content/drive/MyDrive/PARC2026/models/openvla_oft_plus_base
bash /content/Physical_ai/training/openvla_oft_a100/scripts/train_smoke.sh
```

既存Action HeadとProprio Projectorを引き継がずにランダム初期化すると素の性能を失うため、`component_checkpoint_dir`を必ず指定する。

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
