# オフライン推論・学習環境 次の実行手順

## 明日最初に行うこと

### 1. A100 40GBを確認

```bash
nvidia-smi
```

A100 40GB以外の場合は学習設定を変更する。

### 2. 学習環境構築

```bash
bash training/openvla_oft_a100/scripts/bootstrap_colab.sh
```

### 3. CheckpointをGoogle Driveへ保存

Colab学習用にはHFから取得してよい。提出物では外部通信しない。

### 4. 公式Fork基準出力を保存

固定した2画像、Proprio、Instructionを用意し、以下を保存する。

- Action chunk
- Peak VRAM
- Cold start
- First inference
- Warm inference 20回
- Package versions
- Git commit

### 5. Offline runtimeとのParity

同じ入力を`submission/openvla_oft_offline`へ与える。

合格条件:

- Action shape `(8, 7)`
- NaN / Infなし
- `max_abs_diff <= 1e-4`を初期目標
- 差がある場合はAttention Mask、画像処理、Normalizationを順に確認

### 6. Smoke Fine-tuning

```bash
export DATA_ROOT_DIR=/content/drive/MyDrive/PARC2026/datasets/rlds
export RUN_ROOT_DIR=/content/drive/MyDrive/PARC2026/experiments
export DATASET_NAME=<prepared_dataset_name>
bash training/openvla_oft_a100/scripts/train_smoke.sh
```

### 7. Go / No-Go

GO:

- 100 steps以上OOMなし
- Checkpoint保存・再ロード成功
- Offline runtime parity合格
- L4 24GBでPeak VRAM 22GB未満
- 最大推論8秒未満

No-Go:

- 2日以内に上記を満たせない
- MolmoAct2へ切り替える
