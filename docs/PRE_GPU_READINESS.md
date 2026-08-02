# PARC2026 GPU実行前の準備状況

更新日: 2026-08-02  
対象ブランチ: `feat/parc2026-foundation`

## 1. 目的

A100 40GBを利用できる時間を、環境構築や形式エラーの調査ではなく、モデル実測と学習に集中させる。そのため、GPUを使わずに確認できる処理を事前に自動化する。

## 2. 事前準備済みの項目

### 2.1 3 Episode Mini Dataset

800 Episodeを処理する前に、同一Taskから次を選ぶ。

```text
train       2 Episode
validation  1 Episode
合計        3 Episode
```

実装:

```text
configs/datasets/mini_e2e_v001.yaml
src/data/build_mini_selection.py
```

Mini Datasetは本番800 Episodeと同じConverter、Parity検証、OpenVLA Batch Transformを通す。

### 2.2 Synthetic LeRobot→RLDS E2E

外部Datasetを取得しなくても、Parquet、Front MP4、Wrist MP4を生成して次を検証できる。

```text
Synthetic LeRobot payload
→ TFDS/RLDS
→ train / val split
→ State・Action parity
→ Front・Wrist orientation parity
```

実装:

```text
src/data/generate_synthetic_lerobot_fixture.py
tests/test_synthetic_rlds_e2e.py
```

### 2.3 GitHub Actions

PushとPull Requestで次を自動確認する。

```text
Python compile
YAML parse
Notebook JSON parse
Shell syntax
提出requirements禁止依存
Dataset・RLDS・Runtime Unit Test
Synthetic RLDS E2E
```

Workflow:

```text
.github/workflows/ci.yml
```

TensorFlowを使うE2Eは軽量Unit Jobと分離し、問題箇所を特定しやすくしている。

### 2.4 Checkpoint Manifest

Base Checkpoint取得後に、全ファイルのSHA256、容量、役割、提出対象を自動記録する。

実装:

```text
configs/models/openvla_oft_plus_checkpoint.yaml
schemas/checkpoint_manifest.schema.json
training/openvla_oft_a100/scripts/build_checkpoint_manifest.py
training/openvla_oft_a100/scripts/download_base_checkpoint.py
```

主なGate:

- Resolved revisionが固定されている
- Model shardが存在する
- Action Headが存在する
- Proprio Projectorが存在する
- Dataset statisticsが存在する
- Symlinkがない
- Optimizer等の学習状態を提出対象へ含めない
- Model directoryが内部19GB基準以内

### 2.5 Colab薄型Notebook

Notebookは学習ロジックを持たず、Repo内の固定Scriptを呼ぶだけとする。

```text
notebooks/00_environment_check.ipynb
notebooks/01_model_feasibility.ipynb
notebooks/02_dataset_prepare.ipynb
notebooks/03_stage_a_train.ipynb
notebooks/04_submission_validate.ipynb
```

これにより、VS Code、Shell、Colabの実行経路を共通化する。

### 2.6 学習条件の固定

```text
S0: Base inference only
S1: Action Head＋Proprio Projector、100 steps
S2: Action Head＋Proprio Projector、500 steps
```

Config:

```text
training/openvla_oft_a100/configs/stage_a_s0_base_inference.yaml
training/openvla_oft_a100/configs/stage_a_s1_head_proprio_100.yaml
training/openvla_oft_a100/configs/stage_a_s2_head_proprio_500.yaml
```

S1がGateを通過しない場合、S2へ進まない。

## 3. 明日の実行順

### Step 1: 環境Gate

```bash
python training/openvla_oft_a100/scripts/preflight.py \
  --project-root /content/Physical_ai \
  --work-root /content/work \
  --drive-root /content/drive/MyDrive/PARC2026 \
  --require-a100-40gb \
  --output /content/drive/MyDrive/PARC2026/00_admin/preflight.json
```

確認:

- A100、約40GB VRAM
- Google Drive mount
- `/content/work`空き容量
- Drive空き容量
- Git Commit
- 必須Script・Config

### Step 2: Base Checkpoint取得

```bash
python training/openvla_oft_a100/scripts/download_base_checkpoint.py \
  --output /content/drive/MyDrive/PARC2026/30_models/openvla_oft_plus_base
```

生成物:

```text
model_source_manifest.json
checkpoint_manifest.json
```

### Step 3: 3 Episode Mini Dataset

```bash
python -m src.data.build_mini_selection \
  --selection artifacts/datasets/libero_plus_selection_v001.json \
  --train-count 2 \
  --validation-count 1 \
  --output artifacts/datasets/mini_selection_v001.json
```

Payload取得後、`prepare_stage_a_rlds.sh`を実行する。

### Step 4: OpenVLA Batch互換性

次の3 Reportが全てPassであることを確認する。

```text
rlds_conversion_report.json
rlds_source_parity.json
openvla_rlds_compatibility.json
```

`dataset_manifest.payload_validated.json`が生成されるまで学習しない。

### Step 5: S0→S1→S2

1. S0で素モデルの固定入力Actionを保存
2. S1を100 steps実行
3. Loss、VRAM、Checkpoint再ロード、Action差分を確認
4. 問題がなければS2を500 steps実行

## 4. 中止条件

以下の場合は800 Episode変換またはS2へ進まない。

- Mini DatasetのSource parity失敗
- Front／Wristの向きが一致しない
- Stateが8次元でない
- Actionが7次元でない
- Action chunkが8×7でない
- Checkpoint ManifestがFail
- A100 40GBでStage AがOOM
- S1でLossが低下しない
- NaNまたはInfが発生
- 学習後にGripper出力が崩れる

## 5. 未実測事項

現時点でコードとCIを準備したが、次は実環境で確定する。

- GitHub Actionsの全Job成功
- 実Source Datasetによる3 Episode Mini E2E
- 800 Episodeの変換時間とDisk使用量
- OpenVLA-OFT+ Base Checkpointの実ファイル構成
- A100 40GBのPeak VRAM
- L4 24GBのCold startと最大Latency
- 運営Validatorによる最終ZIP確認

## 6. 原則

> Mini Dataset、Source parity、OpenVLA Batch compatibility、Checkpoint Manifestの4 Gateを通してから、800 Episode変換とStage A学習へ進む。
