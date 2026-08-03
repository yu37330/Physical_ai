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

Mini Datasetは本番800 Episodeと同じConverter、Parity検証、OpenVLA Batch Transformを通す。Mini実行時は`PROMOTE_MANIFEST=0`とし、本番Dataset Manifestを昇格しない。

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

GitHub Actions上で、Synthetic Parquet・MP4生成からTFDS/RLDS shard生成、Train/Val読込、Source parityまで成功済みである。

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
- Hugging Face download cacheを提出対象へ含めない
- Model directoryが内部19GB基準以内
- Model cardで確認したMITライセンスを記録する

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

`04_submission_validate.ipynb`は運営Repoを固定Commitで取得し、次を分けて実行する。

```text
Source directory static validation
ZIP static validation
pip dry-run
Dynamic health/reset/act smoke test
```

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

## 3. CIで事前に解消した問題

実装後のCIにより、次をGPU利用前に検出・修正した。

1. `PYTHONPATH`未設定による`src`・`submission` Import失敗
2. TensorFlow 2.15と最新TF Metadata／Protobufの不整合
3. TFDSがImplicit Namespace PackageからCustom Builderの位置を解決できない問題
4. Mini Validation 1件に対してParity 2件を要求する問題
5. Mini実行から本番Dataset Manifestを誤って昇格し得る問題

採用したRLDS変換依存（Python 3.10/3.11）:

```text
TensorFlow 2.15.1
TensorFlow Datasets 4.9.3
TensorFlow Metadata 1.15.0
Protobuf 3.20.3
RLDS 0.1.8
```

現行ColabのPython 3.12にはTensorFlow 2.15のWheelが存在しないため、3.12では次を採用する。`requirements-data.txt`のEnvironment markerで自動選択する。

```text
TensorFlow 2.19系
TensorFlow Datasets 4.9.9以上4.10未満
TensorFlow Metadata 1.17.2以上1.18未満
Protobuf 4.25.2以上6未満
importlib_resources 6以上
RLDS 0.1.8
```

加えて、固定OpenVLA-OFT Commitの`pyproject.toml`は3.12で解決できない。`bootstrap_colab.sh`が`patch_openvla_oft_dependencies.py`でTensorFlow Pinを緩め、`tensorflow-addons`にPython 3.12 Wheelが無い`tensorflow_graphics`をDependencyから外し、`--no-deps`で再導入する。Python 3.10/3.11ではこのPatchはNo-opとなる。

## 4. 明日の実行順

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

Payload取得後、Mini modeで`prepare_stage_a_rlds.sh`を実行する。

```bash
PARITY_EPISODES_PER_SPLIT=1 \
COMPATIBILITY_SAMPLES_PER_SPLIT=4 \
PROMOTE_MANIFEST=0 \
bash training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh
```

### Step 4: OpenVLA Batch互換性

次の3 Reportが全てPassであることを確認する。

```text
rlds_conversion_report.json
rlds_source_parity.json
openvla_rlds_compatibility.json
```

Miniが通過した後、800 Episodeで`PROMOTE_MANIFEST=1`として`dataset_manifest.payload_validated.json`を生成する。

### Step 5: S0→S1→S2

1. S0で素モデルの固定入力Actionを保存
2. S1を100 steps実行
3. Loss、VRAM、Checkpoint再ロード、Action差分を確認
4. 問題がなければS2を500 steps実行

## 5. 中止条件

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

## 6. 未実測事項

CPU／Synthetic範囲のCIは通過した。次は実環境で確定する。

- 公開Datasetによる3 Episode Mini E2E
- 800 Episodeの変換時間とDisk使用量
- OpenVLA-OFT+ Base Checkpointの実ファイル構成・SHA256・容量
- OpenVLA-OFT実CheckpointによるBatch Transform
- A100 40GBのPeak VRAM
- L4 24GBのCold startと最大Latency
- 運営Validatorによる最終ZIP確認

## 7. 原則

> CI、Mini Dataset、Source parity、OpenVLA Batch compatibility、Checkpoint Manifestの5 Gateを通してから、800 Episode変換とStage A学習へ進む。
