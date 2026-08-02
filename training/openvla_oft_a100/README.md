# OpenVLA-OFT+ A100 40GB Training Environment

A100 40GBでOpenVLA-OFT+の素の性能を維持しながら、必要最小限の独自学習を行う構成です。

## Stage A: 本命の最小調整

- 選定CheckpointのVLA本体を完全Freeze
- 新しいLoRAを追加しない
- 事前学習済みAction Headを読み込んで微調整
- 事前学習済みProprio Projectorを読み込んで微調整
- `torchrun --nproc-per-node 1`で単GPU実行

新しいLoRAを追加して凍結する方法は採用しません。初期化直後のLoRAでもForwardを変える可能性があり、「素の性能を守る」という方針に反するためです。

## Stage B: Stage Aで不足する場合のみ

- `USE_LORA=True`
- `TRAIN_VLA_LORA=True`
- LoRA rank 8
- Vision BackboneのLoRAはFreeze
- Language/統合側だけを低学習率で更新

## 学習データ

Stage Aの初期構成は次です。

```text
LIBERO-plus selected RLDS   800 Episode
通常LIBERO Replay           200 Episode
合計                        1,000 Episode
```

先にLIBERO-plus 800 EpisodeだけをTFDS/RLDSへ変換し、OpenVLA-OFTの`RLDSBatchTransform`とCollatorまで通します。Plus単体の互換性が確認できてから通常LIBERO Replayを混合します。

学習データ全体と変換仕様は次を参照してください。

```text
docs/LEARNING_DATA_REPORT.md
docs/LEROBOT_TO_RLDS_AND_BATCH_COMPATIBILITY.md
```

## 原則

- 学習は固定Commitの公式OpenVLA-OFT Repoと公式Transformers Forkを使う
- 2画像とProprioceptionを維持する
- Plus 800＋通常LIBERO Replay 200をStage Aの初期配分とする
- Full Fine-tuningとVision Encoder全面解凍は行わない
- 学習成果物はGoogle Driveへ保存する
- Colabは固定Git CommitをCheckoutする
- Dataset Manifestが`payload_validated`になるまで学習を開始しない

## 実行順

1. `scripts/bootstrap_colab.sh`で環境を作る
2. 選択済みLeRobot EpisodeのParquetとFront/Wrist動画を配置する
3. `scripts/prepare_stage_a_rlds.sh`でRLDS変換・Batch互換検証を実行する
4. `dataset_manifest.payload_validated.json`を確認する
5. `configs/smoke_lora_r8.yaml`のPathを確認する
6. `scripts/train_smoke.sh`を実行する
7. 100〜500 stepsでLoss、VRAM、Checkpoint再ロードを確認する
8. 推論Parity確認後に1,000 steps以上へ進む

## RLDS変換・互換検証

```bash
export PROJECT_ROOT=/content/Physical_ai
export OPENVLA_ROOT=/content/openvla-oft
export SOURCE_ROOT=/content/data/libero_plus_selected
export SELECTION_FILE=/content/drive/MyDrive/PARC2026/artifacts/libero_plus_selection_v001.json
export TFDS_ROOT=/content/work/rlds
export BASE_CHECKPOINT=/content/drive/MyDrive/PARC2026/models/openvla_oft_plus_base
export MANIFEST_FILE=/content/drive/MyDrive/PARC2026/artifacts/dataset_manifest.json
export ARTIFACT_ROOT=/content/drive/MyDrive/PARC2026/artifacts/parc_stage_a_balanced_v001

bash training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh
```

TFDS shardはColabローカルの`/content/work`へ生成し、検証後にGoogle DriveへCopyする方がI/O面で安全です。

## 注意

公式OpenVLA-OFT Fine-tuningはRLDS形式を前提とします。LeRobot形式データをそのまま渡さず、固定Selectionと変換Report、Dataset Manifestを残します。

v001では独自No-op除去を行いません。LeRobotのLIBERO ProcessorがDataset保存時に画像を180度回転するため、ConverterとOpenVLA OXE transformでは追加回転しません。State分割とGripper Action変換は、公式OpenVLAの`libero_dataset_transform`と同じ処理を使用します。
