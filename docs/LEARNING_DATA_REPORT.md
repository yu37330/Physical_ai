# PARC2026 学習データ統合レポート

更新日: 2026-08-02  
対象モデル: OpenVLA-OFT+ 7B  
対象ブランチ: `feat/parc2026-foundation`

## 1. 目的

本レポートは、PARC2026 Track 1向け追加学習で使用するデータについて、次を一つにまとめる。

1. 公開データの実体と固定Revision
2. 800 Episodeの選定方法
3. Train / Validation分離
4. LeRobotからTFDS/RLDSへの変換
5. OpenVLA-OFT `RLDSBatchTransform`との互換性検証
6. Dataset Manifestによる再現性・コンプライアンス管理
7. 現在の実装状態と、実データで未確認の項目

詳細は次の2文書を正本とする。

- `docs/DATASET_INVENTORY_AND_MANIFEST.md`
- `docs/LEROBOT_TO_RLDS_AND_BATCH_COMPATIBILITY.md`

## 2. 採用する学習データ構成

Stage Aの初期構成は次とする。

```text
LIBERO-plus selected data   800 Episode
Original LIBERO Replay      200 Episode
合計                       1,000 Episode
```

最初にLIBERO-plus 800 Episode単体で変換・互換性を確認する。通常LIBERO Replay 200 Episodeは、Plus単体のGateを通過した後に混合する。

### LIBERO-plus 800 Episode

- Source: `Sylvest/libero_plus_lerobot`
- Resolved revision: `22c57433fef692b5b9ecc0795344daac7fa867a5`
- Selection unit: Episode
- Task count: 40
- Per task: 20 Episode
- Train: 16 × 40 = 640
- Validation: 4 × 40 = 160
- Suites: Spatial / Object / Goal / Longを各200 Episode

### 通常LIBERO Replay 200 Episode

- Source: `openvla/modified_libero_rlds`
- 4 Suiteを各50 Episode
- 元の通常LIBERO能力を維持するReplay用途
- Plus単体の互換性と学習安定性を確認してから追加

## 3. Episode選定

全14,347 Episodeをそのまま学習へ入れず、二段階で選定する。

### Phase 1: Metadata選定

各Taskから40 Episode、合計1,600候補を選ぶ。

```text
Train候補       32 × 40 = 1,280
Validation候補   8 × 40 =   320
```

Episode Lengthの分位点を使い、短い・中間・長いTrajectoryを含める。

### Phase 2: Initial-state diversity

候補Parquetの最初の8次元StateをTask・Splitごとに標準化し、Farthest Point Samplingで20 Episodeへ絞る。

```text
Final Train       16 × 40 = 640
Final Validation   4 × 40 = 160
```

8次元Stateは以下である。

```text
EEF position XYZ       3
EEF axis-angle         3
Gripper qpos           2
合計                    8
```

これはRobot initial state多様性のProxyであり、正式なPerturbation labelではない。公開MetadataにないCamera、Layout、Light等のラベルを推定で埋めない。

## 4. LeRobotからTFDS/RLDSへの変換

選択したEpisodeの次だけを取得する。

```text
Parquet
Front camera MP4
Wrist camera MP4
```

変換後Dataset:

```text
name: parc_libero_plus_selected
version: 1.0.0
splits:
  train: 640 Episode
  val:   160 Episode
```

OpenVLA-OFTのDataset loaderがValidationを`val`として読むため、Selection Manifest上の`validation`をTFDSでは`val`へ写像する。

### RLDS Episode契約

```text
episode
├─ steps
│  ├─ observation.image             uint8 [256,256,3]
│  ├─ observation.wrist_image       uint8 [256,256,3]
│  ├─ observation.state             float32 [8]
│  ├─ action                        float32 [7]
│  ├─ language_instruction          string
│  ├─ reward / discount
│  └─ is_first / is_last / is_terminal
└─ episode_metadata
   ├─ source_episode_index
   ├─ suite / task
   └─ source_parquet_sha256
```

Action chunkは変換時に保存しない。OpenVLA-OFTのTrajectory transformが連続Stepから8×7のAction chunkを生成する。

## 5. 公式OpenVLA LIBERO処理との整合

### 画像

LeRobotのLIBERO Processorは、Dataset保存時に画像をHuggingFaceVLA/LIBERO規約へ合わせて180度回転する。そのため、LeRobot→RLDS ConverterとOpenVLA OXE transformでは追加回転を行わない。

実データ実行時には、Source MP4のFrameとRLDS Raw Frameを比較し、変換で向きが変わっていないことを確認する。

### State

公式OpenVLAの`libero_dataset_transform`と同じく、8次元Stateを次へ分ける。

```text
EEF_state      = state[:, :6]
gripper_state  = state[:, -2:]
```

OXE configはこの2つを連結し、8次元Proprioceptionを作る。

### Gripper Action

Source ActionはLIBERO規約の`-1=open, +1=close`である。公式OpenVLA transformと同じく、最後の1次元をClip・Invertし、`+1=open, 0=close`へ変換する。

連続6軸はNormalization対象、Gripperは絶対ActionかつNormalization対象外とする。

### No-op

v001では独自No-op除去を行わない。Source Datasetでの処理状態を確認せず、独自ThresholdでStepを消すと、Gripper保持や微小補正を誤って除去するためである。

必要になった場合だけ、OpenVLA公式の次の条件をそのまま再現する。

- 先頭6軸のNormが`1e-4`未満
- Gripper Actionが前Stepと同一

## 6. `RLDSBatchTransform`互換性検証

Pinned OpenVLA-OFTへCustom Datasetを登録し、以下の公式クラスを使う。

```text
RLDSDataset
RLDSBatchTransform
PaddedCollatorForActionPrediction
```

期待するBatch Key:

```text
pixel_values
pixel_values_wrist
input_ids
labels
actions
proprio
```

合格条件:

| 項目 | 条件 |
|---|---|
| Front camera | `pixel_values`が存在し有限 |
| Wrist camera | `pixel_values_wrist`が存在し有限 |
| State | `proprio`最終次元が8 |
| Action | `actions`最終2次元が8×7 |
| Language | `input_ids`と`labels`を生成 |
| Split | TrainとValを読める |
| Collator | Batch size 1で成功 |
| Numerical | Action / ProprioにNaN・Infなし |

結果は`openvla_rlds_compatibility.json`へ保存する。

## 7. Dataset Manifest

Dataset Manifestは次の段階で状態を変える。

### Metadata選定後

```text
quality.status = metadata_selected_pending_payload_validation
```

### 実Payload・RLDS・OpenVLA互換性検証後

```text
quality.status = payload_validated
structure.format = tfds_rlds
```

Manifestへ残す情報:

- Source RepoとResolved revision
- Inventory / Selection SHA256
- Train / Validation Episode ID
- Converter version
- TFDS Dataset version
- State / Action / Camera契約
- OpenVLA Transform契約
- Action chunk shape
- Front / Wrist Tensor確認
- NaN / Inf確認
- 公式評価データを含まないこと

## 8. 実装済み

```text
configs/datasets/lerobot_to_rlds_v001.yaml
src/data/rlds_contract.py
src/data/convert_selected_lerobot_to_rlds.py
src/data/update_dataset_manifest_after_rlds.py
training/openvla_oft_a100/requirements-data.txt
training/openvla_oft_a100/scripts/patch_parc_dataset_registry.py
training/openvla_oft_a100/scripts/validate_rlds_batch_transform.py
training/openvla_oft_a100/scripts/prepare_stage_a_rlds.sh
tests/test_rlds_contract.py
```

実装内容:

- 選択EpisodeのみParquet・2 Cameraを変換
- Train / Val Split生成
- EpisodeごとのSource SHA256記録
- OpenVLA local TFDS builder読込Patch
- 公式LIBERO State / Gripper transformの再現
- 2 Camera Batch Key検証
- 8×7 Action chunk検証
- Dataset Manifest自動昇格

## 9. 実データで未確認

現時点では、公開PayloadとA100環境を使った次の実行はまだ行っていない。

- 800 Episode全体の動画Decode
- TFDS shard生成
- SourceとRLDS画像向きの実Frame比較
- Dataset statistics計算
- 実Checkpoint ProcessorによるToken・画像変換
- `RLDSBatchTransform`の実Batch通過
- 8×7 Action chunkの実値確認
- Disk使用量と変換時間

したがって、コード完成は「実行可能状態」であり、実データ互換性の最終判定は明日の実行結果で確定する。

## 10. 実行コマンド

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

出力:

```text
rlds_conversion_report.json
openvla_rlds_compatibility.json
dataset_manifest.payload_validated.json
```

## 11. 学習開始Gate

Stage A学習は、以下を全て満たした場合だけ開始する。

1. 640 Train / 160 Validation Episodeを変換
2. Train / Validation Episode重複なし
3. 2 Camera、State 8D、Action 7D成立
4. SourceとRLDSで画像向きが一致
5. 公式LIBERO State / Gripper Action transform成立
6. `RLDSBatchTransform`がTrain / Valで成功
7. Front / Wrist Tensorが別々に生成される
8. Action chunkが8×7
9. NaN / Infなし
10. Dataset Manifestが`payload_validated`

Gate通過後、まずPlus 800単体で100〜500 StepsのStage A Smoke trainingを行う。その後に通常LIBERO Replay 200を混合し、元性能保持効果を比較する。
