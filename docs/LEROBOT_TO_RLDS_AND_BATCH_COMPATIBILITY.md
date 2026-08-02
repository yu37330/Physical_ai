# LeRobot選択EpisodeのRLDS変換・OpenVLA-OFT互換性設計

更新日: 2026-08-02  
対象Dataset: `parc_stage_a_balanced_v001`  
対象モデル: OpenVLA-OFT+ 7B  
対象ブランチ: `feat/parc2026-foundation`

## 1. 結論

選択したLIBERO-plus 800 Episodeを、OpenVLA-OFTが利用するTFDS/RLDSへ変換し、`RLDSDataset`、`RLDSBatchTransform`、公式Collatorまで通して互換性を検証するパイプラインを実装した。

```text
LeRobot v2.1 selected payload
  ├─ Episode Parquet: state 8D / action 7D
  ├─ Front MP4
  └─ Wrist MP4
        ↓
TFDS GeneratorBasedBuilder
        ↓
parc_libero_plus_selected/1.0.0
  ├─ train: 640 Episode
  └─ val:   160 Episode
        ↓
OpenVLA-OFT local builder / OXE registry patch
        ↓
RLDSDataset
        ↓
RLDSBatchTransform
        ↓
PaddedCollatorForActionPrediction
        ↓
Batch compatibility report
        ↓
Dataset Manifestをpayload_validatedへ昇格
```

公開データを実際にダウンロードしていない現時点では、変換コード、契約、合成データ向けUnit testまでを準備済みである。800 Episodeの実変換と実Checkpointを使ったBatch検証は、データ・GPU環境で実行して最終確定する。

## 2. 変換後RLDS契約

```text
episode
├─ steps
│  ├─ observation.image          uint8 [256,256,3]
│  ├─ observation.wrist_image    uint8 [256,256,3]
│  ├─ observation.state          float32 [8]
│  ├─ action                     float32 [7]
│  ├─ language_instruction       string
│  ├─ reward / discount
│  └─ is_first / is_last / is_terminal
└─ episode_metadata
   ├─ episode_id
   ├─ source_episode_index
   ├─ suite / task
   └─ source_parquet_sha256
```

Stateの意味順序:

```text
EEF position XYZ       3
EEF axis-angle         3
Gripper qpos           2
合計                    8
```

Actionの意味順序:

```text
Delta XYZ              3
Delta roll/pitch/yaw   3
Gripper                 1
合計                    7
```

Action chunkはRLDSへ事前保存しない。OpenVLA-OFTのTrajectory transformが連続Stepから8×7 Chunkを作る。

## 3. TFDS Split

Selection Manifest上のSplit名は`train`と`validation`である。一方、OpenVLA-OFTのRLDS loaderはValidationを`val`として読むため、変換時に次へ写像する。

```text
selection train       → TFDS train
selection validation  → TFDS val
```

Custom TFDS builderは生成後のDirectoryから読み込む必要があるため、対象Datasetだけ`tfds.builder_from_directory()`を使うPatchを適用する。

## 4. 画像方向

LeRobotのLIBERO Processorは、環境から得た画像をDataset規約へ合わせて180度回転して保存する。したがって、`Sylvest/libero_plus_lerobot`からRLDSへ変換するときは、画像方向をそのまま保持する。

v001の責任分担:

- Converter: MP4 Decode、RGB化、256×256確認。回転しない
- OpenVLA OXE transform: 回転しない
- RLDSDataset: モデル標準画像変換
- 実データGate: Source MP4 FrameとRLDS Raw Frameの向きを比較

Converterには診断用`--rotate-180`を残すが、標準Workflowでは指定しない。

## 5. State・Gripper Actionの公式互換処理

OpenVLA-OFTの公式`libero_dataset_transform`と同じ処理を使う。

### State

```text
EEF_state      = observation.state[:, :6]
gripper_state  = observation.state[:, -2:]
```

OXE configの`state_obs_keys`は次とする。

```text
["EEF_state", "gripper_state"]
```

これを連結して8次元Proprioceptionを生成する。

### Gripper Action

SourceのGripper Actionは`-1=open, +1=close`である。公式処理と同じく、最後の1次元を0〜1へClipし、Invertして`+1=open, 0=close`へ変換する。

```text
absolute_action_mask      [F,F,F,F,F,F,T]
action_normalization_mask [T,T,T,T,T,T,F]
```

連続6軸のみをNormalizationし、Gripperは絶対Actionとして扱う。

## 6. No-op処理

v001では独自No-op除去を行わない。

理由:

- Source Datasetの作成段階でどのNo-op処理が行われたかを実Payloadで確認する必要がある
- 独自Thresholdを追加すると、Gripper開閉・保持や微小補正を誤って削除する可能性がある
- Stage Aでは時間方向の整合を優先する

必要になった場合だけ、OpenVLA公式条件を再現する。

```text
Norm(action[:6]) < 1e-4
かつ
現在のGripper Action == 前StepのGripper Action
```

## 7. `RLDSBatchTransform`互換性Gate

TrainとValから複数Sampleを取得し、公式Collatorまで通す。

必須出力:

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
| Front | `pixel_values`が存在し有限 |
| Wrist | `pixel_values_wrist`が存在し有限 |
| State | `proprio`最終次元8 |
| Action | `actions`最終2次元8×7 |
| Language | `input_ids`と`labels`生成 |
| Split | TrainとValを読める |
| Collator | Batch size 1で成功 |
| Numerical | Action・ProprioにNaN/Infなし |

結果は`openvla_rlds_compatibility.json`へ保存する。

## 8. Dataset Manifest更新

Metadata選定段階:

```text
quality.status = metadata_selected_pending_payload_validation
```

次を全て満たした場合のみ昇格する。

1. 800 EpisodeのPayloadがそろう
2. State 8D、Action 7D、2 Cameraが全Episodeで成立
3. TFDS/RLDS変換完了
4. Train / Valの両Splitを読める
5. `RLDSBatchTransform`成功
6. Front / Wrist Tensorを別々に生成
7. Action chunk 8×7成立
8. OpenVLA Collator成功
9. NaN / Infなし

更新後:

```text
quality.status = payload_validated
structure.format = tfds_rlds
```

## 9. 実装ファイル

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

## 10. 実行方法

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

生成物:

```text
rlds_conversion_report.json
openvla_rlds_compatibility.json
dataset_manifest.payload_validated.json
```

## 11. 実データ実行時の注意

- TFRecordは`/content/work`へ生成し、完成・検証後にDriveへコピーする
- 一度に800 EpisodeをRAMへ保持せず、Episode単位でDecodeする
- 最初は1 Task・Train 2 Episode・Val 1 EpisodeのMini Datasetで確認する
- Source MP4とRLDS Raw Frameを目視比較する
- Dataset statisticsとSelection SHA256をCheckpointと一緒に固定する
- Converter、OpenVLA Repo、Dataset、CheckpointのRevisionをRun Manifestへ残す

## 12. 次の判断

このGateを通過した後に、通常LIBERO Replay 200 Episodeを混合する。Plus 800単体とPlus 800＋Replay 200を比較し、通常LIBERO能力の保持とLIBERO-plus適応のバランスを確認してから本学習へ進む。
