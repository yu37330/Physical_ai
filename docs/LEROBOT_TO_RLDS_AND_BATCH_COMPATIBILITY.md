# LeRobot選択EpisodeのRLDS変換・OpenVLA-OFT互換性設計

更新日: 2026-08-02  
対象Dataset: `parc_stage_a_balanced_v001`  
対象モデル: OpenVLA-OFT+ 7B  
対象ブランチ: `feat/parc2026-foundation`

## 1. 結論

選択したLIBERO-plus 800 Episodeを、OpenVLA-OFTが標準利用するTFDS/RLDSへ変換し、`RLDSBatchTransform`と公式Collatorまで通して互換性を検証するパイプラインを実装した。

変換・検証は次の順に行う。

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
  └─ validation: 160 Episode
        ↓
OpenVLA-OFT OXE registry patch
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

公開データを実際にダウンロードしていない現時点では、コード、契約、Synthetic unit testまでを準備済みである。800 Episodeの実変換、TFDS shard生成、OpenVLA-OFT実Batch検証は、データとA100環境を取得後に実行して確定する。

## 2. 変換後RLDS契約

### Episode構造

```text
episode
├─ steps
│  ├─ observation
│  │  ├─ image: uint8 [256, 256, 3]
│  │  ├─ wrist_image: uint8 [256, 256, 3]
│  │  └─ state: float32 [8]
│  ├─ action: float32 [7]
│  ├─ language_instruction: string
│  ├─ reward: float32
│  ├─ discount: float32
│  ├─ is_first: bool
│  ├─ is_last: bool
│  └─ is_terminal: bool
└─ episode_metadata
   ├─ episode_id
   ├─ source_episode_index
   ├─ suite
   ├─ task
   └─ source_parquet_sha256
```

### State意味順序

```text
EEF position XYZ       3
EEF axis-angle         3
Gripper qpos           2
合計                    8
```

### Action意味順序

```text
delta XYZ              3
delta roll/pitch/yaw   3
gripper                 1
合計                    7
```

Action chunkはRLDSへ事前保存しない。OpenVLA-OFTのRLDS loaderが連続Stepから8×7 Chunkを作る。これにより、Chunk生成の意味を公式実装へ委ね、独自変換による時間方向のずれを避ける。

## 3. 画像変換の責任分担

画像の180度回転をConverterとOpenVLA transformの両方で行うと元に戻ってしまうため、責任を一か所へ固定する。

### v001の方針

- LeRobot→RLDS Converter: RGB Decode、256×256統一のみ
- OpenVLA OXE transform: Front/Wristを180度回転
- RLDSDataset: モデル標準解像度へのResize・画像Augmentation
- PARC 128×128情報損失の再現は、後続Dataset v002で比較実験する

`--rotate-180`はConverterに残しているが、標準Workflowでは指定しない。使用時はOXE transform側の回転を無効化しなければならない。

## 4. No-op処理

v001ではNo-op除去を行わない。

理由:

- LeRobot版LIBERO-plusがどの段階でNo-op処理されたかをEpisode単位Metadataだけでは断定できない
- OpenVLAの公式modified LIBEROと同一のNo-op判定を再現せずに独自Thresholdを適用すると、Gripper保持や微小補正を誤って削除する可能性がある
- Action Headを最小調整するStage Aでは、時間整合性を優先する

実データ比較でNo-op比率が問題になった場合だけ、公式`is_noop`実装と一致する処理をv002へ追加する。

## 5. OpenVLA-OFTへの登録

Pinned OpenVLA-OFT Repoへ次を追加するPatchを用意した。

### Dataset config

```text
image primary: image
image wrist: wrist_image
state key: state
state encoding: POS_EULER
Action encoding: EEF_POS
absolute action mask: [F,F,F,F,F,F,T]
action normalization mask: [T,T,T,T,T,T,F]
```

Gripper Actionは絶対値として扱い、連続6軸だけをAction Normalization対象にする。

### Standardization transform

- Front imageを180度回転
- Wrist imageを180度回転
- State、Action、Languageは変更しない

### Mixture

```text
parc_stage_a_plus_only:
  parc_libero_plus_selected = 1.0
```

通常LIBERO Replay 200 Episodeを混ぜる本番Mixtureは、Plus単体の互換性検証後に追加する。最初からMixすると、どちらのSourceに問題があるか判別しにくいためである。

## 6. `RLDSBatchTransform`互換性Gate

TrainとValidationからそれぞれ複数Sampleを取得し、公式Collatorまで通す。

必須出力:

```text
pixel_values
input_ids
labels
actions
proprio
```

合格条件:

| 項目 | 条件 |
|---|---|
| Camera | 2画像が`pixel_values`へ入る |
| State | 最終次元8 |
| Action | 最終2次元が8×7 |
| Token | `input_ids`と`labels`が生成される |
| 数値 | Action・ProprioにNaN/Infなし |
| Split | TrainとValidationの両方を読める |
| Collator | Batch size 1で正常にCollateできる |

検証結果は`openvla_rlds_compatibility.json`へ保存する。

## 7. Dataset Manifest更新

Metadata選定段階では次のStatusである。

```text
metadata_selected_pending_payload_validation
```

以下が全て成功した場合のみ、Manifestを更新する。

1. 800 EpisodeのPayloadがそろう
2. State 8D、Action 7D、2 Cameraが全Episodeで成立
3. TFDS/RLDS変換完了
4. Train/Validationの両Splitが読める
5. `RLDSBatchTransform`成功
6. Action chunk 8×7成立
7. OpenVLA Collator成功

更新後:

```text
quality.status = payload_validated
structure.format = tfds_rlds
```

Transformation履歴には、Selection SHA256、Converter version、画像回転の実施場所、OpenVLA Batch Transform契約を残す。

## 8. 実装ファイル

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

## 9. 実行方法

```bash
export PROJECT_ROOT=/content/Physical_ai
export OPENVLA_ROOT=/content/openvla-oft
export SOURCE_ROOT=/content/data/libero_plus_selected
export SELECTION_FILE=/content/drive/MyDrive/PARC2026/artifacts/libero_plus_selection_v001.json
export TFDS_ROOT=/content/drive/MyDrive/PARC2026/datasets/rlds
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

## 10. 実データ実行時の注意

- Google Drive上でTFRecordを直接大量生成するとI/Oが遅くなる可能性がある。`/content/work`へ生成して完成後にDriveへCopyする方が安全
- Video DecodeはEpisode単位で行い、一度に800 EpisodeをMemoryへ保持しない
- TFDS builderの途中失敗に備えて十分なローカルDiskを確保する
- Converter、OpenVLA Repo、Source Dataset、Selectionの各Revision/SHA256をRun Manifestへ記録する
- 変換後のDataset statisticsは、追加学習Checkpointと一緒に固定する
- 実データで問題が出た場合、まず1 Task・Train 2 Episode・Validation 1 EpisodeのMini Datasetで原因を切り分ける

## 11. 次の判断

このGateを通過した後に、通常LIBERO Replay 200 Episodeを同一RLDS Mixへ追加する。Plus 800とReplay 200を混ぜた最終Stage A Datasetは、Source別のDataset statisticsとSampling weightを明示し、Plus単体との比較を行ってから学習に使用する。
