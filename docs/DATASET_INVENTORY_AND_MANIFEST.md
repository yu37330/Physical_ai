# PARC2026 学習データ実体調査・Dataset Manifest設計

更新日: 2026-08-02  
対象モデル: OpenVLA-OFT+ 7B  
対象ブランチ: `feat/parc2026-foundation`

## 1. 結論

Stage Aでは、次の構成を採用する。

- 主データ: `Sylvest/libero_plus_lerobot`
- Replay: `openvla/modified_libero_rlds`
- 選定単位: フレームではなくEpisode
- 初期候補: 1,000 Episode
  - LIBERO-plus: 800 Episode
  - 通常LIBERO Replay: 200 Episode
- 40タスクを均等化
- Spatial / Object / Goal / Longを各250 Episode
- Train / ValidationはEpisode単位で分離
- 公開メタデータに存在しない摂動ラベルは推定で埋めず、`unknown`として管理

最も重要な判断は、`Sylvest/libero_plus_lerobot`を主なEpisode選定元とすることである。このデータは14,347 Episode、2,238,036 Frame、40 Task、20 FPS、2 Camera、8次元State、7次元Actionを持つ。さらに、Parquetと動画がEpisode単位で配置されているため、選定したEpisodeだけを取得できる。

一方、`lerobot/libero_plus`は同じ規模・Featureを持ち、公式PARC Notebookが参照する重要な基準データだが、LeRobot v3のChunk単位構成である。少数Episodeだけを取得する用途では、Episode単位構成のSylvest版が扱いやすい。

## 2. データソース棚卸し

### 2.1 `Sylvest/libero_plus_lerobot`

固定Revision:

```text
22c57433fef692b5b9ecc0795344daac7fa867a5
```

| 項目 | 内容 |
|---|---:|
| Format | LeRobot v2.1 |
| Robot | Panda |
| Episodes | 14,347 |
| Frames | 2,238,036 |
| Tasks | 40 |
| FPS | 20 |
| Front camera | 256×256×3 |
| Wrist camera | 256×256×3 |
| State | float32, 8次元 |
| Action | float32, 7次元 |
| License | MIT |

配置:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-XXX/episode_XXXXXX.parquet
videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4
```

利点:

- `meta/episodes.jsonl`だけで全EpisodeのInstructionとLengthを確認できる
- 選択したEpisodeのParquetと2本の動画だけを取得できる
- 40タスクを正確に均等化できる
- 8次元Stateと7次元ActionがOpenVLA-OFTのLIBERO構成と一致する

制約:

- 公開メタデータには各Episodeの摂動カテゴリがない
- Collision、minimum clearance、成功余裕などの品質ラベルがない
- Perturbation categoryをInstructionやEpisode番号から推定しない

### 2.2 `lerobot/libero_plus`

固定Revision:

```text
f3f49f426d75030177b18778374005bc12ccd588
```

実体は14,347 Episode、2,238,036 Frame、40 Task、20 FPS、Front/Wrist 256×256、State 8、Action 7である。公式PARC Notebookが使用するRevisionであり、Feature互換確認の基準とする。

ただし、DataとVideoがChunkファイルへ統合されており、少数Episodeだけの取得では不要データも一緒に落とす可能性がある。またDataset cardにLicenseが明示されていないため、学習使用時は出典側のLicense確認を別途記録する。

### 2.3 `openvla/modified_libero_rlds`

固定Revision:

```text
6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551
```

OpenVLAがLIBERO Fine-tuningで使ったRLDS形式のデータで、約10.2GB。以下の4 Datasetを含む。

```text
libero_spatial_no_noops
libero_object_no_noops
libero_goal_no_noops
libero_10_no_noops
```

元の再生成処理では、256×256画像を保存し、No-op Transitionと失敗Demonstrationを除外している。Stage Aでは通常LIBERO性能を壊さないReplayデータとして20%程度使う。

### 2.4 `Sylvest/libero_plus_rlds`

データファイル追加Revision:

```text
0cd38d5a756c7a60eca669286f2b53a1aaa94707
```

選定したOpenVLA-OFT+のMix-SFTに関連するRLDS Archiveで、分割ZIP合計約75.5GB。モデルとの整合性は最も高いが、次の理由からStage Aの直接入力にはしない。

- 全Archive取得が必要
- Episode選択前に大容量Downloadが必要
- Split ZIPで中身の事前棚卸しが難しい
- 小規模独自学習に対してI/O負荷が大きい

まずLeRobot版でEpisodeを選び、選択分だけRLDSへ変換する。RLDS Archiveは変換結果とのFeature・Statistics照合に使う。

## 3. 40タスクのSuite分類

40タスクは次の4 Suiteへ10タスクずつ分類する。

- Spatial: Black bowlの初期位置・空間関係が異なる10タスク
- Object: 異なる対象物をBasketへ置く10タスク
- Goal: 同一・類似SceneでGoalが異なる10タスク
- Long: 複数物体、Drawer、Microwaveなどを含む長手順10タスク

分類はInstructionの完全一致で行い、曖昧なキーワード推定を使用しない。正本は次とする。

```text
configs/datasets/libero_task_suite_map.json
```

## 4. Stage A Dataset v001

### 4.1 Episode配分

```text
LIBERO-plus         800 Episode
通常LIBERO Replay   200 Episode
合計               1,000 Episode
```

| Suite | Plus | Replay | 合計 |
|---|---:|---:|---:|
| Spatial | 200 | 50 | 250 |
| Object | 200 | 50 | 250 |
| Goal | 200 | 50 | 250 |
| Long | 200 | 50 | 250 |

LIBERO-plus側は、まず1タスク40 Episodeの候補をMetadataから選ぶ。

```text
Candidate Train:       32 Episode / Task = 1,280
Candidate Validation:   8 Episode / Task =   320
Candidate合計:         40 Episode / Task = 1,600
```

候補Parquetだけを取得し、初期8次元StateによるFarthest Point Sampling後に次へ絞る。

```text
Final Train:       16 Episode / Task = 640
Final Validation:   4 Episode / Task = 160
Final合計:         20 Episode / Task = 800
```

通常LIBERO Replayは1タスク5 Episodeを目安とする。

### 4.2 選定手順

#### Phase 1: Metadata-only選定

最初は全データを取得せず、`meta/info.json`、`tasks.jsonl`、`episodes.jsonl`だけを取得する。

選定軸:

1. 40タスク均等
2. 4 Suite均等
3. Episode Lengthの分布を広く取る
4. 固定Seedで再現可能
5. Train / ValidationのEpisode重複なし

Metadata-only段階では`episode_length_quantiles`を使う。短い、中央値付近、長いTrajectoryを均等に選び、先頭Episode偏重を避ける。

#### Phase 2: Candidate payload取得後

候補Parquetから初期8次元Stateを読み、タスク・Splitごとに標準化した空間でFarthest Point Samplingを行う。

対象:

```text
EEF position
Axis-angle orientation
Gripper initial state
```

これはRobot initial stateの多様性を確保するためのProxyであり、正式なPerturbation labelではない。

必要に応じて初期画像Embeddingも追加検討するが、画像クラスタをCamera/Layoutの正解ラベルとして扱わない。

### 4.3 Perturbationラベル方針

公開LeRobotメタデータだけでは、Episodeごとの以下を判定できない。

- Camera
- Robot initial state
- Language
- Light
- Background
- Noise
- Layout

Manifestでは次のように記録する。

```json
{
  "status": "unknown",
  "label_source": "not_in_public_lerobot_metadata",
  "counts": {"unknown": 800}
}
```

ファイルPath、Generation log、公式MetadataなどAuthoritativeな根拠が得られた場合だけ更新する。

## 5. Dataset Manifest v2

従来の最小Schemaを拡張し、次を必須にする。

### Provenance

- Source repository
- Requested / resolved revision
- License
- Inventory fileとSHA256
- Selection fileとSHA256

### Compliance

- Training利用可否
- Official evaluation dataを含まないこと
- Official evaluation領域と分離されていること
- Generation method
- Reportへの開示事項

### Structure

- Format、Robot、FPS
- Task / Episode / Frame数
- Camera key、Shape、Dtype
- StateとActionの意味順序

### Taxonomy

- Suite別Episode数
- Perturbation labelの状態と根拠
- Unknownを含むCategory別件数

### Selection / Split

- Episode単位選定
- SeedとStrategy
- Train / Validation Episode ID
- Overlapなし

### Transformations

- 180度回転
- 128×128へのDownsample
- LeRobot→RLDS変換
- Action chunk長

### Quality

- Metadata整合
- NaN / Inf
- Camera stream存在
- State 8次元
- Action 7次元
- 40タスク存在
- Payload検査状態

## 6. 実装ツール

```text
src/data/download_hf_metadata.py
src/data/inspect_lerobot_metadata.py
src/data/select_balanced_episodes.py
src/data/build_episode_download_plan.py
src/data/download_selected_episodes.py
src/data/refine_selection_by_initial_state.py
src/data/build_dataset_manifest.py
src/data/validate_dataset_manifest.py
src/data/libero_taxonomy.py
```

実行例:

```bash
python src/data/download_hf_metadata.py \
  --repo-id Sylvest/libero_plus_lerobot \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 \
  --format lerobot_v2_1 \
  --output-dir /content/data/libero_plus

python src/data/inspect_lerobot_metadata.py \
  --meta-dir /content/data/libero_plus/meta \
  --suite-map configs/datasets/libero_task_suite_map.json \
  --repo-id Sylvest/libero_plus_lerobot \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 \
  --license mit \
  --output artifacts/datasets/libero_plus_inventory.json

python src/data/select_balanced_episodes.py \
  --inventory artifacts/datasets/libero_plus_inventory.json \
  --per-task 40 \
  --train-per-task 32 \
  --seed 20260802 \
  --output artifacts/datasets/libero_plus_candidates_v001.json

python src/data/build_episode_download_plan.py \
  --selection artifacts/datasets/libero_plus_candidates_v001.json \
  --output artifacts/datasets/candidate_parquet_plan.json

python src/data/download_selected_episodes.py \
  --plan artifacts/datasets/candidate_parquet_plan.json \
  --output-dir /content/data/libero_plus_candidates

python src/data/refine_selection_by_initial_state.py \
  --candidate-selection artifacts/datasets/libero_plus_candidates_v001.json \
  --dataset-root /content/data/libero_plus_candidates \
  --train-per-task 16 \
  --validation-per-task 4 \
  --output artifacts/datasets/libero_plus_selection_v001.json

python src/data/build_dataset_manifest.py \
  --inventory artifacts/datasets/libero_plus_inventory.json \
  --selection artifacts/datasets/libero_plus_selection_v001.json \
  --dataset-id parc_stage_a_balanced_v001 \
  --output artifacts/datasets/dataset_manifest.json

python src/data/validate_dataset_manifest.py \
  artifacts/datasets/dataset_manifest.json \
  --schema schemas/dataset_manifest.schema.json
```

## 7. 実行前Gate

Stage A学習は、Manifest作成だけでは開始しない。次を通過してから行う。

1. 40タスクがすべて分類されている
2. Train / ValidationのEpisode重複がない
3. 選択ParquetのState 8次元、Action 7次元を確認
4. Front / Wrist動画が存在する
5. NaN / Infがない
6. LeRobot→RLDS変換が固定Revision・固定コードで再現できる
7. OpenVLA-OFTの`RLDSBatchTransform`で1 Batchを正常に生成できる
8. Manifest statusを`payload_validated`へ更新する

次の実装は、選択EpisodeだけのLeRobot→RLDS変換と、OpenVLA-OFT Batch Transform互換検査である。
