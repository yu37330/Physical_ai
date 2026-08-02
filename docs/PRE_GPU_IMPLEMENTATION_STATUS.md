# PARC2026 事前実装ステータス

更新日: 2026-08-02  
ブランチ: `feat/parc2026-foundation`

## 実装済み

| 項目 | 状態 | 主なファイル |
|---|---|---|
| 3 Episode Mini Dataset | 実装済み | `src/data/build_mini_selection.py` |
| Synthetic LeRobot fixture | 実装済み | `src/data/generate_synthetic_lerobot_fixture.py` |
| Synthetic LeRobot→RLDS E2E | 実装済み | `tests/test_synthetic_rlds_e2e.py` |
| GitHub Actions | 実装済み | `.github/workflows/ci.yml` |
| 提出requirements検査 | 実装済み | `scripts/validate_submission_requirements.py` |
| Checkpoint Manifest | 実装済み | `build_checkpoint_manifest.py`、`checkpoint_manifest.schema.json` |
| Checkpoint取得後の自動棚卸し | 実装済み | `download_base_checkpoint.py` |
| Colab Preflight | 実装済み | `training/openvla_oft_a100/scripts/preflight.py` |
| Colab薄型Notebook | 5本実装済み | `notebooks/00`〜`04` |
| S0 / S1 / S2 Config | 実装済み | `training/openvla_oft_a100/configs/stage_a_*` |
| Notebook・YAML・Shell検査 | CI組込み済み | `.github/workflows/ci.yml` |
| 公式Validator固定Commit実行 | Notebook組込み済み | `notebooks/04_submission_validate.ipynb` |

## CI構成

### unit-and-contracts

- Python compile
- YAML parse
- Notebook JSON parse
- Shell syntax
- 提出requirements禁止依存
- Dataset pipeline tests
- RLDS contract tests
- Offline runtime tests
- Mini selection tests
- Checkpoint Manifest tests

### synthetic-rlds-e2e

- TensorFlow 2.15.1
- TFDS 4.9.3
- TF Metadata 1.15.0
- Protobuf 3.20.3
- Synthetic ParquetとFront/Wrist MP4生成
- TFDS/RLDS shard生成
- Train / Val読込
- Source State / Action parity
- Front / Wrist orientation parity

## CIで事前に検出・修正した事項

1. Repo rootが`PYTHONPATH`に入っておらず、`src`と`submission`をImportできなかった
   - Workflowの`PYTHONPATH`と`pytest.ini`で修正
2. TensorFlow 2.15と最新TF Metadata／Protobufの組合せが不整合だった
   - TF Metadata 1.15.0、Protobuf 3.20.3へ固定
3. TFDSが`src`をImplicit Namespace Packageとして解決し、Custom Builderのコード位置を特定できなかった
   - `src/__init__.py`を追加しRegular Package化
4. Mini Validationが1 EpisodeなのにParityを2 Episode要求していた
   - `PARITY_EPISODES_PER_SPLIT`を外部設定化
5. Mini実行で本番Dataset Manifestを昇格し得た
   - `PROMOTE_MANIFEST=0`のMini modeを追加

## 実環境待ち

- 公開Datasetによる3 Episode Mini E2E
- 800 Episode変換
- Base Checkpoint Manifest実値
- OpenVLA-OFT実Batch Transform
- A100 40GB学習
- L4 24GB推論
- 運営Validator

## Go条件

次を全て満たしてから800 EpisodeとStage Aへ進む。

```text
CI green
Mini Dataset payload complete
Source parity pass
RLDSBatchTransform pass
Checkpoint Manifest pass
A100 40GB preflight pass
```
