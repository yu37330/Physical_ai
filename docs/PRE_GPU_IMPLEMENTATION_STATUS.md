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
- Synthetic ParquetとFront/Wrist MP4生成
- TFDS/RLDS shard生成
- Train / Val読込
- Source State / Action parity
- Front / Wrist orientation parity

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
