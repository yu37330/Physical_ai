# Physical AI / PARC2026

PARC2026予選に向けたVLAモデル開発、Google Colab学習、Google Drive成果物管理、ローカル評価、提出物作成、実験知識蓄積のための個人開発リポジトリです。

## 目的

- 運営配布環境と提出制約に適合する学習済みPolicy Serverを構築する
- GitHubをコード・設定・履歴の正本とする
- Google ColabをGPU学習・重い評価の実行環境とする
- Google Driveをモデル、データ、ログ、提出物の保管先とする
- GradioでColab上のRLDS確認、OpenVLA推論、Agent Action提案を実行できる検証UIを構築する
- 将来はReact/FastAPIで実験結果を比較・管理できる常設基盤へ拡張する
- Markdown/JSONに実験知識を蓄積し、LLMが過去実験を参照して次の実験を提案できるようにする
- 公式評価由来の情報を学習データや自動最適化へ流入させない

## ドキュメント

- [GPU実行前の準備状況](docs/PRE_GPU_READINESS.md)
- [GPU実行前の実装ステータス](docs/PRE_GPU_IMPLEMENTATION_STATUS.md)
- [基盤構想](docs/PLATFORM_ARCHITECTURE.md)
- [Agent Cockpit設計・実行ガイド](docs/AGENT_COCKPIT.md)
- [モデル選定・最小調整方針](docs/MODEL_STRATEGY.md)
- [OpenVLA-OFT+の依存関係・オフライン推論成立性](docs/OPENVLA_OFFLINE_RUNTIME_ASSESSMENT.md)
- [オフライン推論・学習環境の次の実行手順](docs/OFFLINE_AND_TRAINING_NEXT_STEPS.md)
- [学習データ統合レポート](docs/LEARNING_DATA_REPORT.md)
- [学習データ実体調査・Dataset Manifest設計](docs/DATASET_INVENTORY_AND_MANIFEST.md)
- [LeRobot→RLDS変換・OpenVLA Batch互換性設計](docs/LEROBOT_TO_RLDS_AND_BATCH_COMPATIBILITY.md)
- [運営Repo調査結果](docs/OFFICIAL_REPOSITORY_NOTES.md)
- [提出・禁止事項チェック](docs/COMPLIANCE_AND_SUBMISSION.md)
- [提出までの実行計画](docs/DELIVERY_PLAN.md)

## Colab実行順

```text
00_environment_check.ipynb
→ 01_model_feasibility.ipynb
→ 02_dataset_prepare.ipynb
→ 03_stage_a_train.ipynb
→ 04_submission_validate.ipynb
→ 05_agent_cockpit.ipynb（RLDS・推論・Agent検証）
```

詳細は[`notebooks/README.md`](notebooks/README.md)を参照してください。

## Agent Cockpitの起動

`notebooks/05_agent_cockpit.ipynb`をColabで開き、Google Driveをマウントして起動します。初期状態は`propose`モードで、AIは次のActionを提案しますがExecutorへ自動送信しません。

主要画面：

- Dashboard
- Dataset Explorer
- Inference
- Agent Cockpit
- Run Trace

現在のMVPでは、Dataset Manifestと選定Episodeの表示、変換済みRLDSのFront/Wrist画像・State・Action chunk読込、既存OpenVLAオフラインRuntimeによる推論、教師Actionとの誤差比較、Safety確認、Google Drive Trace保存まで実装しています。

Traceは`MyDrive/PARC2026/40_experiments/agent_cockpit/`へ保存します。観測画像も各Step配下へPNGとしてコピーするため、一時的なGradio upload pathには依存しません。

## 事前CI Gate

次の2 JobはGitHub Actionsで通過実績があります。

```text
unit-and-contracts
synthetic-rlds-e2e
```

Synthetic E2Eでは、Parquet・Front/Wrist MP4生成、TFDS/RLDS shard生成、Train/Val読込、State/Action parity、画像方向保持までをCPUで検証します。

## 実装ディレクトリ

- `submission/openvla_oft_offline/`: OpenVLA-OFT+提出用オフライン推論ランタイム
- `training/openvla_oft_a100/`: Google Colab A100 40GB向け学習環境、RLDS変換・Batch互換検証
- `src/data/`: Dataset棚卸し、Episode選定、Mini E2E、LeRobot→RLDS変換、Manifest生成・昇格
- `src/agent_cockpit/`: Dataset Explorer、OpenVLA Adapter、Planner、Safety Validator、Evaluator、Trace保存、Agent Orchestrator
- `frontend/gradio_app.py`: Colab上で起動するRLDS・推論・Agent検証UI
- `configs/agent_cockpit.example.yaml`: 自律レベル、Action制約、Drive保存先の設定例
- `configs/models/`: Checkpointの必要ファイル・容量・提出対象契約
- `notebooks/`: Scriptを順番に呼び出すColab薄型Notebook
- `.github/workflows/`: Unit、契約、Synthetic RLDS E2Eの自動検証
- `tests/`: 前処理、Action chunk、Dataset pipeline、RLDS契約、Checkpoint Manifest、Agent制約、Policy Adapterのテスト

## 予定構成

```text
Physical_ai/
├── .github/workflows/
├── docs/
├── official_reference/
├── configs/
├── schemas/
├── notebooks/
├── src/
│   ├── data/
│   └── agent_cockpit/
├── submission/
│   └── openvla_oft_offline/
├── training/
│   └── openvla_oft_a100/
├── frontend/
├── backend/
└── tests/
```

## 運用原則

1. 実装はVS Codeで行い、GitHubへCommitする
2. Colabは固定CommitをCheckoutして実行する
3. 学習・評価条件はYAML/JSONで管理する
4. 大容量データとモデル重みはGitHubへ保存しない
5. 各実験に`run_manifest.json`を残す
6. 各データセットに`dataset_manifest.json`を残す
7. 各モデルに`checkpoint_manifest.json`を残す
8. Agentの各StepにObservation、Proposal、Safety、Execution、EvaluationのTraceを残す
9. 公式評価の観測、Seed、非公開タスク情報は保存・学習利用しない
10. 最終提出物とレポートの内容を一致させる
11. Mini Datasetと全事前Gateを通過してから800 Episode変換とStage A学習へ進む
12. 実機ExecutorはSafety制約と人の承認を実装するまで接続しない
