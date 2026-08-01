# Physical AI / PARC2026

P​ARC2026予選に向けたVLAモデル開発、Google Colab学習、Google Drive成果物管理、ローカル評価、提出物作成、実験知識蓄積のための個人開発リポジトリです。

## 目的

- 運営配布環境と提出制約に適合する学習済みPolicy Serverを構築する
- GitHubをコード・設定・履歴の正本とする
- Google ColabをGPU学習・重い評価の実行環境とする
- Google Driveをモデル、データ、ログ、提出物の保管先とする
- React/FastAPIで実験結果を比較・管理できる基盤を構築する
- Markdown/JSONに実験知識を蓄積し、LLMが過去実験を参照して次の実験を提案できるようにする
- 公式評価由来の情報を学習データや自動最適化へ流入させない

## ドキュメント

- [基盤構想](docs/PLATFORM_ARCHITECTURE.md)
- [モデル対応方針](docs/MODEL_STRATEGY.md)
- [運営Repo調査結果](docs/OFFICIAL_REPOSITORY_NOTES.md)
- [提出・禁止事項チェック](docs/COMPLIANCE_AND_SUBMISSION.md)
- [提出までの実行計画](docs/DELIVERY_PLAN.md)

## 予定構成

```text
Physical_ai/
├── docs/
├── official_reference/
├── configs/
│   ├── datasets/
│   ├── experiments/
│   └── evaluation/
├── schemas/
├── notebooks/
├── src/
│   ├── data/
│   ├── training/
│   ├── policy/
│   ├── evaluation/
│   └── reporting/
├── submission/
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
7. 公式評価の観測、Seed、非公開タスク情報は保存・学習利用しない
8. 最終提出物とレポートの内容を一致させる
