# PARC2026 個人開発基盤構想

## 1. 目的

本基盤は、P​ARC2026予選に向けたVLAモデル開発を、GitHub・VS Code・Google Colab・Google Drive・React/FastAPI・LLMで一貫管理するためのものです。学習データ、モデル、評価結果、判断履歴を追跡可能にしつつ、公式評価由来の情報を学習へ戻さない構造を採用します。

## 2. 基本アーキテクチャ

```text
VS Code
  ├─ Python実装
  ├─ React/FastAPI
  ├─ Config / Schema
  └─ Test
       │
       ▼
GitHub
  ├─ コード・設定の正本
  ├─ Commit / Tag / PR
  └─ CI・提出前検査
       │ 固定Commitを取得
       ▼
Google Colab
  ├─ データ準備
  ├─ LoRA学習
  ├─ Checkpointマージ
  ├─ GPU推論試験
  └─ ローカル公開タスク評価
       │ 成果物
       ▼
Google Drive
  ├─ Dataset
  ├─ Checkpoint / Adapter
  ├─ Run log / Metrics
  ├─ Submission ZIP
  └─ Report
       │ JSON/Markdownを参照
       ▼
FastAPI + React
  ├─ 実験一覧
  ├─ モデル比較
  ├─ Dataset台帳
  ├─ 失敗分析
  └─ 提出候補管理
       │ 許可済みデータのみ
       ▼
LLM Experiment Assistant
  ├─ 過去実験検索
  ├─ 差分分析
  ├─ 次実験提案
  └─ レポート下書き
```

## 3. 各コンポーネントの責務

### GitHub

- Python、Notebook、React、FastAPI、Config、Schema、Testを管理
- Colab実行時に使用するCommit SHAを固定
- 実験設定を`configs/experiments/*.yaml`として保存
- 提出モデルをGit TagとSHA256で識別
- 大容量モデル、軌道動画、データセットは保存しない

### VS Code

- 日常の実装・レビュー・テスト
- Google Drive同期フォルダ内ではなく通常のローカルフォルダでGitを運用
- 例：`C:\work\Physical_ai`

### Google Colab

- GitHubから固定CommitをClone
- 学習中のCheckpointは`/content/work`へ保存
- Best/Last Checkpointと結果だけをGoogle Driveへ転送
- Notebookは薄い実行ラッパーとし、ロジックは`src/`へ配置

### Google Drive

推奨フォルダ構成：

```text
MyDrive/PARC2026/
├── 00_admin/
├── 10_raw/
├── 20_processed/
├── 30_models/
├── 40_experiments/
├── 50_reports/
├── 60_submissions/
└── 90_restricted_official_eval/
```

`90_restricted_official_eval/`は、公式評価の提出日時、提出ZIPのHash、公開されたスコアのみを保存する隔離領域です。評価時の観測画像、Seed、非公開タスクを識別できる情報は保存しません。

### FastAPI / React

最初はGoogle Drive for Desktopの同期フォルダをFastAPIから読み取ります。Drive APIへの移行は、複数端末・自動実行が必要になった段階で行います。

主要画面：

- Dashboard
- Experiments
- Experiment Comparison
- Datasets
- Models
- Submissions

### LLM

LLMの「継続学習」は、当初はFine-tuningではなく、実験Markdown/JSONを検索するRAG方式とします。

入力可能：

- 学習設定
- Dataset Manifest
- Training logの要約
- ローカル評価結果
- コード差分
- 人間の判断理由

入力禁止：

- 公式評価時の観測
- 非公開タスク情報
- 評価Seed
- Fingerprintingに利用できる情報
- 評価結果を直接教師データ化した情報

## 4. データ管理

### Dataset Manifest

すべてのデータセットに以下を持たせます。

```json
{
  "dataset_id": "ds_example_v001",
  "source": "public_or_self_generated",
  "source_uri": "",
  "revision": "",
  "license": "",
  "generation_method": "public_dataset",
  "task_count": 0,
  "episode_count": 0,
  "allowed_for_training": true,
  "contains_official_evaluation_data": false,
  "sha256": ""
}
```

### Run Manifest

```json
{
  "run_id": "run_YYYYMMDD_NNN",
  "git_sha": "",
  "base_model": {
    "name": "",
    "revision": ""
  },
  "datasets": [],
  "training": {},
  "inference": {},
  "metrics": {},
  "checkpoint_drive_path": "",
  "checkpoint_sha256": "",
  "eligible_for_submission": false
}
```

## 5. 実験知識の蓄積

各Runに以下を保存します。

```text
run_manifest.json
config.yaml
metrics.json
task_metrics.csv
training_summary.md
failure_analysis.json
experiment_decision.md
```

`experiment_decision.md`には、仮説、変更内容、結果、採用判断、次の実験を残します。LLMはこの情報から過去実験を比較します。

## 6. CI方針

GitHub Actionsでは次を確認します。

- Ruff / Pyright / Pytest
- JSON Schema検証
- Dataset Manifestの学習利用可否
- `policy_server.py`の構文と必須Endpoint
- `/health` → `/reset` → `/act`のスモークテスト
- Actionが`float32 shape (7,)`
- 外部URL依存や危険なRequirement指定の検出
- 運営`validate_submission.py`による提出ZIP検証

GPUモデル本体を使う試験はColabで行い、CIではDummy Policyまたは小型Fixtureを使います。

## 7. 開発順序

1. 運営Repoの評価環境を再現
2. ランダムPolicyで公開評価を実行
3. SmolVLAをPolicy Serverへ接続
4. 公式50エピソードでLoRA学習
5. 提出ZIPをValidation PASS
6. ManifestとDrive保存を自動化
7. React/FastAPIへ結果を表示
8. 独自学習データ生成
9. LLMによる実験比較・提案
