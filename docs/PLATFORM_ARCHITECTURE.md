# PARC2026 個人開発基盤構想

## 1. 目的

本基盤は、PARC2026予選に向けたVLAモデル開発を、GitHub・VS Code・Google Colab・Google Drive・Gradio・将来のReact/FastAPI・LLMで一貫管理するためのものです。学習データ、モデル、評価結果、判断履歴を追跡可能にしつつ、公式評価由来の情報を学習へ戻さない構造を採用します。

## 2. 基本アーキテクチャ

```text
VS Code
  ├─ Python実装
  ├─ Gradio / 将来のReact・FastAPI
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
  ├─ Agent Cockpit
  └─ ローカル公開タスク評価
       │ 成果物・Trace
       ▼
Google Drive
  ├─ Dataset
  ├─ Checkpoint / Adapter
  ├─ Run log / Metrics
  ├─ Agent step trace
  ├─ Submission ZIP
  └─ Report
       │ JSON/Markdownを参照
       ▼
Gradio Agent Cockpit（MVP）
  ├─ Dashboard
  ├─ Dataset Explorer
  ├─ Inference
  ├─ Agent Cockpit
  └─ Run Trace
       │ 将来の常設化
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

- Python、Notebook、Gradio、将来のReact/FastAPI、Config、Schema、Testを管理
- Colab実行時に使用するCommit SHAを固定
- 実験設定を`configs/experiments/*.yaml`として保存
- 提出モデルをGit TagとSHA256で識別
- 大容量モデル、軌道動画、データセットは保存しない

### VS Code

- 日常の実装・レビュー・テスト
- Google Drive同期フォルダ内ではなく通常のローカルフォルダでGitを運用
- 公式Colab接続またはNotebook経由で、重い処理だけColab GPUへ寄せる

### Google Colab

- GitHubから固定CommitをClone
- 学習中のCheckpointは`/content/work`へ保存
- Best/Last Checkpointと結果だけをGoogle Driveへ転送
- Notebookは薄い実行ラッパーとし、ロジックは`src/`へ配置
- Agent Cockpitは`notebooks/05_agent_cockpit.ipynb`から起動する

### Google Drive

推奨フォルダ構成：

```text
MyDrive/PARC2026/
├── 00_admin/
├── 10_raw/
├── 20_processed/
├── 30_models/
├── 40_experiments/
│   └── agent_cockpit/
├── 50_reports/
├── 60_submissions/
└── 90_restricted_official_eval/
```

`90_restricted_official_eval/`は、公式評価の提出日時、提出ZIPのHash、公開されたスコアのみを保存する隔離領域です。評価時の観測画像、Seed、非公開タスクを識別できる情報は保存しません。

### Gradio Agent Cockpit

初期フェーズでは、Colab上のPython処理と同じプロセスで動くGradioを採用します。ReactとFastAPIを先に分離せず、まずデータ、モデル、Action、Traceの契約を確定します。

主要画面：

- Dashboard
- Dataset Explorer
- Inference
- Agent Cockpit
- Run Trace

Agent Cockpitは次の責務を分離します。

```text
Observation
  ↓
High-level Planner
  ↓
OpenVLA Policy
  ↓
Safety Validator
  ↓
Executor
  ↓
Evaluator
  ↓
Drive Trace / Replan
```

初期状態は`propose`モードとし、Actionは表示・保存するだけで自動実行しません。実機Executorは、Safety制約、緊急停止、人の承認が実装されるまで接続しません。

### FastAPI / React

モデル、Dataset、Trace形式が安定した後に常設UIへ移行します。最初はGoogle Drive for Desktopの同期フォルダまたはDrive APIをFastAPIから読み取り、Gradioで確定したドメインモデルとAPI契約を再利用します。

主要画面：

- Dashboard
- Experiments
- Experiment Comparison
- Datasets
- Models
- Agent Trace
- Submissions

### LLM

LLMの「継続学習」は、当初はFine-tuningではなく、実験Markdown/JSONを検索するRAG方式とします。

入力可能：

- 学習設定
- Dataset Manifest
- Training logの要約
- ローカル評価結果
- AgentのGoal、Subgoal、Action Proposal、Safety、Evaluation
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

### Agent Step Trace

```text
run_manifest.json
timeline.jsonl
steps/step_0000/observation.json
steps/step_0000/proposal.json
steps/step_0000/safety.json
steps/step_0000/execution.json
steps/step_0000/evaluation.json
```

Agentが次のActionを自己決定する場合でも、内部の自由記述思考過程ではなく、Goal、Subgoal、候補Action、Score、選択Action、制約結果、期待結果、実結果を構造化して残します。

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
- Agent Actionの次元、有限値、上限制約
- ProposalモードでExecutorが呼ばれないこと
- Agent TraceがRun/Step単位で保存されること
- `policy_server.py`の構文と必須Endpoint
- `/health` → `/reset` → `/act`のスモークテスト
- Actionが`float32 shape (7,)`
- 外部URL依存や危険なRequirement指定の検出
- 運営`validate_submission.py`による提出ZIP検証

GPUモデル本体を使う試験はColabで行い、CIではDummy Policyまたは小型Fixtureを使います。

## 7. 開発順序

1. 運営Repoの評価環境を再現
2. ランダムPolicyで公開評価を実行
3. OpenVLA-OFTをPolicy Serverへ接続
4. 選定EpisodeでLoRA学習
5. 提出ZIPをValidation PASS
6. ManifestとDrive保存を自動化
7. Gradio Agent CockpitでDataset、推論、Action Traceを確認
8. OpenVLA Policy AdapterとSimulation Executorを接続
9. 制約付き自律ループを実装
10. React/FastAPIへ常設UIを拡張
11. 独自学習データ生成
12. LLMによる実験比較・次Action提案
