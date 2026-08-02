# Physical AI Agent Cockpit 設計・実行ガイド

## 目的

この改修は、Google ColabをGPU実行環境、Google Driveを永続ストレージ、GitHubをコード・設定の正本として、VLAの観測、計画、Action提案、安全確認、実行後評価、再計画を一つのUIから追跡できるようにするものです。既存の学習、RLDS変換、提出用オフライン推論コードには手を入れず、独立した検証レイヤーとして追加します。

## 画面構成

- Dashboard: Model、Dataset、Run ID、Drive保存先を確認する
- Dataset Explorer: Episode、画像、State、Action、Action chunk、Tensor shapeを確認する接続点
- Inference: OpenVLA-OFT単発推論、正解Actionとの差、推論時間を確認する接続点
- Agent Cockpit: 最終目標、サブゴール、候補Action、Safety、Evaluatorを確認する
- Run Trace: Step履歴、失敗分析、モデル比較を表示する接続点

## Agentの責務分離

```text
Observation
  ↓
High-level Planner
  ↓
OpenVLA Policy Adapter（次フェーズで接続）
  ↓
Safety Validator
  ↓
Human Approval / Constrained Executor
  ↓
Evaluator
  ↓
Google Drive Trace
  ↓
Replan
```

`RuleBasedPlanner`はUI、保存形式、承認フローをGPUなしで検証するための仮実装です。OpenVLA接続時は、同じ`propose(observation, goal)`契約を満たすPolicy/Plannerへ差し替えます。

## Google Colabでの起動

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone -b agent/add-colab-agent-cockpit https://github.com/yu37330/Physical_ai.git
%cd /content/Physical_ai
!pip install -r requirements-frontend.txt
```

```python
import os
os.environ['PHYSICAL_AI_DRIVE_ROOT'] = '/content/drive/MyDrive/PARC2026'

from frontend.gradio_app import build_app
build_app().launch(share=True, debug=True)
```

## Trace保存先

```text
MyDrive/PARC2026/40_experiments/agent_cockpit/
└── agent_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    ├── timeline.jsonl
    └── steps/
        └── step_0000/
            ├── observation.json
            ├── proposal.json
            ├── safety.json
            ├── execution.json
            └── evaluation.json
```

画像本体は大きいため、Agent Traceでは原則として参照パスを記録します。必要なSample画像だけをRun配下へコピーする運用にします。

## 自律レベル

- `observe`: 観測と表示のみ
- `propose`: 次Actionを提案し、実行しない
- `approval`: 人の承認後だけExecutorへ渡す
- `constrained_auto`: Safety制約を通過したActionのみ自動実行する

現段階のMVPは`propose`を標準とします。実機Executorは未接続であり、`approval`または`constrained_auto`を選んでも、Executorが設定されていなければ実行されません。

## 可視化の優先順位

1. Observation画像、Instruction、State
2. 現在GoalとSubgoal
3. 候補Action、Score、選択Action、Action chunk
4. Action各軸の時系列と急変
5. Safety制約結果
6. 期待状態と実状態の差
7. 同一Action反復、停滞、不確実性
8. Run間の成功率、平均ステップ数、推論時間、GPUメモリ

## 次の接続作業

1. Dataset ManifestからEpisode一覧を読み込む
2. RLDS SampleをDataset Explorerへ表示する
3. OpenVLA-OFTのPolicy Adapterを実装する
4. Action chunkの軸別プロットを追加する
5. Simulation Executorを接続する
6. Evaluatorを画像・Stateベースへ拡張する
7. 制約内自律ループと停止条件を実装する

公式評価由来の観測、Seed、非公開タスク情報はTraceへ保存せず、学習や自動最適化にも利用しません。
