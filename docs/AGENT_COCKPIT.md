# Physical AI Agent Cockpit 設計・実行ガイド

## 目的

この改修は、Google ColabをGPU実行環境、Google Driveを永続ストレージ、GitHubをコード・設定の正本として、VLAの観測、Action chunk推論、安全確認、実行後評価、次Action提案を一つのUIから追跡できるようにするものです。既存の学習、LeRobot→RLDS変換、提出用オフライン推論ランタイムは正本として維持し、Agent Cockpitはそれらを呼び出す独立した検証レイヤーとして実装します。

## 現在の実装範囲

実装済み:

- Dataset Manifest、変換レポート、OpenVLA互換性レポートの統合表示
- 選定Episode一覧の表示
- TFDS Builder directoryからのRLDS Episode・Frame読込
- Front/Wrist画像、State `(8,)`、Action `(7,)`、教師Action chunk `(8, 7)`の表示
- 既存`OpenVLAOfflineRuntime`を再利用したAction chunk推論
- 予測Action chunkと教師Action chunkのMAE・RMSE・軸別誤差
- Action chunkの時系列プロット
- OpenVLA Action chunk先頭を次Actionとして提案するPlanner
- Safety Validator、承認Gate、Evaluator、Google Drive Trace
- 観測画像をStep配下へPNGとして永続保存

未接続:

- Simulation Executor
- 実機Executor
- 画像ベースの成功判定
- 複数Stepを自動反復するAgent Loop
- OpenVLAの校正済みConfidence

## 画面構成

### Dashboard

Model、Dataset、Run ID、Google Drive保存先を設定します。最初にRunを作成してください。

### Dataset Explorer

次を読み込みます。

- `dataset_manifest.json`
- RLDS変換レポート
- OpenVLA Batch互換性レポート
- Episode選定JSON
- TFDS Builder directory

Episode offsetとFrame IDを指定すると、Front/Wrist画像、State、Action、終端Padding済み教師Action chunkを表示します。読み込んだSampleはGradio Stateに保持され、InferenceとAgent Cockpitで再利用できます。

### Inference

Google Drive上のCheckpoint directoryを指定し、既存の提出用`OpenVLAOfflineRuntime`からAction chunkを取得します。Dataset Explorer Sampleを使用する場合は、画像・State・Instruction・教師Action chunkが自動的に引き継がれます。

表示内容:

- 予測Action chunk
- 推論時間
- Checkpoint path
- 教師Action chunkとのMAE・RMSE
- 7軸ごとのMAE・RMSE
- 予測／教師Action chunkプロット

### Agent Cockpit

Plannerは次から選択できます。

- `rule_based`: GPUなしでUIとTraceを確認する仮Planner
- `openvla`: OpenVLA Action chunkの先頭Actionを次Actionとして提案

OpenVLA PlannerのAction Proposalには、先頭Actionだけでなく、Action chunk全体、Shape、推論時間、Checkpoint pathをMetadataとして残します。

### Run Trace

各Stepの観測、提案、安全判定、実行状態、評価と画像をGoogle Driveへ保存します。

## Agentの責務分離

```text
Observation
  ├─ Front / Wrist image
  ├─ State
  └─ Instruction
       ↓
Planner
  ├─ RuleBasedPlanner
  └─ OpenVLAPolicyPlanner
       ↓
OpenVLA Policy Adapter
  ├─ Existing OpenVLAOfflineRuntime
  ├─ Action chunk
  └─ Inference latency
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

`AgentOrchestrator`はPlannerをProtocol化しており、Rule-basedとOpenVLAを同じ`propose(observation, goal)`契約で差し替えます。

## 画像方向の扱い

LeRobot版LIBERO画像は保存時点で180度回転済みであり、RLDS変換では追加回転を行いません。提出用Runtimeの`build_policy_input()`は生のLIBERO観測を対象として180度回転を含むため、変換済みRLDS画像へそのまま適用すると二重回転になります。

Agent Cockpitの`OpenVLAPolicyAdapter.predict_rlds()`では、RLDS画像を再回転せず、ResizeとCenter Cropだけを実施して`PolicyInput`を構築します。生観測を使う提出用Policy Serverの前処理は変更していません。

## Google Colabでの起動

推奨は[`notebooks/05_agent_cockpit.ipynb`](../notebooks/05_agent_cockpit.ipynb)です。

手動起動する場合:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone -b agent/add-colab-agent-cockpit https://github.com/yu37330/Physical_ai.git
%cd /content/Physical_ai
!pip install -r training/openvla_oft_a100/requirements-data.txt
!pip install -r requirements-frontend.txt
```

```python
import os
os.environ['PHYSICAL_AI_DRIVE_ROOT'] = '/content/drive/MyDrive/PARC2026'

from frontend.gradio_app import build_app
build_app().launch(share=True, debug=True)
```

OpenVLA推論を使う場合は、`01_model_feasibility.ipynb`で確認した依存環境とCheckpointを利用してください。Dataset ExplorerとRule-based PlannerはGPUモデルをロードせずに確認できます。

## Google Driveの推奨配置

```text
MyDrive/PARC2026/
├── 20_processed/
│   ├── dataset_manifest.json
│   ├── libero_plus_selection_v001.json
│   ├── rlds_conversion_report.json
│   ├── openvla_compatibility_report.json
│   └── rlds/
│       └── parc_libero_plus_selected/1.0.0/
├── 30_models/
│   └── openvla_oft_plus/
└── 40_experiments/
    └── agent_cockpit/
```

Dataset Explorerの`TFDS Builder directory`には、変換レポートの`data_dir`、または`dataset_info.json`が存在するBuilder directoryを指定します。

## Trace保存先

```text
MyDrive/PARC2026/40_experiments/agent_cockpit/
└── agent_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    ├── timeline.jsonl
    └── steps/
        └── step_0000/
            ├── front_image.png
            ├── wrist_image.png
            ├── observation.json
            ├── proposal.json
            ├── safety.json
            ├── execution.json
            └── evaluation.json
```

Dataset Explorer Sampleと手動アップロード画像のどちらも、Agent Step実行時にRun配下へコピーします。Traceが一時的なGradio upload pathへ依存しないようにしています。

## 自律レベル

- `observe`: 観測と表示のみ
- `propose`: 次Actionを提案し、実行しない
- `approval`: 人の承認後だけExecutorへ渡す
- `constrained_auto`: Safety制約を通過したActionのみExecutorへ渡す

現段階の標準は`propose`です。実機Executorは未接続であり、`approval`または`constrained_auto`を選んでも、`executor_not_configured`として停止します。

## OpenVLA Scoreと不確実性

現時点のOpenVLA Runtimeは校正済みConfidenceを返しません。`OpenVLAPolicyPlanner`の候補Scoreは候補内の優先順位を表すだけで、成功確率ではありません。Proposal Metadataには`confidence_available: false`を記録し、`uncertainty: -1.0`を「未提供」のSentinelとして使用します。

## 次の実装

1. Simulation Executorを接続する
2. `Observe → Plan → Act → Evaluate → Replan`の複数Step Loopを追加する
3. 最大Step、同一Action反復、停滞、Safety違反による停止条件を実装する
4. State差分だけでなく画像・物体位置を使うEvaluatorを追加する
5. Run Trace Timelineと失敗分析をUIへ表示する
6. 複数CheckpointのAction chunk・成功率・推論時間を比較する

公式評価由来の観測、Seed、非公開タスク情報はTraceへ保存せず、学習や自動最適化にも利用しません。
