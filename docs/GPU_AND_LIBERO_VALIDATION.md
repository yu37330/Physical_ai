# 実Checkpoint GPU検証・LIBERO閉ループ評価ガイド

## 目的

このフェーズでは、Google Drive上の実Checkpointと変換済みRLDSをGoogle Colab GPUで読み込み、OpenVLA-OFT推論が実データで成立することを機械判定します。その後、公式LIBEROの`OffScreenRenderEnv`へ予測Actionを入力し、次観測で再推論する因果的な閉ループ1 Trialへ進みます。

GitHub ActionsではGPU、実Checkpoint、公式LIBERO Assetを利用できないため、CIは契約・停止条件・Trace保存をFake Runtimeで検証し、実GPU GateはColabで実施します。

## 実装ファイル

```text
src/agent_cockpit/gpu_validation.py
src/agent_cockpit/libero_executor.py
src/agent_cockpit/libero_closed_loop.py
scripts/run_colab_gpu_validation.py
scripts/run_libero_closed_loop.py
training/openvla_oft_a100/scripts/bootstrap_libero_colab.sh
notebooks/06_gpu_libero_validation.ipynb
tests/test_gpu_validation_and_libero.py
```

## 1. 実Checkpoint＋実RLDS GPU Gate

`run_colab_gpu_validation.py`は次を一つのJSONへ保存します。

- Git Commit
- Python、PyTorch、Transformers、TensorFlow、TFDSのVersion
- GPU名、Compute Capability、総VRAM
- Checkpoint pathとRLDS Builder directory
- Warm-up回数と検証Frame数
- Action chunk shape
- NaN／Inf検査
- 推論Latencyの平均、中央値、最小、最大
- Peak allocated／reserved VRAM
- 教師Action chunkに対するMAE、RMSE、軸別誤差

実行例:

```bash
python scripts/run_colab_gpu_validation.py \
  --checkpoint-dir /content/work/openvla_oft_plus \
  --dataset-dir /content/drive/MyDrive/PARC2026/20_processed/rlds/parc_libero_plus_selected/1.0.0 \
  --split val \
  --episode-offset 0 \
  --start-frame 0 \
  --num-frames 3 \
  --warmup-runs 1 \
  --output /content/drive/MyDrive/PARC2026/40_experiments/gpu_validation/latest_gpu_validation.json
```

標準ではCUDA必須です。`--allow-cpu`はTest Doubleや軽量Runtime確認用であり、実Checkpoint Gateには使いません。

## 2. LIBERO環境の準備

公式LIBEROをCommit固定でCloneします。

```bash
LIBERO_WORKDIR=/content/LIBERO \
OPENVLA_OFT_WORKDIR=/content/openvla-oft \
bash training/openvla_oft_a100/scripts/bootstrap_libero_colab.sh
```

標準Commit:

```text
Lifelong-Robot-Learning/LIBERO
8f1084e3132a39270c3a13ebe37270a43ece2a01
```

OpenVLA-OFT側は既存の`bootstrap_colab.sh`でCommit固定されています。Headless描画は`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`を使用します。

## 3. 因果的な閉ループ1 Trial

```bash
python scripts/run_libero_closed_loop.py \
  --checkpoint-dir /content/work/openvla_oft_plus \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --init-state-id 0 \
  --seed 7 \
  --max-steps 300
```

処理順:

```text
LIBERO Observation
  ↓
OfflinePolicy.get_action
  ↓
Action Safety Validator
  ↓
OffScreenRenderEnv.step
  ↓
Reward / Done / Next Observation
  ↓
再推論
```

これはRLDS Replayと異なり、予測Actionが次の環境状態を生成します。成功判定はLIBEROのSparse Rewardが正値になった場合です。

## Traceとコンプライアンス

標準では公式Benchmark観測画像を保存しません。Traceには、Instruction、観測KeyとShape、予測Action、推論時間、Safety結果、Reward、Done、Successを保存します。

```text
MyDrive/PARC2026/40_experiments/agent_cockpit/
└── libero_YYYYMMDD_HHMMSS/
    ├── run_manifest.json
    ├── timeline.jsonl
    ├── libero_closed_loop_summary.json
    └── steps/
        └── step_0000/
            ├── simulation_observation.json
            ├── policy_action.json
            ├── safety.json
            └── execution.json
```

`--persist-observation-images`を指定した場合だけ画像を保存します。公式Benchmark初期状態の数値本体は保存しません。評価結果や観測を学習データへ自動追加する処理も実装していません。

## 推奨Gate

1. GPU Reportが`pass`
2. Action chunkが`(8, 7)`で有限値
3. Peak VRAMがColab GPU容量内
4. 同一条件の再実行でLatencyとActionが大きく変動しない
5. LIBERO Task 0／Init state 0の1 Trialが例外なく終了
6. 10初期状態へ拡張
7. Task全体へ拡張
8. Checkpoint間の成功率・Latency・停止理由を比較

## 現在の制約

- 実GPUと実Checkpointの結果はColabで実行するまで確定しない
- LIBERO AssetとMuJoCo描画環境が必要
- 1 Trial CLIであり、全Task×複数初期状態の並列Benchmark集計は次フェーズ
- 成功判定は現時点ではLIBERO Reward依存
- 画像・物体位置を利用する独立Evaluatorは未実装
