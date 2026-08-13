# PARC2026 Colab Notebooks

Notebookは実装ロジックを持たず、Repo内のScriptを順番に実行するための薄いUIです。実処理は`training/openvla_oft_a100/scripts/colab_*.sh`にあり、NotebookとColab Terminalの両方から同じScriptを呼びます。

VS Codeの公式Colab拡張機能とColab Terminalを使う手順は[VS Code + Colab拡張機能での実行](../docs/VSCODE_COLAB_WORKFLOW.md)を参照してください。

## 実行順

| Notebook | 呼び出すScript | 内容 |
|---|---|---|
| `00_environment_check.ipynb` | `colab_preflight.sh` | Drive mount、`main`取得、GPU・Python・Disk・Commitの事前確認 |
| `01_model_feasibility.ipynb` | `colab_setup.sh` | 固定OpenVLA-OFT環境、提出用Prismatic Vendor、Base Checkpoint、Checkpoint Manifest |
| `02_dataset_prepare.ipynb` | `colab_dataset_prepare.sh` | Mini Selection、Episode取得、LeRobot→RLDS、Source parity、Batch compatibility |
| `03_stage_a_train.ipynb` | `colab_stage_a.sh` | S1 100 steps、Gate通過時のみS2 500 steps |
| `04_submission_validate.ipynb` | `colab_submission_validate.sh` | 提出依存検査、ZIP、SHA256、運営Validator |
| `05_agent_cockpit.ipynb` | — | Gradio UI（Dataset Explorer、推論、Agent、自律Replay） |
| `06_gpu_libero_validation.ipynb` | `run_colab_gpu_validation.py` | 実Checkpoint GPU Gate、任意でLIBERO因果閉ループ1 Trial |
| `07_patch_submission_requirements.ipynb` | `colab_patch_submission.sh` | 既存の提出ZIP内の`requirements.txt`だけを差し替え、運営Validatorで再検査 |

`07`は通常の実行順には入りません。重みは正しいまま依存だけを直したい場合に使います。組み立てからやり直す`04`と違い、Base重みの再取得もCheckpointの再組み立ても不要です。

## 無料T4での事前スモーク

A100を確保する前に、`colab_smoke.sh`で配管だけ確認できます。7B Checkpointのダウンロードは不要で、Processorとconfigの約2.5MBだけ取得します。

```bash
bash training/openvla_oft_a100/scripts/colab_smoke.sh
```

Stage A学習、Notebook 06のGPU Gate、提出ZIPは実7B Checkpointが必要なためT4では確認できません。

## 保存先

```text
/content/work   Dataset、Base Checkpoint、学習中間生成物、提出ZIP（揮発、約100GB）
Google Drive    Manifest、レポート、学習済みcomponent、SHA256（永続、小容量）
```

提出ZIPは重み込みで十数GBになるためDriveへ置かず、`/content/work/submission`から直接ローカルPCへダウンロードします。

## 原則

- Notebook内で学習コードを直接編集しない
- 実装変更はVS CodeとGitHubで行う
- Notebook実行前にBranchとCommitを確認する
- Terminalにしか存在しない実行手順を作らない
- 大容量処理は`/content/work`で行い、完成物だけGoogle Driveへ保存する
- S1が失敗した場合はS2を実行しない
- Agent Cockpitは初期状態では`propose`モードとし、実機Executorを接続しない
- 実Checkpoint GPU Gateが通る前にLIBERO Task数を増やさない
- 公式LIBERO観測や初期状態を学習データへ追加しない
