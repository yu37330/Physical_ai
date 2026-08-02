# PARC2026 Colab Notebooks

Notebookは実装ロジックを持たず、Repo内のScriptを順番に実行するための薄いUIです。

## 実行順

1. `00_environment_check.ipynb`
   - Google Drive mount
   - Branch更新
   - A100 40GB、Disk、Git CommitのPreflight
2. `01_model_feasibility.ipynb`
   - 固定OpenVLA-OFT環境
   - Base Checkpoint取得
   - Checkpoint Manifest生成
   - 提出用Prismatic Vendor準備
3. `02_dataset_prepare.ipynb`
   - 3 Episode Mini Selection
   - 選択Payload取得
   - LeRobot→RLDS
   - Source parity
   - OpenVLA Batch compatibility
4. `03_stage_a_train.ipynb`
   - S0 Base inference
   - S1 100 steps
   - Gate通過時のみS2 500 steps
5. `04_submission_validate.ipynb`
   - 提出依存検査
   - 最終Checkpoint配置
   - ZIP、容量、SHA256
   - 運営Validator

## 原則

- Notebook内で学習コードを直接編集しない
- 実装変更はVS CodeとGitHubで行う
- Notebook実行前にBranchとCommitを確認する
- 大容量処理は`/content/work`で行い、完成物だけGoogle Driveへ保存する
- S1が失敗した場合はS2を実行しない
