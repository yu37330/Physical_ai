# PARC2026 運営リポジトリ調査結果

調査対象：`matsuolab/PARC2026_pre`

確認時点：2026-08-02

## 1. リポジトリの役割

運営Repoは、完成済み競技モデルではなく、以下を提供する自己評価・提出準備キットです。

- Track 1の公開Exampleタスク評価
- 提出用HTTP Policy Serverテンプレート
- 提出ZIPの静的検査・起動スモークテスト
- 本番と同じ成功判定・推論タイムアウト
- SmolVLA LoRA学習の参考Notebook

本番は公開されていないタスクを含む別タスクセットで実施され、ランキング用の正規化・重み設定は含まれません。

## 2. 参照元

- Official repository: `https://github.com/matsuolab/PARC2026_pre`
- 調査Commit: `fcac626949f52c0edb4b0701bc75e87464e36d64`

運営RepoをこのRepoへ丸ごと複製せず、必要な仕様情報・固定Commit・取得手順を管理します。ライセンスや更新追従を明確にするためです。

## 3. 公開Exampleタスク

`compe/t1/T1_TASKS.csv`では4タスクが公開されています。

| No. | 内容 | Suite | 変化 | 難易度 |
|---:|---|---|---|---:|
| 1 | キャビネット上段の黒いボウルを皿へ | libero_spatial | Background Textures | L3 |
| 2 | トマトソースをバスケットへ | libero_object | Background Textures | L5 |
| 3 | 牛乳をバスケットへ | libero_object | Light Conditions | L2 |
| 4 | ボウルをコンロへ | libero_goal | Light Conditions | L4 |

公開4タスクだけへの過学習を避け、Spatial/Object/Goalと見た目変化を広くカバーする必要があります。

## 4. Policy Server仕様

必須Endpoint：

- `GET /health`
- `POST /reset`
- `POST /act`

観測：

- `agentview_image`: `(128, 128, 3) uint8`
- `robot0_eye_in_hand_image`: `(128, 128, 3) uint8`
- `robot0_joint_pos`: `(7,)`
- `robot0_eef_pos`: `(3,)`
- `robot0_eef_quat`: `(4,)`
- `robot0_gripper_qpos`: `(2,)`

Action：

- `(7,) float32`
- `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

テンプレートでは原則`MyPolicy`の中を置き換えます。サーバー・シリアライゼーション部分は変更しない方針とします。

## 5. 評価仕様

- 画像解像度：128×128
- 既定最大ステップ：600
- 既定評価回数：1タスク20エピソード
- 制御周期：20Hz、`dt=0.05`
- `/act`・`/reset`：1リクエスト10秒以内
- サーバー起動：既定120秒以内
- Track全体：1時間超でタイムアウト
- 評価GPU：単一NVIDIA L4、VRAM 24GB

## 6. 成功・衝突判定

成功は以下の両方を満たす必要があります。

1. LIBEROのタスクゴールを達成
2. 操作対象以外の物体を動かしていない

非対象物体について、初期位置からのXYZ各軸絶対差の合計が1mmを超えた場合、衝突扱いとなり失敗します。途中で元へ戻しても、最大変位で判定されるため失敗のままです。

## 7. 公開評価指標

公開Scorerは次を記録します。

- success rate
- collision rate
- average steps to success
- average episode time
- Cartesian path length
- Joint path length
- EEF orientation path length
- Cartesian/Joint jerk
- SPARC

ただし、公開コードの`overall_score`は現状、タスク平均成功率です。JerkやSPARCを含む本番用の重み・正規化設定は非公開です。

## 8. 運営Exampleの学習条件

Notebook：`examples/smolvla_libero_spatial_lora.ipynb`

- Base model: `lerobot/smolvla_libero_plus`
- Base revision: `7bb70aa5bc92b82c9239142775d3a173103567ff`
- Dataset: `lerobot/libero_plus`
- Dataset revision: `f3f49f426d75030177b18778374005bc12ccd588`
- VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- LeRobot: v0.6.0
- Spatial 10タスク×5エピソード＝50エピソード
- 3,000 steps
- batch size 1
- learning rate `3e-4 → 3e-5`
- warmup 100
- LoRA rank 16
- LoRA alpha 16
- Vision Encoder freeze
- Expert側を中心に学習

Notebook評価は256×256、各タスク3エピソードであり、提出評価とは異なります。出力モデルもそのまま提出ZIPにはならず、`MyPolicy`への組込みが必要です。

## 9. 提出制約

- ZIP最大20GB
- 展開後最大40GB
- `policy_server.py`必須
- `requirements.txt`必須
- 暗号化ZIP不可
- Zip bomb不可
- Symbolic link / Hard link不可
- Path traversal不可
- RequirementsでGit/HTTP/File URLなど外部ソース指定不可
- モデル重みと必要な独自コードはZIPへ同梱する

## 10. 取得・更新方針

ローカル開発時は運営Repoを別ディレクトリへCloneします。

```bash
git clone https://github.com/matsuolab/PARC2026_pre.git vendor/PARC2026_pre
cd vendor/PARC2026_pre
git checkout fcac626949f52c0edb4b0701bc75e87464e36d64
```

`vendor/`は原則Git管理対象外とし、このRepoには以下だけを残します。

- 固定Commit
- 必要ファイル一覧
- SHA256
- 差分調査記録
- 取得スクリプト

運営Repo更新時は、固定Commitを明示的に変更し、仕様差分をレビューしてから採用します。
