# PARC2026 モデル対応方針

## 1. 競技の対象

本競技は、ロボット本体の設計ではなく、言語指示・カメラ画像・ロボット状態から7次元Actionを生成するVLA/Policyの性能を競います。

提出Policyの入力：

- `agentview_image`: 128×128×3
- `robot0_eye_in_hand_image`: 128×128×3
- `robot0_joint_pos`: 7次元
- `robot0_eef_pos`: 3次元
- `robot0_eef_quat`: 4次元
- `robot0_gripper_qpos`: 2次元
- エピソード開始時の言語指示

出力：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

## 2. 初期モデル方針

最初は運営Exampleと互換性が高い以下を固定します。

- Base model: `lerobot/smolvla_libero_plus`
- VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Framework: LeRobot v0.6.0系
- 学習方法: LoRA
- 初期条件: 3,000 steps、batch size 1、LoRA rank 16、alpha 16
- Vision Encoder: Freeze
- Expert側を中心に学習

モデル比較は、SmolVLAのEnd-to-End提出が成立してから行います。

## 3. 最優先課題

1. SmolVLAを提出用`MyPolicy`へ接続する
2. 128×128の2視点画像と状態をSmolVLA Featureへ変換する
3. 出力Actionの正規化・逆正規化を一致させる
4. Action chunkをキャッシュし、1回の`/act`で1Actionを返す
5. `/act`と`/reset`を10秒以内にする
6. 外部ネットワークなしで起動・推論できるよう重みを同梱する

## 4. 学習データ方針

### Baseline

運営Notebookの最小構成：

- `lerobot/libero_plus`
- LIBERO Spatial 10タスク
- 各5エピソード
- 合計50エピソード
- 主に黒いボウルを異なる初期位置から皿へ移動するデモ

これは学習・マージ・推論・提出を一周させるためのBaselineであり、競技全体をカバーするデータではありません。

### 拡張方針

優先順位：

1. Spatial・Object・Goalのタスクカバレッジ
2. 対象物体の多様性
3. 目的地の多様性
4. 背景テクスチャ
5. 照明条件
6. 初期配置・高さ・向き
7. 衝突しない安全軌道
8. 言語指示の表現多様性

初期候補：

```text
Dataset v1: Spatial/Object/Goal 各50 episode = 150
Dataset v2: Spatial/Object/Goal 各100 episode = 300
```

Train/ValidationはFrame単位ではなく、EpisodeまたはTask Variant単位で分離します。

## 5. 独自データ生成

許可された方法を利用します。

- Simulator
- Teleoperation
- Scripted Policy
- データ拡張
- 生成モデルによるデータ生成

Scripted Policyは教師データ生成のみに利用し、提出時のAction生成には利用しません。

生成パイプライン：

```text
Task template
→ Scene/Lighting/Texture/Position randomization
→ Scripted Policy or Teleoperation
→ Success/Collision/Trajectory validation
→ LeRobot形式へ変換
→ Dataset Manifest生成
→ 学習
```

## 6. Action制御

SmolVLAのAction chunkをすべてOpen-loopで実行せず、再計画間隔を比較します。

候補：

- 5
- 10
- 25
- 50

比較指標：

- 成功率
- 衝突率
- 平均ステップ数
- 最大推論時間
- Action振動
- Cartesian/Joint path length
- Jerk
- SPARC

Action Clipや平滑化はタスク非依存の一般処理に限定し、FSMやタスク固有ルールへ発展させません。

## 7. 実験計画

| Run | データ | 設定 | 目的 |
|---|---|---|---|
| R0 | 追加学習なし | 公開重み | 参考値 |
| R1 | 公式50 | 3k steps, r16 | E2E Baseline |
| R2 | 3 Suite 150 | 6k steps, r16 | カバレッジ |
| R3 | 3 Suite 300 | 10k steps, r16 | データ量 |
| R4 | R3 | 10k steps, r32 | LoRA容量 |
| R5 | 最良データ＋画像拡張 | 最良設定 | Texture/Light対策 |
| R6 | 最良条件 | Seed変更 | 再現性 |

一度に複数要因を変更せず、Ablation可能な形で実施します。

## 8. 評価方針

段階的に評価回数を増やします。

- Smoke: 各1 episode
- Screening: 各3 episodes
- Candidate: 各10 episodes
- Final: 各20 episodes

選定順序：

1. Error/Timeoutなし
2. 成功率
3. 衝突率
4. 最低タスク成功率
5. 最大推論時間
6. ステップ数・経路長
7. Jerk/SPARC

本番の正規化・重みが非公開のため、独自総合スコアだけで判断しません。

## 9. 提出候補Hard Gate

- `/act`最大8秒未満
- `/reset`最大8秒未満
- 起動90秒未満
- `float32 shape (7,)`
- NaN/Infなし
- ZIP 18GB未満を目標
- `validate_submission.py` PASS
- 外部ネットワーク依存なし
- 独自学習重みが推論時に実質的に使用される
- レポートと実装・重み・データ情報が一致する

## 10. 禁止事項を踏まえた実装境界

禁止：

- タスク固有FSM
- ハードコードAction列
- Scene/Task/SeedをキーにしたActionテーブル
- 成功条件・報酬の直接参照
- 非公開タスクFingerprinting
- 公開重みを実質変更せずそのまま推論
- 学習済みモデルを使わないFallback Policy
- 公式評価観測を保存して追加学習
- 評価中の外部ネットワークアクセス

安全な構成：

```text
公開VLA
→ 許可されたデータで独自学習
→ 学習済みAction生成
→ タスク非依存の正規化・Clip・Chunk Cache
→ 7次元Action
```
