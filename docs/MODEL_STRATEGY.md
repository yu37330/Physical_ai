# PARC2026 モデル選定・最小調整方針

更新日: 2026-08-02  
対象: PARC 2026 予選 Track 1  
開発環境: VS Code / GitHub / Google Colab / Google Drive  
最大学習GPU: NVIDIA A100 40GB  
提出時GPU: NVIDIA L4 24GB

---

## 1. 結論

本プロジェクトのモデル方針は次で固定する。

> **今回の評価条件に近い環境ですでに強い事前学習済みVLAを選び、その素の性能を壊さない範囲で、競技要件を満たす最小限の独自学習だけを行う。**

### 採用順位

1. **本命: OpenVLA-OFT+ 7B**  
   `Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`
2. **Fallback: MolmoAct2-LIBERO-LeRobot 5B**  
   `allenai/MolmoAct2-LIBERO-LeRobot`
3. **軽量Fallback: X-VLA 0.9B**  
   `lerobot/xvla-libero`
4. **緊急提出用: SmolVLA 0.5B**  
   `lerobot/smolvla_libero_plus`

### 採用判断

第一候補は **OpenVLA-OFT+ 7B** とする。

主な理由は次のとおり。

- 今回の評価環境に近い **LIBERO-plusで直接評価された公開モデル**である。
- LIBERO-plus総合値が **79.6** で、公開された比較表の中で最上位である。
- 公開Exampleに含まれる条件と近いLightで **94.9**、Backgroundで **93.9** を記録している。
- 2カメラ、Proprioception、連続Action chunkを扱うOpenVLA-OFT構成が今回の入力仕様と合いやすい。
- OpenVLA-OFT公式のLIBERO推論要件は約16GB VRAMで、提出時のL4 24GBに収まる可能性が高い。
- 公開Checkpointリポジトリは約15.9GBで、提出ZIP上限20GBに収められる可能性がある。
- A100 40GBで、全面学習ではなくAction系モジュールと小規模LoRAの追加学習を試せる範囲にある。

ただし、公開Benchmarkだけで採用確定とはしない。次をすべて実測して初めて最終採用とする。

1. L4 24GBで安定してロード・推論できる
2. `/act`と`/reset`の最大応答時間が8秒未満
3. 提出ZIPが20GB未満
4. A100 40GBで独自学習を実行できる
5. 独自学習後も元のロバスト性を維持できる

---

## 2. 今回競うもの

本競技はロボット本体の設計競技ではなく、言語指示、カメラ画像、ロボット状態から7次元Actionを生成するVLA Policyの性能を競う。

### 入力

- `agentview_image`: 128×128×3
- `robot0_eye_in_hand_image`: 128×128×3
- `robot0_joint_pos`: 7次元
- `robot0_eef_pos`: 3次元
- `robot0_eef_quat`: 4次元
- `robot0_gripper_qpos`: 2次元
- エピソード開始時の言語指示

### 出力

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

### 特に重要な能力

1. 対象物体と目的地の認識
2. 言語指示と視覚情報の対応
3. 照明、背景、配置、視点変化への汎化
4. 周辺物体へ接触しない把持・搬送
5. 安定した7次元Action生成
6. Action chunkによる効率的な再計画

操作対象以外の物体を初期位置から1mm超動かすと失敗になるため、成功率だけでなくCollision rateを重視する。

---

## 3. 提出Hard Gate

公開されているPARC 2026評価Repoから、以下を絶対条件とする。

| 項目 | 公式上限 | 内部目標 |
|---|---:|---:|
| 提出ZIP | 20GB | 18GB以下 |
| 展開後 | 40GB | 35GB以下 |
| `/act` 1回 | 10秒 | 最大8秒未満 |
| `/reset` 1回 | 10秒 | 最大8秒未満 |
| サーバー起動 | 120秒 | 90秒未満 |
| 推論VRAM | L4 24GB | 22GB未満 |
| Action | float32, shape (7,) | NaN/Infなし |
| 外部通信 | 不可 | 完全オフライン |

1回でも10秒を超えるとTrack全体がerror扱いとなるため、平均Latencyではなく最大Latencyで判定する。

---

## 4. モデル候補比較

Benchmark条件が異なるため、通常LIBEROとLIBERO-plusの数値は単純比較しない。

| モデル | 規模 | 強み | 弱み | 位置づけ |
|---|---:|---|---|---|
| **OpenVLA-OFT+** | 7B | LIBERO-plus総合79.6、Light/Backgroundに強い | 容量と依存が重い | **本命** |
| **MolmoAct2-LIBERO** | 5B | 通常LIBERO平均98.25%、LeRobot対応 | LIBERO-plus直接値なし | Fallback |
| **X-VLA** | 0.9B | 軽量、LeRobot対応、適応しやすい | 大型モデルより素の性能が低い | 軽量Fallback |
| **SmolVLA** | 0.5B | 公式Notebookと互換、T4でも学習可能 | モデル容量が小さい | 緊急提出用 |

### 4.1 OpenVLA-OFT+のLIBERO-plus結果

公開Model Cardの列順に従った値は次のとおり。

| 摂動 | 成功率 |
|---|---:|
| Camera | 92.8 |
| Robot initial state | 30.3 |
| Language | 85.8 |
| Light | 94.9 |
| Background | 93.9 |
| Noise | 89.3 |
| Layout | 77.6 |
| **Total** | **79.6** |

相対的な弱点はRobot initial stateであり、次いでLayoutである。追加学習を行う場合も、LightやBackgroundをさらに強化するより、この弱点を小さく補正する方が合理的である。

### 4.2 MolmoAct2の通常LIBERO結果

| Suite | 成功率 |
|---|---:|
| Spatial | 98.4% |
| Object | 100.0% |
| Goal | 98.0% |
| LIBERO 10 | 96.6% |
| **Average** | **98.25%** |

MolmoAct2は非常に強いが、今回重要なLIBERO-plusでの直接結果が公開されていない。そのため第一候補にはせず、OpenVLA-OFT+の提出適合性に問題がある場合のFallbackとする。

---

## 5. 学習データ方針

### 5.1 利用可能な公開データ

#### コンペ公式Notebookが参照するデータ

```text
lerobot/libero_plus
```

公式NotebookはSpatial 10タスクから各5Episode、合計50Episodeだけを抽出する。これは学習・マージ・推論を確認するための最小例であり、本学習データとしては狭すぎる。

#### LIBERO-plus公開LeRobotデータ

```text
Sylvest/libero_plus_lerobot
```

- Robot: Panda
- Tasks: 40
- Episodes: 14,347
- Frames: 2,238,036
- Cameras: 2
- FPS: 20
- License: MIT

今回の主データ候補とする。

#### 通常LIBERO公開データ

```text
lerobot/libero
```

通常タスク性能の維持とCatastrophic forgetting防止のReplayデータとして使用する。

### 5.2 基本配合

初期配合は次とする。

```text
LIBERO-plus: 70%
通常LIBERO:   30%
```

ただし、OpenVLA-OFT+はすでにLIBERO-plus mix-SFT済みである。追加学習では全量を再学習せず、タスクと摂動を均等化した小さなBalanced subsetを使用する。

### 5.3 PARC入力条件への適合

評価入力は128×128である。学習時も一度128×128へ縮小した画像をモデル標準Processorへ入力し、評価時の情報損失を再現する。この処理を行わない高解像度学習だけでは、公開Benchmark値がそのまま再現される保証はない。

### 5.4 優先するデータ

追加学習では次を優先する。

1. Robot initial state変化
2. Layout・物体配置変化
3. Camera viewpoint変化
4. Spatial / Object / Goal / Longの均等化
5. 128×128へ縮小した2カメラ画像
6. 通常LIBERO Replay

公開4タスクだけを大量学習せず、非公開タスクへの汎化を維持する。

---

## 6. 素の性能を守る最小調整方針

### 6.1 基本思想

OpenVLA-OFT+はすでにLIBERO-plusへ適応されている。大きく学習し直すと、次の性能を壊す可能性がある。

- 照明・背景ロバスト性
- 言語理解
- 通常LIBEROのタスク遂行能力
- 安定したAction分布
- 未知条件への汎化

したがって目標はモデルを作り直すことではない。

> **高い既存性能を保持しながら、PARCの128×128入力、7次元Action、厳しい衝突条件、弱い摂動カテゴリだけを補正する。**

### 6.2 更新対象の優先順位

1. Action Head
2. Proprioception Projector
3. 小規模LoRA rank 8
4. 必要な場合のみLoRA rank 16
5. Vision Encoderは原則Freeze

実装上、Action HeadとProprioception Projectorだけの分離学習が難しい場合は、OpenVLA-OFT公式の学習構成を維持したまま、低学習率・短Steps・小規模LoRAで代替する。

### 6.3 初期学習案

```yaml
model:
  checkpoint: Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata
  dtype: bfloat16
  freeze_vision_encoder: true
  train_action_head: true
  train_proprio_projector: true
  use_lora: false

training:
  batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  learning_rate: 1.0e-5
  max_steps: 500
  save_best_only: true
```

このYAMLは初期方針であり、使用するOpenVLA-OFTのConfig項目へ正確に変換して管理する。

### 6.4 段階的な調整

#### Stage 0: 素のモデル

- 追加学習なし
- L4でのVRAM、Latency、成功率、衝突率を測定
- 以降のすべての比較基準とする

#### Stage A: 最小更新

- Action Head
- Proprioception Projector
- 200〜500 steps
- 500〜1,000 Balanced episodes
- Learning rate 1e-5以下

目的は、独自学習部分がAction生成に寄与することを確認しつつ、元性能を維持することである。

#### Stage B: 小規模LoRA

Stage Aで不足が残る場合だけ実施する。

- LoRA rank 8
- 1,000〜3,000 steps
- Vision Encoder Freeze
- 通常LIBERO Replayを維持

#### Stage C: 限定拡張

Robot initial state、Layout、Cameraで改善がなく、元性能が維持されている場合のみ実施する。

- LoRA rank 16
- VLM後段の限定層のみ
- 低学習率

### 6.5 行わないこと

- 7B全パラメータのFull Fine-tuning
- Vision Encoder全層の解凍
- 高学習率での全面更新
- 公開4タスクだけを用いた長時間学習
- 評価結果を自動的に次の学習データへ戻すこと
- タスク固有FSMやPlannerによる性能補完

### 6.6 Early Stop条件

次のいずれかで学習を停止する。

- Validation成功率が2評価連続で低下
- Collision rateが悪化
- LightまたはBackground条件が低下
- 通常LIBERO Replay性能が低下
- ActionにNaN、Inf、過大値が発生
- 素のモデルとの差が統計的に改善と判断できない

---

## 7. A100 40GBでの実行方針

OpenVLA-OFT公式では、LIBERO推論は約16GB VRAM、学習は構成により27〜80GBとされている。A100 40GBでは部分学習は射程内だが、Full Fine-tuningには余裕がない。

使用する省メモリ設定は次のとおり。

- BF16
- Batch size 1
- Gradient accumulation
- Gradient checkpointing
- Vision Encoder Freeze
- Action系モジュール優先
- 小規模LoRA
- Optimizer stateを提出物へ含めない

学習中のCheckpointはColabローカルへ保存し、BestとLastだけをGoogle Driveへコピーする。

---

## 8. 推論・Action chunk

OpenVLA-OFTの入力対応を次のように接続する。

```text
agentview_image
  -> full_image

robot0_eye_in_hand_image
  -> wrist_image

joint / eef / quaternion / gripper
  -> state / proprioception

instruction
  -> task_description
```

Action chunkはPolicy Server内にキャッシュし、`/act`ごとに7次元Actionを1つ返す。

比較する再計画間隔は次とする。

```text
5 / 10 / 20 / full chunk
```

短い間隔は追従性が高いが推論回数が増え、長い間隔は高速だがOpen-loop誤差が増える。成功率、Collision rate、最大Latencyで決定する。

---

## 9. 採用GateとFallback

### 9.1 OpenVLA-OFT+採用Gate

#### 学習

- A100 40GBでOOMなし
- 100 steps以上連続実行
- Lossが低下
- 独自Checkpointを再ロードできる
- 独自学習部分を無効化した場合とAction出力に差がある
- Dataset revision、Episode ID、Configを追跡できる

#### 提出

- L4 24GBで推論可能
- Peak VRAM 22GB未満を目標
- Cold start 90秒未満
- `/act`最大8秒未満
- `/reset`最大8秒未満
- ZIP 20GB未満
- `float32 shape (7,)`
- NaN / Infなし
- `validate_submission.py` PASS
- 完全オフライン

#### 性能

- 素のOpenVLA-OFT+を下回らない
- Collision rateが悪化しない
- 最低タスク成功率が悪化しない
- Light・Background性能を維持
- 未学習Episodeで改善または同等

### 9.2 MolmoAct2への切替条件

OpenVLA-OFT+検証開始後、2日以内に次のいずれかが解消しない場合はMolmoAct2へ切り替える。

- A100 40GBで学習OOM
- L4 24GBで推論OOM
- `/act`最大8秒超
- 起動90秒超
- ZIP 20GB超
- 2画像・状態入力のPolicy Server統合が成立しない
- 追加学習後に元性能が大幅に低下

MolmoAct2ではAction Expert only fine-tuningを第一選択とする。LeRobot公式計測では、この構成は16.5〜21.4GiB程度であり、A100 40GBで十分実行可能である。

### 9.3 X-VLAとSmolVLA

MolmoAct2でも容量・依存・提出時間に問題がある場合はX-VLAへ切り替える。SmolVLAは公式PARC Notebookと最も互換性が高いため、緊急提出可能な最小構成として並行して維持する。

---

## 10. 実験順序

| 順序 | 実験 | 主な判断 |
|---:|---|---|
| 1 | OpenVLA-OFT+素モデル推論 | L4適合、VRAM、Latency、ZIP |
| 2 | Policy Server接続 | 2画像・状態・言語・7D Action |
| 3 | 公開4タスクSmoke | APIとActionの正常性 |
| 4 | Stage A 200〜500 steps | 最小学習が成立するか |
| 5 | 500〜1,000 Episode評価 | 元性能を維持できるか |
| 6 | LoRA rank 8 | Stage Aより改善するか |
| 7 | Replan 5/10/20/full | 成功率・衝突・Latency |
| 8 | Candidate 10 Episode | 上位モデル比較 |
| 9 | Final 20 Episode | 提出候補確定 |
| 10 | Model Freeze | Hash、ZIP、レポート固定 |

一度に変更する主要因子は1つとする。

---

## 11. 実験管理

各Runで次を記録する。

```text
run_id
git_commit
base_model_repo
base_model_revision
base_model_license
dataset_repo
dataset_revision
episode_ids
training_config
trainable_parameters
checkpoint_sha256
submission_zip_sha256
latency_max
latency_p95
peak_vram
success_rate
collision_rate
task_metrics
human_decision
```

モデルとデータはGoogle Driveへ保存し、コード、Config、Manifest、レポートはGitHubへ保存する。Hugging Faceの`main`を直接参照せず、モデルとデータのRevisionを固定する。

公式評価結果は学習データ、RAG、LLMによる自動最適化から分離する。

---

## 12. 禁止事項への対応

以下を実装しない。

- タスク固有FSM
- ハードコードAction列
- `task_id`やSeedによるAction切替
- 報酬・成功条件を直接参照するPlanner
- 非公開タスクFingerprinting
- Scripted Policyを評価時に実行
- 公式評価観測を保存して再学習
- 外部LLM・APIへの評価時通信

Scripted Policyは独自教師データ生成にだけ使用し、提出Policyには含めない。

---

## 13. 最終方針

```text
本命モデル:
OpenVLA-OFT+ 7B

モデル利用思想:
素のLIBERO-plus性能を最大限利用し、
PARC適合に必要な範囲だけを小さく調整する。

最初の更新対象:
Action Head
Proprioception Projector

必要な場合のみ:
LoRA rank 8 -> rank 16

主データ:
LIBERO-plus Balanced subset 70%
通常LIBERO Replay 30%
128×128入力を再現

学習環境:
Google Colab A100 40GB

提出環境:
NVIDIA L4 24GB

Fallback:
MolmoAct2 5B
-> X-VLA 0.9B
-> SmolVLA 0.5B
```

---

## 14. 参照資料

- PARC2026配布Repo: https://github.com/matsuolab/PARC2026_pre
- OpenVLA-OFT: https://github.com/moojink/openvla-oft
- OpenVLA-OFT+ Checkpoint: https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata
- LIBERO-plus LeRobot Dataset: https://huggingface.co/datasets/Sylvest/libero_plus_lerobot
- MolmoAct2 LIBERO LeRobot: https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot
- MolmoAct2 LeRobot Guide: https://huggingface.co/docs/lerobot/main/molmoact2
- X-VLA LIBERO: https://huggingface.co/lerobot/xvla-libero
- SmolVLA LIBERO-plus: https://huggingface.co/lerobot/smolvla_libero_plus
