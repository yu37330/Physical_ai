# ロボットアクションモデルの全体像とPARC2026モデル選定

更新日: 2026-08-03  
対象: PARC 2026 予選 Track 1  
提出時GPU: NVIDIA L4 24GB  
最大学習GPU: Google Colab NVIDIA A100 40GB

---

## 1. この文書の目的

本書は、ロボットアクション生成モデルの主要な系統を整理し、次の位置付けを明確にする。

1. 運営サンプルで採用されているSmolVLAの役割
2. 本プロジェクトでOpenVLA-OFT+を第一候補とする理由
3. MolmoAct2、X-VLA、SmolVLAをFallbackとして残す理由
4. 7Bを超える大型VLA・Embodied Modelが今回の競技に向かない理由

結論は次のとおりである。

> **運営SmolVLAは、学習・評価・提出フローを理解するための軽量Baselineである。OpenVLA-OFT+ 7Bは、LIBERO-plus適合性と提出制約の両方を満たしながら性能上限を狙う本命である。7B超の大型モデルは存在するが、公開性、容量、VRAM、起動時間、推論Latencyの点で今回の提出条件には適さない。**

---

## 2. ロボットアクションモデルの全体像

ロボット操作モデルは、概ね次の4層に分類できる。

```text
特定作業・軽量・高速
        │
        ├─ 古典制御 / ルールベース
        ├─ 模倣学習Policy
        │    ├─ Behavior Cloning
        │    ├─ ACT
        │    └─ Diffusion Policy
        ├─ 軽量VLA
        │    ├─ SmolVLA 450M
        │    └─ X-VLA 0.9B
        ├─ 中～大型VLA
        │    ├─ MolmoAct2 5B
        │    └─ OpenVLA / OpenVLA-OFT+ 7B
        └─ Frontier VLA / Embodied Model
             ├─ RT-2 PaLM-E 12B
             ├─ RT-2 / RT-2-X PaLI-X 55B
             ├─ Gemini Robotics系（規模非公開）
             └─ PaLM-E 562B（主にEmbodied reasoning）

汎用性・意味理解・計算負荷
```

モデルサイズが大きいほど自動的にロボット操作性能が上がるわけではない。競技では次の適合性が重要である。

- 対象Benchmarkに近い事前学習・追加学習
- カメラ数と画像解像度
- Robot State / Proprioceptionの形式
- Action次元とControl mode
- Action chunkの長さ
- 推論Latency
- GPUメモリ
- Checkpoint容量
- 完全オフライン実行
- 提出テンプレートへの統合容易性

---

## 3. 非VLA系Policy

### 3.1 Behavior Cloning

画像やRobot Stateから、教師デモのActionを直接回帰する最も基本的な模倣学習である。

```text
画像 + Robot State
        ↓
Policy
        ↓
1 stepのAction
```

軽量で高速だが、言語理解や未知物体への汎化は限定的である。

### 3.2 ACT

ACTは、複数ステップ分のAction chunkをTransformerでまとめて予測する。1 stepずつ予測する方式より、長い軌道での誤差蓄積を抑えやすい。

```text
複数カメラ + Robot State
        ↓
Transformer
        ↓
Action chunk
```

特定ロボット・特定作業で強く、推論も比較的軽い。一方、通常は大規模VLMの意味理解を持たない。

### 3.3 Diffusion Policy

ノイズから連続Action列を反復的に生成する。複数の妥当な軌道が存在する操作を表現しやすいが、複数回のDenoisingが必要なため推論時間が増えやすい。

これらは強いタスク専用Policyであるが、今回のような言語指示、複数カメラ、未知条件への汎化を重視する競技ではVLAが有利になりやすい。

---

## 4. VLAの主要なAction生成方式

VLAは、画像、言語、Robot Stateをまとめて解釈し、ロボットActionを生成する。

```text
Front camera
+ Wrist camera
+ Robot State
+ Language instruction
        ↓
Vision-Language-Action Model
        ↓
7D Action / Action chunk
```

### 4.1 Action Token型

連続Actionを離散Tokenへ変換し、言語Tokenと同様に自己回帰生成する。

代表例:

- OpenVLA
- RT-2

意味理解と既存VLM資産を活用しやすいが、逐次Token生成は高周波制御に不利になりやすい。

### 4.2 連続Action Head型

VLMの表現から専用Headで連続Actionを直接生成する。

代表例:

- OpenVLA-OFT / OpenVLA-OFT+

OpenVLA-OFTは、並列Decoding、Action chunking、連続Action表現、L1回帰を組み合わせる。公式プロジェクトでは、Base OpenVLAよりAction生成を大幅に高速化し、LIBEROで97.1%の平均成功率を報告している。

### 4.3 Flow Matching型

ノイズを正しい連続Action chunkへ変換するVector fieldを学習する。

代表例:

- SmolVLA
- X-VLA
- π0系
- MolmoAct2のContinuous Action Expert

Action Token型より非自己回帰化しやすく、連続制御とAction chunkに向いている。

---

## 5. 運営サンプル: SmolVLA 450M

### 5.1 サンプルの構成

運営サンプルは次を使用する。

```text
Base model:
lerobot/smolvla_libero_plus

追加学習データ:
lerobot/libero_plus
LIBERO-plus Spatial 10 tasks × 5 episodes = 50 episodes

学習:
LoRA rank 16
3,000 steps
Batch size 1
Vision Encoder freeze
Expert側を追加学習
```

学習後はLoRAをBase Weightへマージし、追加学習前後を同じLIBERO-plus Spatial環境で評価する。

### 5.2 SmolVLAの特徴

SmolVLAはHugging Faceが公開する450Mパラメータの軽量VLAである。

- 複数カメラ画像
- Robot State
- 自然言語指示
- Flow Matching Action Expert
- 連続Action chunk
- LeRobotとの高い統合性

公式資料では、対象タスクへのFine-tuning開始点として約50 Episodeが推奨されている。

### 5.3 運営がSmolVLAをサンプルにした意味

運営サンプルは「競技で最も強いモデル」を示すものではなく、次を一通り体験できるReference Pipelineと解釈する。

```text
公開Checkpoint取得
→ Dataset選択
→ LoRA追加学習
→ Adapter merge
→ 学習前後評価
→ モデルZIP出力
```

利点:

- Colab T4でも試せる
- LeRobot形式のまま学習できる
- Dataset、Processor、Policyの関係を理解しやすい
- 学習前後の比較をすぐ行える
- 提出用Policy Serverへ組み込む前段を確認できる

制約:

- 約450Mで意味理解・表現容量は限定的
- サンプル学習はSpatial 50 Episodeだけ
- Notebook内評価は256×256、Spatial 10 tasksで、本番Track 1評価とは異なる
- 出力ZIPはモデルWeightであり、競技提出ZIPではない

したがって、SmolVLAは本プロジェクトでは次の役割を持つ。

> **運営基準のBaseline、開発フロー理解用モデル、OpenVLAが提出制約を満たせない場合の緊急Fallback。**

---

## 6. 現在の候補モデル

| 優先度 | モデル | 規模 | Action方式 | 位置付け |
|---:|---|---:|---|---|
| 1 | OpenVLA-OFT+ LIBERO-plus | 7B | 連続Action Head | 本命 |
| 2 | MolmoAct2-LIBERO-LeRobot | 5B | Flow Matching + Action Token | 高性能Fallback |
| 3 | X-VLA-LIBERO | 0.9B | Flow Matching | 軽量Fallback |
| 4 | SmolVLA-LIBERO-plus | 450M | Flow Matching | 運営Baseline / 緊急提出 |

### 6.1 本命: OpenVLA-OFT+ 7B

Checkpoint:

```text
Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata
```

採用理由:

1. LIBERO-plus向けにFine-tuneされた公開Checkpointである
2. 2カメラ + Proprioception + Language + 7D Action chunkの構成が競技入力に近い
3. OpenVLAの7B表現能力を利用できる
4. OFT方式によりBase OpenVLAの逐次Action Token生成より高速である
5. 公式OpenVLA-OFT資料では、LIBEROの2画像 + Proprioception推論に約16.2GB VRAMを示しており、L4 24GBで成立する可能性がある
6. Checkpoint一式が約15.9GBで、20GB提出上限へ収められる可能性がある
7. Action HeadとProprioception系の最小追加学習で既存性能を維持する戦略を取れる

主なリスク:

- 20GB提出ZIP上限までの余裕が小さい
- L4での最大Latencyを実測する必要がある
- Git版Transformersを含む依存関係を完全オフライン化する必要がある
- 公式OFT学習は重く、A100 40GBではBatch size 1とGradient accumulationが前提になる

### 6.2 高性能Fallback: MolmoAct2 5B

Checkpoint:

```text
allenai/MolmoAct2-LIBERO-LeRobot
```

特徴:

- 5Bパラメータ
- Multi-view RGB、Robot State、Language instruction
- Continuous Flow Matching Action Expert
- LIBERO LeRobot Checkpointは約10.9GB
- LeRobot版の通常LIBERO平均成功率は98.25%

本命にしない理由:

- 通常LIBEROでは非常に強いが、LIBERO-plusの直接評価が不足している
- 今回はLight、Background、Camera、Layoutなどの摂動耐性が重要である
- 2026年公開の新しい構成で、提出環境での依存・Latency検証がまだ必要である

### 6.3 軽量Fallback: X-VLA 0.9B

Checkpoint:

```text
lerobot/xvla-libero
```

特徴:

- 0.9Bパラメータ
- Flow Matching
- Soft PromptでRobot embodimentとDataset domainを表現
- 29万Episode、7 Robot Platformでの事前学習
- 約1%のParameterだけを調整する適応方式

SmolVLAより容量があり、OpenVLAより大幅に軽い。L4推論、提出容量、Colab学習のバランスがよい中間案である。

### 6.4 緊急Fallback: SmolVLA 450M

運営サンプルとの互換性が最も高い。OpenVLA-OFT+とMolmoAct2が容量・依存・Latency Gateを満たさない場合でも、提出まで到達しやすい。

---

## 7. モデル選定はサイズ順ではない

今回の選定軸は次の順である。

### Gate 1: 競技規約と提出形式

- ZIP 20GB未満
- 展開後40GB未満
- 完全オフライン
- Policy Serverとして起動
- `/act`、`/reset`が各10秒以内
- Actionは`float32`, shape `(7,)`

### Gate 2: L4 24GBでの推論成立性

- Cold start 120秒未満
- Peak VRAM 22GB未満を内部目標とする
- 平均ではなく最大Latency 8秒未満を内部目標とする

### Gate 3: 入出力適合

- 128×128 Front image
- 128×128 Wrist image
- Robot State / Proprioception
- Language instruction
- 7D relative Action
- Action chunk

### Gate 4: Benchmark適合

- 通常LIBEROだけでなくLIBERO-plusへの適合
- Camera、Initial state、Language、Light、Background、Noise、Layoutへの耐性
- Collisionを抑えた操作

### Gate 5: 独自学習

- 公開Checkpointをそのまま使うだけではなく、Action生成に寄与する追加学習を実施
- Base性能を壊さない
- 学習前後を同一条件で比較

このため、単純な「最大モデルを使う」戦略は採用しない。

---

## 8. 7Bを超える大型モデル

### 8.1 RT-2 PaLM-E 12B

Google DeepMindのRT-2には、PaLM-E 12BをBackboneとする構成がある。Web-scale Vision-Language知識とRobot Dataを共同学習し、Robot ActionをText Tokenとして生成する。

今回使わない主な理由:

- 公開Checkpointと完全な学習・推論一式が提供されていない
- 今回のLIBERO/Panda Action schemaへ直接適合したモデルではない
- 12BをBF16で保持するとWeightだけで概算24GBとなり、L4 24GBへ余裕を持って載せにくい
- 提出ZIP 20GBへ収めるには量子化がほぼ必須になる

### 8.2 RT-2 / RT-2-X PaLI-X 55B

RT-2 PaLI-X版とRT-2-Xは55B級である。RT-2-XはOpen X-Embodimentの複数Robot Dataを用い、Cross-embodiment Generalizationを狙った研究モデルである。

55Bの概算Weight容量:

```text
BF16 / FP16: 約110GB
INT8:         約55GB
INT4:         約27.5GB
```

INT4でもWeightだけでL4 24GBを超えるため、KV Cache、Activation、Action Head、Runtimeを含める余地がない。提出ZIP 20GBにも収まらない。

また、公開された競技用Checkpointではなく、今回の128×128・2カメラ・7D relative Actionへ直接統合できない。

### 8.3 Gemini Robotics / Gemini Robotics 1.5

Google DeepMindはGemini Roboticsを高度なVLA、Gemini Robotics 1.5を視覚情報と指示からMotor Commandへ変換するVLAとして公開している。

ただし、Parameter数、Checkpoint、完全オフラインRuntimeは一般公開されていない。クラウドまたは限定提供モデルは、外部通信禁止の今回の提出環境では使用できない。

### 8.4 PaLM-E 562B

PaLM-Eの最大構成は562Bである。ただし、PaLM-Eは主としてEmbodied multimodal reasoning、Planning、Visual question answeringを扱うモデルであり、今回のように10秒以内で7D Actionを直接返す軽量Policy Serverと同じカテゴリではない。

562Bは仮にBF16ならWeightだけで1TBを超える。今回の競技で使う対象ではなく、将来的な階層型構成における高レベルPlannerの参考モデルである。

---

## 9. 大型モデルを使う場合の一般的な構成

7Bを超える巨大モデルは、すべてをControl loopへ直接入れるより、階層化して使う方が現実的である。

```text
大型VLM / Embodied Model
高レベル理解・計画・Subgoal生成
        ↓
中小型VLA / Action Policy
数Hz～数十HzのAction chunk生成
        ↓
Robot Controller
安全制約・軌道補間・Servo制御
```

今回の競技では外部モデル呼び出しができず、単一提出Package内で低Latency推論する必要がある。そのため、高レベルPlannerを別GPUやCloudへ分離する一般的な構成を採れない。

---

## 10. 今回の三段構え

```text
第一候補: OpenVLA-OFT+ 7B
性能上限を狙う

第二候補: MolmoAct2 5B / X-VLA 0.9B
OpenVLAが容量・Latency・依存Gateを満たさない場合

第三候補: SmolVLA 450M
運営サンプルを基準に、確実に提出へ到達する
```

採否はモデル名や公開Benchmarkだけでは確定しない。次を実測して決める。

1. L4 24GB Peak VRAM
2. Cold start
3. 最大`/act` Latency
4. 提出ZIP容量
5. 完全オフライン起動
6. 運営Validator
7. 公開ローカル評価の成功率とCollision
8. 独自学習前後の性能差

---

## 11. 現在の判断

現時点の判断は次である。

> **7Bより大きいモデルは存在する。しかし今回の20GB提出、L4 24GB、10秒応答、完全オフラインという条件では、OpenVLA-OFT+ 7Bが高性能を狙える実質的な上限に近い。**

SmolVLAは小さいから不要なのではなく、運営Referenceとして重要である。OpenVLA-OFT+は大きいから採用するのではなく、LIBERO-plus適合、Action chunk、2カメラ、Proprioception、公開Checkpoint、L4推論可能性を総合して採用している。

---

## 12. 参考資料

### PARC2026 / 運営サンプル

- https://github.com/matsuolab/PARC2026_pre
- https://github.com/matsuolab/PARC2026_pre/tree/main/examples

### SmolVLA

- https://huggingface.co/docs/lerobot/smolvla
- https://huggingface.co/blog/smolvla
- https://huggingface.co/lerobot/smolvla_base

### X-VLA

- https://huggingface.co/docs/lerobot/xvla
- https://huggingface.co/lerobot/xvla-libero

### OpenVLA / OpenVLA-OFT

- https://openvla.github.io/
- https://openvla-oft.github.io/
- https://github.com/moojink/openvla-oft
- https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata

### MolmoAct2

- https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot
- https://github.com/allenai/molmoact2
- https://arxiv.org/abs/2605.02881

### 大型VLA / Embodied Model

- https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/
- https://deepmind.google/blog/scaling-up-learning-across-many-different-robot-types
- https://arxiv.org/abs/2307.15818
- https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/
- https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/
- https://arxiv.org/abs/2303.03378
