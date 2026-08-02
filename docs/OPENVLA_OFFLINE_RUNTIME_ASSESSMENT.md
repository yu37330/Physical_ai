# OpenVLA-OFT+ 提出依存関係・オフライン推論成立性調査

更新日: 2026-08-02  
対象ブランチ: `feat/parc2026-foundation`

## 1. 結論

OpenVLA-OFT+の採用判断は **Conditional GO** とする。

モデル構造、PARC入力、7次元Action、A100 40GBでの部分学習、L4 24GBでの推論、20GB提出上限はいずれも成立見込みがある。一方、公式OpenVLA-OFT環境をそのまま提出物へ持ち込むことはできない。

主な理由は次の3点である。

1. 公式環境はGit版Transformers Forkへ依存する
2. PARCの`requirements.txt`検査はGit/HTTP/HTTPS/ローカルパス依存を拒否する
3. 公式Helperは対応Checkpoint IDをハードコードし、選定したSylvest版CheckpointをHF IDのままでは読み込めない

ただし、いずれも提出用ランタイムを分離すれば回避できる。

## 2. Git版Transformers依存

OpenVLA-OFTの`pyproject.toml`は、通常のPyPI版ではなく以下を要求する。

```text
transformers @ git+https://github.com/moojink/transformers-openvla-oft.git
```

このForkはTransformers 4.40.1を基に、LlamaのAttentionをOFT用の双方向Attentionへ変更している。主要差分は`modeling_llama.py`で、因果Maskを無効化し、Paddingだけを除外したMaskを使って`is_causal=False`でSDPAを実行する部分である。

### 対応方針

- 学習環境では公式Forkを使う
- 提出環境では`transformers==4.40.1`をPyPIから導入する
- 必要な双方向Attention差分を提出用コード内で適用する
- 公式ForkとPyPI版＋Patchへ同一入力を与え、Action差分を比較する

## 3. PARC提出制約との衝突

PARCのValidatorはGit、HTTP/HTTPS、Editable install、ローカルパス依存、独自Indexを拒否する。学習環境と提出環境を分離する。

```text
Google Colab / Training
- 公式OpenVLA-OFT Repo
- 公式Transformers Fork
- TensorFlow / RLDS / dlimp
- 必要な場合のみFlash Attention

Submission / Inference
- PyPI依存のみ
- 完全ローカルCheckpoint
- VendorしたPrismaticコード
- 双方向Attention Patch
- TensorFlow/RLDS/W&Bなし
```

## 4. Checkpoint名ハードコード

公式Helperが認識するHF IDには`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`が含まれない。提出時はHF IDを使わず、Action Head、Proprio Projector、Dataset statisticsをローカルPathから明示的にロードする。

## 5. 完全オフライン化

提出サーバー起動時に次を設定する。

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```

モデルはVendorした次のクラスから直接ロードする。

- `OpenVLAConfig`
- `PrismaticProcessor`
- `OpenVLAForActionPrediction`

`AutoModel`、`trust_remote_code=True`、`HfApi`、`hf_hub_download`は提出時に使用しない。

## 6. 推論時に削除する依存

- TensorFlow
- TensorFlow Datasets
- TensorFlow Graphics
- dlimp
- wandb
- diffusers（L1 Action Head利用時）
- Flash Attention（まずSDPAで検証）

画像処理はPILで再実装し、公式TensorFlow処理との画素差・Action差を実測する。

## 7. 学習環境のカスタム

Stage Aでは選定CheckpointのVLA本体へ新しいLoRAを追加しない。既存Action HeadとProprio Projectorをロードして、この2つだけを調整する。

Stage Bへ進む場合のみLoRA rank 8を追加し、Vision Backbone側LoRAをFreezeする。

公式スクリプトは単GPUでも分散Barrierを使用するため、`python`ではなく`torchrun --nproc-per-node 1`で起動する。

## 8. 実測が必要な項目

- PyPI Transformers＋Patchと公式ForkのAction一致
- PIL前処理と公式TensorFlow前処理のAction一致
- L4でのPeak VRAM
- Cold startと最大Latency
- PARC Docker内でのTorch/CUDA互換性
- 20回連続推論の最大時間
- 最終ZIP容量

## 9. Go / No-Go基準

次をすべて満たせば本命として継続する。

- A100 40GBで100 steps以上学習できる
- L4 24GBで推論できる
- `/act`最大8秒未満
- 起動90秒未満
- ZIP 20GB未満
- 完全オフラインで再ロードできる
- PyPI版＋Patchが公式Forkと等価なActionを生成する

2日以内に解消できない場合はMolmoAct2へ切り替える。
