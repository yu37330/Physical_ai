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

したがって、通常のTransformersへ単純に置き換えるだけではOFT本来の並列Action生成を再現できない可能性が高い。

### 対応方針

- 学習環境では公式Forkを使う
- 提出環境では`transformers==4.40.1`をPyPIから導入する
- Fork全体ではなく、必要な双方向Attention差分を提出用コード内で適用する
- 公式ForkとPyPI版＋Patchへ同一入力を与え、Action差分を比較する

## 3. PARC提出制約との衝突

PARCのValidatorは次を拒否する。

- `git:`
- `http:` / `https:`
- Editable install
- ローカルパス依存
- 独自Index
- `--find-links`

したがって、以下は提出用`requirements.txt`へ記載できない。

```text
transformers @ git+https://...
dlimp @ git+https://...
-e .
```

学習環境と提出環境を分離する。

```text
Google Colab / Training
- 公式OpenVLA-OFT Repo
- 公式Transformers Fork
- TensorFlow / RLDS / dlimp
- Flash Attention

Submission / Inference
- PyPI依存のみ
- 完全ローカルCheckpoint
- 推論用Prismaticコード
- 双方向Attention Patch
- TensorFlow/RLDS/W&Bなし
```

## 4. Checkpoint名ハードコード

公式`openvla_utils.py`は、HF上のCheckpoint IDに応じてAction HeadとProprio Projectorのファイル名を辞書で選択する。

登録対象は主に`moojink/openvla-7b-oft-finetuned-libero-*`であり、今回選定した以下は含まれない。

```text
Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata
```

HF IDを直接渡すと`Unsupported HF Hub pretrained checkpoint found!`になる可能性がある。

### 回避策

Checkpointを事前にローカルディレクトリへ保存し、明示的なファイルパスで読む。

```text
model_weights/openvla_oft_plus/
├── config.json
├── model-*.safetensors
├── model.safetensors.index.json
├── tokenizer files
├── action_head--150000_checkpoint.pt
├── proprio_projector--150000_checkpoint.pt
└── dataset_statistics.json
```

提出時はHF ID、`HfApi().model_info()`、`hf_hub_download()`を使わない。

## 5. 完全オフライン化

提出サーバー起動時に次を設定する。

```python
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```

すべてのモデルロードへ`local_files_only=True`を指定する。モデル存在判定もローカルPathだけで行う。

## 6. 推論時に削除できる依存

公式プロジェクトには学習用依存が多数含まれるが、提出用推論では次を削除可能である。

- TensorFlow
- TensorFlow Datasets
- TensorFlow Graphics
- dlimp
- wandb
- diffusers（L1 Action Head利用時）
- Flash Attention（まずSDPAで検証）

画像のCenter CropとResizeはPILまたはPyTorchで再実装する。

## 7. 推奨提出構成

```text
submission/
├── policy_server.py
├── requirements.txt
├── runtime/
│   ├── offline_env.py
│   ├── bidirectional_attention.py
│   ├── checkpoint_layout.py
│   ├── preprocessing.py
│   ├── action_postprocess.py
│   └── policy.py
├── vendor/
│   └── prismatic/             # 推論に必要な最小部分
└── model_weights/
    └── openvla_oft_plus/
```

## 8. 確認済みと未確認

### 事前確認済み

- PARC観測からOpenVLA-OFTの2画像＋8次元Proprioを構築できる
- OpenVLA-OFTは7次元Actionを8ステップChunkで生成する
- 選定CheckpointにAction Head、Proprio Projector、統計情報が存在する
- A100 40GBでBatch 1 LoRA学習は公称上可能
- L4 24GBでBF16推論は公称上可能
- Checkpoint総量は20GB提出制限内に収まる見込み

### 実測が必要

- PyPI Transformers＋Patchと公式ForkのAction一致
- L4でのPeak VRAM
- Cold startと最大Latency
- 公式Checkpointから不要LoRAファイルを除外できるか
- PARC Docker内でのTorch/CUDA互換性
- 20回連続推論の最大時間

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
