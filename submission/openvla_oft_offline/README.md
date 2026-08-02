# OpenVLA-OFT+ Offline Submission Runtime

PARC2026提出用の完全オフライン推論ランタイムの初期実装です。

## 方針

- モデル重みは`model_weights/openvla_oft_plus/`へ配置する
- Hugging Face Hubへの通信は行わない
- `requirements.txt`にはPyPIパッケージだけを書く
- 公式OpenVLA-OFTの学習依存を提出物へ持ち込まない
- Action chunkをサーバー内に保持し、`/act`ごとに7次元Actionを1つ返す

## 現在の状態

以下は実装済みです。

- オフライン環境変数
- Checkpoint構成検査
- PARC観測からOpenVLA用入力への変換
- QuaternionからAxis-Angleへの変換
- 90% Center Cropと224×224 Resize
- Action chunk cache
- 公式Fork差分を再現するAttention Patchの隔離ポイント
- FastAPI Policy Serverの骨格

以下はモデル配置後に接続が必要です。

- OpenVLA/Prismatic本体の最小Vendor
- Action Head / Proprio Projectorの実ロード
- モデル固有のAction unnormalization
- 公式Forkとの出力一致試験

## モデル配置

```text
model_weights/openvla_oft_plus/
├── config.json
├── model-*.safetensors
├── model.safetensors.index.json
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
├── action_head--150000_checkpoint.pt
├── proprio_projector--150000_checkpoint.pt
└── dataset_statistics.json
```

## 起動

```bash
python policy_server.py --port 8000
```

モデル未配置時は起動時に明示的なエラーを返します。

## 次の確認

1. CheckpointをGoogle Driveから配置
2. `python -m runtime.checkpoint_layout model_weights/openvla_oft_plus`
3. 公式Fork環境で基準Actionを保存
4. Offline runtimeで同一入力を推論
5. Action差分、VRAM、Latencyを記録
