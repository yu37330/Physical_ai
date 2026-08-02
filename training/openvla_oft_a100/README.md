# OpenVLA-OFT+ A100 40GB Training Environment

A100 40GBでOpenVLA-OFT+の素の性能を維持しながら、必要最小限の独自学習を行うための構成です。

## 原則

- 学習は公式OpenVLA-OFT Repoと公式Transformers Forkを使う
- 追加学習はLoRA rank 8から始める
- 2画像とProprioceptionを維持する
- LIBERO-plus 70%＋通常LIBERO Replay 30%を基本とする
- Full Fine-tuningとVision Encoder全面解凍は行わない
- 学習成果物はGoogle Driveへ保存する
- Colabは固定Git CommitをCheckoutする

## 実行順

1. `scripts/bootstrap_colab.sh`で環境を作る
2. Google Drive上のRLDSデータまたは変換済みSubsetを配置する
3. `configs/smoke_lora_r8.yaml`のPathを更新する
4. `scripts/train_smoke.sh`を実行する
5. 100〜500 stepsでLoss、VRAM、Checkpoint再ロードを確認する
6. 推論Parity確認後に1,000 steps以上へ進む

## 注意

公式OpenVLA-OFT Fine-tuningはRLDS形式を前提とする。LeRobot形式データをそのまま渡せないため、公開RLDSデータを使うか、事前にRLDSへ変換する。
