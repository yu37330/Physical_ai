# 提出成立性 実測結果

`MODEL_STRATEGY.md` 実験順序#1「OpenVLA-OFT+素モデル推論 → L4適合、VRAM、Latency、ZIP」の実測記録。

## 測定条件

| 項目 | 値 |
|---|---|
| 測定日 | 2026-08-04 |
| GPU | NVIDIA L4（採点環境と同一。総容量 22.49 GiB） |
| transformers | **PyPI 4.40.1**（`matches_scoring_environment: true`） |
| Checkpoint | `Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata` @ `a85655ec941bae6644c9fbdf62db02b9726d7cf5` |
| 学習 | なし（Base 重みのみ。**この構成は提出不可**） |
| 観測 | 合成乱数。運営`pipeline/remote_policy.py`と同一のmsgpack形式 |
| 計測 | Warm-up 2回を除く30リクエスト |
| Script | `scripts/measure_submission_viability.py` |

## 結果

| Gate | 実測 | 上限 | 判定 | 余裕 |
|---|---:|---:|---|---|
| `/act` 最大 | **0.234 秒** | 10 秒 | pass | **43倍** |
| `/act` p95 | 0.230 秒 | 10 秒 | pass | |
| `/reset` | 0.002 秒 | 10 秒 | pass | |
| サーバー起動 | **47.6 秒** | 120 秒 | pass | 2.5倍 |
| Peak VRAM | **15.33 GiB** | 22.49 GiB | pass | 7.2 GiB 余 |
| 提出ディレクトリ | **14.85 GiB** | 20 GB (ZIP) | pass | |
| Action契約 | 30/30 | `(7,) float32` 有限 | pass | |

### Latency の読み方

`act_mean` 0.024秒に対し `act_max` 0.234秒と10倍の差がある。これは Action chunk が効いているためで、**チャンクが空のリクエストだけが実推論**し、残りはバッファから取り出すだけになる。

予選ルールもこの構成を想定している。

> アクションチャンクをサーバー内にキャッシュするモデルの場合、推論が実行される「重い」リクエストのみが上限の対象となる（実質的な制約は「チャンク 1 回分の推論 ≤ 10 秒」である）

したがって判定に使うべきは `act_max` = **0.234秒**。10秒制限に対して43倍の余裕がある。

### Track 1時間予算

推論のみの理論上限は約147,000回。環境ステップとレンダリングは含まない。1推論0.234秒でチャンク長分のステップを賄うため、推論が1時間予算を圧迫する可能性は低い。実際の制約はMuJoCoシミュレーション側になる見込み。

## 判定

**Go.** `MODEL_STRATEGY.md` 9.1「提出Gate」のうち、実測を要する5項目が全て通過した。

- L4 24GBで推論可能
- Peak VRAM 22GB未満（目標値に対し15.33 GiB）
- Cold start 90秒未満（47.6秒）
- `/act`最大8秒未満（0.234秒）
- ZIP 20GB未満の見込み（提出対象 14.85 GiB）

Stage A学習で増えるのは Action Head と Proprio Projector のみのため、これらの数値は学習後も大きくは変わらない見込み。

## 未検証の残件

### 1. Action値の一致（最重要）

合成乱数入力に対する shape と有限性しか検証していない。ForkとPyPI版＋Patchが**同一のActionを出すか**の照合（`OPENVLA_OFFLINE_RUNTIME_ASSESSMENT.md` 8章）は未実施。

`submission/openvla_oft_offline/tools/compare_action_chunks.py` は2つの`.npy`を比較する実装があるが、**比較対象を生成する側が存在しない**ため、照合は実行できていない。

参考として、同一Checkpointに対しForkとPyPI版＋Patchで測った次の値は一致した。同じ計算が走っている傍証にはなるが、Action値の一致を示すものではない。

| | Fork | PyPI + Patch |
|---|---:|---:|
| `act_max` | 0.23451 秒 | 0.23447 秒 |
| Peak VRAM | 16,460,546,048 | 16,460,546,048 |

PIL前処理と公式TensorFlow前処理のAction一致（同8章）も未実施。

### 2. 提出物としての成立

Base重みのみの構成は「公開重みを実質的に変更せず推論する」に該当し提出できない。運営Validatorによる静的検査・ZIP作成・動的スモークも未実施。

### 3. 滑らかさ・実行効率

予選スコアには ジャーク、SPARC、EEF回転総量、ステップ数、軌道総距離 が含まれる（`OFFICIAL_RULES.md` 3章）。本測定はこれらを一切見ていない。
