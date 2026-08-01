# PARC2026 禁止事項・提出チェック

## 1. モデル・学習ルール

公開モデル、モデル構造、事前学習済み重み、Tokenizerは利用できます。ただし、参加者自身による独自学習が必要です。

独自学習として認められる例：

- LoRA / Adapter
- 独自学習したAction head / Decoder
- 独自学習した視覚・言語・状態統合モジュール
- Full fine-tuning

独自学習として認められない例：

- ファイル形式の変換だけ
- モデル名・ファイル名の変更だけ
- Prompt・推論パラメータの変更だけ
- 公開重みを実質的に変更せず、そのまま推論

独自学習要素は、最終Action生成に実質的に寄与しなければなりません。

## 2. 禁止されるPolicy

- 外部または手続き的なタスクレベルPlanner
- タスク固有の有限状態機械
- ハードコードされたAction列
- Scene ID、Task ID、Seed等をキーにしたAction table
- 成功条件や報酬を直接参照するPlanner
- 評価環境専用のif文攻略
- 非公開タスクを識別するFingerprinting
- 学習済みモデルを実質的に使用しないFallback Policy

## 3. データ利用ルール

利用可能：

- 講座提供データ
- 公開データセット
- 独自収集・独自生成データ
- Simulator
- Teleoperation
- Scripted Policyによる教師データ生成
- データ拡張
- 生成モデルを利用したデータ生成

必要事項：

- 出所
- License
- 生成方法
- 使用目的
- データ件数、タスク数、Episode数

禁止：

- 他参加者の非公開データ
- 利用権限のないデータ
- 法令・License違反データ
- 公式評価結果を利用した追加学習
- 公式評価時の観測を保存して後続学習へ利用

## 4. 評価環境へのアクセス

- 評価中の外部Network accessは禁止
- 許可される通信は評価ClientとPolicy Server間のみ
- 評価環境へのAttackやGuardrail回避は禁止
- 非公開タスク、評価設定、内部ファイルの取得を試みない

## 5. ZIP禁止事項

- 暗号化ZIP
- 入れ子の圧縮ファイル
- Zip bomb
- Symbolic link / Hard link
- 展開先外へのPath traversal
- 不要に難読化されたファイル
- `.git`、仮想環境、`node_modules`等の不要物
- RequirementsからのGit/HTTP/File URL依存

## 6. レポート整合性

提出レポートには以下を記載します。

### 学習・推論時の工夫

- 仮説
- 試行錯誤
- 採用・不採用理由

### モデル情報

- モデル名・Version
- Base model名・入手元
- Base weight revision / commit hash
- 提出Checkpoint hash
- モデル構成・圧縮方式
- Action chunk有無・Chunk長

### 学習情報

- Dataset名・出所
- 独自生成データの概要
- データ件数・タスク数・Episode数
- 学習方法
- Step数 / Epoch数
- 学習対象Parameter
- Full fine-tuning / LoRA / Adapter区分
- 主なHyperparameter

### 権利関係

- Base model license
- Dataset license
- 第三者Code license

レポートと提出モデルの内容が異ならないよう、Run Manifestから自動生成します。

## 7. 提出前チェックリスト

### 構造

- [ ] `policy_server.py`がルート直下
- [ ] `requirements.txt`がルート直下
- [ ] Model weightがZIP内に存在
- [ ] `/health`、`/reset`、`/act`が実装済み
- [ ] Server部の変更禁止範囲を変更していない

### 推論

- [ ] Actionは`np.ndarray`
- [ ] dtypeは`float32`
- [ ] shapeは`(7,)`
- [ ] NaN / Infなし
- [ ] `/act`最大8秒未満を目標
- [ ] `/reset`最大8秒未満を目標
- [ ] 起動90秒未満を目標
- [ ] 外部Network accessなし

### 学習実体

- [ ] 独自学習Weightが実際にLoadされる
- [ ] Public weightだけのFallbackがない
- [ ] Task固有FSM・Action tableがない
- [ ] Official evaluation由来の学習データがない

### ZIP

- [ ] 20GB以内
- [ ] 18GB以内を目標
- [ ] 暗号化なし
- [ ] Linkなし
- [ ] 不要物なし
- [ ] `validate_submission.py` PASS
- [ ] ZIP SHA256記録済み

### 提出

- [ ] 締切直前を避ける
- [ ] 1日1回制限を確認
- [ ] 採点完了時点で回数が消費される点を確認
- [ ] 提出したZIPを変更せずGoogle Driveへ保管
- [ ] Git Tag、Model Hash、ZIP Hashを紐づける
