# PARC2026 提出までの実行計画

前提：モデル提出期限 2026-08-14 23:59、レポート提出期限 2026-08-17 23:59。

## 8月2日

- GitHub構成作成
- 運営Repo固定Commitの記録
- Google Driveフォルダ構成作成
- Colab用Training/Evaluation Notebookを分離
- Dataset/Run Manifest Schemaを作成
- ランダムPolicyで公開評価を実行

完了条件：`results/*.json`が生成される。

## 8月3日

- 公式50エピソードでR1学習
- LoRAマージ
- Checkpoint、設定、Dataset revision、Episode IDs、SHA256をDriveへ保存

完了条件：R1のMerged modelとRun Manifestが存在する。

## 8月4日

- SmolVLAを`MyPolicy`へ接続
- 2視点画像・状態・言語指示の変換
- Action unnormalize
- Action chunk cache
- `/health`、`/reset`、`/act`のSmoke Test

## 8月5日

- 公開4タスクを各3エピソード評価
- Replan interval 5/10/25/50を比較
- 最初の提出ZIPを作成
- `validate_submission.py`をPASSさせる

## 8月6日

- 最初の有効モデルを午前中に公式提出
- `lerobot/libero_plus`のTask inventory作成
- Spatial/Object/Goalから150 Episode抽出
- R2学習開始

## 8月7日

- R2評価
- R1/R2比較
- 失敗を認識・把持・運搬・配置・衝突へ分類
- 300 Episode版R3を作成して学習

## 8月8日

- R3 Screening
- Chunk再計画間隔の再比較
- 推論時間最大値・p95の測定
- ローカル失敗軌道の分析

## 8月9日

- 2回目の公式提出
- 認識・外観変化が弱い場合は画像拡張R5
- 学習容量不足ならLoRA rank 32のR4
- レポート用メタデータ生成開始

## 8月10日

- 上位2モデルを各10 Episodeで比較
- 最終候補を選定
- 最良条件をSeed変更でR6再学習
- この日以降、大きな構造変更を原則停止

## 8月11日

- 上位モデルを各20 Episode評価
- 600 stepsの長時間試験
- 起動・推論・ZIP展開を複数回確認
- Colab再起動後の再現性確認

## 8月12日

- 3回目の公式提出
- Model/Dataset/LoRA/Chunk/Dependenciesを原則固定

## 8月13日

- Model Freeze
- Git Tag作成
- Git SHA、Model SHA256、Dataset Manifest SHA256、ZIP SHA256を確定
- End-to-End評価を2回
- 提出レポート用Evidence生成

## 8月14日

- 午前から昼までに最終提出
- 提出日時・回数・ZIP Hashを記録
- 提出ZIPをDriveへ保存して変更しない

## 8月15日から17日

- Run Manifestからレポートを生成
- モデル・データ・学習方法・License・試行錯誤を確認
- 提出物との整合性チェック
- 8月17日23:59までにPDF提出

## 中止・切替基準

- 8月5日までにSmolVLA提出ZIPが動かない場合：モデル比較を中止し、Adapter/Feature変換の修正へ集中
- R2がR1より悪化：データ配分、Action変換、正規化を再確認
- Trainは良いがValidationが悪い：Steps/Rankを増やさずデータ多様性を改善
- Texture/Lightだけ弱い：軽い画像拡張を導入
- 8月10日以降：新規モデル系統への変更を原則禁止
