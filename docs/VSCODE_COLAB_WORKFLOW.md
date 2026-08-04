# VS Code + Colab拡張機能での実行

公式のColab VS Code拡張機能（発行元Google、拡張機能ID`Google.colab`）を使い、ローカルのVS CodeからColabのGPUランタイムへ接続して実行します。以前必要だったSSHトンネルは不要です。無料Colabではノートブックを迂回するSSHシェルやリモート制御が制限対象になるため、拡張機能経由で使います。

## 役割分担

```text
VS Code          コード編集、Git、レビュー
GitHub           コードの正本
Notebook 00-06   正式な実行Runbook、Drive認証、Gate結果の確認
Colab Terminal   実処理の実行、デバッグ、失敗したStepだけの再実行
/content/work    Dataset、Base Checkpoint、学習中間生成物（揮発、約100GB）
Google Drive     Manifest、レポート、学習済みcomponent、SHA256（永続、小容量）
```

NotebookとTerminalは**同じScriptを呼びます**。実処理は`training/openvla_oft_a100/scripts/colab_*.sh`に集約してあるので、どちらの入口から実行しても内容は一致します。

## セットアップ

1. VS Codeで`Google Colab`拡張機能をインストールします。`Jupyter`拡張機能は依存関係として一緒に入ります。
2. `notebooks/00_environment_check.ipynb`を開きます。
3. 右上の`Select Kernel` → `Colab` → `New Colab Server`でA100を選び、Googleアカウントで認証します。既定のランタイムでよければ`Auto Connect`を選びます。

## 実行

### 1. Notebookで認証とGate

`00_environment_check.ipynb`を上から実行します。Driveマウント、`main`の取得、`colab_preflight.sh`によるGPU・Disk・Commit確認まで行います。

Driveのマウントは`Ctrl + Shift + P` → `Colab: Mount Google Drive to Server...`でも実行できます。

### 2. Colab Terminalで実処理

`Ctrl + Shift + P` → `Colab: Open Terminal`でColabランタイムに繋がったターミナルを開きます（experimental機能のため設定から有効化が必要な場合があります）。

**通常のVS CodeターミナルはローカルPC上で動きます。`Colab: Open Terminal`で開いたものだけがColab側です。**

```bash
cd /content/Physical_ai
bash training/openvla_oft_a100/scripts/colab_setup.sh
bash training/openvla_oft_a100/scripts/colab_dataset_prepare.sh
bash training/openvla_oft_a100/scripts/colab_stage_a.sh s1
bash training/openvla_oft_a100/scripts/colab_stage_a.sh s2
bash training/openvla_oft_a100/scripts/colab_submission_validate.sh
```

Terminal中心にすると、`%cd`と`!cd`の状態差、`set -euo pipefail`の効き方、複数行の環境変数といったNotebook特有の問題が構造的に起きません。

### 3. Notebookで結果確認

各NotebookのGate確認Cellで、Manifestとレポートの`status`を確認します。

## 無料T4での事前スモーク

A100を使う前に、無料枠のT4で配管だけを確認できます。7B Checkpointのダウンロード（十数GB）もDriveも不要です。

```bash
cd /content/Physical_ai
bash training/openvla_oft_a100/scripts/colab_smoke.sh
```

確認できる範囲:

- `bootstrap_colab.sh`（Python 3.12のTensorFlow依存Patchを含む）
- 提出用Prismatic Vendor
- 合成LeRobot Fixture → RLDS変換 → Source parity
- PARC Dataset Registry Patch
- `RLDSBatchTransform`。`AutoProcessor`と`AutoConfig`しか読まないため、Checkpointは**Processorとconfigの約2.5MBだけ**取得します

確認できない範囲:

- Stage A学習、Notebook 06のGPU Gate、提出ZIP

これらは実7B Checkpointが必要で、T4の16GB VRAMには載りません。OpenVLA-OFTに小型の差し替えモデルは存在しないため、A100専用のままです。

Notebook 00をT4で通したい場合は、A100要求だけ外します。

```bash
REQUIRE_A100=0 bash training/openvla_oft_a100/scripts/colab_preflight.sh
```

`a100_40gb`の判定自体はレポートに残り、必須Checkから外れるだけです。

## セッションが落ちる場合

Colabのセッションは頻繁に落ちます。落ち方によって失うものが違うので、切り分けて対処します。

### ターミナルだけ切れた（VMは生きている）

前景で走らせたコマンドはターミナルと一緒に死にます。**長いものは`nohup`で切り離してログへ流してください。**

```bash
cd /content/Physical_ai
nohup bash training/openvla_oft_a100/scripts/colab_setup.sh > /content/work/setup.log 2>&1 &
tail -f /content/work/setup.log
```

`tail`は`Ctrl+C`で抜けても本体は走り続けます。ターミナルが切れたら、繋ぎ直して`tail -f`し直すだけです。

```bash
tail -f /content/work/setup.log      # 進捗を再表示
pgrep -af colab_setup.sh             # まだ走っているか確認
```

### VMは生きているがスクリプトをやり直したい

`bootstrap_colab.sh`は冪等です。環境が揃っていれば10分のpip installを飛ばします。

```bash
python training/openvla_oft_a100/scripts/check_openvla_env.py
```

これが`status: ok`なら再構築は不要です。判定はimportの成否、numpyのメジャーバージョン、そしてtransformersがForkかPyPI版かまで見ます（`colab_action_parity.sh`はForkをPyPI版へ入れ替えるため、学習へ戻る前に検出できるようにしてあります）。

強制的に作り直す場合は`FORCE_BOOTSTRAP=1`を付けます。

### VMごと落ちた

`/content`は消えるので作り直しです。Checkpointの再取得は1分程度、pip installが10分程度かかります。

Driveの空きが足りず、15GBのCheckpointを退避しておくことはできません。復旧は次の1ブロックで済みます。

```bash
cd /content && rm -rf Physical_ai openvla-oft work
git clone -b main https://github.com/yu37330/Physical_ai.git
cd Physical_ai
nohup bash -c '
  REQUIRE_DRIVE=0 REQUIRE_A100=0 bash training/openvla_oft_a100/scripts/colab_setup.sh
' > /content/setup.log 2>&1 &
tail -f /content/setup.log
```

### アイドルで落とさない

実行中のジョブがあればアイドル判定は起きにくいですが、待ち時間が長い場合はターミナルを開いたままにしてください。作業を終えたら`Colab: Remove Server`で明示的に落とすと、コンピューティングユニットの浪費を防げます。

## 環境変数

`colab_env.sh`が全Scriptの共通契約です。既定値を変える場合だけ上書きします。

| 変数 | 既定値 | 用途 |
|---|---|---|
| `PROJECT_ROOT` | `/content/Physical_ai` | Repo |
| `WORK_ROOT` | `/content/work` | 揮発領域 |
| `DRIVE_ROOT` | `/content/drive/MyDrive/PARC2026` | 永続領域 |
| `OPENVLA_ROOT` | `/content/openvla-oft` | 固定OpenVLA-OFT |
| `DATASET_PROFILE` | `mini` | `mini`（3 Episode）または`full`（800 Episode） |
| `SKIP_PREFLIGHT` | `0` | `1`でPreflight全体を省略 |
| `REQUIRE_A100` | `1` | `0`でA100 40GBを必須Checkから外す（T4スモーク用） |
| `REQUIRE_DRIVE` | `1` | `0`でDriveのmount要求を外す（計測のみの実行用） |
| `FORCE_BOOTSTRAP` | `0` | `1`でOpenVLA-OFT環境を無条件に作り直す |
| `USE_PYPI_TRANSFORMERS` | `1` | 提出計測時にForkをPyPI版へ入れ替える |
| `RUN_DYNAMIC_SMOKE` | `0` | `1`で提出物の動的スモークを実行 |

`colab::persist`が既定256MB、Stage Aは既定1024MBを超えるDriveコピーを拒否します。大きな成果物をDriveへ入れて容量を枯渇させる事故を防ぐためです。

## 拡張機能の既知の制限

公式Wikiに記載のあるものです。

- `userdata.get()`はエラーになります。Colab Secretsは使えないので、必要な値は環境変数かipywidgetの入力で渡します。**このリポジトリは現時点で`userdata.get()`を使っていません。**
- `files.upload()`、`files.download()`はWeb版と挙動が異なります。File Upload IPyWidgetで代替します。
- Plotlyは`pio.renderers.default = "vscode"`の設定が必要です。
- `sheets.InteractiveSheet()`はサインインで止まります。
- `drive.mount()`はv0.2.1以降、`auth.authenticate_user()`はv0.3.0以降サポートされています。
- Colab TerminalとServer Mountingはexperimentalです。Server Mountingは外部で変更されたファイルが自動反映されないことがあります。

拡張機能側で不具合が出た場合は、Web版Colabで同じNotebookを開けば同じScriptを実行できます。Terminalにしか存在しない実行手順を作らないでください。

## 容量の前提

- `/content`は**永続化されません**。アイドルや最大稼働時間でVMごと削除されます。インストールした依存、cloneしたRepo、学習中間生成物は全て消えます。
- Drive上で大量の小さなファイルを直接読み書きすると遅くなるため、DatasetとCheckpointは`/content/work`で扱います。
- 提出ZIPはOpenVLA-7Bの重みを含み、`configs/models/openvla_oft_plus_checkpoint.yaml`が最大20GBを見込んでいます。Driveには置かず、`/content/work/submission`で組み立ててローカルPCへ直接ダウンロードします。DriveにはBuild ManifestとSHA256だけを残します。
- Stage Aはaction headとproprio projectorだけを学習するため、永続化すべき成果物は小さく収まります。Base重みは`checkpoint_manifest.json`に記録された固定Revisionから再取得できます。
