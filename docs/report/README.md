# 予選レポート

`parc2026_report.html` が原稿、`parc2026_report.pdf` が提出物です。PDFは**A4 2ページ**（運営の上限）に収まるよう調整してあります。提出は Omnicampus のレポートフォームへ、8/17 23:59 まで（締切までなら複数回提出可）。

## 差し替えが必要な箇所

HTML内で `class="fill"`（黄色ハイライト）が付いている2箇所は、実際に提出したArchiveから確定させてください。

| 箇所 | 取得元 |
|---|---|
| 提出ZIPのSHA256 | `MyDrive/PARC2026/60_submissions/patched_submission_sha256.txt` |
| Stage Aのステップ数（S1=100 / S2=500） | 提出物内の `model_weights/openvla_oft_plus/parc_submission_manifest.json` の `trained_run_dir` |

後者はColabで次のように取り出せます。ZIP全体を展開せず、該当エントリだけ読みます。

```bash
unzip -p /content/drive/MyDrive/PARC2026/60_submissions/parc2026_track1_openvla_oft_plus_patched.zip \
  model_weights/openvla_oft_plus/parc_submission_manifest.json
```

`trained_run_dir` が `stage_a_s1_head_proprio_100` なら100 step、`stage_a_s2_head_proprio_500` なら500 stepです。同じJSONに `base_resolved_revision` と学習済みcomponentのSHA256も入っています。

## PDFの作り直し

HTMLを編集したあと、Chromiumのヘッドレス印刷で再生成します。日本語はIPAGothicで描画されます。

```bash
chromium --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=docs/report/parc2026_report.pdf \
  file://$PWD/docs/report/parc2026_report.html
```

手元にChromiumが無ければ、ブラウザでHTMLを開いて Ctrl+P →「PDFに保存」でも同じ結果になります。用紙A4、余白「なし」、背景グラフィックを有効にしてください。

**2ページに収まっているか毎回確認すること。** 加筆すると3ページ目が出ます。その場合は本文を削るか、`body` の `font-size` / `line-height` を詰めます。

## タイトル

Omnicampus のレポートフォームには「レポートのタイトル」欄があります。PDFのH1と揃えてください。

```
事前学習済みVLAを壊さない最小適応 ― OpenVLA-OFT+によるTrack 1提出
```
