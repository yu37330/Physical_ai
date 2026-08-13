#!/usr/bin/env python3
"""提出ZIP内の`requirements.txt`だけを差し替え、別名のZIPを書き出す。

重みは変わらないのに依存だけを直したい場合、`build_submission_zip.py`で作り直すには
Base重みの再取得とCheckpointの再組み立てが要る。ここは既にあるArchiveを読み、
1エントリだけ差し替えて書き出す。

    python scripts/patch_submission_zip.py \\
      --input  /content/drive/MyDrive/PARC2026/60_submissions/parc2026_track1.zip \\
      --output /content/work/submission/parc2026_track1_patched.zip \\
      --requirements submission/openvla_oft_offline/requirements.txt

入力は読むだけで変更しない。差し替えに失敗しても元のArchiveが提出可能な状態で残る。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

TARGET_NAME = "requirements.txt"
COPY_BUFFER = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_BUFFER), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_requirements_entry(names: list[str]) -> str:
    """運営Validatorと同じ2つのレイアウトを許す。

    `validate_submission.py`の`_find_root_prefix`は、ルート直下と、単一フォルダで
    包んだ`<top>/`の両方を受け付ける。どちらで作られたArchiveでも同じように直せる
    ように、ここも同じ2つだけを候補にする。深い階層の同名ファイルは、提出物が
    読む`requirements.txt`ではないので候補にしない。
    """
    candidates = [
        name
        for name in names
        if name == TARGET_NAME
        or (name.count("/") == 1 and name.endswith(f"/{TARGET_NAME}"))
    ]
    if not candidates:
        raise SystemExit(
            f"No {TARGET_NAME} at the archive root or under a single top folder.\n"
            "Check the archive with: unzip -l <zip> | grep requirements.txt"
        )
    if len(candidates) > 1:
        raise SystemExit(f"Ambiguous {TARGET_NAME} entries: {candidates}")
    return candidates[0]


def _entry_for_writing(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """書き込み用に、保持したい属性だけを写した新しいZipInfoを作る。

    元のZipInfoをそのまま渡すと`flag_bits`（data descriptorや暗号化のビット）まで
    引き継ぎ、CRCとサイズを書き直す今回の経路と食い違う。`compress_type`は必ず
    引き継ぐ。重みは`build_submission_zip.py`がZIP_STOREDで格納しており、
    ここでDEFLATEDに変えると展開後40GB制限とは別に、書き出しが数十分に伸びる。
    """
    entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    entry.compress_type = info.compress_type
    entry.external_attr = info.external_attr
    entry.internal_attr = info.internal_attr
    entry.create_system = info.create_system
    # `file_size`も必ず写す。zipfileは書き込み前のZipInfoが申告するサイズだけを見て
    # ZIP64拡張を書くかどうかを決めるため、既定の0のままだと4GB超のEntryが
    # `RuntimeError: File size too large, try using force_zip64` で落ちる。
    # 7B Checkpointのshardは4GBを超える。close()が実測値で書き直すので、ここでの
    # 値はZIP64の判断にだけ効く。
    entry.file_size = info.file_size
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    requirements = args.requirements.resolve()

    if not source.is_file():
        raise SystemExit(f"Submission archive not found: {source}")
    if not requirements.is_file():
        raise SystemExit(f"Replacement requirements.txt not found: {requirements}")
    # 元のArchiveが唯一の提出可能な現物であることがある。上書きは許さない。
    if output == source:
        raise SystemExit("--output must differ from --input; the original is the fallback")

    payload = requirements.read_bytes()
    source_bytes = source.stat().st_size
    output.parent.mkdir(parents=True, exist_ok=True)

    # 空き容量を見る前に消す。どうせ上書きするので、既にある分は「要る空き」ではなく
    # 「使える空き」。失敗した前回の書きかけを残したまま数えると、実際には入るのに
    # 足りないと言って止まる。
    output.unlink(missing_ok=True)

    # 途中でディスクが尽きると、中途半端なArchiveが残った空きを占める。始める前に言う。
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < source_bytes + 1024**3:
        raise SystemExit(
            f"Not enough space for the patched archive: needs about "
            f"{source_bytes / 1024**3:.1f} GiB plus margin, "
            f"{free_bytes / 1024**3:.1f} GiB free at {output.parent}."
        )

    try:
        with zipfile.ZipFile(source, "r") as original:
            target = find_requirements_entry(original.namelist())
            infos = original.infolist()
            with zipfile.ZipFile(output, "w", allowZip64=True) as patched:
                for info in infos:
                    entry = _entry_for_writing(info)
                    if info.filename == target:
                        patched.writestr(entry, payload)
                        continue
                    if info.is_dir():
                        patched.writestr(entry, b"")
                        continue
                    # 14GBを一度に読まずに流す。重みはZIP_STOREDなので、この経路は
                    # 実質ファイルコピーになる。
                    with original.open(info, "r") as reader, patched.open(entry, "w") as writer:
                        shutil.copyfileobj(reader, writer, COPY_BUFFER)
    except BaseException:
        # 書き損じたArchiveは、ちょうど埋まったディスクの上の死荷重にしかならない。
        # OSErrorに限らない。zipfileはZIP64が要る場面をRuntimeErrorで知らせるし、
        # 14GBの書き出しはCtrl-Cされる余地も十分にある。どの経路でも残さない。
        output.unlink(missing_ok=True)
        raise

    with zipfile.ZipFile(output, "r") as verify:
        written = verify.read(target)
        entries = len(verify.namelist())
    if written != payload:
        output.unlink(missing_ok=True)
        raise SystemExit(f"Patched archive did not keep the new {target}")

    report = {
        "input": str(source),
        "output": str(output),
        "replaced_entry": target,
        "entries": entries,
        "source_entries": len(infos),
        "input_bytes": source_bytes,
        "zip_bytes": output.stat().st_size,
        "zip_gib": output.stat().st_size / 1024**3,
        "requirements_source": str(requirements),
        "input_sha256": sha256(source),
        # build_submission_zip.py と同じキー名。Drive へ記録を残す側が両方を
        # 同じように読めるようにしておく。
        "sha256": sha256(output),
    }
    report_path = args.report or output.with_suffix(output.suffix + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
