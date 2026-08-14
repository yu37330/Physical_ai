#!/usr/bin/env python3
"""提出ZIPのコードを、requirements.txtだけを入れたvenvで実際にimportする。

運営Validatorの静的検査は必須ファイルとZIP構造しか見ず、`--pip-dry-run`は依存が
解決できるかしか見ない。**解決できた依存で提出物のコードが動くか**は誰も見ていない
ため、そこが採点で初めて分かる状態だった。実際に次の2つが採点で出た。

    ImportError: libnvJitLink.so.12: cannot open shared object file
    ModuleNotFoundError: No module named 'draccus'

どちらもColabでは出ない。Colab側にはOpenVLA-OFTの依存一式が入っていて、Vendorした
`prismatic`が自己完結して見えるからである。

    python scripts/verify_submission_imports.py --archive /content/work/submission/x.zip

ZIPからテキストファイルだけを取り出す（重みは読まない）。venvは採点環境と同じく
`--system-site-packages`で作る。これがnvjitlinkの分裂を再現する唯一の条件である。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

REQUIREMENTS_NAME = "requirements.txt"
# 重みは読まない。ZIPの14GBのうち、importに要るのは数MBのテキストだけ。
SKIP_ROOTS = ("model_weights",)
TEXT_SUFFIXES = {".py", ".txt", ".json", ".yaml", ".yml", ".typed", ".md", ""}

# 提出物が起動時に触る入口。policy_server.pyがMyPolicyを作った時点でここまで走る。
IMPORT_PROBE = """
import sys, traceback

failures = []
# 先にこれ。torchより前にlibnvJitLink.so.12を開く経路そのものを通す。
try:
    from runtime.offline_env import configure_offline_environment
    configure_offline_environment()
except Exception:
    failures.append(("runtime.offline_env", traceback.format_exc()))

for name in [
    "runtime.preprocessing",
    "runtime.checkpoint_layout",
    "runtime.bidirectional_attention",
    "runtime.model_runtime",
    "prismatic.extern.hf.configuration_prismatic",
    "prismatic.extern.hf.modeling_prismatic",
    "prismatic.extern.hf.processing_prismatic",
    "prismatic.models.action_heads",
    "prismatic.models.projectors",
    "prismatic.vla.constants",
]:
    try:
        __import__(name)
    except Exception:
        failures.append((name, traceback.format_exc()))

# パッチ本体。transformersの版が合っているかは、これを呼ばないと分からない。
try:
    from runtime.bidirectional_attention import apply_bidirectional_attention_patch
    apply_bidirectional_attention_patch()
except Exception:
    failures.append(("apply_bidirectional_attention_patch", traceback.format_exc()))

for name, detail in failures:
    print(f"--- {name} ---", file=sys.stderr)
    print(detail, file=sys.stderr)
print(f"FAILURES={len(failures)}")
sys.exit(1 if failures else 0)
"""


def extract_code(archive: Path, destination: Path) -> str:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        candidates = [
            name
            for name in names
            if name == REQUIREMENTS_NAME
            or (name.count("/") == 1 and name.endswith(f"/{REQUIREMENTS_NAME}"))
        ]
        if len(candidates) != 1:
            raise SystemExit(f"Cannot locate {REQUIREMENTS_NAME} in {archive}: {candidates}")
        prefix = candidates[0][: -len(REQUIREMENTS_NAME)]
        for info in zf.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            relative = info.filename[len(prefix):]
            if not relative or relative.split("/")[0] in SKIP_ROOTS:
                continue
            if Path(relative).suffix not in TEXT_SUFFIXES:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    return prefix


def create_venv(env_dir: Path) -> tuple[Path, list[str]]:
    """`--system-site-packages`のvenvと、そこへインストールするpipコマンドを返す。

    Colabのシステム Python 3.12 には`ensurepip`が入っておらず、`with_pip=True`は

        CalledProcessError: ... '-m', 'ensurepip', '--upgrade', '--default-pip'

    で落ちる。Debian系がpipの同梱をdistroパッケージへ切り出しているためで、
    ここで直せる話ではない。pip自身を持たないvenvを作り、外側のpipに`--python`で
    そこへ入れさせる。`--python`はpip 23.3以降にある。
    """
    python = env_dir / "bin" / "python"
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
        return python, [str(python), "-m", "pip"]
    except subprocess.CalledProcessError:
        print("[check] no ensurepip; driving the venv from the outer pip", flush=True)
        shutil.rmtree(env_dir, ignore_errors=True)
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(env_dir)
        probe = subprocess.run(
            [sys.executable, "-m", "pip", "--python", str(python), "--version"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise SystemExit(
                "This interpreter's pip has neither ensurepip nor --python "
                f"(needs pip 23.3+):\n{probe.stderr.strip()}"
            )
        # `--python`はサブコマンドより前でなければpipが受け付けない。
        return python, [sys.executable, "-m", "pip", "--python", str(python)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="提出ZIP")
    source.add_argument("--directory", type=Path, help="展開済みの提出ディレクトリ")
    parser.add_argument("--keep", action="store_true", help="作業ディレクトリを残す")
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="parc_import_check_"))
    try:
        code_root = workspace / "code"
        if args.archive:
            code_root.mkdir()
            extract_code(args.archive.resolve(), code_root)
        else:
            code_root = args.directory.resolve()

        requirements = code_root / REQUIREMENTS_NAME
        if not requirements.is_file():
            raise SystemExit(f"No {REQUIREMENTS_NAME} under {code_root}")
        if not (code_root / "prismatic").is_dir():
            raise SystemExit(
                f"No vendored prismatic under {code_root}. "
                "Import checking is the point of this script, so a submission "
                "without it is not the thing that gets scored."
            )

        # 採点環境と同じ作り方。--system-site-packages でないと、nvidia-*の
        # wheelがvenv側とイメージ側へ分かれる状況そのものが再現しない。
        env_dir = workspace / "venv"
        print(f"[check] creating venv --system-site-packages at {env_dir}", flush=True)
        python, pip = create_venv(env_dir)

        print("[check] pip install -r requirements.txt", flush=True)
        install = subprocess.run(
            [*pip, "install", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
        # 「... requires X, but you have Y」を残さないことは運営READMEの明示的な指示。
        conflicts = [
            line for line in (install.stdout + install.stderr).splitlines()
            if "requires" in line and "but you have" in line
        ]
        if install.returncode != 0:
            print(install.stdout[-4000:], file=sys.stderr)
            print(install.stderr[-4000:], file=sys.stderr)
            raise SystemExit("pip install failed")
        for line in conflicts:
            print(f"[check] CONFLICT {line}", file=sys.stderr)

        print("[check] importing the submission's startup path", flush=True)
        probe = subprocess.run(
            [str(python), "-c", IMPORT_PROBE],
            cwd=str(code_root),
            capture_output=True,
            text=True,
        )
        sys.stderr.write(probe.stderr)
        print(probe.stdout.strip())

        report = {
            "source": str(args.archive or args.directory),
            "python": subprocess.run(
                [str(python), "-V"], capture_output=True, text=True
            ).stdout.strip(),
            "pip_conflicts": conflicts,
            "imports_ok": probe.returncode == 0,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if probe.returncode != 0:
            return 1
        if conflicts:
            print(
                "[check] imports passed but pip reported version conflicts; the "
                "official README asks for none.",
                file=sys.stderr,
            )
        return 0
    finally:
        if args.keep:
            print(f"[check] kept {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
