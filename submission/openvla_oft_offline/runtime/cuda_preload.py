"""torchを読み込む前に`libnvJitLink.so.12`を先に開く。

採点環境で`import torch`が次で落ちた。

    ImportError: libnvJitLink.so.12: cannot open shared object file

`libcusparse.so.12`は`libnvJitLink.so.12`をDT_NEEDEDに持ち、RUNPATHは
`$ORIGIN:$ORIGIN/../../nvjitlink/lib`、つまり**自分と同じsite-packagesの中しか
見ない**。torch 2.2.0の`_load_global_deps()`が先読みするのはcublas、cudnn、nvrtc、
cudart、cupti、cufft、curand、cusolver、cusparse、nccl、nvtxの11個で、nvjitlinkは
入っていない（torch 2.4で追加された）。

採点環境のvenvは`--system-site-packages`付きで作られ、`nvidia-*`のwheelがvenv側と
イメージ側に分かれて入る。requirements.txtへ`nvidia-nvjitlink-cu12`を厳密指定しても
この症状は消えなかったので、pipがどちらへ入れるかを当てにするのをやめる。

sonameで先にRTLD_GLOBALで開いておけば、以降`libnvJitLink.so.12`を要求する共有
ライブラリはロード済みのものに解決され、RUNPATHを辿らない。どのsite-packagesに
あっても効く。

見つからない場合は何もしない。ローダーの標準検索パスに置かれている環境では先読み
自体が不要で、そこで例外にすると動く構成を壊す。何を見て何が見つかったかはstderr
へ出す。採点ログに残る唯一の手掛かりになる。
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys

SONAME = "libnvJitLink.so.12"
# `nvidia/nvjitlink/lib/`が正規の位置。CUDA 13側は`libnvJitLink.so.13`を同じ場所へ
# 置くため、バージョン付きの名前まで拾わないように`.so.12`で始まるものだけを見る。
_PATTERNS = (
    os.path.join("nvidia", "nvjitlink", "lib", SONAME + "*"),
    # レイアウトが変わった場合の保険。nvidia配下の他のパッケージが同梱していても拾う。
    os.path.join("nvidia", "*", "lib", SONAME + "*"),
)


def _site_paths() -> list[str]:
    """sys.path上の実在ディレクトリ。venv側とイメージ側の両方が含まれる。"""
    seen: set[str] = set()
    paths: list[str] = []
    for entry in sys.path:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        if os.path.isdir(entry):
            paths.append(entry)
    return paths


def find_nvjitlink() -> str | None:
    for entry in _site_paths():
        for pattern in _PATTERNS:
            for candidate in sorted(glob.glob(os.path.join(entry, pattern))):
                if os.path.isfile(candidate):
                    return candidate
    return None


def describe_nvidia_layout() -> list[str]:
    """どのsite-packagesにどの`nvidia/*`が入っているかを列挙する。

    先読みが空振りしたときに、venv側とイメージ側のどちらに何が入ったのかを
    採点ログから読み取れるようにするためだけのもの。
    """
    lines: list[str] = []
    for entry in _site_paths():
        nvidia_root = os.path.join(entry, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        try:
            packages = sorted(
                name for name in os.listdir(nvidia_root)
                if os.path.isdir(os.path.join(nvidia_root, name))
            )
        except OSError:
            continue
        lines.append(f"{nvidia_root}: {', '.join(packages) or '(empty)'}")
        jit = os.path.join(nvidia_root, "nvjitlink", "lib")
        if os.path.isdir(jit):
            try:
                lines.append(f"  nvjitlink/lib: {', '.join(sorted(os.listdir(jit)))}")
            except OSError:
                pass
    return lines


def preload_nvjitlink() -> str | None:
    """見つかれば開いてそのパスを返す。見つからなければNone。例外は投げない。"""
    path = find_nvjitlink()
    if path is None:
        print(f"[parc] {SONAME} not found on sys.path", file=sys.stderr)
        for line in describe_nvidia_layout():
            print(f"[parc]   {line}", file=sys.stderr)
        return None
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        print(f"[parc] could not preload {path}: {error}", file=sys.stderr)
        return None
    print(f"[parc] preloaded {path}", file=sys.stderr)
    return path
