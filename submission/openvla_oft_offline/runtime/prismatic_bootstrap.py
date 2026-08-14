"""Vendorした`prismatic`の学習・データ系チェーンを走らせずにimportできるようにする。

提出物が使うのは6つのモジュールだけである。

    prismatic.extern.hf.configuration_prismatic
    prismatic.extern.hf.modeling_prismatic
    prismatic.extern.hf.processing_prismatic
    prismatic.models.action_heads
    prismatic.models.projectors
    prismatic.vla.constants

ところが、そのどれを読んでも親パッケージの`__init__.py`が先に走る。

    prismatic/__init__.py       -> .models
    prismatic/models/__init__.py -> .load, .materialize（LLM/Vision backbone一式）
    prismatic/vla/__init__.py    -> .materialize -> .datasets -> dlimp

この経路が引くものは提出物では満たせない。`dlimp`はgit依存であり、運営Validatorは
`git+`を禁止している（`BANNED_REQ_SCHEMES`）。TensorFlowも道連れになる。採点環境で
実際に次が出た。

    ModuleNotFoundError: No module named 'dlimp'

そこで、この3つのパッケージだけを`__init__.py`を実行しないモジュールとして先に
`sys.modules`へ置く。`__path__`は本物のディレクトリを指すので、下位モジュールの
importは通常どおり解決される。Vendorしたツリーには手を入れない——`prepare_vendor.sh`
が入れたものをそのまま残すほうが、Colabで作り直したときの食い違いが起きない。

`available_models`や`get_vla_dataset_and_collator`といった、握り潰した`__init__.py`が
公開していた名前は提出物のどこからも使っていない。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# `__init__.py`が重い依存を引くパッケージだけ。extern と extern.hf は空なので
# 触らない。握り潰す範囲は狭いほどよい。
STUBBED = (
    "prismatic",
    "prismatic.models",
    "prismatic.vla",
    # modeling_prismatic は prismatic.training.train_utils を読む。その親の
    # __init__.py は FSDP の学習戦略一式を引き、jsonlines と wandb まで辿る。
    "prismatic.training",
)


def vendored_root() -> Path:
    """`runtime/`の隣にVendorされた`prismatic/`。提出物の展開先を基準にする。"""
    return Path(__file__).resolve().parent.parent


def install_lightweight_packages(root: Path | None = None) -> list[str]:
    """`__init__.py`を実行しないパッケージとして先に登録する。

    既にimport済みのものには触らない。戻り値は今回登録した名前。
    """
    base = (root or vendored_root()).resolve()
    if not (base / "prismatic").is_dir():
        return []

    installed: list[str] = []
    for dotted in STUBBED:
        if dotted in sys.modules:
            continue
        directory = base.joinpath(*dotted.split("."))
        if not directory.is_dir():
            continue
        module = types.ModuleType(dotted)
        # `__path__`があるとPythonはパッケージとして扱い、下位モジュールを
        # ここから探す。`__init__.py`は読まれない。
        module.__path__ = [str(directory)]
        module.__package__ = dotted
        module.__doc__ = "Lightweight stand-in; see runtime/prismatic_bootstrap.py"
        sys.modules[dotted] = module
        parent, _, leaf = dotted.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, module)
        installed.append(dotted)
    if installed:
        print(
            f"[parc] bypassed heavy prismatic __init__ for: {', '.join(installed)}",
            file=sys.stderr,
        )
    return installed
