from __future__ import annotations

import sys
from pathlib import Path

import pytest

from submission.openvla_oft_offline.runtime import prismatic_bootstrap


@pytest.fixture()
def vendored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`prepare_vendor.sh`が置くツリーの、壊れる部分だけを写したもの。

    親の`__init__.py`は openvla-oft e4287e94 と同じく重い依存へ降りていく。
    ここでは提出物では満たせない import として書いておく。dlimp は git 依存で、
    運営Validatorが `git+` を禁じているため requirements.txt には足せない。
    """
    root = tmp_path / "submission"
    package = root / "prismatic"
    for sub in ("models", "vla", "training", "extern/hf"):
        (package / sub).mkdir(parents=True)

    (package / "__init__.py").write_text("from .models import load\n", encoding="utf-8")
    (package / "models/__init__.py").write_text(
        "import definitely_not_installed_llm_backbones\n", encoding="utf-8"
    )
    (package / "vla/__init__.py").write_text("import dlimp\n", encoding="utf-8")
    (package / "training/__init__.py").write_text("import jsonlines\n", encoding="utf-8")
    (package / "extern/__init__.py").write_text("", encoding="utf-8")
    (package / "extern/hf/__init__.py").write_text("", encoding="utf-8")

    # 提出物が実際に読む葉。親の__init__.pyを通らなければ読める。
    (package / "models/action_heads.py").write_text("VALUE = 'action heads'\n", encoding="utf-8")
    (package / "vla/constants.py").write_text("ACTION_DIM = 7\n", encoding="utf-8")
    (package / "extern/hf/modeling_prismatic.py").write_text(
        "from prismatic.vla.constants import ACTION_DIM\nVALUE = ACTION_DIM\n", encoding="utf-8"
    )

    monkeypatch.syspath_prepend(str(root))
    for name in [n for n in sys.modules if n == "prismatic" or n.startswith("prismatic.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return root


def _forget(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in list(sys.modules) if n == "prismatic" or n.startswith("prismatic.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_without_the_bootstrap_the_heavy_chain_runs(
    vendored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """これが採点環境で出た失敗そのもの。先に確かめておく。"""
    with pytest.raises(ModuleNotFoundError):
        __import__("prismatic.extern.hf.modeling_prismatic")


def test_the_leaf_modules_import_once_the_parents_are_bypassed(
    vendored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forget(monkeypatch)
    prismatic_bootstrap.install_lightweight_packages(vendored)

    import prismatic.extern.hf.modeling_prismatic as modeling
    import prismatic.models.action_heads as action_heads

    assert modeling.VALUE == 7
    assert action_heads.VALUE == "action heads"


def test_only_the_named_packages_are_stubbed(
    vendored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """握り潰す範囲は狭いほどよい。空の`extern`まで置き換える理由はない。"""
    _forget(monkeypatch)
    installed = prismatic_bootstrap.install_lightweight_packages(vendored)

    assert installed == list(prismatic_bootstrap.STUBBED)
    assert "prismatic.extern" not in sys.modules


def test_an_already_imported_package_is_left_alone(
    vendored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本物が先に入っているなら、それを差し替えるのは越権。"""
    _forget(monkeypatch)
    sentinel = type(sys)("prismatic")
    monkeypatch.setitem(sys.modules, "prismatic", sentinel)

    installed = prismatic_bootstrap.install_lightweight_packages(vendored)

    assert "prismatic" not in installed
    assert sys.modules["prismatic"] is sentinel


def test_a_tree_without_the_vendor_is_a_no_op(tmp_path: Path) -> None:
    """Repoのチェックアウトには`prismatic/`が無い。そこで呼ばれても壊さない。"""
    assert prismatic_bootstrap.install_lightweight_packages(tmp_path) == []


def test_configure_offline_environment_bootstraps_prismatic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torchの先読みと同じ位置に置く。`prismatic`のどれかに触る前でなければ効かない。"""
    from submission.openvla_oft_offline.runtime import offline_env

    calls: list[str] = []
    monkeypatch.setattr(offline_env, "preload_nvjitlink", lambda: calls.append("preload"))
    monkeypatch.setattr(
        offline_env, "install_lightweight_packages", lambda: calls.append("bootstrap")
    )

    offline_env.configure_offline_environment()

    assert calls == ["preload", "bootstrap"]
