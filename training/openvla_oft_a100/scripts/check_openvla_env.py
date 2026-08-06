#!/usr/bin/env python3
"""OpenVLA-OFT環境が学習可能な状態で揃っているかを判定する。

`bootstrap_colab.sh`が再実行を省略してよいかの判断に使う。Colabセッションは頻繁に
落ちるが、VMが生きていれば10分のpip installをやり直す必要はない。ただし
「入っていそう」で省略すると壊れた環境で学習を始めることになるため、実際にimportし、
バージョンと出所まで確認する。

    python training/openvla_oft_a100/scripts/check_openvla_env.py            # 学習用
    python .../check_openvla_env.py --expect-pypi-transformers               # 提出計測用

exit 0 で健全、1 で要再構築。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _transformers_is_fork() -> bool | None:
    """The OpenVLA-OFT fork and the PyPI build both report 4.40.1.

    pip records direct_url.json only for VCS/URL installs, so that is what
    separates them.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return distribution("transformers").read_text("direct_url.json") is not None
    except PackageNotFoundError:
        return None


def inspect() -> dict[str, Any]:
    report: dict[str, Any] = {"imports": {}, "versions": {}}

    # prismatic comes from the editable openvla-oft install; dlimp and
    # tensorflow_graphics are the two dependencies bootstrap installs --no-deps.
    for module_name in (
        "torch",
        "transformers",
        "prismatic",
        "dlimp",
        "tensorflow_graphics.geometry.transformation",
        "numpy",
    ):
        try:
            __import__(module_name)
        except Exception as error:  # noqa: BLE001 - any import failure means rebuild
            report["imports"][module_name] = f"{type(error).__name__}: {error}"
        else:
            report["imports"][module_name] = "ok"

    from importlib.metadata import PackageNotFoundError, version

    for package in ("torch", "transformers", "numpy", "tensorflow"):
        try:
            report["versions"][package] = version(package)
        except PackageNotFoundError:
            report["versions"][package] = None

    report["transformers_is_fork"] = _transformers_is_fork()

    # patch_parc_dataset_registry.py edits the OpenVLA-OFT checkout, which is
    # rebuilt with every VM. Without it training dies on
    # KeyError: 'parc_stage_a_plus_only', and the bootstrap skip would otherwise
    # keep an environment that only looks complete.
    try:
        from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
        from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

        report["parc_dataset_registered"] = (
            "parc_stage_a_plus_only" in OXE_NAMED_MIXTURES
            and "parc_libero_plus_selected" in OXE_DATASET_CONFIGS
        )
    except Exception as error:  # noqa: BLE001 - any failure means not registered
        report["parc_dataset_registered"] = False
        report["parc_dataset_error"] = f"{type(error).__name__}: {error}"
    return report


def evaluate(
    report: dict[str, Any],
    expect_pypi_transformers: bool,
    require_parc_dataset: bool = False,
) -> list[str]:
    problems = [
        f"{name}: {status}"
        for name, status in report["imports"].items()
        if status != "ok"
    ]

    numpy_version = report["versions"].get("numpy")
    if numpy_version and int(numpy_version.split(".")[0]) >= 2:
        # torch 2.2.0 is built against the NumPy 1.x C API.
        problems.append(f"numpy {numpy_version} breaks torch tensor/ndarray conversion")

    is_fork = report["transformers_is_fork"]
    if is_fork is None:
        problems.append("transformers is not installed")
    elif expect_pypi_transformers and is_fork:
        problems.append("transformers is the git fork; the scoring environment uses PyPI")
    elif not expect_pypi_transformers and not is_fork:
        # colab_action_parity.sh swaps the fork out; training needs it back.
        problems.append("transformers is the PyPI build; training needs the OpenVLA-OFT fork")

    if require_parc_dataset and not report.get("parc_dataset_registered"):
        problems.append(
            "the PARC dataset is not registered in the OpenVLA-OFT checkout; "
            "training would fail with KeyError: 'parc_stage_a_plus_only'"
        )

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-pypi-transformers",
        action="store_true",
        help="Require the PyPI build instead of the OpenVLA-OFT fork.",
    )
    parser.add_argument(
        "--require-parc-dataset",
        action="store_true",
        help="Also require the PARC dataset and mixture to be registered.",
    )
    args = parser.parse_args()

    report = inspect()
    problems = evaluate(
        report, args.expect_pypi_transformers, args.require_parc_dataset
    )
    report["problems"] = problems
    report["status"] = "ok" if not problems else "rebuild_required"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
