from __future__ import annotations

import pytest

from training.openvla_oft_a100.scripts.check_openvla_env import evaluate

OK_IMPORTS = {
    "torch": "ok",
    "transformers": "ok",
    "prismatic": "ok",
    "dlimp": "ok",
    "tensorflow_graphics.geometry.transformation": "ok",
    "numpy": "ok",
}


def _report(**overrides):
    report = {
        "imports": dict(OK_IMPORTS),
        "versions": {"torch": "2.2.0", "transformers": "4.40.1", "numpy": "1.26.4"},
        "transformers_is_fork": True,
        "parc_dataset_registered": True,
    }
    report.update(overrides)
    return report


def test_an_unregistered_parc_dataset_blocks_training() -> None:
    """patch_parc_dataset_registry.py edits the OpenVLA-OFT checkout, which is
    rebuilt with every VM. Restoring a converted dataset from Drive used to skip
    it, and training died on KeyError: 'parc_stage_a_plus_only'."""
    problems = evaluate(
        _report(parc_dataset_registered=False),
        expect_pypi_transformers=False,
        require_parc_dataset=True,
    )

    assert any("parc_stage_a_plus_only" in problem for problem in problems)


def test_the_registry_is_not_required_unless_asked() -> None:
    """The submission measurement does not train, so it has no use for it."""
    assert evaluate(
        _report(parc_dataset_registered=False), expect_pypi_transformers=False
    ) == []


def test_a_complete_training_environment_has_no_problems() -> None:
    assert evaluate(_report(), expect_pypi_transformers=False) == []


def test_failed_imports_are_reported() -> None:
    imports = dict(OK_IMPORTS)
    imports["dlimp"] = "VersionError: gencode 6.31.1 runtime 5.29.6"

    problems = evaluate(_report(imports=imports), expect_pypi_transformers=False)

    assert len(problems) == 1
    assert "dlimp" in problems[0]


def test_numpy_2_is_rejected() -> None:
    """torch 2.2.0 is built against the NumPy 1.x C API."""
    problems = evaluate(
        _report(versions={"torch": "2.2.0", "transformers": "4.40.1", "numpy": "2.0.2"}),
        expect_pypi_transformers=False,
    )

    assert any("numpy 2.0.2" in problem for problem in problems)


def test_training_requires_the_fork() -> None:
    """colab_action_parity.sh swaps the fork out; training must not continue on
    the PyPI build."""
    problems = evaluate(_report(transformers_is_fork=False), expect_pypi_transformers=False)

    assert any("needs the OpenVLA-OFT fork" in problem for problem in problems)


def test_submission_measurement_requires_the_pypi_build() -> None:
    problems = evaluate(_report(transformers_is_fork=True), expect_pypi_transformers=True)

    assert any("scoring environment uses PyPI" in problem for problem in problems)


@pytest.mark.parametrize("expect_pypi", [False, True])
def test_missing_transformers_is_always_a_problem(expect_pypi: bool) -> None:
    problems = evaluate(_report(transformers_is_fork=None), expect_pypi_transformers=expect_pypi)

    assert any("not installed" in problem for problem in problems)
