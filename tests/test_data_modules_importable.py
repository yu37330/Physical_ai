from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "data"

MODULES = sorted(
    path.stem
    for path in DATA_DIR.glob("*.py")
    if path.stem != "__init__"
)


@pytest.mark.parametrize("module", MODULES)
def test_every_data_module_imports_as_a_package_member(module: str) -> None:
    """These are invoked as `python -m src.data.<name>`.

    A bare `from libero_taxonomy import ...` resolves only when src/data itself
    is on sys.path, which it is not under `-m`, so the module fails at import.
    """
    importlib.import_module(f"src.data.{module}")


@pytest.mark.parametrize("module", MODULES)
def test_every_data_module_runs_under_dash_m(module: str) -> None:
    """pytest.ini puts the repo root on sys.path, which can mask an import that
    only works in-process. Run the real command line instead."""
    completed = subprocess.run(
        [sys.executable, "-m", f"src.data.{module}", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert "ModuleNotFoundError" not in completed.stderr, completed.stderr
    # --help exits 0; a module without argparse is still fine as long as it imported.
    assert completed.returncode in (0, 1, 2), completed.stderr
