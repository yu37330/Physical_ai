from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from submission.openvla_oft_offline.runtime.model_runtime import _register_vendored_prismatic


class _FakeAuto:
    """Records registrations and rejects duplicates, like the real Auto classes."""

    def __init__(self, name: str, log: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._name = name
        self._log = log
        self._seen: set[tuple[Any, ...]] = set()

    def register(self, *args: Any) -> None:
        if args in self._seen:
            raise ValueError(f"{args[0]} is already registered for {self._name}")
        self._seen.add(args)
        self._log.append((self._name, args))


class _OpenVLAConfig:
    pass


class _OpenVLAForActionPrediction:
    pass


class _PrismaticImageProcessor:
    pass


class _PrismaticProcessor:
    pass


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    log: list[tuple[str, tuple[Any, ...]]] = []

    transformers = types.ModuleType("transformers")
    for name in ("AutoConfig", "AutoImageProcessor", "AutoProcessor", "AutoModelForVision2Seq"):
        setattr(transformers, name, _FakeAuto(name, log))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    for path, attributes in {
        "prismatic.extern.hf.configuration_prismatic": {"OpenVLAConfig": _OpenVLAConfig},
        "prismatic.extern.hf.modeling_prismatic": {
            "OpenVLAForActionPrediction": _OpenVLAForActionPrediction
        },
        "prismatic.extern.hf.processing_prismatic": {
            "PrismaticImageProcessor": _PrismaticImageProcessor,
            "PrismaticProcessor": _PrismaticProcessor,
        },
    }.items():
        module = types.ModuleType(path)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, path, module)

    for package in ("prismatic", "prismatic.extern", "prismatic.extern.hf"):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))

    return log


def test_registers_every_auto_class_the_checkpoint_auto_map_names(stubbed) -> None:
    """config.json, preprocessor_config.json and processor_config.json each carry an
    auto_map; every target needs a local registration or transformers demands
    trust_remote_code."""
    _register_vendored_prismatic()

    assert stubbed == [
        # AutoConfig first: the others resolve the config to find their entry.
        ("AutoConfig", ("openvla", _OpenVLAConfig)),
        ("AutoImageProcessor", (_OpenVLAConfig, _PrismaticImageProcessor)),
        ("AutoProcessor", (_OpenVLAConfig, _PrismaticProcessor)),
        ("AutoModelForVision2Seq", (_OpenVLAConfig, _OpenVLAForActionPrediction)),
    ]


def test_registering_twice_is_tolerated(stubbed) -> None:
    """The policy server may build more than one runtime in a process."""
    _register_vendored_prismatic()
    _register_vendored_prismatic()  # must not raise

    assert len(stubbed) == 4


def test_model_type_matches_the_checkpoint(stubbed) -> None:
    """The registered model_type has to equal config.json's, or lookup misses."""
    _register_vendored_prismatic()

    model_type = next(args[0] for name, args in stubbed if name == "AutoConfig")
    assert model_type == "openvla"
