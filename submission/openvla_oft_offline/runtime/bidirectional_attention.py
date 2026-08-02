from __future__ import annotations

"""Compatibility boundary for the OpenVLA-OFT Transformers fork.

The upstream fork changes Llama SDPA attention from causal to bidirectional while
preserving padding masking. Direct monkey-patching of Transformers internals is
version-sensitive, so this module intentionally exposes an explicit gate rather
than silently patching unknown versions.

Tomorrow's feasibility test must validate one of these paths:
1. vendor the exact patched Llama module from the MIT/Apache-compatible fork, or
2. implement a narrowly scoped patch for transformers==4.40.1 and prove output
   parity against the official fork.
"""

SUPPORTED_TRANSFORMERS_VERSION = "4.40.1"


def assert_supported_transformers_version() -> None:
    import transformers

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "OpenVLA-OFT bidirectional attention requires a validated Transformers "
            f"runtime. Expected {SUPPORTED_TRANSFORMERS_VERSION}, got "
            f"{transformers.__version__}."
        )


def apply_bidirectional_attention_patch() -> None:
    """Apply the validated patch once implemented.

    This raises by design until parity testing is complete. It prevents accidental
    evaluation with ordinary causal Llama attention, which would silently change
    the policy computation.
    """
    assert_supported_transformers_version()
    raise RuntimeError(
        "Bidirectional attention patch is not validated yet. Run the official-fork "
        "vs PyPI parity test before enabling the offline submission runtime."
    )
