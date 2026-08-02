from __future__ import annotations

import argparse
from pathlib import Path

DATASET_NAME = "parc_libero_plus_selected"
MIXTURE_NAME = "parc_stage_a_plus_only"
CONFIG_MARKER = "PARC2026_SELECTED_LIBERO_CONFIG"
TRANSFORM_MARKER = "PARC2026_SELECTED_LIBERO_TRANSFORM"
MIXTURE_MARKER = "PARC2026_SELECTED_LIBERO_MIXTURE"


def _append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def patch_openvla_oft(root: Path) -> None:
    oxe_dir = root / "prismatic/vla/datasets/rlds/oxe"
    configs = oxe_dir / "configs.py"
    transforms = oxe_dir / "transforms.py"
    mixtures = oxe_dir / "mixtures.py"
    for path in (configs, transforms, mixtures):
        if not path.is_file():
            raise FileNotFoundError(f"Pinned OpenVLA-OFT layout not found: {path}")

    _append_once(
        configs,
        CONFIG_MARKER,
        f'''
# {CONFIG_MARKER}
# Source TFDS steps already expose image, wrist_image, state, action, and language_instruction.
OXE_DATASET_CONFIGS["{DATASET_NAME}"] = {{
    "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": "wrist_image"}},
    "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
    "absolute_action_mask": [False, False, False, False, False, False, True],
    "action_normalization_mask": [True, True, True, True, True, True, False],
}}
''',
    )

    _append_once(
        transforms,
        TRANSFORM_MARKER,
        f'''
# {TRANSFORM_MARKER}
def parc_libero_plus_selected_transform(trajectory):
    """Match the official OpenVLA LIBERO visual convention without changing state/action semantics."""
    trajectory["observation"]["image"] = tf.image.rot90(
        trajectory["observation"]["image"], k=2
    )
    trajectory["observation"]["wrist_image"] = tf.image.rot90(
        trajectory["observation"]["wrist_image"], k=2
    )
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS["{DATASET_NAME}"] = parc_libero_plus_selected_transform
''',
    )

    _append_once(
        mixtures,
        MIXTURE_MARKER,
        f'''
# {MIXTURE_MARKER}
OXE_NAMED_MIXTURES["{MIXTURE_NAME}"] = [("{DATASET_NAME}", 1.0)]
''',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Register PARC selected RLDS dataset in pinned OpenVLA-OFT")
    parser.add_argument("--openvla-root", type=Path, required=True)
    args = parser.parse_args()
    patch_openvla_oft(args.openvla_root.resolve())
    print(f"Registered {DATASET_NAME} under {args.openvla_root}")


if __name__ == "__main__":
    main()
