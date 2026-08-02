from __future__ import annotations

import argparse
from pathlib import Path

DATASET_NAME = "parc_libero_plus_selected"
DATASET_VERSION = "1.0.0"
MIXTURE_NAME = "parc_stage_a_plus_only"
CONFIG_MARKER = "PARC2026_SELECTED_LIBERO_CONFIG"
TRANSFORM_MARKER = "PARC2026_SELECTED_LIBERO_TRANSFORM"
MIXTURE_MARKER = "PARC2026_SELECTED_LIBERO_MIXTURE"
LOADER_MARKER = "PARC2026_SELECTED_LIBERO_LOCAL_BUILDER"


def _append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _patch_loader(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if LOADER_MARKER in text:
        return
    old = "    builder = tfds.builder(name, data_dir=data_dir)\n"
    new = f'''    # {LOADER_MARKER}\n    if name == "{DATASET_NAME}":\n        builder_dir = tf.io.gfile.join(data_dir, name, "{DATASET_VERSION}")\n        builder = tfds.builder_from_directory(builder_dir)\n    else:\n        builder = tfds.builder(name, data_dir=data_dir)\n'''
    if old not in text:
        raise RuntimeError("Could not locate tfds.builder call; pinned OpenVLA-OFT source changed")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_openvla_oft(root: Path) -> None:
    dataset_dir = root / "prismatic/vla/datasets/rlds"
    oxe_dir = dataset_dir / "oxe"
    dataset_loader = dataset_dir / "dataset.py"
    configs = oxe_dir / "configs.py"
    transforms = oxe_dir / "transforms.py"
    mixtures = oxe_dir / "mixtures.py"
    for path in (dataset_loader, configs, transforms, mixtures):
        if not path.is_file():
            raise FileNotFoundError(f"Pinned OpenVLA-OFT layout not found: {path}")

    _patch_loader(dataset_loader)

    _append_once(
        configs,
        CONFIG_MARKER,
        f'''
# {CONFIG_MARKER}
# Match the official modified-LIBERO OXE contract.
OXE_DATASET_CONFIGS["{DATASET_NAME}"] = {{
    "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": "wrist_image"}},
    "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
    "state_obs_keys": ["EEF_state", "gripper_state"],
    "language_key": "language_instruction",
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
    """Mirror OpenVLA-OFT's official libero_dataset_transform."""
    # Source action uses -1=open and +1=close. OpenVLA trains with +1=open and 0=close.
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))
    trajectory["action"] = tf.concat(
        [trajectory["action"][:, :6], gripper_action], axis=1
    )

    # LeRobot state is EEF xyz + axis-angle + two gripper qpos values.
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]

    # Do not rotate here: LeRobot's LIBERO processor stores images in the
    # HuggingFaceVLA/LIBERO 180-degree-rotated training convention already.
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
