from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "PARC_COMPONENT_CHECKPOINT_PATCH"


def patch(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return

    source = source.replace(
        '    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)\n',
        '    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)\n'
        '    component_checkpoint_dir: Optional[Path] = None   # PARC_COMPONENT_CHECKPOINT_PATCH\n',
        1,
    )

    old = '''    if cfg.resume:\n        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)\n        module.load_state_dict(state_dict)\n'''
    new = '''    if cfg.resume:\n        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)\n        module.load_state_dict(state_dict)\n    elif cfg.component_checkpoint_dir is not None:\n        component_root = Path(cfg.component_checkpoint_dir)\n        matches = sorted(component_root.glob(f"{module_name}--*checkpoint.pt"))\n        if len(matches) != 1:\n            raise ValueError(\n                f"Expected exactly one pretrained {module_name} checkpoint under "\n                f"{component_root}, found {len(matches)}"\n            )\n        state_dict = torch.load(matches[0], weights_only=True, map_location="cpu")\n        module.load_state_dict(remove_ddp_in_checkpoint(state_dict))\n        print(f"Loaded pretrained {module_name} from {matches[0]}")\n'''
    if old not in source:
        raise RuntimeError("Could not locate init_module checkpoint block; pinned source changed")
    source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finetune_py", type=Path)
    args = parser.parse_args()
    patch(args.finetune_py)
    print(f"Patched: {args.finetune_py}")


if __name__ == "__main__":
    main()
