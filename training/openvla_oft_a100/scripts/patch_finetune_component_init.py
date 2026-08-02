from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "PARC_COMPONENT_CHECKPOINT_PATCH"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Could not locate {label}; pinned OpenVLA-OFT source changed")
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return

    source = _replace_once(
        source,
        '    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)\n',
        '    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)\n'
        '    component_checkpoint_dir: Optional[Path] = None   # PARC_COMPONENT_CHECKPOINT_PATCH\n'
        '    train_vla_lora: bool = False                      # Stage A: head/projector only\n'
        '    freeze_vision_lora: bool = True                   # Stage B: preserve visual robustness\n',
        "FinetuneConfig insertion point",
    )

    source = _replace_once(
        source,
        '    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"\n',
        '    if cfg.train_vla_lora:\n'
        '        assert cfg.use_lora, "train_vla_lora=True requires --use_lora=True"\n',
        "LoRA-only assertion",
    )

    source = _replace_once(
        source,
        '''def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:\n''',
        '''class FrozenModuleWrapper(nn.Module):\n    """Expose `.module` like DDP without requiring trainable VLA parameters."""\n\n    def __init__(self, module: nn.Module):\n        super().__init__()\n        self.module = module\n\n    def forward(self, *args, **kwargs):\n        return self.module(*args, **kwargs)\n\n\ndef wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:\n''',
        "wrap_ddp definition",
    )

    source = _replace_once(
        source,
        '''    if cfg.resume:\n        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)\n        module.load_state_dict(state_dict)\n''',
        '''    if cfg.resume:\n        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)\n        module.load_state_dict(state_dict)\n    elif cfg.component_checkpoint_dir is not None:\n        component_root = Path(cfg.component_checkpoint_dir)\n        matches = sorted(component_root.glob(f"{module_name}--*checkpoint.pt"))\n        if len(matches) != 1:\n            raise ValueError(\n                f"Expected exactly one pretrained {module_name} checkpoint under "\n                f"{component_root}, found {len(matches)}"\n            )\n        state_dict = torch.load(matches[0], weights_only=True, map_location="cpu")\n        module.load_state_dict(remove_ddp_in_checkpoint(state_dict))\n        print(f"Loaded pretrained {module_name} from {matches[0]}")\n''',
        "component checkpoint initialization",
    )

    source = _replace_once(
        source,
        '''    # LoRA setup\n    if cfg.use_lora:\n        lora_config = LoraConfig(\n            r=cfg.lora_rank,\n            lora_alpha=min(cfg.lora_rank, 16),\n            lora_dropout=cfg.lora_dropout,\n            target_modules="all-linear",\n            init_lora_weights="gaussian",\n        )\n        vla = get_peft_model(vla, lora_config)\n        vla.print_trainable_parameters()\n''',
        '''    # LoRA setup\n    if cfg.use_lora:\n        lora_config = LoraConfig(\n            r=cfg.lora_rank,\n            lora_alpha=min(cfg.lora_rank, 16),\n            lora_dropout=cfg.lora_dropout,\n            target_modules="all-linear",\n            init_lora_weights="gaussian",\n        )\n        vla = get_peft_model(vla, lora_config)\n        if not cfg.train_vla_lora:\n            raise ValueError("use_lora=True requires train_vla_lora=True in the PARC training path")\n        if cfg.freeze_vision_lora:\n            for name, parameter in vla.named_parameters():\n                if "vision_backbone" in name and "lora_" in name:\n                    parameter.requires_grad = False\n            print("PARC Stage B: vision-backbone LoRA parameters frozen")\n        vla.print_trainable_parameters()\n    else:\n        for parameter in vla.parameters():\n            parameter.requires_grad = False\n        print("PARC Stage A: base VLA frozen without adding random LoRA weights")\n''',
        "LoRA setup block",
    )

    source = _replace_once(
        source,
        '''    # Wrap VLA with DDP\n    vla = wrap_ddp(vla, device_id, find_unused=True)\n''',
        '''    # Wrap VLA with DDP only when VLA parameters are trainable.\n    # PyTorch DDP rejects modules that have no parameters requiring gradients.\n    if cfg.train_vla_lora:\n        vla = wrap_ddp(vla, device_id, find_unused=True)\n    else:\n        vla = FrozenModuleWrapper(vla)\n''',
        "VLA DDP wrapping block",
    )

    source = _replace_once(
        source,
        '''        # Save processor and LoRA adapter\n        processor.save_pretrained(checkpoint_dir)\n        vla.module.save_pretrained(adapter_dir)\n''',
        '''        # Save processor and, only for Stage B, the trained LoRA adapter.\n        # Saving a frozen non-PEFT VLA here would duplicate the full 15GB model.\n        processor.save_pretrained(checkpoint_dir)\n        if cfg.use_lora and cfg.train_vla_lora:\n            vla.module.save_pretrained(adapter_dir)\n''',
        "checkpoint adapter save block",
    )

    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("finetune_py", type=Path)
    args = parser.parse_args()
    patch(args.finetune_py)
    print(f"Patched: {args.finetune_py}")


if __name__ == "__main__":
    main()
