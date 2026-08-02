from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rlds_contract import (  # noqa: E402
    ACTION_CHUNK_LENGTH,
    ACTION_DIM,
    DATASET_NAME,
    STATE_DIM,
    validate_batch_contract,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_split(
    *,
    data_root: Path,
    checkpoint: Path,
    dataset_name: str,
    split: str,
    sample_count: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoProcessor

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    dataset = RLDSDataset(
        data_root,
        dataset_name,
        transform,
        resize_resolution=tuple(model_config.image_sizes),
        shuffle_buffer_size=max(128, sample_count * 8),
        image_aug=False,
        train=split == "train",
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    sample_reports: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        report = validate_batch_contract(batch)
        report["index"] = index
        report["actions_dtype"] = str(batch["actions"].dtype)
        report["proprio_dtype"] = str(batch["proprio"].dtype)
        sample_reports.append(report)
        if len(sample_reports) >= sample_count:
            break
    if len(sample_reports) != sample_count:
        raise ValueError(
            f"Requested {sample_count} samples from {split}, received {len(sample_reports)}"
        )

    return {
        "split": split,
        "sample_count": len(sample_reports),
        "samples": sample_reports,
        "dataset_statistics": _jsonable(dataset.dataset_statistics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TFDS/RLDS output through OpenVLA-OFT RLDSBatchTransform")
    parser.add_argument("--openvla-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.openvla_root.resolve()
    if not (root / "prismatic").is_dir():
        raise FileNotFoundError(f"OpenVLA-OFT prismatic package not found under {root}")
    sys.path.insert(0, str(root))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    split_reports = [
        validate_split(
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            dataset_name=args.dataset_name,
            split=split,
            sample_count=args.samples_per_split,
        )
        for split in ("train", "validation")
    ]
    payload = {
        "status": "pass",
        "dataset_name": args.dataset_name,
        "contract": {
            "action_chunk_length": ACTION_CHUNK_LENGTH,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "front_tensor_key": "pixel_values",
            "wrist_tensor_key": "pixel_values_wrist",
            "two_camera_input": True,
        },
        "splits": split_reports,
        "checks": {
            "rlds_dataset_constructed": True,
            "rlds_batch_transform_passed": True,
            "collator_passed": True,
            "front_and_wrist_tensors_present": True,
            "finite_action_and_proprio": True,
            "action_chunk_shape_passed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2))


if __name__ == "__main__":
    main()
