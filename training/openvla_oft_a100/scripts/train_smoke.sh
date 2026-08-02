#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_DIR="${DATA_ROOT_DIR:?Set DATA_ROOT_DIR to the RLDS dataset root}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:?Set RUN_ROOT_DIR to the output directory}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to the prepared smoke dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the downloaded local checkpoint}"
MAX_STEPS="${MAX_STEPS:-500}"
LORA_RANK="${LORA_RANK:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
TRAIN_VLA_LORA="${TRAIN_VLA_LORA:-False}"
FREEZE_VISION_LORA="${FREEZE_VISION_LORA:-True}"

export WANDB_MODE="${WANDB_MODE:-offline}"

python vla-scripts/finetune.py \
  --vla_path "$CHECKPOINT_DIR" \
  --component_checkpoint_dir "$CHECKPOINT_DIR" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
  --learning_rate 1e-5 \
  --lr_warmup_steps 50 \
  --num_steps_before_decay "$MAX_STEPS" \
  --max_steps "$MAX_STEPS" \
  --save_freq 100 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --use_lora True \
  --lora_rank "$LORA_RANK" \
  --train_vla_lora "$TRAIN_VLA_LORA" \
  --freeze_vision_lora "$FREEZE_VISION_LORA" \
  --merge_lora_during_training False \
  --wandb_project parc2026-openvla-oft \
  --run_id_note smoke-stage-a-a100-40gb
