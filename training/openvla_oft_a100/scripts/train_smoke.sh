#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_DIR="${DATA_ROOT_DIR:?Set DATA_ROOT_DIR to the RLDS dataset root}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:?Set RUN_ROOT_DIR to the output directory}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to the prepared smoke dataset}"
CHECKPOINT="${CHECKPOINT:-Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata}"
MAX_STEPS="${MAX_STEPS:-500}"
LORA_RANK="${LORA_RANK:-8}"

python vla-scripts/finetune.py \
  --vla_path "$CHECKPOINT" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --num_steps_before_decay "$MAX_STEPS" \
  --max_steps "$MAX_STEPS" \
  --save_freq 100 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank "$LORA_RANK" \
  --wandb_project parc2026-openvla-oft \
  --run_id_note smoke-lora-r8-a100-40gb
