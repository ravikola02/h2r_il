#!/usr/bin/env bash
# Fine-tune GR00T N1.7 via lerobot-train.
# Defaults are overridable through env vars; any extra --flags are passed
# straight through to lerobot-train (last occurrence wins in draccus).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET=${DATASET:-lerobot/svla_so101_pickplace}
BASE_MODEL=${BASE_MODEL:-nvidia/GR00T-N1.7-3B}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
JOB_NAME=${JOB_NAME:-groot_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/$JOB_NAME}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-32}
WANDB=${WANDB:-false}

exec uv run lerobot-train \
    --dataset.repo_id="$DATASET" \
    --policy.type=groot \
    --policy.base_model_path="$BASE_MODEL" \
    --policy.embodiment_tag="$EMBODIMENT_TAG" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable="$WANDB" \
    "$@"
