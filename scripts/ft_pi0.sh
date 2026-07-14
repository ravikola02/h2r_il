#!/usr/bin/env bash
# Fine-tune pi0 via lerobot-train.
# Defaults are overridable through env vars; any extra --flags are passed
# straight through to lerobot-train (last occurrence wins in draccus).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional per-dataset config (e.g. CONFIG=configs/gear_left.env) setting
# DATASET, DATASET_ROOT, PI0_RENAME_MAP, ...
if [[ -n "${CONFIG:-}" ]]; then source "$CONFIG"; fi

DATASET=${DATASET:-lerobot/svla_so101_pickplace}
PRETRAINED=${PRETRAINED:-lerobot/pi0_base}
JOB_NAME=${JOB_NAME:-pi0_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/$JOB_NAME}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-32}
WANDB=${WANDB:-false}

EXTRA_ARGS=()
[[ -n "${DATASET_ROOT:-}" ]] && EXTRA_ARGS+=(--dataset.root="$DATASET_ROOT")
[[ -n "${PI0_RENAME_MAP:-}" ]] && EXTRA_ARGS+=(--rename_map="$PI0_RENAME_MAP")

exec uv run lerobot-train \
    --dataset.repo_id="$DATASET" \
    --policy.path="$PRETRAINED" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable="$WANDB" \
    "${EXTRA_ARGS[@]}" \
    "$@"
