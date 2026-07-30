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

# Relative EEF actions: the policy predicts (action - current state) for the wrist
# dims, while RELATIVE_EXCLUDE_JOINTS keeps the gripper absolute (matched
# substring/case-insensitive against the dataset's action feature names; pi0
# already defaults to ["gripper"], groot does not).
# LeRobot's conversion is a plain subtraction, which is wrong at the +pi/-pi seam,
# so RELATIVE_ANGLE_DIMS names the Euler dims to wrap back into [-pi, pi]
# (see src/h2r_il/relative_angles.py). Set it whenever RELATIVE_ACTIONS is on and
# the action space has Euler angles.
RELATIVE_ACTIONS=${RELATIVE_ACTIONS:-}
RELATIVE_EXCLUDE_JOINTS=${RELATIVE_EXCLUDE_JOINTS:-'["gripper"]'}
RELATIVE_ANGLE_DIMS=${RELATIVE_ANGLE_DIMS:-}
if [[ -n "$RELATIVE_ACTIONS" ]]; then
    EXTRA_ARGS+=(--policy.use_relative_actions=true)
    EXTRA_ARGS+=(--policy.relative_exclude_joints="$RELATIVE_EXCLUDE_JOINTS")
fi

# Optional inpainting methods (arm inpainting, ...). When INPAINTING is set,
# route training through h2r_il.train, which injects the methods on the fly via
# LeRobot's set_image_transforms hook (dataset stays read-only). Same CLI flags.
# Tip: warm the cache first so DataLoader workers just read cached frames:
#   uv run --no-sync python scripts/viz_inpainting.py --dataset "$DATASET" \
#       --dataset-root "$DATASET_ROOT" --methods $INPAINTING \
#       --num-frames -1 --fill-cache --no-panels
INPAINTING=${INPAINTING:-}

if [[ -n "$INPAINTING" || -n "$RELATIVE_ANGLE_DIMS" ]]; then
    if [[ -n "$INPAINTING" ]]; then export H2R_INPAINTING="$INPAINTING"; fi
    if [[ -n "${INPAINTING_CACHE:-}" ]]; then export H2R_INPAINTING_CACHE="$INPAINTING_CACHE"; fi
    if [[ -n "$RELATIVE_ANGLE_DIMS" ]]; then export H2R_RELATIVE_ANGLE_DIMS="$RELATIVE_ANGLE_DIMS"; fi
    ENTRY=(-m h2r_il.train)
else
    ENTRY=(-m lerobot.scripts.lerobot_train)  # == the lerobot-train console script
fi

# NUM_GPUS>1 launches under accelerate (LeRobot's own multi-GPU path; a bare
# `python -m` run is a single process and only ever uses one GPU). Note BATCH_SIZE
# is *per process*: effective batch = BATCH_SIZE x NUM_GPUS. LeRobot deliberately
# does not auto-scale lr/steps, so pass --optimizer.lr / --steps yourself.
NUM_GPUS=${NUM_GPUS:-1}
if (( NUM_GPUS > 1 )); then
    TRAIN=(uv run --no-sync accelerate launch --multi_gpu --num_processes="$NUM_GPUS" "${ENTRY[@]}")
else
    TRAIN=(uv run --no-sync python "${ENTRY[@]}")
fi

exec "${TRAIN[@]}" \
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
