#!/usr/bin/env bash
# Fine-tune GR00T N1.7 via lerobot-train.
# Defaults are overridable through env vars; any extra --flags are passed
# straight through to lerobot-train (last occurrence wins in draccus).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional per-dataset config (e.g. CONFIG=configs/gear_left.env) setting
# DATASET, DATASET_ROOT, ... (PI0_RENAME_MAP is ignored here: groot's
# new_embodiment adopts the dataset's camera keys directly.)
if [[ -n "${CONFIG:-}" ]]; then source "$CONFIG"; fi

DATASET=${DATASET:-lerobot/svla_so101_pickplace}
BASE_MODEL=${BASE_MODEL:-nvidia/GR00T-N1.7-3B}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
JOB_NAME=${JOB_NAME:-groot_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/$JOB_NAME}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-32}
WANDB=${WANDB:-false}
# DataLoader workers *per process* (LeRobot's default of 4 starves the GPUs:
# every sample decodes three 720x1280 videos). Total processes = NUM_WORKERS x
# NUM_GPUS, so keep it under nproc/NUM_GPUS with room for the trainer itself.
NUM_WORKERS=${NUM_WORKERS:-8}

EXTRA_ARGS=()
[[ -n "${DATASET_ROOT:-}" ]] && EXTRA_ARGS+=(--dataset.root="$DATASET_ROOT")

# Relative EEF actions: the policy predicts (action - current state) for the wrist
# dims, while RELATIVE_EXCLUDE_JOINTS keeps the gripper absolute (matched
# substring/case-insensitive against the dataset's action feature names).
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

# Optional visual methods (arm inpainting, ...). When VISUAL_METHODS is set,
# route training through h2r_il.train, which injects the methods on the fly via
# LeRobot's set_image_transforms hook (dataset stays read-only). Same CLI flags.
# Tip: warm the cache first so DataLoader workers just read cached frames:
#   uv run python scripts/viz_visual_methods.py --dataset "$DATASET" \
#       --dataset-root "$DATASET_ROOT" --methods $VISUAL_METHODS \
#       --num-frames -1 --fill-cache --no-panels
VISUAL_METHODS=${VISUAL_METHODS:-}

# Optional action interpolation: train a policy that moves slower than the demos.
# ACTION_SLOWDOWN=k resamples every action chunk to 1/k the demonstrated speed at
# dataloader time (dataset stays read-only; see src/h2r_il/action_interp.py). The
# policy still emits chunk_size actions at the dataset's fps, so it traces the
# same path k times slower. A bare factor, or JSON to declare Euler-angle dims so
# they are unwrapped rather than blended through a +pi/-pi flip, e.g.
#   ACTION_SLOWDOWN='{"factor":2,"angle_dims":[3,4,5,10,11,12]}'
ACTION_SLOWDOWN=${ACTION_SLOWDOWN:-}

if [[ -n "$VISUAL_METHODS" || -n "$ACTION_SLOWDOWN" || -n "$RELATIVE_ANGLE_DIMS" ]]; then
    if [[ -n "$VISUAL_METHODS" ]]; then export H2R_VISUAL_METHODS="$VISUAL_METHODS"; fi
    if [[ -n "${VISUAL_CACHE:-}" ]]; then export H2R_VISUAL_CACHE="$VISUAL_CACHE"; fi
    if [[ -n "$ACTION_SLOWDOWN" ]]; then export H2R_ACTION_SLOWDOWN="$ACTION_SLOWDOWN"; fi
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
    TRAIN=(uv run accelerate launch --multi_gpu --num_processes="$NUM_GPUS" "${ENTRY[@]}")
else
    TRAIN=(uv run python "${ENTRY[@]}")
fi

exec "${TRAIN[@]}" \
    --dataset.repo_id="$DATASET" \
    --policy.type=groot \
    --policy.base_model_path="$BASE_MODEL" \
    --policy.embodiment_tag="$EMBODIMENT_TAG" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --steps="$STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable="$WANDB" \
    "${EXTRA_ARGS[@]}" \
    "$@"
