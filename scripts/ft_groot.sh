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

# Optional inpainting methods (arm inpainting, ...). When INPAINTING is set,
# route training through h2r_il.train, which injects the methods on the fly via
# LeRobot's set_image_transforms hook (dataset stays read-only). Same CLI flags.
# Tip: warm the cache first so DataLoader workers just read cached frames:
#   uv run --no-sync python scripts/viz_inpainting.py --dataset "$DATASET" \
#       --dataset-root "$DATASET_ROOT" --methods $INPAINTING \
#       --num-frames -1 --fill-cache --no-panels
INPAINTING=${INPAINTING:-}

# Optional object-pose target. OBJECT_POSE points at a store built up front by
# `object_pose_dataset track` + `store`; training only looks it up by the frame
# index each sample already carries. OBJECT_POSE_TARGET is position (default) or
# pose6d -- position only, because rotation comes from an arbitrary SAM3D body
# frame that nothing in the pipeline validates.
OBJECT_POSE=${OBJECT_POSE:-}

if [[ -n "$INPAINTING" || -n "$RELATIVE_ANGLE_DIMS" || -n "$OBJECT_POSE" ]]; then
    if [[ -n "$INPAINTING" ]]; then export H2R_INPAINTING="$INPAINTING"; fi
    if [[ -n "${INPAINTING_CACHE:-}" ]]; then export H2R_INPAINTING_CACHE="$INPAINTING_CACHE"; fi
    if [[ -n "$RELATIVE_ANGLE_DIMS" ]]; then export H2R_RELATIVE_ANGLE_DIMS="$RELATIVE_ANGLE_DIMS"; fi
    if [[ -n "$OBJECT_POSE" ]]; then export H2R_OBJECT_POSE="$OBJECT_POSE"; fi
    if [[ -n "${OBJECT_POSE_TARGET:-}" ]]; then export H2R_OBJECT_POSE_TARGET="$OBJECT_POSE_TARGET"; fi
    # Poses for the whole action chunk, so the pose channel is a trajectory rather
    # than a constant. Must be >= the policy's chunk_size or the tail of every
    # chunk goes unsupervised (the policy warns once if it is short). GR00T's
    # chunk_size defaults to 40; override CHUNK_SIZE and this together.
    if [[ -n "$OBJECT_POSE" ]]; then
        export H2R_OBJECT_POSE_HORIZON="${OBJECT_POSE_HORIZON:-${CHUNK_SIZE:-40}}"
    fi
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

# Attaching the target puts it in the batch; something has to consume it. With
# OBJECT_POSE set we therefore train the h2r_groot variant -- GR00T plus an
# object-pose head on the backbone -- rather than stock groot, which would carry
# the target through the batch and ignore it. OBJECT_POSE_HEAD=0 opts out, for
# checking the plumbing without the head.
POLICY_TYPE=groot
if [[ -n "$OBJECT_POSE" && "${OBJECT_POSE_HEAD:-1}" != "0" ]]; then
    POLICY_TYPE=h2r_groot
    # The store is passed twice on purpose and they mean different things: the env
    # var attaches the per-sample target, this one standardises it in the head.
    EXTRA_ARGS+=(--policy.object_pose_store="$OBJECT_POSE")
    EXTRA_ARGS+=(--policy.object_pose_weight="${OBJECT_POSE_WEIGHT:-1.0}")
    EXTRA_ARGS+=(--policy.object_pose_target="${OBJECT_POSE_TARGET:-position}")
    if [[ -n "${CHUNK_SIZE:-}" ]]; then EXTRA_ARGS+=(--policy.chunk_size="$CHUNK_SIZE"); fi
fi

exec "${TRAIN[@]}" \
    --dataset.repo_id="$DATASET" \
    --policy.type="$POLICY_TYPE" \
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
