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

# Optional object-pose target; see the same block in ft_groot.sh.
OBJECT_POSE=${OBJECT_POSE:-}

if [[ -n "$INPAINTING" || -n "$RELATIVE_ANGLE_DIMS" || -n "$OBJECT_POSE" ]]; then
    if [[ -n "$INPAINTING" ]]; then export H2R_INPAINTING="$INPAINTING"; fi
    if [[ -n "${INPAINTING_CACHE:-}" ]]; then export H2R_INPAINTING_CACHE="$INPAINTING_CACHE"; fi
    if [[ -n "$RELATIVE_ANGLE_DIMS" ]]; then export H2R_RELATIVE_ANGLE_DIMS="$RELATIVE_ANGLE_DIMS"; fi
    if [[ -n "$OBJECT_POSE" ]]; then export H2R_OBJECT_POSE="$OBJECT_POSE"; fi
    if [[ -n "${OBJECT_POSE_TARGET:-}" ]]; then export H2R_OBJECT_POSE_TARGET="$OBJECT_POSE_TARGET"; fi
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

# pi0 loads its weights through --policy.path, and LeRobot takes the policy class
# from that checkpoint's config.json `type` (it pops the field before applying CLI
# overrides), so --policy.type cannot switch it to h2r_pi0. Re-stage the checkpoint
# once instead -- scripts/stage_h2r_policy.py hard-links the weights and rewrites
# only the type -- and point PRETRAINED at the staged directory.
if [[ -n "$OBJECT_POSE" && "${OBJECT_POSE_HEAD:-1}" != "0" ]]; then
    STAGED_TYPE=$(python - "$PRETRAINED/config.json" <<'PY' 2>/dev/null || true
import json, sys
print(json.load(open(sys.argv[1])).get("type", ""))
PY
)
    if [[ "$STAGED_TYPE" != "h2r_pi0" ]]; then
        echo "OBJECT_POSE is set, but $PRETRAINED has type '${STAGED_TYPE:-unknown}', not h2r_pi0." >&2
        echo "Stage it once, then point PRETRAINED at the result:" >&2
        echo "    python scripts/stage_h2r_policy.py --checkpoint $PRETRAINED \\" >&2
        echo "        --type h2r_pi0 --out <staged-dir>" >&2
        echo "(or set OBJECT_POSE_HEAD=0 to attach the target without a head)" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--policy.object_pose_store="$OBJECT_POSE")
    EXTRA_ARGS+=(--policy.object_pose_weight="${OBJECT_POSE_WEIGHT:-1.0}")
    EXTRA_ARGS+=(--policy.object_pose_target="${OBJECT_POSE_TARGET:-position}")
    if [[ "${OBJECT_POSE_DETACH:-0}" != "0" ]]; then
        EXTRA_ARGS+=(--policy.object_pose_detach=true)
    fi
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
