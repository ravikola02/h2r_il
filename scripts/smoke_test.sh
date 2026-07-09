#!/usr/bin/env bash
# Phase-0 smoke test: a few training steps for both pi0 and GR00T N1.7 in the
# same env, on a small public dataset, with one checkpoint saved each.
# The dataset on disk is never modified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export STEPS=${STEPS:-10}
export BATCH_SIZE=${BATCH_SIZE:-2}
export WANDB=false

echo "=== smoke test: pi0 ==="
# pi0_base expects openpi camera names; map the dataset's two cameras onto a
# subset of them (either-direction subset passes lerobot's visual check).
PI0_RENAME='{"observation.images.up": "observation.images.base_0_rgb", "observation.images.side": "observation.images.left_wrist_0_rgb"}'
JOB_NAME=smoke_pi0 ./ft_pi0.sh --save_freq="$STEPS" --log_freq=1 --rename_map="$PI0_RENAME"

echo "=== smoke test: groot ==="
JOB_NAME=smoke_groot ./ft_groot.sh --save_freq="$STEPS" --log_freq=1

echo "=== checkpoints ==="
ls -d ../outputs/smoke_pi0/checkpoints/* ../outputs/smoke_groot/checkpoints/*
echo "SMOKE TEST PASSED"
