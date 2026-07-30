#!/usr/bin/env bash
# Pausable GR00T training chain: the same dataset trained twice, raw vs inpainted.
#
#   ./scripts/chain.sh start <dataset> [run-name] [dataset-root]   start a new chain
#   ./scripts/chain.sh resume    pick up from the last checkpoint (no args)
#   ./scripts/chain.sh status    where it is: stage, step, ETA, cache, GPU, disk
#   ./scripts/chain.sh logs      follow the live output (Ctrl-C is safe)
#   ./scripts/chain.sh pause     stop training and FREE THE GPUS; fully resumable
#   ./scripts/chain.sh stop      abandon the chain (no resume)
#
# Five stages: train raw -> validate -> warm inpaint cache -> train inpaint ->
# validate. Both trainings are GR00T on the same dataset with the same steps and
# batch size; the only difference is that the second sees arm-inpainted frames
# (H2R_INPAINTING=arm_inpaint), so the pair is a clean baseline/treatment ablation.
#
# <dataset> is the HF repo id (e.g. user/my_dataset). Its basename picks the
# per-dataset config `configs/<name>.env` (modalities, relative actions, rename
# map) and the inpainting cache `$CACHE_ROOT/<name>`.
#
# <run-name> names both policies of the pair: the two jobs (and therefore their
# output dirs and validation reports) are `<run-name>_raw` and
# `<run-name>_inpaint`, so the ablation stays together under one name. Defaults
# to `groot_<dataset name>_<timestamp>`.
#
# The dataset root is argument 3, or `$DATASETS_ROOT/<name>` if that exists, or
# the HF cache. Everything is recorded in the state file, so `resume` takes no
# arguments.
#
# Pause really stops training -- the processes exit and GPU memory is released.
# Training restarts from the last checkpoint, so pausing costs at most SAVE_FREQ
# steps of work (see the note on SAVE_FREQ below). Ctrl-C while following the log
# only detaches your terminal; it never touches training.
#
# GPU stages wait for the GPUs to be mostly idle before launching, so you can
# start the chain while someone else's job is still finishing (see GPU_FREE_PCT).
#
# Knobs (override on the command line, e.g. `WANDB=true ./scripts/chain.sh start ...`):
WANDB=${WANDB:-false}          # true -> live loss curves; you are already logged in
BATCH_SIZE=${BATCH_SIZE:-64}   # per GPU; effective batch = BATCH_SIZE x NUM_GPUS = 128
NUM_GPUS=${NUM_GPUS:-2}
# DataLoader workers per GPU process. LeRobot's default of 4 starves the GPUs
# (three 720x1280 video streams decoded per sample): ~15 samples/s. 16 hit ~32
# samples/s but filled shm with decoded video -- throughput collapsed to ~4 by
# step 500 with 45 GB shared and one GPU idling. 8 keeps the win, halves the
# footprint. Total loader processes = NUM_WORKERS x NUM_GPUS. If a run dies with
# "Killed" (host OOM, not CUDA OOM), this is the first knob to lower: prefetched
# batches are ~BATCH_SIZE x 3 cameras of full-res frames per worker.
NUM_WORKERS=${NUM_WORKERS:-8}
STEPS=${STEPS:-20000}
# Checkpoint cadence, and therefore how much a pause can cost: at ~6 s/step,
# 250 steps is ~25 min of work at risk. Lower = cheaper pause, more 24 GB writes.
SAVE_FREQ=${SAVE_FREQ:-250}
EPISODES=${EPISODES:-"14 15 19"}
# GPU work waits until every GPU it will use is at least this % free (memory).
# Lets you queue the chain behind someone else's job. 0 disables the wait;
# GPU_WAIT_TIMEOUT=0 waits forever, otherwise give up (and fail) after N seconds.
GPU_FREE_PCT=${GPU_FREE_PCT:-80}
GPU_WAIT_TIMEOUT=${GPU_WAIT_TIMEOUT:-0}
GPU_POLL=${GPU_POLL:-30}

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Machine-specific paths (VAL_DS, CACHE_ROOT, DATASETS_ROOT and the default
# DATASET) live in the untracked configs/local.env -- see LOCAL_SETUP.md.
# Anything already in the environment wins over it.
if [[ -f configs/local.env ]]; then source configs/local.env; fi
: "${VAL_DS:?set VAL_DS (validation dataset path) in configs/local.env}"
: "${CACHE_ROOT:?set CACHE_ROOT (inpainting cache root) in configs/local.env}"
STATE_DIR=outputs/chain
STATE=$STATE_DIR/state.env
LOG=$STATE_DIR/chain.log
CKPT_DIR=$(printf "%06d" "$STEPS")

# uv would re-sync the venv on every `uv run` and fails on packages installed
# into it as root; the env is already built, so skip the sync.
UV=(uv run --no-sync)

mkdir -p "$STATE_DIR"
say()  { echo "[chain $(date '+%F %T')] $*"; }
die()  { say "ABORT: $*"; exit 1; }
load() { [[ -f $STATE ]] && source "$STATE"; }
save() { cat > "$STATE" <<EOF
STAGE=$STAGE
JOB_RAW=$JOB_RAW
JOB_INPAINT=$JOB_INPAINT
DATASET=$DATASET
DATASET_ROOT=${DATASET_ROOT:-}
CONFIG=$CONFIG
CACHE=$CACHE
EOF
}

# Derive everything that follows from the dataset id: the per-dataset config and
# the inpainting cache. Called once when a chain is created; afterwards the
# values come from the state file, so a resume can never drift onto another
# dataset or cache.
configure_dataset() {  # configure_dataset <dataset> [dataset-root]
    DATASET=$1
    local name=${DATASET##*/}
    if [[ -n ${2:-} ]]; then
        DATASET_ROOT=$2
    elif [[ -n ${DATASETS_ROOT:-} && -d $DATASETS_ROOT/$name ]]; then
        DATASET_ROOT=$DATASETS_ROOT/$name
    else
        DATASET_ROOT=""   # not local -> lerobot resolves it from the HF cache
    fi
    [[ -n $DATASET_ROOT && ! -d $DATASET_ROOT ]] && die "dataset root not found: $DATASET_ROOT"
    CONFIG=configs/$name.env
    [[ -f $CONFIG ]] || die "no config for '$name': create $CONFIG (copy configs/gear_left.env)"
    CACHE=$CACHE_ROOT/$name
}

# --- helpers ----------------------------------------------------------------

# PIDs of anything this chain runs on the GPUs (not the supervisor itself).
work_pids() { pgrep -f "h2r_il\.train|accelerate launch|viz_inpainting|eval_openloop" 2>/dev/null; }
sup_pid()   { pgrep -f "chain\.sh __run" 2>/dev/null | head -1; }

# Newest *resumable* checkpoint, or empty. A run killed mid-save leaves a
# directory with the 12 GB model written but no optimizer state; resuming from
# one of those fails (or silently drops the optimizer), so skip it.
latest_ckpt() {  # latest_ckpt <job> -> newest complete checkpoint dir, or empty
    local d
    while read -r d; do
        [[ -f $d/training_state/optimizer_state.safetensors ]] || continue
        [[ -f $d/pretrained_model/train_config.json ]]        || continue
        echo "$d"; return
    done < <(ls -1d "outputs/$1/checkpoints"/[0-9]* 2>/dev/null | sort -r)
}

# Space needed before a training stage: two checkpoints (the pruner briefly
# holds the new one alongside the old) plus headroom. A full disk used to
# surface as a crash an hour in, mid-checkpoint-write; this fails at step 0.
CKPT_GB=${CKPT_GB:-25}
check_disk() {
    local need=$((CKPT_GB * 2)) free
    free=$(df -BG --output=avail "$(readlink -f outputs)" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -z $free ]] && { say "warning: could not read free space; continuing"; return 0; }
    if (( free < need )); then
        say "only ${free}G free where checkpoints land; need ~${need}G (2 x ${CKPT_GB}G)"
        say "free space or lower CKPT_GB, then resume"
        return 1
    fi
    say "disk ok: ${free}G free (need ~${need}G)"
}

# The H2R_* hooks live in env vars, not in train_config.json, so they must be
# re-exported on every launch -- including resumes -- or a resumed run would
# silently drop the angle fix (or, worse, train the inpainting arm on raw frames).
export_hooks() {  # export_hooks [with_inpainting]
    set -a; source "$CONFIG"; set +a
    [[ -n "${RELATIVE_ANGLE_DIMS:-}" ]] && export H2R_RELATIVE_ANGLE_DIMS="$RELATIVE_ANGLE_DIMS"
    if [[ "${1:-}" == "with_inpainting" ]]; then
        export H2R_INPAINTING=arm_inpaint
        export H2R_INPAINTING_CACHE="$CACHE"
    else
        unset H2R_INPAINTING H2R_INPAINTING_CACHE
    fi
}

# Checkpoints are ~24 GB and LeRobot never prunes them. At SAVE_FREQ=250 a run
# would write 80 of them; this keeps only the newest (plus the final) so a run
# costs ~24-48 GB instead of filling the disk.
start_pruner() {
    ( while true; do
        sleep 120
        local d="outputs/$1/checkpoints"
        [[ -d $d ]] || continue
        ls -1dt "$d"/[0-9]* 2>/dev/null | grep -v "/$CKPT_DIR\$" | tail -n +2 \
            | while read -r old; do rm -rf "$old"; done
      done ) &
    PRUNER=$!
}
stop_pruner() { [[ -n "${PRUNER:-}" ]] && kill "$PRUNER" 2>/dev/null; PRUNER=""; }

# Block until GPUs 0..NUM_GPUS-1 are each >= GPU_FREE_PCT% free memory, so the
# chain can be queued behind another job instead of OOM-ing next to it.
wait_for_gpus() {
    (( GPU_FREE_PCT <= 0 )) && return 0
    local waited=0 busy
    while :; do
        busy=$(nvidia-smi --query-gpu=index,memory.free,memory.total \
                 --format=csv,noheader,nounits 2>/dev/null \
               | head -n "$NUM_GPUS" \
               | awk -F', ' -v p="$GPU_FREE_PCT" \
                   '$3 > 0 && 100*$2/$3 < p { printf "gpu%s %d%% free; ", $1, 100*$2/$3 }')
        if [[ -z $busy ]]; then
            [[ $waited -gt 0 ]] && say "GPUs free after ${waited}s -- starting"
            return 0
        fi
        if (( GPU_WAIT_TIMEOUT > 0 && waited >= GPU_WAIT_TIMEOUT )); then
            say "gave up waiting for GPUs after ${waited}s (${busy%; })"
            return 1
        fi
        # One line on entry, then a reminder every ~10 minutes.
        if (( waited == 0 || waited % 600 < GPU_POLL )); then
            say "waiting for >=${GPU_FREE_PCT}% free on $NUM_GPUS GPU(s): ${busy%; }"
        fi
        sleep "$GPU_POLL"
        waited=$((waited + GPU_POLL))
    done
}

# --- stages -----------------------------------------------------------------

run_train() {  # run_train <job> [with_inpainting]
    local job=$1 inpaint=${2:-}
    check_disk    || return 1
    wait_for_gpus || return 1
    export_hooks "$inpaint"
    local ckpt; ckpt=$(latest_ckpt "$job")
    # LeRobot refuses to start fresh into an existing output_dir (train.py
    # validate(): FileExistsError). After a crash before the first good
    # checkpoint, the dir exists but holds nothing resumable -- clear it so the
    # chain restarts the stage instead of wedging on every retry.
    if [[ -z $ckpt && -d outputs/$job ]]; then
        say "clearing unusable output dir outputs/$job ($(du -sh "outputs/$job" 2>/dev/null | cut -f1), no resumable checkpoint)"
        rm -rf "outputs/$job"
    fi
    start_pruner "$job"
    if [[ -n $ckpt && -f $ckpt/pretrained_model/train_config.json ]]; then
        # Resume: the checkpoint's own config carries batch size, steps, relative
        # actions, etc. Passing them again risks contradicting it, so we don't.
        # num_workers is the one exception: it is a throughput knob, not part of
        # the training semantics (the sampler order is seeded independently of it),
        # so it is safe to override and must be, or a resume would restore
        # whatever worker count the checkpoint was written with.
        say "resuming $job from $(basename "$ckpt") (sample-exact, workers=$NUM_WORKERS)"
        "${UV[@]}" accelerate launch --multi_gpu --num_processes="$NUM_GPUS" -m h2r_il.train \
            --config_path="$ckpt/pretrained_model/train_config.json" --resume=true \
            --num_workers="$NUM_WORKERS"
    else
        say "starting $job from scratch"
        CONFIG=$CONFIG DATASET=$DATASET DATASET_ROOT=${DATASET_ROOT:-} \
        NUM_GPUS=$NUM_GPUS BATCH_SIZE=$BATCH_SIZE \
        NUM_WORKERS=$NUM_WORKERS WANDB=$WANDB JOB_NAME="$job" \
        INPAINTING=${H2R_INPAINTING:-} INPAINTING_CACHE=${H2R_INPAINTING_CACHE:-} \
            ./scripts/ft_groot.sh --steps="$STEPS" --save_freq="$SAVE_FREQ"
    fi
    local rc=$?
    stop_pruner
    return $rc
}

run_val() {  # run_val <job>
    local job=$1
    local ckpt="outputs/$job/checkpoints/$CKPT_DIR/pretrained_model"
    [[ -d $ckpt ]] || die "no final checkpoint for $job at $ckpt"
    say "validating $job on episodes $EPISODES"
    # The policy as-is against the recorded data as-is.
    "${UV[@]}" python scripts/eval_openloop.py \
        --checkpoint "$ckpt" --dataset "$VAL_DS" --episodes $EPISODES \
        --out "outputs/validation/${job}__$(basename "$VAL_DS")"
}

run_warm() {
    wait_for_gpus || return 1
    export H2R_INPAINTING_CACHE="$CACHE"
    local have; have=$(cached_frames)
    say "warming arm-inpaint cache (have $have/$(cache_target) frames; content-addressed, resumable)"
    local root_args=()
    [[ -n ${DATASET_ROOT:-} ]] && root_args=(--dataset-root "$DATASET_ROOT")
    for i in $(seq 0 $((NUM_GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$i "${UV[@]}" python scripts/viz_inpainting.py \
            --dataset "$DATASET" "${root_args[@]}" \
            --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
            --num-shards "$NUM_GPUS" --shard $i &
    done
    wait
}

cached_frames() { find "$CACHE/arm_inpaint" -name '*.png' 2>/dev/null | wc -l; }

# Frames the warm has to produce: every frame of every camera.
cache_target() {
    local info="${DATASET_ROOT:-}/meta/info.json"
    [[ -f $info ]] || { echo "?"; return; }
    python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
cams = [k for k in d.get("features", {}) if k.startswith("observation.images.")]
print(d.get("total_frames", 0) * len(cams))' "$info" 2>/dev/null || echo "?"
}

# --- supervisor (internal; runs detached) -----------------------------------

__run() {
    load
    trap 'stop_pruner' EXIT
    while :; do
        case $STAGE in
            train_raw)     say "[1/5] train $JOB_RAW (raw frames)"
                           run_train "$JOB_RAW"     || die "training $JOB_RAW failed"; STAGE=val_raw ;;
            val_raw)       say "[2/5] validate $JOB_RAW"
                           run_val   "$JOB_RAW"     || die "validation $JOB_RAW failed"; STAGE=warm ;;
            warm)          say "[3/5] warm arm-inpaint cache"
                           run_warm                 || die "cache warm failed";  STAGE=train_inpaint ;;
            train_inpaint) say "[4/5] train $JOB_INPAINT (arm-inpainted frames)"
                           run_train "$JOB_INPAINT" with_inpainting \
                                                    || die "training $JOB_INPAINT failed"; STAGE=val_inpaint ;;
            val_inpaint)   say "[5/5] validate $JOB_INPAINT"
                           run_val   "$JOB_INPAINT" || die "validation $JOB_INPAINT failed"; STAGE=done ;;
            done)          say "chain complete -- raw vs inpainted, same dataset and hyperparameters"
                           say "  raw      $JOB_RAW     -> outputs/validation/${JOB_RAW}__$(basename "$VAL_DS")"
                           say "  inpaint  $JOB_INPAINT -> outputs/validation/${JOB_INPAINT}__$(basename "$VAL_DS")"
                           save; exit 0 ;;
            *)             die "unknown stage '$STAGE'" ;;
        esac
        save
    done
}

# --- commands ---------------------------------------------------------------

cmd_start() {  # cmd_start [dataset] [run-name] [dataset-root]
    [[ -n $(sup_pid) ]] && { say "already running (use 'status' or 'logs')"; exit 0; }
    if [[ ! -f $STATE ]]; then
        [[ -n ${1:-} ]] || die "usage: $0 start <dataset> [run-name] [dataset-root]  (e.g. $0 start user/my_dataset gear_v1)"
        configure_dataset "$1" "${3:-}"
        local ts name run; ts=$(date +%Y%m%d_%H%M%S); name=${DATASET##*/}
        run=${2:-groot_${name}_$ts}
        STAGE=train_raw
        JOB_RAW=${run}_raw
        JOB_INPAINT=${run}_inpaint
        save
        say "new chain '$run' on $DATASET (config $CONFIG)"
        say "  raw     -> $JOB_RAW"
        say "  inpaint -> $JOB_INPAINT"
    else
        load
        [[ -n ${JOB_RAW:-} ]] || die "state file $STATE predates this script; run '$0 stop' to discard it"
        [[ -n ${1:-} ]] && die "a chain on $DATASET is already staged at '$STAGE'; run '$0 stop' first to switch dataset"
        say "continuing $DATASET from stage '$STAGE'"
    fi
    say "batch=$BATCH_SIZE x $NUM_GPUS (effective $((BATCH_SIZE*NUM_GPUS))), workers=$NUM_WORKERS/gpu, steps=$STEPS, save_freq=$SAVE_FREQ, wandb=$WANDB"
    nohup setsid "$0" __run >> "$LOG" 2>&1 &
    sleep 2
    say "started. following $LOG -- Ctrl-C detaches, training keeps going."
    echo
    tail -f "$LOG"
}

cmd_pause() {
    local pids; pids=$(work_pids)
    [[ -z $pids && -z $(sup_pid) ]] && { say "nothing running"; exit 0; }
    say "pausing: stopping the supervisor, then the GPU work"
    local sp; sp=$(sup_pid); [[ -n $sp ]] && kill -TERM "$sp" 2>/dev/null
    [[ -n $pids ]] && kill -TERM $pids 2>/dev/null
    for _ in $(seq 1 20); do [[ -z $(work_pids) ]] && break; sleep 1; done
    pids=$(work_pids); [[ -n $pids ]] && { say "forcing"; kill -KILL $pids 2>/dev/null; sleep 2; }
    pkill -f "chain\.sh __run" 2>/dev/null
    load
    say "paused at stage '$STAGE'. GPU memory:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
    local ckpt job="${JOB_RAW:-}"
    [[ $STAGE == train_inpaint || $STAGE == val_inpaint ]] && job="${JOB_INPAINT:-}"
    ckpt=$(latest_ckpt "$job")
    [[ -n $ckpt ]] && say "will resume from $(basename "$ckpt")"
    say "resume with: ./scripts/chain.sh resume"
}

cmd_status() {
    load
    echo "dataset:    ${DATASET:-<not started>}${DATASET_ROOT:+  ($DATASET_ROOT)}"
    echo "stage:      ${STAGE:-<not started>}    (raw=${JOB_RAW:-} inpaint=${JOB_INPAINT:-})"
    if [[ -n $(sup_pid) ]]; then echo "supervisor: RUNNING (pid $(sup_pid))"; else echo "supervisor: stopped"; fi
    local n; n=$(work_pids | wc -l); echo "gpu work:   $n process(es)"
    echo "progress:   $(grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' "$LOG" 2>/dev/null | tail -1 || echo n/a)"
    for j in "${JOB_RAW:-}" "${JOB_INPAINT:-}"; do
        [[ -z $j ]] && continue
        local c; c=$(latest_ckpt "$j"); [[ -n $c ]] && echo "checkpoint: $j -> $(basename "$c")"
    done
    [[ -n ${CACHE:-} ]] && echo "cache:      $(cached_frames) / $(cache_target) frames"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/gpu:        /'
    df -h "$(readlink -f outputs)" | tail -1 | awk '{print "disk:       "$4" free"}'
}

cmd_stop() {
    say "stopping chain and discarding its state"
    local sp; sp=$(sup_pid); [[ -n $sp ]] && kill -TERM "$sp" 2>/dev/null
    local pids; pids=$(work_pids); [[ -n $pids ]] && kill -TERM $pids 2>/dev/null
    sleep 5
    pids=$(work_pids); [[ -n $pids ]] && kill -KILL $pids 2>/dev/null
    pkill -f "chain\.sh __run" 2>/dev/null
    [[ -f $STATE ]] && mv "$STATE" "$STATE.abandoned"
    say "stopped. checkpoints kept; state moved to $STATE.abandoned"
}

case "${1:-}" in
    start)  shift; cmd_start "$@" ;;
    resume) cmd_start ;;                       # same path: continue from saved stage
    pause)  cmd_pause ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    logs)   tail -f "$LOG" ;;
    __run)  __run ;;
    *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
