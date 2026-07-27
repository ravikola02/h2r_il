#!/usr/bin/env bash
# Pausable training chain: train -> validate -> warm cache -> train -> validate.
#
#   ./scripts/chain.sh start     start (or continue) the chain, then follow the log
#   ./scripts/chain.sh status    where it is: stage, step, ETA, GPU, disk
#   ./scripts/chain.sh logs      follow the live output (Ctrl-C is safe)
#   ./scripts/chain.sh pause     stop training and FREE THE GPUS; fully resumable
#   ./scripts/chain.sh resume    pick up from the last checkpoint
#   ./scripts/chain.sh stop      abandon the chain (no resume)
#
# Pause really stops training -- the processes exit and GPU memory is released.
# Training restarts from the last checkpoint, so pausing costs at most SAVE_FREQ
# steps of work (see the note on SAVE_FREQ below). Ctrl-C while following the log
# only detaches your terminal; it never touches training.
#
# GPU stages wait for the GPUs to be mostly idle before launching, so you can
# start the chain while someone else's job is still finishing (see GPU_FREE_PCT).
#
# Knobs (override on the command line, e.g. `WANDB=true ./scripts/chain.sh start`):
WANDB=${WANDB:-false}          # true -> live loss curves; you are already logged in
BATCH_SIZE=${BATCH_SIZE:-64}   # per GPU; effective batch = BATCH_SIZE x NUM_GPUS = 128
NUM_GPUS=${NUM_GPUS:-2}
# DataLoader workers per GPU process. LeRobot's default of 4 starves the GPUs
# (three 720x1280 video streams decoded per sample): ~15 samples/s. 16 hit ~32
# samples/s but filled shm with decoded video -- throughput collapsed to ~4 by
# step 500 with 45 GB shared and one GPU idling. 8 keeps the win, halves the
# footprint. Total loader processes = NUM_WORKERS x NUM_GPUS.
NUM_WORKERS=${NUM_WORKERS:-8}
STEPS=${STEPS:-20000}
# Checkpoint cadence, and therefore how much a pause can cost: at ~12 s/step,
# 250 steps is ~50 min of work at risk. Lower = cheaper pause, more 24 GB writes.
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

VAL_DS=/mnt/shared_data/datasets/h2r_val/lerobot_datasets/h2r_val/pick_cube_eef_optical
CACHE=/mnt/shared_data/h2r_il/visual_cache/gear_left
STATE_DIR=outputs/chain
STATE=$STATE_DIR/state.env
LOG=$STATE_DIR/chain.log
CKPT_DIR=$(printf "%06d" "$STEPS")
STAGES=(train1 val1 warm train2 val2)

mkdir -p "$STATE_DIR"
say()  { echo "[chain $(date '+%F %T')] $*"; }
die()  { say "ABORT: $*"; exit 1; }
load() { [[ -f $STATE ]] && source "$STATE"; }
save() { cat > "$STATE" <<EOF
STAGE=$STAGE
JOB1=$JOB1
JOB2=$JOB2
EOF
}

# --- helpers ----------------------------------------------------------------

# PIDs of anything this chain runs on the GPUs (not the supervisor itself).
work_pids() { pgrep -f "h2r_il\.train|accelerate launch|viz_visual_methods|eval_openloop" 2>/dev/null; }
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
# silently drop the slowdown / angle fix.
export_hooks() {  # export_hooks [with_visual]
    set -a; source configs/gear_left.env; set +a
    [[ -n "${ACTION_SLOWDOWN:-}"    ]] && export H2R_ACTION_SLOWDOWN="$ACTION_SLOWDOWN"
    [[ -n "${RELATIVE_ANGLE_DIMS:-}" ]] && export H2R_RELATIVE_ANGLE_DIMS="$RELATIVE_ANGLE_DIMS"
    if [[ "${1:-}" == "with_visual" ]]; then
        export H2R_VISUAL_METHODS=arm_inpaint
        export H2R_VISUAL_CACHE="$CACHE"
    else
        unset H2R_VISUAL_METHODS H2R_VISUAL_CACHE
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

run_train() {  # run_train <job> [with_visual]
    local job=$1 visual=${2:-}
    check_disk    || return 1
    wait_for_gpus || return 1
    export_hooks "$visual"
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
        uv run accelerate launch --multi_gpu --num_processes="$NUM_GPUS" -m h2r_il.train \
            --config_path="$ckpt/pretrained_model/train_config.json" --resume=true \
            --num_workers="$NUM_WORKERS"
    else
        say "starting $job from scratch"
        CONFIG=configs/gear_left.env NUM_GPUS=$NUM_GPUS BATCH_SIZE=$BATCH_SIZE \
        NUM_WORKERS=$NUM_WORKERS WANDB=$WANDB JOB_NAME="$job" \
        VISUAL_METHODS=${H2R_VISUAL_METHODS:-} VISUAL_CACHE=${H2R_VISUAL_CACHE:-} \
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
    # --slowdown 1: the policy as-is against the recorded data as-is.
    uv run python scripts/eval_openloop.py \
        --checkpoint "$ckpt" --dataset "$VAL_DS" --episodes $EPISODES \
        --slowdown 1 --out "outputs/validation/${job}__pick_cube"
}

run_warm() {
    wait_for_gpus || return 1
    export H2R_VISUAL_CACHE="$CACHE"
    local have; have=$(find "$CACHE/arm_inpaint" -name '*.png' 2>/dev/null | wc -l)
    say "warming arm-inpaint cache (have $have frames; content-addressed, resumable)"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$i uv run python scripts/viz_visual_methods.py \
            --dataset ravioli02/gear_left \
            --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
            --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
            --num-shards "$NUM_GPUS" --shard $i &
    done
    wait
}

# --- supervisor (internal; runs detached) -----------------------------------

__run() {
    load
    trap 'stop_pruner' EXIT
    while :; do
        case $STAGE in
            train1) say "[1/5] train $JOB1 (relative EEF + interpolation)"
                    run_train "$JOB1"                 || die "training $JOB1 failed"; STAGE=val1  ;;
            val1)   say "[2/5] validate $JOB1"
                    run_val   "$JOB1"                 || die "validation $JOB1 failed"; STAGE=warm ;;
            warm)   say "[3/5] warm arm-inpaint cache"
                    run_warm                          || die "cache warm failed";      STAGE=train2;;
            train2) say "[4/5] train $JOB2 (visual methods + interpolation)"
                    run_train "$JOB2" with_visual     || die "training $JOB2 failed"; STAGE=val2  ;;
            val2)   say "[5/5] validate $JOB2"
                    run_val   "$JOB2"                 || die "validation $JOB2 failed"; STAGE=done ;;
            done)   say "chain complete"
                    say "  $JOB1 -> outputs/validation/${JOB1}__pick_cube"
                    say "  $JOB2 -> outputs/validation/${JOB2}__pick_cube"; save; exit 0 ;;
            *)      die "unknown stage '$STAGE'" ;;
        esac
        save
    done
}

# --- commands ---------------------------------------------------------------

cmd_start() {
    [[ -n $(sup_pid) ]] && { say "already running (use 'status' or 'logs')"; exit 0; }
    if [[ ! -f $STATE ]]; then
        local ts; ts=$(date +%Y%m%d_%H%M%S)
        STAGE=train1; JOB1=groot_relinterp_$ts; JOB2=groot_visinterp_$ts; save
        say "new chain: job1=$JOB1 job2=$JOB2"
    else
        load; say "continuing from stage '$STAGE'"
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
    local ckpt; ckpt=$(latest_ckpt "${JOB1:-}")
    [[ $STAGE == train2 ]] && ckpt=$(latest_ckpt "${JOB2:-}")
    [[ -n $ckpt ]] && say "will resume from $(basename "$ckpt")"
    say "resume with: ./scripts/chain.sh resume"
}

cmd_status() {
    load
    echo "stage:      ${STAGE:-<not started>}    (${JOB1:-} / ${JOB2:-})"
    if [[ -n $(sup_pid) ]]; then echo "supervisor: RUNNING (pid $(sup_pid))"; else echo "supervisor: stopped"; fi
    local n; n=$(work_pids | wc -l); echo "gpu work:   $n process(es)"
    echo "progress:   $(grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' "$LOG" 2>/dev/null | tail -1 || echo n/a)"
    for j in "${JOB1:-}" "${JOB2:-}"; do
        [[ -z $j ]] && continue
        local c; c=$(latest_ckpt "$j"); [[ -n $c ]] && echo "checkpoint: $j -> $(basename "$c")"
    done
    echo "cache:      $(find "$CACHE/arm_inpaint" -name '*.png' 2>/dev/null | wc -l) / ~20112 frames"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/gpu:        /'
    df -h /mnt/shared_data | tail -1 | awk '{print "disk:       "$4" free"}'
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
    start)  cmd_start ;;
    resume) cmd_start ;;                       # same path: continue from saved stage
    pause)  cmd_pause ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    logs)   tail -f "$LOG" ;;
    __run)  __run ;;
    *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
