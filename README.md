# h2r_il

Imitation learning for robot policies from **annotated human demonstration datasets**. The core idea: layer custom frame-level manipulations (masking-style transforms driven by annotations) and auxiliary loss functions on top of standard VLA fine-tuning, and evaluate with **pi0** and **GR00T N1.7**. Everything is built as a pipeline on top of [LeRobot](https://github.com/huggingface/lerobot).

## Design principles

- **Datasets are read-only.** All frame manipulation (masking etc.) happens on the fly in the training pipeline — nothing is ever written back to the dataset on disk. LeRobot's dataloader-time `--dataset.image_transforms` is the injection point.
- **Build on LeRobot, never fork it.** LeRobot lives here as a git submodule pinned to a known-good commit, installed editable, and treated as a read-only dependency. All custom code (transforms, aux losses, policy variants) lives in this repo's own package and hooks in through LeRobot's extension points: the image-transforms config and policy subclassing via custom `--policy.type` registration.
- **One environment, one entrypoint for both models.** Current LeRobot fine-tunes both through the same `lerobot-train` command: `--policy.type=pi0` ([docs](https://huggingface.co/docs/lerobot/pi0)) and `--policy.type=groot` for GR00T N1.7, base model `nvidia/GR00T-N1.7-3B` ([docs](https://huggingface.co/docs/lerobot/en/groot)). Note: LeRobot dropped GR00T N1.5 — we target N1.7 on current LeRobot.

## Repo layout

```
h2r_il/
├── lerobot/            # git submodule, pinned commit (read-only dependency)
├── src/h2r_il/
│   ├── inpainting/     # annotation-driven frame manipulations (arm inpainting, ...)
│   ├── relative_angles.py  # angle-aware relative-action conversion
│   ├── train.py        # lerobot-train wrapper that injects the above
│   ├── policies/       # pi0/groot subclasses with aux losses (custom --policy.type)
│   └── losses/         # auxiliary loss functions
├── scripts/
│   ├── ft_pi0.sh, ft_groot.sh   # thin wrappers over lerobot-train
│   ├── chain.sh                 # raw vs inpainted ablation, pausable
│   ├── viz_inpainting.py        # inspect methods / warm the cache
│   ├── eval_openloop.py         # open-loop action-prediction validation
│   └── convert_v21_to_v30_local.py
├── configs/            # per-dataset config (<name>.env); local.env holds machine paths
└── pyproject.toml      # uv-managed env; lerobot installed editable with [groot] extra
```

## Roadmap

### Phase 0 — logistics (done)

Wrap the pipeline plumbing so both models train end-to-end in one env:

- [x] uv environment (Python 3.12, torch cu128); `lerobot` submodule pinned to v0.6.0, editable install with `[pi,groot,training]` extras
- [x] Fine-tune wrappers: `scripts/ft_pi0.sh` (from `lerobot/pi0_base` via `--policy.path`) and `scripts/ft_groot.sh` (`--policy.type=groot`, base `nvidia/GR00T-N1.7-3B`, `embodiment_tag=new_embodiment`); `scripts/smoke_test.sh` runs both for a few steps on `lerobot/svla_so101_pickplace`
- [x] Verify checkpoint save/load for both

**Exit criterion (met):** both models complete training steps in the same environment, dataset untouched on disk.

Notes from the smoke runs:

- pi0_base expects openpi camera names (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); datasets with other camera keys need `--rename_map`. LeRobot accepts either-direction subsets between dataset and policy cameras. GR00T with `embodiment_tag=new_embodiment` adapts to any camera set — no rename needed.
- ~35 GB GPU memory per model at batch size 2, bf16 — leaving headroom for real batch sizes on a large-memory card.

### Phase 1 — annotation-driven frame manipulation (current)

**Inpainting framework** (`src/h2r_il/inpainting/`). An *inpainting method* is a cache-backed, per-frame image manipulation. `InpaintingMethod` (base) handles tensor⇄numpy conversion (uint8/float, CHW/`(T,C,H,W)`, RGB), a content-addressed disk cache, and a registry; subclasses implement `apply(rgb_uint8) -> rgb_uint8`. New methods self-register with `@register_inpainting_method("name")` and are immediately usable everywhere.

**Injection into training.** LeRobot's `--dataset.image_transforms` config only accepts torchvision-v2 augmentations, so a custom class can't be named there. Instead we use LeRobot's own extension point — `LeRobotDataset.set_image_transforms` — from a thin wrapper (`h2r_il.train`) that never forks LeRobot: the raw frame flows through our methods first (deterministic, cached), then any photometric augmentation. The dataset on disk is never touched.

```bash
# Fine-tune with arm inpainting active (works with ft_pi0.sh too):
CONFIG=configs/gear_left.env INPAINTING=arm_inpaint ./scripts/ft_groot.sh
```

**Arm inpainting** (`arm_inpaint`, replicating [HumanEgo](https://github.com/TX-Leo/HumanEgo)): Grounding DINO (`"arm. hand."`) → SAM2 mask → dilate → LaMa inpaint, removing the demonstrator's arm/hand while keeping the manipulated object. Grounding DINO + SAM2 come from `transformers`; LaMa is `simple-lama-inpainting`. Models load lazily only on a cache miss.

**Test / visualization tooling** (`scripts/viz_inpainting.py`). Dumps original vs. each method's intermediates and result side by side (to `outputs/`, never into the dataset), for any registered method. Also warms the cache so training-time DataLoader workers just read cached frames.

```bash
# Inspect arm inpainting on a few frames (all cameras):
uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --num-frames 6

# Render an original|result mp4 over a consecutive clip (one per method+camera):
uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --cameras observation.images.head \
    --video --start-index 60 --num-frames 60 --fps 12

# Tune the arm_inpaint knobs from the CLI (see --help for all):
uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --num-frames 6 \
    --box-threshold 0.2 --dilation 25 --prompt "arm. hand."
```

The same knobs carry into training via the `INPAINTING` JSON spec, e.g.
`INPAINTING='[{"name":"arm_inpaint","box_threshold":0.2,"dilation":25}]'`.

```bash
# Warm the whole-dataset cache before a training run (recommended). Inpainting
# runs at ~0.65 s/frame on one GPU; shard across GPUs to divide that time
# (the cache is content-addressed, so sharing it is safe):
CUDA_VISIBLE_DEVICES=0 uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
    --num-shards 2 --shard 0 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
    --num-shards 2 --shard 1 &
```

**The raw vs inpainted ablation** (`scripts/chain.sh`). One command trains the same dataset
twice with identical hyperparameters — once on raw frames, once on arm-inpainted frames —
validating each and warming the inpaint cache in between:

```bash
./scripts/chain.sh start <hf-repo-id> <run-name>       # 5 stages, runs detached
./scripts/chain.sh status                              # stage, step, ETA, cache, GPU, disk
./scripts/chain.sh pause                               # frees the GPUs, fully resumable
./scripts/chain.sh resume                              # continues from the last checkpoint
```

`<run-name>` names the pair: the two policies come out as `<run-name>_raw` and
`<run-name>_inpaint`, with matching output dirs and validation reports. The dataset's
basename selects its `configs/<name>.env` (modalities, relative actions, rename map) and
its inpaint cache; a third argument overrides the dataset root. All of it is recorded in
the chain's state file, so a resume can never drift onto a different dataset. Pausing costs at most `SAVE_FREQ` steps:
training exits, GPU memory is released, and the resume restarts from the last complete
checkpoint. A pruner keeps only the newest checkpoint, since LeRobot never prunes and each
is ~24 GB.

> **Cache before training.** The models are too heavy to run inside DataLoader workers per frame (CUDA-in-forked-worker also breaks). Warm the content-addressed cache with `--fill-cache` first; then training reads cached frames (no model load, no GPU contention). The warm streams frame-by-frame (low RAM) and prints progress/ETA. The dataset stays read-only — the cache lives under `H2R_INPAINTING_CACHE` (default `outputs/inpaint_cache`).

**Relative EEF actions** (`src/h2r_il/relative_angles.py`). LeRobot can train a policy on
`action - current_state` (`--policy.use_relative_actions`), with `relative_exclude_joints`
keeping named channels (grippers) absolute. Its conversion is a plain subtraction, which is
wrong at the ±π seam: a wrist yaw that crosses it turns into a ~2π delta and dominates the
relative-action statistics. `H2R_RELATIVE_ANGLE_DIMS` names the Euler dims so they are
wrapped back into [-π, π] instead. Both knobs are set per dataset in `configs/<name>.env`.

**Open-loop validation** (`scripts/eval_openloop.py`). At sampled start frames the policy
sees the recorded observation and predicts a chunk, scored against the recorded actions —
observations always come from the recording, so errors do not compound the way a rollout's
would. Reports per-dim MAE/RMSE/bias, a debiased MAE (is it just a frame shift?), a
frozen-state baseline any useful policy must beat, and per-episode trajectory plots.

### Phase 2 — auxiliary losses (next)

- Policy subclasses registered as custom policy types (e.g. `pi0_h2r`, `groot_h2r`) that add aux loss terms to the training objective
- Loss weights and toggles switchable from config, so baselines and variants share one script

### Phase 3 — experiments

- Real fine-tuning runs on the annotated human-demo dataset
- Ablations: masking on/off, aux losses on/off, per-model (pi0 vs GR00T N1.7)
- Evaluation on the target benchmark/robot

## Open questions

- Exact masking spec: what gets masked (hand/arm? background?), mask source, per-camera behavior
- Annotation format and how annotations are stored/loaded alongside the LeRobot dataset
- Aux loss definitions and where they attach in each model
- Evaluation benchmark / robot setup

## Requirements

Full fine-tuning of pi0 and GR00T-N1.7-3B without LoRA needs large-memory GPUs (see the
memory figures under Phase 0). Dataset paths and any other machine-specific settings go in
an untracked `configs/local.env`, which `configs/<dataset>.env` and `scripts/chain.sh`
source automatically — nothing environment-specific belongs in a tracked file.
