# h2r_il

Imitation learning for robot policies from **annotated human demonstration datasets**. The core idea: layer custom frame-level manipulations (masking-style transforms driven by annotations) and auxiliary loss functions on top of standard VLA fine-tuning, and evaluate with **pi0** and **GR00T N1.7**. Everything is built as a pipeline on top of [LeRobot](https://github.com/huggingface/lerobot).

## Design principles

- **Datasets are read-only.** All frame manipulation (masking etc.) happens on the fly in the training pipeline — nothing is ever written back to the dataset on disk. LeRobot's dataloader-time `--dataset.image_transforms` is the injection point.
- **Build on LeRobot, never fork it.** LeRobot lives here as a git submodule pinned to a known-good commit, installed editable, and treated as a read-only dependency. All custom code (transforms, aux losses, policy variants) lives in this repo's own package and hooks in through LeRobot's extension points: the image-transforms config and policy subclassing via custom `--policy.type` registration.
- **One environment, one entrypoint for both models.** Current LeRobot fine-tunes both through the same `lerobot-train` command: `--policy.type=pi0` ([docs](https://huggingface.co/docs/lerobot/pi0)) and `--policy.type=groot` for GR00T N1.7, base model `nvidia/GR00T-N1.7-3B` ([docs](https://huggingface.co/docs/lerobot/en/groot)). Note: LeRobot dropped GR00T N1.5 — we target N1.7 on current LeRobot.

## Planned repo layout

```
h2r_il/
├── lerobot/            # git submodule, pinned commit (read-only dependency)
├── src/h2r_il/
│   ├── transforms/     # masking & frame manipulations (annotation-driven)
│   ├── policies/       # pi0/groot subclasses with aux losses (custom --policy.type)
│   └── losses/         # auxiliary loss functions
├── scripts/            # ft_pi0.sh, ft_groot.sh — thin wrappers over lerobot-train
├── configs/            # per-model / per-experiment training configs
└── pyproject.toml      # uv-managed env; lerobot installed editable with [groot] extra
```

## Roadmap

### Phase 0 — logistics (current)

Wrap the pipeline plumbing so both models train end-to-end in one env:

- [x] uv environment (Python 3.12, torch cu128); `lerobot` submodule pinned to v0.6.0, editable install with `[pi,groot,training]` extras
- [x] Fine-tune wrappers: `scripts/ft_pi0.sh` (from `lerobot/pi0_base` via `--policy.path`) and `scripts/ft_groot.sh` (`--policy.type=groot`, base `nvidia/GR00T-N1.7-3B`, `embodiment_tag=new_embodiment`); `scripts/smoke_test.sh` runs both for a few steps on `lerobot/svla_so101_pickplace`
- [x] Verify checkpoint save/load for both

**Exit criterion (met):** both models complete training steps in the same environment, dataset untouched on disk.

Notes from the smoke runs:

- pi0_base expects openpi camera names (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); datasets with other camera keys need `--rename_map`. LeRobot accepts either-direction subsets between dataset and policy cameras. GR00T with `embodiment_tag=new_embodiment` adapts to any camera set — no rename needed.
- ~35 GB GPU memory per model at batch size 2, bf16 — plenty of headroom on 96 GB cards for real batch sizes.

## Storage layout (this machine)

- HF cache (models + datasets) lives on `/mnt/shared_data/hf_cache`, symlinked from `~/.cache/huggingface` (root disk is nearly full).
- `outputs/` is a symlink to `/mnt/shared_data/h2r_il/outputs` (checkpoints are ~20 GB each incl. optimizer state).

### Phase 1 — annotation-driven frame manipulation

**Visual methods framework** (`src/h2r_il/transforms/`). A *visual method* is a cache-backed, per-frame image manipulation. `VisualMethod` (base) handles tensor⇄numpy conversion (uint8/float, CHW/`(T,C,H,W)`, RGB), a content-addressed disk cache, and a registry; subclasses implement `apply(rgb_uint8) -> rgb_uint8`. New methods self-register with `@register_visual_method("name")` and are immediately usable everywhere.

**Injection into training.** LeRobot's `--dataset.image_transforms` config only accepts torchvision-v2 augmentations, so a custom class can't be named there. Instead we use LeRobot's own extension point — `LeRobotDataset.set_image_transforms` — from a thin wrapper (`h2r_il.train`) that never forks LeRobot: the raw frame flows through our methods first (deterministic, cached), then any photometric augmentation. The dataset on disk is never touched.

```bash
# Fine-tune with arm inpainting active (works with ft_pi0.sh too):
CONFIG=configs/gear_left.env VISUAL_METHODS=arm_inpaint ./scripts/ft_groot.sh
```

**Arm inpainting** (`arm_inpaint`, replicating [HumanEgo](https://github.com/TX-Leo/HumanEgo)): Grounding DINO (`"arm. hand."`) → SAM2 mask → dilate → LaMa inpaint, removing the demonstrator's arm/hand while keeping the manipulated object. Grounding DINO + SAM2 come from `transformers`; LaMa is `simple-lama-inpainting`. Models load lazily only on a cache miss.

**Test / visualization tooling** (`scripts/viz_visual_methods.py`). Dumps original vs. each method's intermediates and result side by side (to `outputs/`, never into the dataset), for any registered method. Also warms the cache so training-time DataLoader workers just read cached frames.

```bash
# Inspect arm inpainting on a few frames (all cameras):
uv run python scripts/viz_visual_methods.py \
    --dataset ravioli02/gear_left \
    --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
    --methods arm_inpaint --num-frames 6

# Render an original|result mp4 over a consecutive clip (one per method+camera):
uv run python scripts/viz_visual_methods.py \
    --dataset ravioli02/gear_left \
    --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
    --methods arm_inpaint --cameras observation.images.head \
    --video --start-index 60 --num-frames 60 --fps 12

# Tune the arm_inpaint knobs from the CLI (see --help for all):
uv run python scripts/viz_visual_methods.py \
    --dataset ravioli02/gear_left \
    --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
    --methods arm_inpaint --num-frames 6 \
    --box-threshold 0.2 --dilation 25 --prompt "arm. hand."
```

The same knobs carry into training via the `VISUAL_METHODS` JSON spec, e.g.
`VISUAL_METHODS='[{"name":"arm_inpaint","box_threshold":0.2,"dilation":25}]'`.

```bash
# Warm the whole-dataset cache before a training run (recommended).
# ~20k frames (6704 x 3 cams) at ~0.65 s/frame ~= 4-5 h on one GPU; split
# across both GPUs to ~halve it (content-addressed cache = safe to share):
CUDA_VISIBLE_DEVICES=0 uv run python scripts/viz_visual_methods.py \
    --dataset ravioli02/gear_left --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
    --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
    --num-shards 2 --shard 0 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/viz_visual_methods.py \
    --dataset ravioli02/gear_left --dataset-root /mnt/shared_data/h2r_il/datasets/gear_left \
    --methods arm_inpaint --num-frames -1 --fill-cache --no-panels \
    --num-shards 2 --shard 1 &
```

> **Cache before training.** The models are too heavy to run inside DataLoader workers per frame (CUDA-in-forked-worker also breaks). Warm the content-addressed cache with `--fill-cache` first; then training reads cached frames (no model load, no GPU contention). The warm streams frame-by-frame (low RAM) and prints progress/ETA. The dataset stays read-only — the cache lives under `H2R_VISUAL_CACHE` (default `outputs/visual_cache`).

### Phase 2 — auxiliary losses

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

## Hardware

Dev machine: 2× ~96 GB NVIDIA GPUs — enough for full fine-tuning of both pi0 and GR00T-N1.7-3B without LoRA.
