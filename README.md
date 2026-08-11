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
│   ├── object_pose.py  # 6-DoF object pose + overlay; pipeline sourced from v2d
│   ├── train.py        # lerobot-train wrapper that injects the above
│   ├── policies/       # pi0/groot subclasses with aux losses (custom --policy.type)
│   └── losses/         # auxiliary loss functions
├── scripts/
│   ├── ft_pi0.sh, ft_groot.sh   # thin wrappers over lerobot-train
│   ├── chain.sh                 # raw vs inpainted ablation, pausable
│   ├── viz_inpainting.py        # inspect methods / warm the cache
│   ├── eval_openloop.py         # open-loop action-prediction validation
│   ├── attention_overlay.py     # where the policy looked, painted on the frames
│   └── convert_v21_to_v30_local.py
├── configs/            # per-dataset config (<name>.env: gear_left, kitting);
│                       # local.env (untracked) holds machine paths
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

**Training modes so far — two.** Both are trained and validated by `scripts/chain.sh` on
the same dataset with identical hyperparameters, so the pair isolates the manipulation:

| mode | frames the policy sees |
|---|---|
| `raw` | the recorded frames, untouched — the baseline |
| `inpaint` | the demonstrator's arm/hand inpainted out of every frame (`arm_inpaint`) |

Crossing that is a second, independent axis — **what the policy predicts**. It is set per
dataset in `configs/<name>.env`, not by `chain.sh`, so a run name records it:

| action space | policy target |
|---|---|
| relative (`RELATIVE_ACTIONS=1`) | `action - current_state`, grippers kept absolute, angle-aware at the ±π seam |
| absolute (the setting absent) | `action` verbatim, the pose the dataset records |

`ft_groot.sh` tests `RELATIVE_ACTIONS` for *presence*, not truth, so `RELATIVE_ACTIONS=0`
still turns relative actions **on** — absolute means commenting the line out entirely
(along with `RELATIVE_ANGLE_DIMS`, which only has meaning for the relative conversion).

A third frame-level signal, **object grounding**, is still being explored and is not part
of the training pipeline yet — see below.

**Datasets.** Two are configured, both bimanual human demos from `h2r_collection`, both
14-dim (both wrists, then both grippers) and converted to LeRobot v3.0:

| dataset | episodes / frames | cameras | language |
|---|---|---|---|
| `gear_left` | 72 / 6704 @ 30 fps | head, wrist_left, wrist_right | none |
| `kitting` | 141 / 20613 @ 30 fps | head, wrist_left, wrist_right | "pick the black object and place it inside the box" |

`configs/kitting.env` records a measured data-quality caveat: its right-wrist track is
noticeably jumpier than the left (446 frame-to-frame jumps > 5 cm across 56 episodes, vs
23 across 16 on the left, single frames moving up to 0.43 m). At 30 fps those are tracking
glitches rather than motion. They are left in deliberately — the dataset is read-only —
so expect the right arm's relative-action statistics to be wider than the left's.

**Runs so far** (GR00T N1.7, `kitting`, batch 64×2). None is a finished experiment; they
exist to shake out the pipeline, and their checkpoints are archived rather than live:

| run | frames | actions | reached |
|---|---|---|---|
| `kitting_v1_raw` | raw | relative | 17k steps |
| `kitting_v1_inpaint` | arm-inpainted | relative | 20k steps |
| `kitting_v1abs_raw` | raw | absolute | 16k steps, stopped |

Step rates for budgeting a run: ~4.7–5.6 s/step raw, ~6.6 s/step arm-inpainted.
The comparison the ablation is *for* — matched steps, matched episode sets — has not been
run yet, so no accuracy numbers are quoted here on purpose.

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

# ...or over one whole episode, whose frame range is looked up in the metadata
# (--start-index is then an offset within the episode):
uv run python scripts/viz_inpainting.py \
    --dataset <hf-repo-id> --dataset-root <local dataset dir> \
    --methods arm_inpaint --cameras observation.images.head \
    --video --episode 3 --num-frames -1 --fps 12

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

# Stop cleanly after one stage instead of running all five, and don't queue behind
# the GPU-free check (the memory in use is yours):
STOP_AFTER=val_raw ./scripts/chain.sh start <hf-repo-id> <run-name> --no-gpu-wait
```

By default a GPU stage waits until every GPU it will use is ≥ `GPU_FREE_PCT` (80%) free,
so a chain can be queued behind someone else's job; `--no-gpu-wait` starts immediately.
`STOP_AFTER=<stage>` saves state and exits after that stage — worth using when the inpaint
cache has been cleared, since the `warm` stage would otherwise silently spend hours
rebuilding it. A later `resume` picks up from where it stopped either way.

`<run-name>` names the pair: the two policies come out as `<run-name>_raw` and
`<run-name>_inpaint`, with matching output dirs and validation reports. The dataset's
basename selects its `configs/<name>.env` (modalities, relative actions, rename map) and
its inpaint cache; a third argument overrides the dataset root. All of it is recorded in
the chain's state file, so a resume can never drift onto a different dataset. Pausing costs at most `SAVE_FREQ` steps
(1000, ~85 min at ~5 s/step): training exits, GPU memory is released, and the resume
restarts from the last complete checkpoint. LeRobot never prunes and each checkpoint is
~25 GB, so a pruner keeps the newest `KEEP_CKPTS` (2) — the previous one survives the next
save, so a checkpoint corrupted mid-write still leaves a resumable predecessor. A disk
check at step 0 demands room for `KEEP_CKPTS + 1`, since a save briefly holds one extra.

> **Host RAM, not GPU.** LeRobot's default `prefetch_factor=4` parked ~43 GB of decoded
> video in shared memory on `kitting` and left the host with ~4 GB. Training survives that;
> the *first checkpoint save* does not — rank 0 serializing ~12.5 GB under that pressure
> crawls past rank 1's NCCL collective timeout and the job aborts. `PREFETCH_FACTOR=2` (the
> default here) is the knob that fixes it; lowering `NUM_WORKERS` does not.

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

**Attention overlay** (`scripts/attention_overlay.py`). The same open-loop replay, with
GR00T's action→image cross-attention painted over each camera view as a video — a direct
read on whether removing the demonstrator's arm moves the policy's gaze onto the object.

```bash
uv run python scripts/attention_overlay.py \
    --checkpoint outputs/<run>/checkpoints/020000/pretrained_model \
    --dataset <local dataset dir> --episodes 0 1 2 \
    --out outputs/validation/<run>__attn
```

GR00T's `AlternateVLDiT` interleaves its 32 blocks so image cross-attention lives at blocks
2, 6, …, 30. The model runs SDPA, which never materialises attention probabilities, so each
of those blocks is recomputed from its own `to_q`/`to_k`, averaged over heads and chunk
queries, and accumulated across every denoising step. Alongside the video it writes
`attention_summary.json` with per-camera attention share and open-loop MAE.

Read it as allocation, not causation: a bright patch is where the model attended, not proof
it used that information. The heatmap is genuinely 8×8 tokens per 256×256 input (Qwen3VL
merges patches), bilinearly upscaled — coarse regions, no pixel-level precision. Averaging
over 8 layers also hides disagreement between them, which `--layers` isolates. So far the
share splits roughly head ≈ 0.40 / wrist_left ≈ 0.33 / wrist_right ≈ 0.25 across all three
`kitting` runs.

**Object grounding (exploratory, not wired into training).** The third candidate signal is
where the manipulated object is, per frame, as a 6-DoF pose. `src/h2r_il/object_pose.py`
produces it, but **does not implement it**: the pipeline is sourced from the
[video_to_data](https://github.com/nvidia-isaac/video_to_data) *reconstruction modules*
(`v2d.pipelines.run_video_object_tracking` — SAM2 masks → MoGe depth → SAM3D mesh → metric
scale → FoundationPose tracking → mesh-overlay renders). No CAD model and no depth sensor:
the mesh comes from SAM3D and the scale from MoGe.

v2d is not vendored here. It stays a separate checkout (`V2D_ROOT`, see LOCAL_SETUP.md)
whose host packages are installed editable into this env, so only orchestration runs here
and every model runs in v2d's own docker images. This module adds what the raw pipeline
does not give you: run identity, one trajectory array, and a reviewable overlay.

Two outputs per clip — `trajectory.npz` (object→camera 4×4 per frame, plus position,
quaternion and rpy, in the OpenCV **optical** frame) and `overlay.mp4` (the mesh renders
with the pose axes and a numeric readout drawn on).

```bash
uv run --no-sync python -m h2r_il.object_pose --video clip.mp4
```

The first run on a clip has no prompts JSON yet, so it opens v2d's SAM2 annotation UI
(box the object on a frame where it is unoccluded), lists what you drew and asks before
spending GPU time. Later runs reuse `<video_dir>/<stem>_prompts.json` and go straight to
tracking; `--annotate` reopens the UI to edit them.

Whether monocular 6-DoF is *accurate enough to train on* is still the open question —
depth, mesh and metric scale are all estimated from a single view, so check `overlay.mp4`
before believing a trajectory. If it lands, it feeds Phase 2 as an object-motion auxiliary
loss rather than as another frame manipulation. Nothing here is on the training path today.

### Phase 2 — auxiliary losses (next)

- Policy subclasses registered as custom policy types (e.g. `pi0_h2r`, `groot_h2r`) that add aux loss terms to the training objective
- Loss weights and toggles switchable from config, so baselines and variants share one script

### Phase 3 — experiments

- Real fine-tuning runs on the annotated human-demo dataset
- Ablations: raw vs inpainted frames, relative vs absolute actions, aux losses on/off,
  per-model (pi0 vs GR00T N1.7) — matched steps and matched validation episodes, which the
  pipeline-shakeout runs above deliberately are not
- Evaluation on the target benchmark/robot

## Open questions

- Inpainting spec: how much to remove (hand/arm? forearm? shadows?) and per-camera behaviour
- Action space: relative deltas or absolute poses — both train, neither is decided
- Object grounding: whether monocular 6-DoF object pose is accurate enough to train on
- Annotation format and how annotations are stored/loaded alongside the LeRobot dataset
- Aux loss definitions and where they attach in each model
- Evaluation benchmark / robot setup

## Requirements

Full fine-tuning of pi0 and GR00T-N1.7-3B without LoRA needs large-memory GPUs (see the
memory figures under Phase 0). Dataset paths and any other machine-specific settings go in
an untracked `configs/local.env`, which `configs/<dataset>.env` and `scripts/chain.sh`
source automatically — nothing environment-specific belongs in a tracked file.
