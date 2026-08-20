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
│   ├── object_pose.py  # 6-DoF object pose for one clip; pipeline sourced from v2d
│   ├── object_pose_dataset.py  # dataset-scale: alignment, tracking, pose store
│   ├── object_pose_inject.py   # attach the pose target to training samples
│   ├── v2d_ext/        # modules bind-mounted into v2d's docker images
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

**Training modes so far — two.** Both are trained and validated by `scripts/chain.sh` on
the same dataset with identical hyperparameters, so the pair isolates the manipulation:

| mode | frames the policy sees |
|---|---|
| `raw` | the recorded frames, untouched — the baseline |
| `inpaint` | the demonstrator's arm/hand inpainted out of every frame (`arm_inpaint`) |

A third signal, **object grounding**, is now produced and attachable to training samples (see
below). It is not a fourth row in that table: it adds a per-frame *target*, not a change to the
frames, so it composes with either mode. The aux loss that consumes it is Phase 2.

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

**Object grounding — where the object is, per frame, as a 6-DoF pose.** The third signal, and
unlike the other two it is a *target* rather than a frame manipulation. All the perception runs
up front and training only reads a lookup table, so it costs nothing at dataloader time.

`src/h2r_il/object_pose.py` produces the pose for one clip but **does not implement it**: the
pipeline is sourced from the [video_to_data](https://github.com/nvidia-isaac/video_to_data)
reconstruction modules (SAM2 masks → MoGe depth → SAM3D mesh → metric scale → FoundationPose
tracking). No CAD model and no depth sensor. v2d is not vendored; it stays a separate checkout
(`V2D_ROOT`) whose models all run in its own docker images.

**One video, not one clip per episode.** A LeRobot v3 camera concatenates its episodes, in order,
into one mp4 — so that file's frame index *is* the dataset's global frame index. Tracking the
whole file once therefore gives one annotation pass, one SAM3D mesh and one metric-scale solve
for the entire dataset, which is what makes poses comparable *across* episodes; a per-episode
sweep re-estimates focal length per clip and lets absolute scale drift. `object_pose_dataset
prepare` proves that identity (frame count against `total_frames`, and every episode's start
timestamp against the running sum of lengths) and refuses to proceed if it is off by a frame.

**Re-registration is the part that matters.** v2d's tracking step registers once and tracks
everything after it, which is wrong here: at each episode cut the pose prior belongs to a
different take. Left alone it lost the object just after the first cut and held a pose 12x too
close for 98.7% of the dataset — while looking perfectly smooth, so step-continuity metrics
could not see it. `object_pose_dataset track` re-registers at every episode boundary *and* every
N frames within an episode (default 15). Validation uses MoGe depth inside the SAM2 mask, which
is independent of FoundationPose; smoothness is not a quality metric.

Result on kitting's head camera: 20613/20613 frames, 141/141 episodes, **0.48 cm median depth
error**, zero degenerate frames.

```bash
# what it would do, and roughly how long, without doing it
uv run --no-sync python -m h2r_il.object_pose_dataset build \
    --dataset-root <dataset> --dry-run

# build the lookup table
uv run --no-sync python -m h2r_il.object_pose_dataset build --dataset-root <dataset>
```

`build` runs five stages — `prepare` (verify alignment, stage the video),
`reconstruct` (v2d's stock pipeline: crop, masks, depth, mesh, metric scale),
`track` (re-registration), `reduce` (poses → trajectory + overlay), `store` (place on
the global frame axis) — as separate processes, skipping any whose output is already on
disk and current. So it is the resume command as well as the start command: after a
crash, or after re-running one stage by hand with different arguments, calling it again
does exactly the outstanding work. Staleness is decided by mtime against the stage's own
input, not by a state file that could disagree with the disk.

It stops once, at the one genuinely manual step — tagging the object in v2d's SAM2 UI —
printing the command to run and exiting **2**. Re-run `build` afterwards and it continues
from there.

Budget on kitting's head camera: **~7 h** of GPU for a cold build (`--dry-run` scales the
estimate to your dataset's frame count). Roughly 1.7 h of that is waste — v2d's
`run_pipeline` is monolithic, so `reconstruct` also runs a register-once FoundationPose
pass whose poses `track` then replaces, and there is no stage selector to skip it. Paid
once per dataset+camera; every training run afterwards just reads the 2 MB table.

The individual subcommands (`prepare`, `track`, `store`) are what `build` shells out to
and remain available for exactly that re-run-one-stage case.

`track --fix-rotation` (also `build --fix-rotation`) forces rotation to identity during
tracking. It is measured **worse** — translation error 2–3× on every episode tested,
because denying FoundationPose rotation makes it slide translation to maximise mesh/depth
overlap. Kept because the measurement is worth being able to repeat at full-dataset
scale; both commands warn when it is on.

**Injection into training** (`src/h2r_il/object_pose_inject.py`). Every LeRobot sample already
carries its global frame index, and the store is dense on that axis, so attaching the target is
one dict assignment per sample — no decoding, no I/O, ~2 MB resident. The dataset on disk is
untouched and LeRobot is not forked.

The target is attached as **`observation.object_pose`**, and the prefix is not cosmetic.
`lerobot_train` runs `batch = preprocessor(batch)` between the dataloader and the policy, and
that pipeline's dict→transition converter keeps only `observation.`-prefixed keys plus a fixed
whitelist — everything else is dropped silently. A bare `object_pose` key survives collation and
looks correct in any dataloader-level test, then is simply gone by the time `forward` runs.
Under the prefix it reaches the policy, and the normalizer passes it through untouched because
it has no feature spec for it.

```bash
# Fine-tune with the object-pose target attached (works with ft_pi0.sh too):
CONFIG=configs/kitting.env OBJECT_POSE=outputs/object_pose/kitting/head/store \
    ./scripts/ft_groot.sh
```

`OBJECT_POSE_TARGET` is `position` (default, xyz) or `pose6d` (xyz + wxyz quaternion). Prefer
position: rotation comes from an arbitrary SAM3D body frame and nothing here validates it — MoGe
gives depth, not orientation. Every sample also carries `object_pose_valid`, and **the loss must
mask on it**, or frames without a pose train the model towards a pose of all zeros.

### Phase 2 — auxiliary losses

**Object-pose prediction heads** (`src/h2r_il/policies/`, `src/h2r_il/losses/object_pose.py`).
`--policy.type=h2r_pi0` and `h2r_groot` are pi0 and GR00T with a head that regresses the object's
position from the backbone's own features. They register through LeRobot's third-party plugin
path (`_get_policy_cls_from_policy_name`), which resolves the policy and processor classes from
naming conventions — so no fork and no patched factory, but the config class name, module name
and registered type have to stay in step.

*Where the head taps.* pi0: `PI0Pytorch.embed_prefix`, the concatenated image + language
embeddings the action expert cross-attends to, captured by wrapping the bound method on that one
instance. GR00T: a forward hook on `model.backbone`, whose `backbone_features` are the same
thing one stage earlier than the action head's private re-encoding. Both give `(B, T, D)` plus a
padding mask; the head masked-mean-pools and runs a small MLP. Gradients are **not** detached, so
the aux loss shapes the shared trunk — that is the entire point. `--policy.object_pose_detach`
turns it into a passive probe that measures whether position is already encoded without changing
anything, which is a diagnostic, not a training signal.

*Two things the loss has to get right.* It **masks on `object_pose_valid`** — frames without a
pose carry zeros, and zero sits in the middle of the coordinate range, so averaging over them
actively pulls the prediction towards the origin. And it **standardises the target** using
statistics from the store: raw positions are metres, with a variance of 0.017 on kitting, so an
unnormalised aux term is ~50× smaller than the flow-matching loss and `object_pose_weight` would
be silently doing unit conversion instead of expressing a preference. Reported metrics are in
centimetres, comparable with the 0.48 cm the tracker itself was measured at.

```bash
# GR00T + object-pose head. OBJECT_POSE both attaches the target and switches the
# policy type; OBJECT_POSE_WEIGHT tunes it, OBJECT_POSE_HEAD=0 opts out of the head.
CONFIG=configs/kitting.env OBJECT_POSE=outputs/object_pose/kitting/head/store \
    OBJECT_POSE_WEIGHT=1.0 ./scripts/ft_groot.sh
```

pi0 needs one extra step. It loads weights through `--policy.path`, and LeRobot takes the policy
class from that checkpoint's `config.json` (it pops `type` before applying CLI overrides), so
`--policy.type` cannot switch it. Re-stage the checkpoint once — the h2r configs are supersets,
so an existing config parses unchanged and only the head starts fresh:

```bash
python scripts/stage_h2r_policy.py --checkpoint <ckpt>/pretrained_model \
    --type h2r_pi0 --out outputs/staged/<name>     # hard-links weights; ~28 KB, not 22 GB
CONFIG=configs/kitting.env PRETRAINED=outputs/staged/<name> \
    OBJECT_POSE=outputs/object_pose/kitting/head/store ./scripts/ft_pi0.sh
```

`scripts/test_object_pose_head.py` covers the parts that fail quietly — target survival through
the processor pipeline, masking, the all-invalid batch, gradient reaching the trunk, and the
standardisation — without needing a GPU or model weights.

Still open: no training run has used these yet, so the heads are verified structurally rather
than empirically.

### Phase 3 — experiments

- Real fine-tuning runs on the annotated human-demo dataset
- Ablations: raw vs inpainted frames, aux losses on/off, per-model (pi0 vs GR00T N1.7)
- Evaluation on the target benchmark/robot

## Open questions

- Inpainting spec: how much to remove (hand/arm? forearm? shadows?) and per-camera behaviour
- Object grounding: **translation** is measured (0.48 cm median depth error vs MoGe on kitting's
  head camera), but **rotation is unvalidated** — MoGe gives depth, not orientation. Which is why
  the default target is position-only. Also open: which cameras (only `head` is tracked; the wrist
  cameras move with the arm, a different problem), and what exactly the loss regresses — absolute
  position, per-step object motion, or a projected 2D keypoint. The store keeps raw poses so that
  stays a load-time choice.
- Annotation format and how annotations are stored/loaded alongside the LeRobot dataset
- Aux loss definitions and where they attach in each model
- Evaluation benchmark / robot setup

## Requirements

Full fine-tuning of pi0 and GR00T-N1.7-3B without LoRA needs large-memory GPUs (see the
memory figures under Phase 0). Dataset paths and any other machine-specific settings go in
an untracked `configs/local.env`, which `configs/<dataset>.env` and `scripts/chain.sh`
source automatically — nothing environment-specific belongs in a tracked file.
