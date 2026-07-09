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

- [ ] uv environment; add `lerobot` submodule pinned to a commit, editable install with the `[groot]` (+ training) extras
- [ ] Dummy fine-tune wrappers: short smoke-test runs (small `--steps`, small batch) for both `--policy.type=pi0` and `--policy.type=groot` on a small public LeRobot-format dataset (e.g. a LIBERO suite subset)
- [ ] Verify checkpoint save/load for both

**Exit criterion:** both models complete N training steps in the same environment, dataset untouched on disk.

### Phase 1 — annotation-driven frame manipulation

- Custom transform classes that consume per-frame annotations (masks etc.) and apply them at dataloader time via LeRobot's image-transforms pipeline
- Visual verification tooling: dump original vs. transformed frames side by side (to `outputs/`, never into the dataset)

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
