"""Train entrypoint that injects h2r_il methods into LeRobot training.

This is a thin wrapper over ``lerobot-train`` — we never fork LeRobot. LeRobot's
``--dataset.image_transforms`` config only accepts torchvision-v2 augmentations,
so a custom method class can't be named there. Instead we hook LeRobot's own
extension point, ``LeRobotDataset.set_image_transforms``: right after LeRobot
builds its datasets we compose our :class:`InpaintingPipeline` onto each
dataset's transform. The raw frame flows through our methods first (deterministic,
cache-backed), then any LeRobot photometric augmentation runs on top. The dataset
on disk is never modified.

Which methods run is read from the ``H2R_INPAINTING`` env var — a spec passed
to :func:`h2r_il.inpainting.build_pipeline` (a method name, or JSON). With it
unset, this behaves exactly like ``lerobot-train``. The cache dir comes from
``H2R_INPAINTING_CACHE``. Every other CLI arg is LeRobot's and is passed through.

    H2R_INPAINTING=arm_inpaint uv run python -m h2r_il.train \\
        --dataset.repo_id=... --policy.type=groot ...

``H2R_RELATIVE_ANGLE_DIMS`` makes LeRobot's relative-action conversion angle-aware
(see :mod:`h2r_il.relative_angles`); set it whenever ``--policy.use_relative_actions``
is on and the action space contains Euler angles.
"""

from __future__ import annotations

import logging
import os

import lerobot.datasets.factory as _factory
import lerobot.scripts.lerobot_train as _lr_train

from h2r_il.relative_angles import install_from_env as _install_relative_angles
from h2r_il.inpainting import build_pipeline


def _install_inpainting() -> None:
    spec = os.environ.get("H2R_INPAINTING", "").strip()
    if not spec:
        return  # no methods requested -> plain lerobot-train behaviour

    cache_dir = os.environ.get("H2R_INPAINTING_CACHE") or None
    orig = _factory.make_train_eval_datasets

    def _patched(cfg):
        train_ds, eval_ds = orig(cfg)
        for ds in (train_ds, eval_ds):
            if ds is None:
                continue
            # existing photometric augmentation (or None) becomes the pipeline tail
            tail = getattr(ds, "image_transforms", None)
            pipeline = build_pipeline(spec, tail=tail, cache_dir=cache_dir)
            ds.set_image_transforms(pipeline)
        logging.info("h2r_il inpainting methods active: %s", spec)
        return train_ds, eval_ds

    # train.py binds the name locally (`from ...factory import make_train_eval_datasets`),
    # so patch that binding; patch the factory module too for any other caller.
    _factory.make_train_eval_datasets = _patched
    _lr_train.make_train_eval_datasets = _patched


def main() -> None:
    _install_inpainting()
    _install_relative_angles()
    _lr_train.train()  # @parser.wrap(): parses sys.argv, runs LeRobot training


if __name__ == "__main__":
    main()
