"""Train entrypoint that injects h2r_il methods into LeRobot training.

This is a thin wrapper over ``lerobot-train`` — we never fork LeRobot. LeRobot's
``--dataset.image_transforms`` config only accepts torchvision-v2 augmentations,
so a custom method class can't be named there. Instead we hook LeRobot's own
extension point, ``LeRobotDataset.set_image_transforms``: right after LeRobot
builds its datasets we compose our :class:`VisualMethodPipeline` onto each
dataset's transform. The raw frame flows through our methods first (deterministic,
cache-backed), then any LeRobot photometric augmentation runs on top. The dataset
on disk is never modified.

Which methods run is read from the ``H2R_VISUAL_METHODS`` env var — a spec passed
to :func:`h2r_il.transforms.build_pipeline` (a method name, or JSON). With it
unset, this behaves exactly like ``lerobot-train``. The cache dir comes from
``H2R_VISUAL_CACHE``. Every other CLI arg is LeRobot's and is passed through.

    H2R_VISUAL_METHODS=arm_inpaint uv run python -m h2r_il.train \\
        --dataset.repo_id=... --policy.type=groot ...

The same idea applies to actions: ``H2R_ACTION_SLOWDOWN`` wraps
``DatasetReader.get_item`` so every action chunk is interpolated to a fraction of
the demonstrated speed (see :mod:`h2r_il.action_interp`). The dataset stays
read-only here too.

    H2R_ACTION_SLOWDOWN=2 uv run python -m h2r_il.train --dataset.repo_id=... ...

``H2R_RELATIVE_ANGLE_DIMS`` makes LeRobot's relative-action conversion angle-aware
(see :mod:`h2r_il.relative_angles`); set it whenever ``--policy.use_relative_actions``
is on and the action space contains Euler angles.
"""

from __future__ import annotations

import logging
import os

import lerobot.datasets.dataset_reader as _reader
import lerobot.datasets.factory as _factory
import lerobot.scripts.lerobot_train as _lr_train

from h2r_il.action_interp import build_action_slowdown
from h2r_il.relative_angles import install_from_env as _install_relative_angles
from h2r_il.transforms import build_pipeline


def _install_visual_methods() -> None:
    spec = os.environ.get("H2R_VISUAL_METHODS", "").strip()
    if not spec:
        return  # no methods requested -> plain lerobot-train behaviour

    cache_dir = os.environ.get("H2R_VISUAL_CACHE") or None
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
        logging.info("h2r_il visual methods active: %s", spec)
        return train_ds, eval_ds

    # train.py binds the name locally (`from ...factory import make_train_eval_datasets`),
    # so patch that binding; patch the factory module too for any other caller.
    _factory.make_train_eval_datasets = _patched
    _lr_train.make_train_eval_datasets = _patched


def _install_action_slowdown() -> None:
    spec = os.environ.get("H2R_ACTION_SLOWDOWN", "").strip()
    if not spec:
        return  # no slowdown requested -> plain lerobot-train behaviour

    slowdown = build_action_slowdown(spec)
    key = slowdown.key

    # Hook the two reader methods that *both* consumers of action chunks go
    # through, rather than get_item: the training dataloader calls get_item, but
    # GR00T's relative-action stats call _get_query_indices / _query_hf_dataset
    # directly (processor_groot._iter_action_state_training_samples). Hooking
    # get_item alone left the stats computed from full-speed deltas while
    # training saw slowed ones, so relative targets were normalized ~4x wrong.
    orig_query = _reader.DatasetReader._query_hf_dataset
    orig_indices = _reader.DatasetReader._get_query_indices

    def _query_patched(self, query_indices):
        result = orig_query(self, query_indices)
        if key in result:
            result[key] = slowdown.resample_values(result[key])
        return result

    def _indices_patched(self, abs_idx, ep_idx):
        # Source indices are left alone (interpolation needs the raw frames);
        # only the pad mask is remapped onto the resampled timeline.
        query_indices, padding = orig_indices(self, abs_idx, ep_idx)
        pad_key = f"{key}_is_pad"
        if pad_key in padding:
            padding[pad_key] = slowdown.resample_pad(padding[pad_key])
        return query_indices, padding

    # Patch the reader class, not an instance: readers are built lazily per dataset.
    _reader.DatasetReader._query_hf_dataset = _query_patched
    _reader.DatasetReader._get_query_indices = _indices_patched
    logging.info("h2r_il action slowdown active: %s", slowdown.identity())


def main() -> None:
    _install_visual_methods()
    _install_action_slowdown()
    _install_relative_angles()
    _lr_train.train()  # @parser.wrap(): parses sys.argv, runs LeRobot training


if __name__ == "__main__":
    main()
