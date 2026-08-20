#!/usr/bin/env python
"""Attach precomputed object poses to training samples.

All the perception happens up front (`object_pose_dataset track` / `store`), so
training only needs a lookup. The store is dense and indexed by the dataset's
**global frame index**, which every LeRobot sample already carries as
``item["index"]`` -- so injection is one dict assignment per sample, with no
decoding, no I/O and no per-worker state.

Why not write the pose into the dataset itself, as a real feature? It would be
loaded and normalised natively, but it means rewriting the dataset (or copying
it, on a disk with ~26 GB free), and this project treats the exported dataset as
read-only. It would also land in the ``observation.`` namespace where a policy
may try to consume it as an input. Injection keeps the dataset untouched and the
pose clearly a *target*, and the data is no less precomputed for it.

The whole store is 20613 x (4x4 + flags) float32 -- about 2 MB. It is read once
into RAM at startup; DataLoader workers inherit it by fork, so there is no
per-worker cost and no memory-mapping subtlety.

Enable it the same way as inpainting, from the environment::

    H2R_OBJECT_POSE=<store dir or object_pose.npz>
    H2R_OBJECT_POSE_TARGET=position   # position (default) | pose6d

``position`` emits xyz only. That is the recommended target: rotation comes from
an arbitrary SAM3D body frame and nothing in this pipeline validates it (MoGe
gives depth, not orientation). ``pose6d`` emits xyz + wxyz quaternion for when
that changes.

Every sample gets ``observation.object_pose_valid``. Frames with no pose emit zeros
with the flag false, and **the loss must mask on it** -- otherwise uncovered frames
train the model towards a pose of all zeros.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

# LeRobot's processor pipeline runs between the dataloader and the policy
# (`batch = preprocessor(batch)` in lerobot_train). Its dict->transition converter
# keeps only keys under the "observation." prefix plus a fixed whitelist, and drops
# everything else -- so a bare "object_pose" key survives collation, looks correct
# in any dataloader-level test, and is silently gone by the time forward() runs.
# The prefix is what makes the target reach the policy at all.
POSE_KEY = "observation.object_pose"
VALID_KEY = "observation.object_pose_valid"

logger = logging.getLogger(__name__)


class PoseStore:
    """Dense per-frame object poses, looked up by global dataset frame index."""

    def __init__(self, path: str | Path, target: str = "position",
                 horizon: int = 1) -> None:
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "object_pose.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no pose store at {path}")
        if target not in ("position", "pose6d"):
            raise ValueError(f"target must be 'position' or 'pose6d', got {target!r}")

        data = np.load(path)
        self.path = path
        self.target = target
        self.horizon = max(1, int(horizon))
        self.valid = torch.from_numpy(data["valid"].astype(np.bool_))
        # Needed to stop a horizon window from walking past an episode cut: the
        # video is one concatenated file, so index+1 past the last frame of an
        # episode is a different take entirely, and its pose is unrelated.
        self.episode_index = torch.from_numpy(data["episode_index"].astype(np.int64))

        position = data["position"].astype(np.float32)
        if target == "position":
            self.values = torch.from_numpy(position)
        else:
            # xyz + wxyz. Stored quaternions are xyzw; reorder rather than make
            # the consumer guess which convention it received.
            quat_xyzw = data["quat_xyzw"].astype(np.float32)
            quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
            self.values = torch.from_numpy(np.concatenate([position, quat_wxyz], axis=1))

        self.dim = self.values.shape[1]
        covered = int(self.valid.sum())
        logger.info(
            "object pose store %s: %d/%d frames covered (%.1f%%), target=%s, dim=%d, "
            "horizon=%d",
            path, covered, len(self.valid), 100 * covered / len(self.valid), target,
            self.dim, self.horizon
        )

    def __len__(self) -> int:
        return len(self.valid)

    def attach(self, item: dict) -> dict:
        """Add the pose target to one sample, keyed on its global frame index."""
        idx = int(item["index"])
        if idx >= len(self.valid):
            # A store built for a different dataset would otherwise silently
            # attach poses from the wrong frames.
            raise IndexError(
                f"frame index {idx} is outside the pose store ({len(self.valid)} "
                f"frames). Store {self.path} does not match this dataset."
            )
        if self.horizon == 1:
            item[POSE_KEY] = self.values[idx]
            item[VALID_KEY] = self.valid[idx]
            return item

        # A window of poses aligned with the policy's action chunk: step t is the
        # object's pose at the frame that action t acts on.
        stop = min(idx + self.horizon, len(self.valid))
        values = self.values[idx:stop]
        # Same episode only. Frames beyond the cut are dropped rather than
        # clamped: repeating the last pose would look like a stationary object.
        valid = self.valid[idx:stop] & (self.episode_index[idx:stop] == self.episode_index[idx])

        # Zero what the mask rejects, matching the store's own convention that an
        # uncovered frame is zeros. Everything downstream masks anyway; this just
        # removes the chance that a future consumer which forgets to reads a real
        # pose from the *next* episode and never notices.
        values = values * valid.unsqueeze(-1).to(values.dtype)

        missing = self.horizon - values.shape[0]
        if missing > 0:                       # near the end of the dataset
            values = torch.cat([values, values.new_zeros(missing, self.dim)])
            valid = torch.cat([valid, valid.new_zeros(missing)])
        item[POSE_KEY] = values               # (horizon, dim)
        item[VALID_KEY] = valid               # (horizon,)
        return item


def wrap_dataset(dataset, store: PoseStore):
    """Make ``dataset[i]`` also return the pose target.

    Retypes this one instance into a subclass that overrides ``__getitem__``.
    Assigning ``dataset.__getitem__ = ...`` looks like it should work and does
    nothing: Python resolves dunder methods on the *type*, so ``dataset[i]``
    would keep calling the original and the samples would silently arrive
    without the pose. Retyping affects only this object, not other datasets in
    the process, and leaves LeRobot's class untouched.
    """
    if getattr(dataset, "_h2r_pose_wrapped", False):
        return dataset

    base = type(dataset)

    class WithObjectPose(base):
        def __getitem__(self, idx):
            return store.attach(base.__getitem__(self, idx))

    WithObjectPose.__name__ = f"{base.__name__}WithObjectPose"
    dataset.__class__ = WithObjectPose
    dataset._h2r_pose_wrapped = True
    return dataset


def install_from_env() -> None:
    """Patch LeRobot's dataset factory when H2R_OBJECT_POSE is set.

    Mirrors the inpainting hook in :mod:`h2r_il.train`: LeRobot's own factory is
    left in place and its result is decorated, so nothing is forked.
    """
    spec = os.environ.get("H2R_OBJECT_POSE", "").strip()
    if not spec:
        return

    target = os.environ.get("H2R_OBJECT_POSE_TARGET", "position").strip() or "position"
    # Poses for the whole action chunk, not just the observed frame. Set this at
    # least as large as the policy's chunk_size; the policy slices what it needs
    # and masks any shortfall, so over-covering is free (50 x 3 floats a sample).
    horizon = int(os.environ.get("H2R_OBJECT_POSE_HORIZON", "1") or 1)
    store = PoseStore(spec, target, horizon=horizon)

    import lerobot.datasets.factory as _factory
    import lerobot.scripts.lerobot_train as _lr_train

    original = _factory.make_train_eval_datasets

    def make_train_eval_datasets(*args, **kwargs):
        result = original(*args, **kwargs)
        datasets = result if isinstance(result, tuple) else (result,)
        for ds in datasets:
            if ds is not None and hasattr(ds, "__getitem__"):
                wrap_dataset(ds, store)
        return result

    _factory.make_train_eval_datasets = make_train_eval_datasets
    # train.py binds the name locally, so patching the factory module alone
    # would not reach it.
    if hasattr(_lr_train, "make_train_eval_datasets"):
        _lr_train.make_train_eval_datasets = make_train_eval_datasets
