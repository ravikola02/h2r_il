"""Auxiliary object-pose regression: pooling, and the masked loss.

The target is a per-frame object pose attached to the batch by
:mod:`h2r_il.object_pose_inject`. This module holds the parts that do not depend
on which policy is being trained: how a sequence of backbone features becomes one
vector, and how that vector is scored against the target.

Two things here are load-bearing.

**Masking.** Frames with no pose carry zeros and ``valid=False``. Averaging over
them trains the model towards a pose of all zeros -- a target that is not merely
wrong but attracts, because zero sits in the middle of the coordinate range. Every
reduction below divides by the number of *valid* samples, and a batch with none
contributes exactly zero gradient rather than NaN.

**Target normalisation.** Positions are metres in the camera optical frame, so a
raw MSE against them is ~1e-2 while the flow-matching loss is ~1e0: the aux term
would be invisible at weight 1.0, and the weight would silently be doing unit
conversion instead of expressing a preference. The head predicts a standardised
target instead, using statistics taken once from the store, so ``weight`` means
what it says and stays comparable across datasets and across position/pose6d.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_mean_pool(features: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over the token axis, ignoring padding. ``(B, T, D) -> (B, D)``.

    Padding matters more than it looks: pi0 pads the language tokens to a fixed
    length, so a plain ``.mean(1)`` would dilute every sample by however much
    padding its prompt happened to need, making the pooled vector depend on prompt
    length rather than on content.
    """
    if mask is None:
        return features.mean(dim=1)

    mask = mask.to(dtype=features.dtype)
    while mask.dim() < features.dim():
        mask = mask.unsqueeze(-1)
    total = (features * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count


def target_statistics(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std over the covered frames only.

    Uncovered frames are stored as zeros. Including them would drag the mean
    towards the origin and inflate the std by exactly the amount of missing data,
    which is the one thing the statistics must not depend on.
    """
    covered = values[valid.astype(bool)] if valid is not None else values
    if covered.size == 0:
        return np.zeros(values.shape[1], np.float32), np.ones(values.shape[1], np.float32)
    mean = covered.mean(axis=0).astype(np.float32)
    # A frozen dimension (a quaternion component that never moves, say) has std 0
    # and would produce inf on division.
    std = np.maximum(covered.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def object_pose_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    mean: Tensor,
    std: Tensor,
    huber_beta: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Masked regression of ``prediction`` (standardised) onto ``target`` (metres).

    Returns the loss and metrics. The reported error is in centimetres in the
    original units, because a standardised residual is not something anyone can
    sanity-check against the 0.48 cm the tracker was measured at.
    """
    target = target.to(dtype=prediction.dtype)
    valid = valid.to(dtype=prediction.dtype).reshape(-1)
    covered = valid.sum()

    standardised = (target - mean) / std
    # Huber, not MSE: a frame where tracking briefly lost the object is a large
    # residual, and squaring it lets one bad frame dominate the batch gradient.
    per_sample = F.smooth_l1_loss(
        prediction, standardised, reduction="none", beta=huber_beta
    ).mean(dim=-1)

    # clamp(min=1) only guards the division; the numerator is already zero when
    # nothing is valid, so the batch contributes no gradient rather than NaN.
    loss = (per_sample * valid).sum() / covered.clamp(min=1.0)

    with torch.no_grad():
        metres = prediction.detach() * std + mean
        # Position lives in the first three dims under either target; the
        # quaternion tail is not a distance and must not enter a norm with them.
        offset = (metres[:, :3] - target[:, :3]).norm(dim=-1)
        centimetres = (offset * valid).sum() / covered.clamp(min=1.0) * 100.0

    metrics = {
        "object_pose_loss": float(loss.detach()),
        "object_pose_err_cm": float(centimetres),
        "object_pose_valid_frac": float(covered / max(valid.numel(), 1)),
    }
    return loss, metrics
