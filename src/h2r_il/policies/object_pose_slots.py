"""Route the object-pose target through the action head's unused output dimensions.

Shared by both policy subclasses. The mixin owns three things: which slice of the
action vector the pose occupies, how the target is written into the action tensor
the model is trained against, and how the pose channel's loss is separated back
out so it can carry its own weight.

There is no new module and no new parameter. The statistics buffers are
non-persistent, so the state dict is byte-identical to the wrapped policy's and a
stock checkpoint loads with no missing or unexpected keys in either direction.

The slot is placed immediately after the real action dimensions, which are already
zero-padded out to ``max_action_dim``. That keeps the real action channels at the
indices the base checkpoint was trained on -- writing the pose *before* them would
shift every action dimension and invalidate the pretrained head.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from h2r_il.losses.object_pose import target_statistics

logger = logging.getLogger(__name__)

# xyz, or xyz + wxyz.
TARGET_DIMS = {"position": 3, "pose6d": 7}


def standardise(target: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (target - mean) / std


def masked_element_mean(per_element: Tensor, weights: Tensor) -> Tensor:
    """``sum(loss * w) / sum(w)``, and exactly zero when nothing is weighted.

    The clamp guards only the division: the numerator is already zero when every
    weight is zero, so an all-invalid batch yields 0.0 with a finite (zero)
    gradient instead of NaN.
    """
    return (per_element * weights).sum() / weights.sum().clamp(min=1e-6)


class ObjectPoseSlotsMixin:
    """Contract for the subclass:

    * call :meth:`_setup_object_pose_slots` at the end of ``__init__``;
    * make the batch reachable as ``self._pending_batch`` during ``forward``;
    * call :meth:`_write_pose_into_actions` on the action tensor before the model
      sees it, and :meth:`_split_pose_loss` on the per-element loss after.
    """

    _pending_batch: dict | None = None

    def _setup_object_pose_slots(self, action_dim: int, max_action_dim: int) -> None:
        config = self.config
        target = getattr(config, "object_pose_target", "position")
        if target not in TARGET_DIMS:
            raise ValueError(f"object_pose_target must be one of {sorted(TARGET_DIMS)}")

        pose_dim = TARGET_DIMS[target]
        if action_dim + pose_dim > max_action_dim:
            # Better here than as a silent out-of-range write that trains nothing.
            raise ValueError(
                f"no room for a {pose_dim}-dim pose target: the action head emits "
                f"{max_action_dim} dims and the action already uses {action_dim}. "
                "Raise max_action_dim or use a narrower target."
            )

        self._object_pose_target = target
        self._object_pose_slice = slice(action_dim, action_dim + pose_dim)
        self._object_pose_action_dim = action_dim
        self._object_pose_warned = False
        self._object_pose_short_warned = False

        # persistent=False: these are derived from the store, not learned, so
        # keeping them out of the state dict is what makes a stock checkpoint load
        # cleanly in both directions.
        self.register_buffer("object_pose_mean", torch.zeros(pose_dim), persistent=False)
        self.register_buffer("object_pose_std", torch.ones(pose_dim), persistent=False)

        store = getattr(config, "object_pose_store", None)
        if store:
            self._load_target_statistics(store, target)
        else:
            logger.warning(
                "object-pose slots active with no object_pose_store: the target is "
                "NOT standardised, so raw metres share a channel with unit-scale "
                "actions and the pose term is ~50x smaller than it looks."
            )
        logger.info(
            "object-pose slots: action dims 0:%d, pose dims %d:%d (%s), weight %.3f",
            action_dim, self._object_pose_slice.start, self._object_pose_slice.stop,
            target, getattr(config, "object_pose_weight", 1.0),
        )

    def _load_target_statistics(self, store: str, target: str) -> None:
        from h2r_il.object_pose_inject import PoseStore

        pose_store = PoseStore(Path(store).expanduser(), target=target)
        mean, std = target_statistics(pose_store.values.numpy(), pose_store.valid.numpy())
        self.object_pose_mean.copy_(torch.as_tensor(mean))
        self.object_pose_std.copy_(torch.as_tensor(std))
        logger.info("object-pose statistics: mean %s, std %s",
                    mean.round(4).tolist(), std.round(4).tolist())

    # ---------------------------------------------------------------- target

    def _pose_target(self) -> tuple[Tensor, Tensor] | None:
        """``(standardised pose (B, H, dim), valid (B, H))`` for the pending batch.

        Accepts either a per-frame target (B, dim) or a horizon window
        (B, H, dim); the former is promoted to H=1 so both shapes flow through the
        same code below.
        """
        from h2r_il.object_pose_inject import POSE_KEY, VALID_KEY

        batch = self._pending_batch
        if not batch or POSE_KEY not in batch:
            if not self._object_pose_warned:
                logger.warning(
                    "object-pose slots are active but %r is not in the batch. Is "
                    "H2R_OBJECT_POSE set, and is training running through "
                    "h2r_il.train? The pose channel is supervising nothing.", POSE_KEY
                )
                self._object_pose_warned = True
            return None

        target = batch[POSE_KEY]
        valid = batch.get(VALID_KEY)
        if target.dim() == 2:                       # (B, dim) -> one step
            target = target[:, None, :]
            valid = None if valid is None else valid.reshape(-1, 1)
        if valid is None:
            valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        return target, valid

    def _covered_steps(self, horizon_available: int, horizon_needed: int) -> int:
        """How many chunk steps the target actually covers.

        A target shorter than the action chunk is not an error -- the tail is left
        masked -- but it is worth saying once, because the usual cause is
        H2R_OBJECT_POSE_HORIZON being smaller than the policy's chunk_size and the
        silent symptom is an aux signal that covers only part of the horizon.
        """
        if horizon_available < horizon_needed and not self._object_pose_short_warned:
            logger.warning(
                "object-pose target covers %d of the policy's %d chunk steps; the "
                "rest stays masked. Raise H2R_OBJECT_POSE_HORIZON to %d to supervise "
                "the whole chunk.", horizon_available, horizon_needed, horizon_needed
            )
            self._object_pose_short_warned = True
        return min(horizon_available, horizon_needed)

    def _write_pose_into_actions(self, actions: Tensor) -> Tensor:
        """Put the pose trajectory in its slot. ``actions`` is (B, T, D).

        Step t of the chunk gets the object's pose at the frame action t acts on,
        so the channel is a genuine object trajectory rather than a constant. The
        store holds one pose per frame and the window was cut at the episode
        boundary upstream, so steps past the cut arrive zeroed and masked.
        """
        found = self._pose_target()
        if found is None:
            return actions

        target, _ = found
        steps = self._covered_steps(target.shape[1], actions.shape[1])
        standardised = standardise(
            target[:, :steps].to(dtype=actions.dtype, device=actions.device),
            self.object_pose_mean.to(actions.dtype),
            self.object_pose_std.to(actions.dtype),
        )
        # Clone: the caller's tensor may be a view of the batch, and an in-place
        # write would leak the pose into whatever else reads that storage.
        actions = actions.clone()
        actions[:, :steps, self._object_pose_slice] = standardised
        return actions

    # ------------------------------------------------------------------ loss

    def _pose_weights(self, reference: Tensor) -> Tensor:
        """A mask shaped like the model's per-element loss, selecting the pose slot."""
        weights = torch.zeros_like(reference)
        found = self._pose_target()
        if found is None:
            return weights
        _, valid = found
        steps = min(valid.shape[1], reference.shape[1])
        column = valid[:, :steps].to(dtype=reference.dtype, device=reference.device)
        weights[:, :steps, self._object_pose_slice] = column[:, :, None]
        return weights

    def _pose_term(self, per_element: Tensor) -> tuple[Tensor, dict[str, float]]:
        """The weighted pose loss and its metrics, from the model's per-element loss."""
        weight = getattr(self.config, "object_pose_weight", 1.0)
        weights = self._pose_weights(per_element)
        pose_loss = masked_element_mean(per_element, weights)
        covered = weights.sum()
        metrics = {
            "object_pose_loss": float(pose_loss.detach()),
            "object_pose_weighted": float((weight * pose_loss).detach()),
            "object_pose_supervised_elems": float(covered),
        }
        return weight * pose_loss, metrics

    def _recompose_losses(
        self, per_element: Tensor, mask: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Split a masked per-element loss into an action term and a pose term.

        ``per_element`` is expected to be ``mse * mask`` already, which is what
        both action heads hand back, so the numerators need no reweighting.

        Each term is normalised over *its own* elements. Averaging them together
        under one denominator -- what the base policy does -- would make the pose's
        influence depend on how many action dimensions the embodiment happens to
        have, so the weight would mean something different on every dataset. The
        arrangement here also makes ``object_pose_weight=0`` reproduce the base
        policy's loss exactly, because that denominator is precisely this one.
        """
        action_slice = slice(0, self._object_pose_action_dim)
        action_loss = per_element[:, :, action_slice].sum() / (
            mask[:, :, action_slice].sum() + 1e-6
        )
        pose_loss, metrics = self._pose_term(per_element)
        total = action_loss + pose_loss

        metrics["action_loss"] = float(action_loss.detach())
        metrics["loss"] = float(total.detach())
        return total, metrics
