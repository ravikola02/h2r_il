"""The object-pose prediction head, and the mixin that hangs it off a policy.

Both pi0 and GR00T are trained here as *black boxes*: their ``forward`` takes a
batch and returns a loss. Neither exposes the intermediate representation an
auxiliary head needs, and this project does not fork LeRobot -- so each subclass
captures its backbone's output in flight (a wrapped method for pi0, a module
forward hook for GR00T) and this mixin does everything downstream of that.

What the head is attached to matters more than its shape. It reads the *fused
vision-language features the action expert attends to*, pools them, and regresses
the object's position. Because the head sits on the shared trunk and gradients are
not detached by default, the aux loss pushes the trunk towards a representation
that knows where the object is -- which is the entire point. Detaching it (see
``object_pose_detach``) turns the head into a passive probe that measures whether
the representation already encodes position without changing it: useful as a
diagnostic, useless as a training signal.

The head is built in ``__init__``, never lazily on the first batch. LeRobot builds
the optimiser from ``policy.parameters()`` right after construction, so a head
created later would train with no optimiser state and silently never update.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from h2r_il.losses.object_pose import masked_mean_pool, object_pose_loss, target_statistics

logger = logging.getLogger(__name__)

TARGET_DIMS = {"position": 3, "pose6d": 7}


class ObjectPoseHead(nn.Module):
    """Pooled backbone features -> standardised object pose.

    Deliberately small. The head is not meant to be capable enough to solve the
    task on its own -- if it were, it could satisfy the aux loss by itself and
    leave the trunk unchanged, which is the failure mode that makes an auxiliary
    loss look fine in the logs and do nothing to the representation.
    """

    def __init__(self, feature_dim: int, target_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        # Buffers, not plain tensors: they must move with .to(device), survive
        # save_pretrained, and be restored on load, or a resumed run would score
        # against different statistics than it trained with.
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    def set_statistics(self, mean, std) -> None:
        self.target_mean.copy_(torch.as_tensor(mean, dtype=self.target_mean.dtype))
        self.target_std.copy_(torch.as_tensor(std, dtype=self.target_std.dtype))

    def forward(self, features: Tensor, mask: Tensor | None) -> Tensor:
        parameter_dtype = self.net[1].weight.dtype
        # The trunk may run under bf16 autocast while the head's parameters are
        # fp32. Cast the activations rather than the parameters: casting the
        # parameters would silently halve the precision of the head's updates.
        pooled = masked_mean_pool(features.to(parameter_dtype), mask)
        return self.net(pooled)


class ObjectPoseAuxMixin:
    """Shared aux-loss machinery. Subclasses supply the captured features.

    Contract for the subclass:
      * call :meth:`_setup_object_pose_aux` at the end of ``__init__``;
      * arrange for :attr:`_captured_features` to be set during ``forward``;
      * call :meth:`_add_object_pose_loss` on the way out of ``forward``.
    """

    _captured_features: tuple[Tensor, Tensor | None] | None = None

    def _setup_object_pose_aux(self, feature_dim: int) -> None:
        config = self.config
        target = getattr(config, "object_pose_target", "position")
        if target not in TARGET_DIMS:
            raise ValueError(f"object_pose_target must be one of {sorted(TARGET_DIMS)}")

        self.object_pose_head = ObjectPoseHead(
            feature_dim=feature_dim,
            target_dim=TARGET_DIMS[target],
            hidden_dim=getattr(config, "object_pose_hidden_dim", 512),
        )
        self._object_pose_target = target
        self._object_pose_warned = False

        store = getattr(config, "object_pose_store", None)
        if store:
            self._load_target_statistics(store, target)
        else:
            # Identity statistics are a real risk, not a cosmetic default: the loss
            # then runs on raw metres and is ~100x smaller than it looks, so say so.
            logger.warning(
                "object-pose head built with no object_pose_store: targets will NOT "
                "be standardised, so object_pose_weight is operating on raw metres."
            )
        logger.info(
            "object-pose head: %d -> %d (%s), weight %.3f, detach=%s",
            feature_dim, TARGET_DIMS[target], target,
            getattr(config, "object_pose_weight", 1.0),
            getattr(config, "object_pose_detach", False),
        )

    def _load_target_statistics(self, store: str, target: str) -> None:
        from h2r_il.object_pose_inject import PoseStore

        path = Path(store).expanduser()
        pose_store = PoseStore(path, target=target)
        mean, std = target_statistics(
            pose_store.values.numpy(), pose_store.valid.numpy()
        )
        self.object_pose_head.set_statistics(mean, std)
        logger.info(
            "object-pose target statistics from %s: mean %s, std %s",
            path, mean.round(4).tolist(), std.round(4).tolist(),
        )

    def _add_object_pose_loss(
        self, loss: Tensor, loss_dict: dict, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict]:
        """Fold the aux term into the policy's own loss and metrics."""
        from h2r_il.object_pose_inject import POSE_KEY, VALID_KEY

        weight = getattr(self.config, "object_pose_weight", 1.0)
        captured = self._captured_features
        # Free the capture whatever happens next: holding it past the backward
        # pass keeps the whole prefix activation alive for the next iteration.
        self._captured_features = None

        if weight == 0.0 or POSE_KEY not in batch:
            if POSE_KEY not in batch and not self._object_pose_warned:
                # The keys are dropped unless they are under "observation." -- the
                # exact mistake this project already made once. Say it loudly
                # rather than training a head on nothing.
                logger.warning(
                    "object-pose head is active but %r is not in the batch. Is "
                    "H2R_OBJECT_POSE set, and does training run through "
                    "h2r_il.train? The aux loss is contributing nothing.", POSE_KEY
                )
                self._object_pose_warned = True
            return loss, loss_dict

        if captured is None:
            raise RuntimeError(
                "object-pose head is active but no backbone features were captured "
                "during forward. The capture hook did not fire -- the upstream "
                "policy's internals have probably changed."
            )

        features, mask = captured
        if getattr(self.config, "object_pose_detach", False):
            features = features.detach()

        prediction = self.object_pose_head(features, mask)
        target = batch[POSE_KEY]
        valid = batch.get(VALID_KEY)
        if valid is None:
            valid = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)

        aux_loss, metrics = object_pose_loss(
            prediction,
            target,
            valid,
            self.object_pose_head.target_mean,
            self.object_pose_head.target_std,
        )

        loss_dict.update(metrics)
        loss_dict["object_pose_weighted"] = float(weight * aux_loss.detach())
        # The base loss may be per-sample (reduction="none" for RA-BC weighting);
        # a scalar aux term broadcasts onto it correctly either way.
        return loss + weight * aux_loss.to(loss.dtype), loss_dict
