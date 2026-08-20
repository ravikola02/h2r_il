"""GR00T with the object pose carried in its action head's unused output dims.

GR00T pads actions to ``max_action_dim`` (132) and zeroes everything past the real
action width in ``action_mask``; the DiT emits all 132 channels regardless. The
object pose goes into the first free ones. No new module, no new parameter, and no
extra compute -- those channels were already being predicted and discarded.

Three facts about GR00T's internals make this work:

* ``action_mask`` is a float multiplier, not a boolean, and it is the only thing
  deciding which dimensions are supervised (``action_loss = mse(...) *
  action_mask``). Enabling a channel means writing a nonzero entry into it.
* the action head returns the *per-element* ``action_loss`` and the mask it used
  alongside the scalar, so the pose channel's contribution can be separated back
  out and given its own weight rather than being averaged in at whatever share of
  the dimensions it happens to occupy.
* ``GrootPolicy.forward`` hands one dict to ``self._groot_model.forward``, and that
  dict already holds the processed ``action`` and ``action_mask``. Wrapping that
  call is a single seam for both the write on the way in and the read on the way
  out.

The supervision is a *velocity* target, not a pose: GR00T's action head is a flow
matching DiT, so these channels learn the pose's velocity field along the noise
path and the pose is recovered by denoising. The aux gradient therefore reaches
the vision backbone through the DiT and the VL self-attention rather than landing
on the trunk directly.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION
from lerobot.policies.groot.modeling_groot import GrootPolicy

from h2r_il.policies.configuration_h2r_groot import H2RGrootConfig
from h2r_il.policies.object_pose_slots import ObjectPoseSlotsMixin

logger = logging.getLogger(__name__)


class H2RGrootPolicy(ObjectPoseSlotsMixin, GrootPolicy):
    name = "h2r_groot"
    config_class = H2RGrootConfig

    def __init__(self, config: H2RGrootConfig, **kwargs):
        super().__init__(config, **kwargs)
        action_dim = config.output_features[ACTION].shape[0]
        self._setup_object_pose_slots(action_dim, config.max_action_dim)
        self._install_action_hook()

    def _install_action_hook(self) -> None:
        """Wrap the inner model's forward: write the target in, capture the loss out."""
        model = self._groot_model
        original = model.forward

        def wrapped(inputs, *args, **kwargs):
            self._inject_pose(inputs)
            outputs = original(inputs, *args, **kwargs)
            self._captured = (outputs.get("action_loss"), outputs.get("action_mask"))
            return outputs

        model.forward = wrapped

    def _inject_pose(self, inputs: dict[str, Tensor]) -> None:
        """Write the pose into the spare action dims and open their mask entries."""
        action, mask = inputs.get("action"), inputs.get("action_mask")
        if action is None or mask is None:
            return                                  # inference path: nothing to supervise
        inputs["action"] = self._write_pose_into_actions(action)
        # The mask arrives with the pose columns zeroed, so without this the
        # channel is written and then multiplied out of existence.
        inputs["action_mask"] = torch.maximum(mask, self._pose_weights(mask))

    def forward(self, batch: dict[str, Tensor], **kwargs) -> tuple[Tensor, dict]:
        self._pending_batch = batch
        self._captured = (None, None)
        try:
            _, loss_dict = super().forward(batch, **kwargs)
        finally:
            self._pending_batch = None

        per_element, mask = self._captured
        if per_element is None or mask is None:
            raise RuntimeError(
                "GR00T did not return 'action_loss'/'action_mask'. The h2r_groot "
                "loss is recomposed from them, so it cannot fall back to the "
                "scalar without silently dropping the pose term's weight."
            )

        # Recompose rather than use GR00T's own scalar: its reduction averages
        # the action and pose channels under one shared denominator. See
        # ObjectPoseSlotsMixin._recompose_losses.
        total, metrics = self._recompose_losses(per_element, mask)
        loss_dict.update(metrics)
        return total, loss_dict
