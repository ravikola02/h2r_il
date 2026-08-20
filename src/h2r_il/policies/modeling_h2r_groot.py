"""GR00T + object-pose head. See :mod:`h2r_il.policies.configuration_h2r_groot`.

GR00T's ``forward`` hands the whole batch to the vendored Isaac model and returns
only a loss, so the tap is a forward hook on ``model.backbone``. That module
returns ``backbone_features`` (B, T, backbone_embedding_dim) with a matching
``backbone_attention_mask`` -- the fused vision-language sequence the action head
cross-attends to, which is the GR00T counterpart of pi0's prefix.

The hook is placed *before* the action head's own ``vlln``/self-attention stage,
so the aux gradient lands on the backbone rather than on the action head's private
re-encoding of it. Shaping the trunk is the point.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.groot.modeling_groot import GrootPolicy

from h2r_il.policies.configuration_h2r_groot import H2RGrootConfig
from h2r_il.policies.object_pose_head import ObjectPoseAuxMixin


class H2RGrootPolicy(ObjectPoseAuxMixin, GrootPolicy):
    name = "h2r_groot"
    config_class = H2RGrootConfig

    def __init__(self, config: H2RGrootConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_object_pose_aux(self._backbone_feature_dim())
        self._install_backbone_capture()

    def _backbone_feature_dim(self) -> int:
        model_config = getattr(self._groot_model, "config", None)
        dim = getattr(model_config, "backbone_embedding_dim", None)
        if dim is None:
            raise RuntimeError(
                "cannot determine GR00T's backbone_embedding_dim; the head's input "
                "width would be a guess and a wrong guess fails only at the first "
                "batch, after the model is loaded."
            )
        return int(dim)

    def _install_backbone_capture(self) -> None:
        backbone = getattr(self._groot_model, "backbone", None)
        if backbone is None:
            raise RuntimeError("GR00T model exposes no `backbone` module to hook.")

        def capture(_module, _inputs, output):
            features = output["backbone_features"]
            mask = output.get("backbone_attention_mask")
            self._captured_features = (features, mask)

        # Kept so the hook can be removed in tests; the handle is not needed in
        # normal training, where the policy and the hook have the same lifetime.
        self._object_pose_hook = backbone.register_forward_hook(capture)

    def forward(self, batch: dict[str, Tensor], **kwargs) -> tuple[Tensor, dict]:
        loss, loss_dict = super().forward(batch, **kwargs)
        # GR00T runs its trunk under bf16 autocast; the head's own cast is inside
        # ObjectPoseHead.forward, so nothing extra is needed here beyond letting
        # the loss come back in the base loss's dtype.
        return self._add_object_pose_loss(loss, loss_dict, batch)
