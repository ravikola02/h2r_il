"""pi0 + object-pose head. See :mod:`h2r_il.policies.configuration_h2r_pi0`.

The tap is ``PI0Pytorch.embed_prefix``, which returns the concatenated image and
language embeddings together with their padding mask -- the exact sequence the
action expert cross-attends to. Capturing there rather than at the vision tower
gets every camera and the language prompt in one tensor, already aligned with a
mask, and costs nothing: the features are computed anyway.

``embed_prefix`` is a plain method, not a submodule, so there is no
``register_forward_hook`` for it; the wrapper below replaces the bound method on
the instance. That keeps the change to this one policy object -- other pi0
instances in the same process are untouched.
"""

from __future__ import annotations

from torch import Tensor

from lerobot.policies.pi0.modeling_pi0 import PI0Policy, get_gemma_config

from h2r_il.policies.configuration_h2r_pi0 import H2RPi0Config
from h2r_il.policies.object_pose_head import ObjectPoseAuxMixin


class H2RPi0Policy(ObjectPoseAuxMixin, PI0Policy):
    name = "h2r_pi0"
    config_class = H2RPi0Config

    def __init__(self, config: H2RPi0Config, **kwargs):
        super().__init__(config, **kwargs)
        # Width from the variant's config, not by walking into the built model:
        # one lookup, and it cannot drift with HF's attribute layout.
        feature_dim = get_gemma_config(config.paligemma_variant).width
        self._setup_object_pose_aux(feature_dim)
        self._install_prefix_capture()

    def _install_prefix_capture(self) -> None:
        inner = self.model
        original = inner.embed_prefix

        def capturing_embed_prefix(*args, **kwargs):
            embeddings, pad_masks, attention_masks = original(*args, **kwargs)
            self._captured_features = (embeddings, pad_masks)
            return embeddings, pad_masks, attention_masks

        inner.embed_prefix = capturing_embed_prefix

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        loss, loss_dict = super().forward(batch, reduction=reduction)
        return self._add_object_pose_loss(loss, loss_dict, batch)
