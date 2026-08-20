"""pi0 with an object-pose auxiliary head: `--policy.type=h2r_pi0`.

Registered through LeRobot's third-party plugin path rather than by patching its
factory. That path derives everything from names, so the three below are load
bearing and must stay in step:

    H2RPi0Config          -> policy class H2RPi0Policy
    configuration_h2r_pi0 -> module modeling_h2r_pi0
    type "h2r_pi0"        -> make_h2r_pi0_pre_post_processors in processor_h2r_pi0

Everything pi0 already accepts still applies; these fields are additions.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config


@PreTrainedConfig.register_subclass("h2r_pi0")
@dataclass
class H2RPi0Config(PI0Config):
    """pi0, plus a head that regresses where the manipulated object is."""

    # Scale of the aux term relative to pi0's flow-matching loss. Meaningful as a
    # preference only because the target is standardised first (see
    # h2r_il.losses.object_pose); on raw metres this would be unit conversion.
    object_pose_weight: float = 1.0

    # "position" (xyz) or "pose6d" (xyz + wxyz). Prefer position: the rotation in
    # the store comes from an arbitrary SAM3D body frame and nothing validates it,
    # so pose6d spends 4 of its 7 dims regressing towards unverified numbers.
    object_pose_target: str = "position"

    object_pose_hidden_dim: int = 512

    # Pose store, used ONLY to standardise the target at construction. The targets
    # themselves arrive per-sample in the batch. Leave unset and the loss runs on
    # raw metres, which is almost never what you want.
    object_pose_store: str | None = None

    # Stop gradients into the trunk, turning the head into a passive probe: it
    # measures whether the representation already encodes object position without
    # shaping it. A diagnostic, not a training signal.
    object_pose_detach: bool = False
