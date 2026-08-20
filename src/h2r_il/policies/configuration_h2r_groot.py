"""GR00T with an object-pose auxiliary head: `--policy.type=h2r_groot`.

Same plugin-path naming contract as :mod:`h2r_il.policies.configuration_h2r_pi0`:

    H2RGrootConfig          -> policy class H2RGrootPolicy
    configuration_h2r_groot -> module modeling_h2r_groot
    type "h2r_groot"        -> make_h2r_groot_pre_post_processors in processor_h2r_groot

The added fields are identical to the pi0 variant on purpose: the two policies
should be swappable in an ablation without rewriting the command line.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("h2r_groot")
@dataclass
class H2RGrootConfig(GrootConfig):
    """GR00T N1.7, plus a head that regresses where the manipulated object is."""

    # Weight on the pose channel, relative to the action loss. Each term is
    # normalised over its own elements, so this is a true weight rather than a
    # share of however many action dimensions the embodiment happens to have --
    # and 0.0 reproduces stock GR00T's loss exactly.
    object_pose_weight: float = 1.0

    # "position" (xyz) or "pose6d" (xyz + wxyz). Prefer position: the rotation in
    # the store comes from an arbitrary SAM3D body frame and nothing validates it.
    object_pose_target: str = "position"

    # Pose store, used ONLY to standardise the target at construction; the targets
    # themselves arrive per-sample in the batch. Without it the pose channel holds
    # raw metres beside unit-scale actions and is a rounding error in the loss.
    object_pose_store: str | None = None
