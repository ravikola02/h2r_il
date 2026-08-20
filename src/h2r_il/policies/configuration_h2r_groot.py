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

    object_pose_weight: float = 1.0
    object_pose_target: str = "position"
    object_pose_hidden_dim: int = 512
    object_pose_store: str | None = None
    object_pose_detach: bool = False
