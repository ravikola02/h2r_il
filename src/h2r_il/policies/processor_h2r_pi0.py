"""Processors for `h2r_pi0`: pi0's own, unchanged.

The aux head changes the loss, not the inputs, so there is nothing to add here.
LeRobot's plugin path looks up ``make_<type>_pre_post_processors`` by name, so the
alias has to exist even though it only forwards.

The pose target reaches the policy without any step of its own: it is attached
under the ``observation.`` prefix, which is what LeRobot's dict->transition
converter preserves, and the normalizer leaves keys it has no feature spec for
untouched.
"""

from __future__ import annotations

from typing import Any

from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

from h2r_il.policies.configuration_h2r_pi0 import H2RPi0Config


def make_h2r_pi0_pre_post_processors(
    config: H2RPi0Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
):
    return make_pi0_pre_post_processors(config, dataset_stats=dataset_stats, **kwargs)
