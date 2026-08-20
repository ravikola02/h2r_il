"""Processors for `h2r_groot`: GR00T's own, unchanged.

One wrinkle GR00T has and pi0 does not: its factory wants ``dataset_meta``, but
LeRobot's plugin path calls ``make_<type>_pre_post_processors(config,
dataset_stats=...)`` with no room for it. ``make_policy`` stashes the metadata on
the config as ``_runtime_dataset_meta`` beforehand, so recover it from there when
the caller does not pass it.
"""

from __future__ import annotations

from typing import Any

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

from h2r_il.policies.configuration_h2r_groot import H2RGrootConfig


def make_h2r_groot_pre_post_processors(
    config: H2RGrootConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    dataset_meta: Any | None = None,
    **kwargs: Any,
):
    if dataset_meta is None:
        dataset_meta = getattr(config, "_runtime_dataset_meta", None)
    return make_groot_pre_post_processors(
        config, dataset_stats=dataset_stats, dataset_meta=dataset_meta, **kwargs
    )
