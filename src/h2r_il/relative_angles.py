"""Angle-aware relative actions.

LeRobot converts absolute actions to relative with a plain subtraction
(``relative = action - state``; see ``lerobot.processor.relative_action_processor``).
That is correct for positions, but wrong for Euler angles: when a dim sits near
the +pi/-pi seam, ``action - state`` returns ~2*pi instead of the small rotation
actually performed. On gear_left, ``right.wrist.yaw`` genuinely crosses the seam,
producing relative deltas of ~6.23 rad against a median of ~0.008 — and because
GR00T normalizes relative action groups with **min/max**, a single such outlier
sets the range and squashes every real yaw delta into ~0.1% of it.

This module wraps the converted delta back into [-pi, pi] for the dims named in
``H2R_RELATIVE_ANGLE_DIMS``, for both directions of the conversion:

* ``to_relative_actions``  -> the delta is the short way round the circle;
* ``to_absolute_actions``  -> state + delta is re-wrapped to the dataset's
  [-pi, pi] convention, so predicted poses come back in the same representation.

Only dims that the caller's ``mask`` actually converts are touched, so an
excluded dim (e.g. the gripper) is never treated as an angle.

The patch is applied to every module that binds these functions, because several
callers do ``from ... import to_relative_actions`` and hold their own reference —
notably GR00T's relative-stats computation, which is what makes the normalization
stats consistent with the training targets.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Callable, Sequence

import torch

_TWO_PI = 2.0 * math.pi


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap radians into [-pi, pi)."""
    return (x + math.pi) % _TWO_PI - math.pi


def _wrap_masked(out: torch.Tensor, mask: Sequence[bool], angle_dims: Sequence[int]) -> torch.Tensor:
    """Wrap ``angle_dims`` of ``out``, but only where ``mask`` converted them."""
    dims = [d for d in angle_dims if d < len(mask) and bool(mask[d]) and d < out.shape[-1]]
    if not dims:
        return out
    out = out.clone()
    out[..., dims] = wrap_to_pi(out[..., dims])
    return out


def parse_angle_dims(spec: str | Sequence[int]) -> list[int]:
    """Parse ``"[3,4,5,9,10,11]"`` (or a list) into a list of ints."""
    if isinstance(spec, str):
        spec = spec.strip()
        parsed: Any = json.loads(spec) if spec.startswith("[") else [int(spec)]
    else:
        parsed = spec
    return [int(d) for d in parsed]


def install_angle_aware_relative_actions(angle_dims: Sequence[int]) -> None:
    """Patch LeRobot's relative<->absolute action conversion to be angle-aware."""
    dims = list(angle_dims)
    if not dims:
        return

    import lerobot.processor.relative_action_processor as _rap

    orig_rel: Callable = _rap.to_relative_actions
    orig_abs: Callable = _rap.to_absolute_actions

    def _rel_patched(actions, state, mask):
        return _wrap_masked(orig_rel(actions, state, mask), mask, dims)

    def _abs_patched(actions, state, mask):
        return _wrap_masked(orig_abs(actions, state, mask), mask, dims)

    # Every module that holds its own reference must be rebound, not just the
    # source module: `from x import f` copies the binding at import time.
    targets = ["lerobot.processor.relative_action_processor", "lerobot.processor"]
    optional = ["lerobot.policies.groot.processor_groot", "lerobot.policies.rtc.relative"]

    patched: list[str] = []
    for name in targets + optional:
        try:
            module = __import__(name, fromlist=["_"])
        except Exception:  # optional policy deps may be absent
            if name in targets:
                raise
            continue
        for attr, fn in (("to_relative_actions", _rel_patched), ("to_absolute_actions", _abs_patched)):
            if hasattr(module, attr):
                setattr(module, attr, fn)
                patched.append(f"{name}.{attr}")

    logging.info("h2r_il angle-aware relative actions: dims=%s, patched %d bindings", dims, len(patched))


def install_from_env() -> None:
    spec = os.environ.get("H2R_RELATIVE_ANGLE_DIMS", "").strip()
    if not spec:
        return
    install_angle_aware_relative_actions(parse_angle_dims(spec))
