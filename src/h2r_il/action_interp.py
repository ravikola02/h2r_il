"""Action-chunk interpolation: train a policy that moves slower than the demos.

The dataset on disk is never touched. LeRobot hands a policy an action *chunk*
per sample — the actions at frames ``t, t+1, ..., t+H-1`` (``delta_timestamps``
expanded in ``DatasetReader.get_item``). :class:`ActionSlowdown` resamples that
chunk at fractional source offsets ``0, 1/k, 2/k, ..., (H-1)/k`` by linear
interpolation, so the same H-step chunk now spans only ``(H-1)/k`` frames of the
original demonstration.

The policy still emits H actions and the robot still executes them at the
dataset's fps, so it traces the same path at ``1/k`` the speed (and a task takes
``k`` times as long). Because the deepest source offset needed is ``(H-1)/k``,
no frames outside the already-fetched chunk are read for ``k >= 1``.

Euler angles must be listed in ``angle_dims``: they are unwrapped before
interpolation so a +pi/-pi crossing is not blended through zero. Positions and
gripper widths interpolate linearly as-is.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import torch


class ActionSlowdown:
    """Resample an action chunk to ``1/factor`` of its original speed.

    Args:
        factor: slowdown ``k``. ``k > 1`` is slower (the useful direction);
            ``k == 1`` is a no-op. ``k < 1`` speeds up and needs actions beyond
            the fetched chunk — those targets are clamped to the last action and
            flagged in ``<key>_is_pad``.
        key: the chunked feature to resample.
        angle_dims: indices along the action dimension holding angles in radians
            (unwrapped over time before interpolation).
    """

    def __init__(
        self,
        factor: float,
        *,
        key: str = "action",
        angle_dims: Sequence[int] = (),
    ) -> None:
        if factor <= 0:
            raise ValueError(f"slowdown factor must be > 0, got {factor}")
        self.factor = float(factor)
        self.key = key
        self.angle_dims = list(angle_dims)

    def resample_indices(self, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(lo, hi, w)`` for target j sampling the source at ``j / factor``."""
        pos = torch.arange(horizon, dtype=torch.float64) / self.factor
        pos = pos.clamp(max=horizon - 1)  # factor < 1: beyond the fetched chunk
        lo = pos.floor().to(torch.long)
        hi = pos.ceil().to(torch.long)
        return lo, hi, pos - lo

    def resample_values(self, actions: torch.Tensor) -> torch.Tensor:
        """Interpolate a ``(T, D)`` action chunk to ``1/factor`` speed."""
        if self.factor == 1.0 or not isinstance(actions, torch.Tensor) or actions.ndim != 2:
            return actions
        lo, hi, w = self.resample_indices(actions.shape[0])
        src = self._unwrap(actions.to(torch.float64))
        out = src[lo] * (1.0 - w.unsqueeze(1)) + src[hi] * w.unsqueeze(1)
        return self._wrap(out).to(actions.dtype)

    def resample_pad(self, pad: torch.Tensor) -> torch.Tensor:
        """A target is padding if either source frame it blends is padding."""
        if self.factor == 1.0 or not isinstance(pad, torch.Tensor) or pad.ndim != 1:
            return pad
        lo, hi, _ = self.resample_indices(pad.shape[0])
        return pad[lo] | pad[hi]

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        actions = item.get(self.key)
        if self.factor == 1.0 or not isinstance(actions, torch.Tensor) or actions.ndim != 2:
            return item  # no chunk (or nothing to do) -> leave the item alone

        item[self.key] = self.resample_values(actions)
        pad_key = f"{self.key}_is_pad"
        pad = item.get(pad_key)
        if isinstance(pad, torch.Tensor):
            item[pad_key] = self.resample_pad(pad)
        return item

    def _unwrap(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.angle_dims:
            return actions
        actions = actions.clone()
        for d in self.angle_dims:
            a = actions[:, d]
            steps = a[1:] - a[:-1]
            steps = steps - 2 * torch.pi * torch.round(steps / (2 * torch.pi))
            actions[:, d] = torch.cat([a[:1], a[:1] + torch.cumsum(steps, dim=0)])
        return actions

    def _wrap(self, actions: torch.Tensor) -> torch.Tensor:
        for d in self.angle_dims:
            actions[:, d] = (actions[:, d] + torch.pi) % (2 * torch.pi) - torch.pi
        return actions

    def identity(self) -> str:
        return (
            f"ActionSlowdown(factor={self.factor}, key={self.key!r}, "
            f"angle_dims={self.angle_dims})"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.identity()


def build_action_slowdown(spec: str | float | dict[str, Any]) -> ActionSlowdown:
    """Build an :class:`ActionSlowdown` from a bare factor or a JSON spec.

    Examples::

        "2"
        "2.5"
        '{"factor": 2, "angle_dims": [3, 4, 5, 10, 11, 12]}'
    """
    if isinstance(spec, str):
        spec = spec.strip()
        parsed: Any = json.loads(spec) if spec.startswith("{") else float(spec)
    else:
        parsed = spec
    if isinstance(parsed, (int, float)):
        return ActionSlowdown(float(parsed))
    parsed = dict(parsed)
    return ActionSlowdown(parsed.pop("factor"), **parsed)
