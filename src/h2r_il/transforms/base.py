"""VisualMethod: pluggable, cache-backed per-frame image manipulations.

A :class:`VisualMethod` is a callable that LeRobot installs on a dataset via
``dataset.set_image_transforms(...)``. LeRobot hands the callable **one camera
frame at a time** as a CHW tensor (uint8 or float in [0, 1], RGB) — see
``lerobot.datasets.dataset_reader`` (``item[cam] = self._image_transforms(item[cam])``).

Subclasses implement :meth:`apply` on an ``HxWx3`` uint8 RGB numpy array and
return the same. The base class handles:

* tensor/ndarray/PIL <-> uint8-RGB-HWC conversion, preserving the input's
  dtype, layout (CHW/HWC), value range, and device on the way out;
* an optional temporal ``(T, C, H, W)`` stack (delta-timestamp observations);
* a content-addressed disk cache so the (expensive) models behind a method run
  **once per unique frame** — the first pass populates the cache, training then
  just reads it. The cache lives outside the dataset, so datasets stay read-only.

Nothing here is specific to arm inpainting; new visual methods (masking, virtual
gripper rendering, ...) subclass :class:`VisualMethod` and register themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type["VisualMethod"]] = {}


def register_visual_method(name: str) -> Callable[[type["VisualMethod"]], type["VisualMethod"]]:
    """Class decorator: register a :class:`VisualMethod` under ``name``."""

    def _decorator(cls: type["VisualMethod"]) -> type["VisualMethod"]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Visual method {name!r} already registered to {_REGISTRY[name]}")
        cls.method_name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def available_methods() -> list[str]:
    """Names of all registered visual methods."""
    return sorted(_REGISTRY)


def build_visual_method(name: str, **kwargs: Any) -> "VisualMethod":
    """Instantiate a registered visual method by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown visual method {name!r}. Available: {available_methods()}")
    return _REGISTRY[name](**kwargs)


def visual_method_class(name: str) -> type["VisualMethod"]:
    """The registered class for ``name`` (e.g. to inspect its constructor)."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown visual method {name!r}. Available: {available_methods()}")
    return _REGISTRY[name]


def default_cache_dir() -> Path:
    """Where computed frames are cached (outside any dataset).

    Overridable with ``H2R_VISUAL_CACHE``. Defaults to ``outputs/visual_cache``
    (``outputs/`` is a symlink to shared storage in this repo).
    """
    env = os.environ.get("H2R_VISUAL_CACHE")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "outputs" / "visual_cache"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class VisualMethod(ABC):
    """A cache-backed, per-frame image manipulation.

    Args:
        cache: enable the on-disk content-addressed cache.
        cache_dir: override the cache root (default :func:`default_cache_dir`).
    """

    #: set by :func:`register_visual_method`
    method_name: str = "visual_method"

    def __init__(self, *, cache: bool = True, cache_dir: str | os.PathLike | None = None) -> None:
        self.cache = cache
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    # --- subclass API ------------------------------------------------------

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform an ``HxWx3`` uint8 RGB image, returning the same shape/dtype."""

    def config_signature(self) -> dict[str, Any]:
        """JSON-serializable knobs that change the output.

        Included in the cache key and in human-readable labels so two configs of
        the same method never collide in the cache. Subclasses should override.
        """
        return {}

    def visualize(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Return named images for inspection (used by the viz script).

        Always includes ``"result"``. Subclasses may add intermediates (masks,
        overlays, ...); the viz tool renders whatever is returned, so new methods
        get inspection for free.
        """
        return {"result": self.apply(image)}

    # --- identity / caching ------------------------------------------------

    def identity(self) -> str:
        """Stable string identifying this method + its config."""
        cfg = json.dumps(self.config_signature(), sort_keys=True, default=str)
        return f"{self.method_name}({cfg})"

    def _cache_path(self, image: np.ndarray) -> Path:
        h = hashlib.blake2b(digest_size=20)
        h.update(self.identity().encode())
        h.update(str(image.shape).encode())
        h.update(np.ascontiguousarray(image).tobytes())
        key = h.hexdigest()
        return self._cache_dir / self.method_name / key[:2] / f"{key}.png"

    def _apply_cached(self, image: np.ndarray) -> np.ndarray:
        if not self.cache:
            return self.apply(image)
        path = self._cache_path(image)
        if path.exists():
            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))
        result = self.apply(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.png")
        Image.fromarray(result).save(tmp)
        os.replace(tmp, path)  # atomic; safe across worker processes
        return result

    # --- callable interface (what LeRobot invokes) -------------------------

    def __call__(self, x: Any) -> Any:
        frames, restore = _to_uint8_rgb_frames(x)
        out = [self._apply_cached(f) for f in frames]
        return restore(out)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.identity()


# ---------------------------------------------------------------------------
# Pipeline (chain several methods, then any existing transform)
# ---------------------------------------------------------------------------


class VisualMethodPipeline:
    """Apply a list of :class:`VisualMethod` in order, then an optional ``tail``.

    ``tail`` is any existing transform (e.g. LeRobot's photometric
    ``ImageTransforms``); it runs *after* the visual methods so the deterministic,
    cacheable manipulations see the raw frame and random augmentation is layered
    on top.
    """

    def __init__(self, methods: list[VisualMethod], tail: Callable | None = None) -> None:
        self.methods = methods
        self.tail = tail

    def __call__(self, x: Any) -> Any:
        for m in self.methods:
            x = m(x)
        if self.tail is not None:
            x = self.tail(x)
        return x

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = [m.identity() for m in self.methods]
        if self.tail is not None:
            parts.append(f"tail={self.tail!r}")
        return "VisualMethodPipeline([" + ", ".join(parts) + "])"


def build_pipeline(spec: str | list[dict[str, Any]], *, tail: Callable | None = None,
                   cache_dir: str | os.PathLike | None = None) -> VisualMethodPipeline:
    """Build a pipeline from a spec.

    ``spec`` is either a JSON string or a list of dicts. Each entry is either a
    bare method name (string) or ``{"name": <str>, **kwargs}``. Examples::

        "arm_inpaint"
        "[\"arm_inpaint\"]"
        [{"name": "arm_inpaint", "prompt": "arm. hand.", "dilation": 21}]
    """
    if isinstance(spec, str):
        spec = spec.strip()
        parsed = json.loads(spec) if spec.startswith(("[", "{")) else [spec]
    else:
        parsed = spec
    if isinstance(parsed, (str, dict)):
        parsed = [parsed]

    methods: list[VisualMethod] = []
    for entry in parsed:
        if isinstance(entry, str):
            name, kwargs = entry, {}
        else:
            entry = dict(entry)
            name = entry.pop("name")
            kwargs = entry
        if cache_dir is not None:
            kwargs.setdefault("cache_dir", cache_dir)
        methods.append(build_visual_method(name, **kwargs))
    return VisualMethodPipeline(methods, tail=tail)


# ---------------------------------------------------------------------------
# Tensor <-> uint8-RGB-HWC conversion
# ---------------------------------------------------------------------------


def _to_uint8_rgb_frames(x: Any) -> tuple[list[np.ndarray], Callable[[list[np.ndarray]], Any]]:
    """Normalize a LeRobot frame to a list of ``HxWx3`` uint8 RGB arrays.

    Returns the frames plus a ``restore`` closure that maps a list of processed
    ``HxWx3`` uint8 arrays back to the original type/layout/dtype/range/device.
    Handles: torch CHW / (T,C,H,W) / HWC, numpy, and PIL; uint8 or float[0,1].
    """
    # PIL
    if isinstance(x, Image.Image):
        arr = np.asarray(x.convert("RGB"))
        return [arr], lambda outs: Image.fromarray(outs[0])

    # numpy -> route through a tensor-like path for uniform handling
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        frames, restore_t = _tensor_to_frames(t)
        return frames, lambda outs: restore_t(outs).numpy()

    if isinstance(x, torch.Tensor):
        return _tensor_to_frames(x)

    raise TypeError(f"VisualMethod received unsupported frame type: {type(x)}")


def _tensor_to_frames(t: torch.Tensor) -> tuple[list[np.ndarray], Callable[[list[np.ndarray]], torch.Tensor]]:
    device, dtype = t.device, t.dtype
    is_float = torch.is_floating_point(t)

    # Identify layout -> produce (T, H, W, C) uint8 and remember how to invert.
    if t.ndim == 4:  # (T, C, H, W)
        layout = "tchw"
        thwc = t.permute(0, 2, 3, 1)
    elif t.ndim == 3 and t.shape[0] in (1, 3):  # (C, H, W)
        layout = "chw"
        thwc = t.permute(1, 2, 0).unsqueeze(0)
    elif t.ndim == 3 and t.shape[-1] in (1, 3):  # (H, W, C)
        layout = "hwc"
        thwc = t.unsqueeze(0)
    else:
        raise ValueError(f"Cannot interpret frame tensor of shape {tuple(t.shape)}")

    thwc_cpu = thwc.detach().to("cpu")
    if is_float:
        u8 = (thwc_cpu.clamp(0, 1) * 255.0).round().to(torch.uint8)
    else:
        u8 = thwc_cpu.to(torch.uint8)

    frames: list[np.ndarray] = []
    for f in u8:  # (H, W, C)
        arr = f.numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        frames.append(np.ascontiguousarray(arr))

    def restore(outs: list[np.ndarray]) -> torch.Tensor:
        stack = torch.from_numpy(np.stack(outs, axis=0))  # (T, H, W, 3) uint8
        if is_float:
            stack = stack.to(dtype) / 255.0
        else:
            stack = stack.to(dtype)
        if layout == "tchw":
            stack = stack.permute(0, 3, 1, 2)
        elif layout == "chw":
            stack = stack[0].permute(2, 0, 1)
        else:  # hwc
            stack = stack[0]
        return stack.to(device)

    return frames, restore
