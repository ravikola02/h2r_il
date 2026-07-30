"""Annotation-driven frame manipulations ("inpainting methods"), applied on the fly
at dataloader time via LeRobot's ``set_image_transforms`` hook.

Public API:

* :class:`InpaintingMethod` — base class for a cache-backed per-frame manipulation.
* :func:`register_inpainting_method`, :func:`build_inpainting_method`,
  :func:`available_methods` — the registry.
* :class:`InpaintingPipeline`, :func:`build_pipeline` — chain methods.

Importing this package registers the built-in methods (arm inpainting, ...).
"""

from .base import (
    InpaintingMethod,
    InpaintingPipeline,
    available_methods,
    build_pipeline,
    build_inpainting_method,
    default_cache_dir,
    register_inpainting_method,
    inpainting_method_class,
)

# Importing method modules registers them via @register_inpainting_method.
from . import arm_inpaint  # noqa: F401,E402  (side-effect: registration)

__all__ = [
    "InpaintingMethod",
    "InpaintingPipeline",
    "available_methods",
    "build_pipeline",
    "build_inpainting_method",
    "default_cache_dir",
    "register_inpainting_method",
    "inpainting_method_class",
]
