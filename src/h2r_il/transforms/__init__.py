"""Annotation-driven frame manipulations ("visual methods"), applied on the fly
at dataloader time via LeRobot's ``set_image_transforms`` hook.

Public API:

* :class:`VisualMethod` — base class for a cache-backed per-frame manipulation.
* :func:`register_visual_method`, :func:`build_visual_method`,
  :func:`available_methods` — the registry.
* :class:`VisualMethodPipeline`, :func:`build_pipeline` — chain methods.

Importing this package registers the built-in methods (arm inpainting, ...).
"""

from .base import (
    VisualMethod,
    VisualMethodPipeline,
    available_methods,
    build_pipeline,
    build_visual_method,
    default_cache_dir,
    register_visual_method,
    visual_method_class,
)

# Importing method modules registers them via @register_visual_method.
from . import arm_inpaint  # noqa: F401,E402  (side-effect: registration)

__all__ = [
    "VisualMethod",
    "VisualMethodPipeline",
    "available_methods",
    "build_pipeline",
    "build_visual_method",
    "default_cache_dir",
    "register_visual_method",
    "visual_method_class",
]
