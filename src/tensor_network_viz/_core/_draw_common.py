"""Shared scale and style parameters for 2D and 3D drawing."""

from __future__ import annotations

from . import draw as _draw

__all__ = _draw.__all__

globals().update({name: getattr(_draw, name) for name in __all__})
