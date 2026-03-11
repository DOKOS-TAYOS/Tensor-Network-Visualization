"""Consolidated axis name to direction and offset sign mappings."""

from __future__ import annotations

_AXIS_DIR_2D: dict[str, tuple[float, float]] = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}

_AXIS_DIR_3D: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "north": (0.0, 0.0, 1.0),
    "south": (0.0, 0.0, -1.0),
    "east": (1.0, 0.0, 0.0),
    "west": (-1.0, 0.0, 0.0),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "in": (0.0, 1.0, 0.0),
    "out": (0.0, -1.0, 0.0),
}

_AXIS_OFFSET_SIGN: dict[str, int] = {
    "up": 1,
    "right": 1,
    "north": 1,
    "east": 1,
    "down": -1,
    "left": -1,
    "south": -1,
    "west": -1,
}
