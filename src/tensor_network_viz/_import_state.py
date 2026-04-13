"""Helpers for restoring import state after code paths with optional side-effect imports."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

_MISSING: Final = object()


@contextmanager
def _preserve_sys_module_entry(module_name: str) -> Iterator[None]:
    """Restore the original ``sys.modules`` entry for *module_name* after the wrapped block."""
    previous: Any = sys.modules.get(module_name, _MISSING)
    try:
        yield
    finally:
        if previous is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


__all__ = ["_preserve_sys_module_entry"]
