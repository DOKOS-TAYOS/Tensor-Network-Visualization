"""Helpers for restoring import state after code paths with optional side-effect imports."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

_MISSING: Final = object()


@contextmanager
def _preserve_sys_module_entry(module_name: str) -> Iterator[None]:
    """Restore the original import binding for *module_name* after the wrapped block."""
    previous: Any = sys.modules.get(module_name, _MISSING)
    parent_name, _, attr_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name, _MISSING) if parent_name else _MISSING
    previous_attr: Any = _MISSING
    if parent_module is not _MISSING:
        previous_attr = getattr(parent_module, attr_name, _MISSING)
    try:
        yield
    finally:
        if previous is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        if parent_module is not _MISSING:
            if previous_attr is _MISSING:
                if hasattr(parent_module, attr_name):
                    delattr(parent_module, attr_name)
            else:
                setattr(parent_module, attr_name, previous_attr)


__all__ = ["_preserve_sys_module_entry"]
