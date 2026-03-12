"""Backend loading and tensor metadata helpers for traced einsum."""

from __future__ import annotations

import importlib
import weakref
from typing import Any


def _load_backend_einsum(backend: str) -> Any:
    if backend == "torch":
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError("torch is required for backend='torch'.") from exc
        return torch.einsum
    if backend == "numpy":
        try:
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ImportError("numpy is required for backend='numpy'.") from exc
        return np.einsum
    raise ValueError(f"Unsupported einsum backend: {backend}")


def _make_weakref(tensor: Any) -> weakref.ReferenceType[Any]:
    try:
        return weakref.ref(tensor)
    except TypeError as exc:
        raise TypeError("Traced einsum operands must support weak references.") from exc


def _shape_tuple(tensor: Any) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except TypeError:
        return None


def _dtype_text(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)
