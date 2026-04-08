from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _seed_from_name_and_shape(name: str, shape: tuple[int, ...]) -> int:
    key = f"{name}|{shape}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def build_demo_numpy_tensor(
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any = float,
) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(_seed_from_name_and_shape(name, shape))
    array = rng.normal(loc=0.0, scale=1.0, size=shape)
    return np.asarray(array, dtype=dtype)


def build_demo_torch_tensor(
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any | None = None,
) -> Any:
    import torch

    resolved_dtype = torch.float32 if dtype is None else dtype
    array = build_demo_numpy_tensor(name=name, shape=shape, dtype=np.float32)
    return torch.tensor(array, dtype=resolved_dtype)


__all__ = ["build_demo_numpy_tensor", "build_demo_torch_tensor"]
