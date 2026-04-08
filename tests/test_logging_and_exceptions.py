from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pytest

from tensor_network_viz import show_tensor_elements, show_tensor_network
from tensor_network_viz.einsum_module._backend import _load_backend_einsum


def test_package_logger_uses_null_handler() -> None:
    tensor_logger = logging.getLogger("tensor_network_viz")

    assert any(isinstance(handler, logging.NullHandler) for handler in tensor_logger.handlers)


def test_public_api_exports_exception_types() -> None:
    package = importlib.import_module("tensor_network_viz")

    assert hasattr(package, "TensorNetworkVizError")
    assert hasattr(package, "VisualizationInputError")
    assert hasattr(package, "UnsupportedEngineError")
    assert hasattr(package, "AxisConfigurationError")
    assert hasattr(package, "TensorDataError")
    assert hasattr(package, "MissingOptionalDependencyError")


def test_show_tensor_network_invalid_view_raises_axis_configuration_error() -> None:
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
    right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
    left["b"] ^ right["b"]

    with pytest.raises(Exception, match="Unsupported tensor network view") as exc_info:
        show_tensor_network(
            network,
            engine="tensorkrowch",
            view="invalid",  # type: ignore[arg-type]
            show=False,
        )

    assert exc_info.type.__name__ == "AxisConfigurationError"
    assert isinstance(exc_info.value, ValueError)


def test_show_tensor_network_invalid_engine_raises_unsupported_engine_error() -> None:
    tk = pytest.importorskip("tensorkrowch")
    network = tk.TensorNetwork(name="test")
    tk.Node(shape=(2,), axes_names=("a",), name="N", network=network)

    with pytest.raises(Exception, match="Unsupported tensor network engine") as exc_info:
        show_tensor_network(
            network,
            engine="unknown_engine",  # type: ignore[arg-type]
            view="2d",
            show=False,
        )

    assert exc_info.type.__name__ == "UnsupportedEngineError"
    assert isinstance(exc_info.value, ValueError)


def test_show_tensor_elements_invalid_input_raises_tensor_data_error() -> None:
    with pytest.raises(Exception, match="Could not infer tensor engine") as exc_info:
        show_tensor_elements(object(), show=False)

    assert exc_info.type.__name__ == "TensorDataError"
    assert isinstance(exc_info.value, ValueError)


class _ExplodingArrayLike:
    shape = (2,)

    def __array__(self) -> Any:
        raise RuntimeError("array boom")


@pytest.mark.parametrize("attr_name", ["nodes", "leaf_nodes", "tensors"])
def test_show_tensor_elements_ambiguous_empty_container_raises_inference_error(
    attr_name: str,
) -> None:
    payload = SimpleNamespace(**{attr_name: []})

    with pytest.raises(Exception, match="Could not infer tensor engine") as exc_info:
        show_tensor_elements(payload, show=False)

    assert exc_info.type.__name__ == "TensorDataError"
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.parametrize("attr_name", ["nodes", "leaf_nodes", "tensors"])
def test_show_tensor_network_ambiguous_empty_container_raises_inference_error(
    attr_name: str,
) -> None:
    payload = SimpleNamespace(**{attr_name: []})

    with pytest.raises(Exception, match="Could not infer tensor network engine") as exc_info:
        show_tensor_network(payload, show=False, show_controls=False)

    assert exc_info.type.__name__ == "VisualizationInputError"
    assert isinstance(exc_info.value, ValueError)


def test_show_tensor_elements_array_like_conversion_error_raises_tensor_data_type_error() -> None:
    with pytest.raises(Exception, match="array boom") as exc_info:
        show_tensor_elements(_ExplodingArrayLike(), show=False, show_controls=False)

    assert exc_info.type.__name__ == "TensorDataTypeError"
    assert isinstance(exc_info.value, TypeError)


def test_load_backend_einsum_wraps_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ImportError("torch missing in test")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(Exception, match="torch is required") as exc_info:
        _load_backend_einsum("torch")

    assert exc_info.type.__name__ == "MissingOptionalDependencyError"
    assert isinstance(exc_info.value, ImportError)


def test_backend_loader_emits_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="tensor_network_viz")

    _load_backend_einsum("numpy")

    assert any(
        record.name == "tensor_network_viz" and "backend='numpy'" in record.getMessage()
        for record in caplog.records
    )


def test_show_tensor_elements_external_axis_for_multiple_records_raises_axis_error() -> None:
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    tensors = [
        SimpleNamespace(
            tensor=[[1.0, 2.0], [3.0, 4.0]],
            name="A",
            axis_names=("row", "col"),
        ),
        SimpleNamespace(
            tensor=[[5.0, 6.0], [7.0, 8.0]],
            name="B",
            axis_names=("row", "col"),
        ),
    ]

    with pytest.raises(Exception, match="explicit ax is only supported") as exc_info:
        show_tensor_elements(tensors, show=False, ax=ax)

    assert exc_info.type.__name__ == "AxisConfigurationError"
    assert isinstance(exc_info.value, ValueError)
