from __future__ import annotations

from collections.abc import Generator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tensor_network_viz as tv

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, None, None]:
    yield
    plt.close("all")


def test_root_package_keeps_callable_einsum_after_backend_import() -> None:
    import tensor_network_viz.einsum_module as einsum_backend

    assert callable(tv.einsum)
    assert hasattr(einsum_backend, "plot_einsum_network_2d")


def test_einsum_trace_auto_registers_pairs_and_renders() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))
    a1 = torch.ones((2, 3, 4))

    r0 = tv.einsum("pa,p->a", a0, x0, trace=trace)
    r1 = tv.einsum("a,apb->pb", r0, a1, trace=trace)
    fig, ax = tv.show_tensor_network(
        trace,
        engine="einsum",
        view="2d",
        show=False,
    )
    pairs = list(trace)

    assert len(trace) == 2
    assert [pair.left_name for pair in pairs] == ["t0", "r0"]
    assert [pair.right_name for pair in pairs] == ["t1", "t2"]
    assert [pair.result_name for pair in pairs] == ["r0", "r1"]
    assert tuple(r0.shape) == (2,)
    assert tuple(r1.shape) == (3, 4)
    assert fig is ax.figure


def test_einsum_trace_bind_uses_human_names_before_first_use() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))

    trace.bind("A0", a0)
    trace.bind("x0", x0)
    tv.einsum("pa,p->a", a0, x0, trace=trace)

    pair = list(trace)[0]
    assert pair.left_name == "A0"
    assert pair.right_name == "x0"
    assert pair.result_name == "r0"


def test_einsum_trace_rejects_binding_after_tensor_was_traced() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))

    tv.einsum("pa,p->a", a0, x0, trace=trace)

    with pytest.raises(ValueError, match="already been traced"):
        trace.bind("A0", a0)


def test_einsum_trace_rejects_reuse_of_consumed_tensor() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))
    x1 = torch.ones((3,))

    tv.einsum("pa,p->a", a0, x0, trace=trace)

    with pytest.raises(ValueError, match="already been consumed"):
        tv.einsum("pa,p->a", a0, x1, trace=trace)
    assert len(trace) == 1


def test_einsum_trace_does_not_mutate_when_backend_call_fails() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((4,))

    with pytest.raises(RuntimeError):
        tv.einsum("pa,p->a", a0, x0, trace=trace)
    assert len(trace) == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda trace, a, b: tv.einsum("ab,b", a, b, trace=trace),
            "explicit output",
        ),
        (
            lambda trace, a, b: tv.einsum("ab,b->a", a, b, a, trace=trace),
            "exactly 2 operands",
        ),
        (
            lambda trace, a, b: tv.einsum(
                "ab,b->a",
                a,
                b,
                trace=trace,
                backend="numpy",
                out=np.empty((2,)),
            ),
            "out=",
        ),
    ],
)
def test_einsum_trace_rejects_unsupported_traced_calls(call, message: str) -> None:
    trace = tv.EinsumTrace()
    a = np.ones((2, 2))
    b = np.ones((2,))

    with pytest.raises((TypeError, ValueError), match=message):
        call(trace, a, b)
    assert len(trace) == 0


def test_einsum_wrapper_is_transparent_without_trace_for_torch_and_numpy() -> None:
    torch_left = torch.arange(6.0).reshape(2, 3)
    torch_right = torch.arange(3.0)
    numpy_left = np.arange(6.0).reshape(2, 3)
    numpy_right = np.arange(3.0)

    torch_result = tv.einsum("ij,j->i", torch_left, torch_right, backend="torch")
    numpy_result = tv.einsum(
        "ij,j->i",
        numpy_left,
        numpy_right,
        backend="numpy",
        optimize=True,
    )

    assert torch.allclose(torch_result, torch.einsum("ij,j->i", torch_left, torch_right))
    assert np.allclose(
        numpy_result,
        np.einsum("ij,j->i", numpy_left, numpy_right, optimize=True),
    )
