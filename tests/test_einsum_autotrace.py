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


def test_einsum_trace_keeps_generated_names_unique_across_multiple_calls() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))
    a1 = torch.ones((2, 3))
    a2 = torch.ones((3, 2))
    x1 = torch.ones((3,))

    trace.bind("r0", a0)
    trace.bind("t0", x0)
    first = tv.einsum("pa,p->a", a0, x0, trace=trace)
    second = tv.einsum("ap,a->p", a1, first, trace=trace)
    tv.einsum("pa,p->a", a2, x1, trace=trace)

    pairs = list(trace)

    assert [pair.result_name for pair in pairs] == ["r1", "r2", "r3"]
    assert pairs[0].left_name == "r0"
    assert pairs[0].right_name == "t0"
    assert pairs[1].right_name == "r1"
    assert tuple(second.shape) == (3,)


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

    # Shape mismatch is rejected by NumPy validation before the backend runs.
    with pytest.raises(ValueError, match="Invalid einsum equation"):
        tv.einsum("pa,p->a", a0, x0, trace=trace)
    assert len(trace) == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda trace, a, b, c: tv.einsum("...ij,...jk", a, b, trace=trace),
            "explicit output subscript",
        ),
        (
            lambda trace, a, b, c: tv.einsum("ab,b->a", trace=trace),
            "at least one operand",
        ),
    ],
)
def test_einsum_trace_rejects_unsupported_traced_calls(call, message: str) -> None:
    trace = tv.EinsumTrace()
    a = torch.ones((2, 3, 4))
    b = torch.ones((2, 4, 5))
    c = torch.ones((2, 2))

    with pytest.raises((TypeError, ValueError), match=message):
        call(trace, a, b, c)
    assert len(trace) == 0


def test_einsum_trace_rejects_mismatched_out_shape_numpy() -> None:
    trace = tv.EinsumTrace()
    a = np.ones((2, 2))
    b = np.ones((2,))
    with pytest.raises(ValueError, match="does not match"):
        tv.einsum("ab,b->a", a, b, trace=trace, backend="numpy", out=np.empty((3,)))
    assert len(trace) == 0


def test_einsum_trace_implicit_binary_stores_canonical_equation() -> None:
    trace = tv.EinsumTrace()
    a = torch.ones((2, 2))
    b = torch.ones((2,))
    tv.einsum("ab,b", a, b, trace=trace)
    pair = list(trace)[0]
    assert str(pair) == "ab,b->a"
    assert pair.equation == "ab,b->a"


def test_einsum_trace_out_kw_torch_matches_reference() -> None:
    trace = tv.EinsumTrace()
    a = torch.ones((2, 2))
    b = torch.ones((2,))
    out = torch.empty((2,))
    result = tv.einsum("ij,j->i", a, b, trace=trace, out=out)
    assert result is out
    expected = torch.einsum("ij,j->i", a, b)
    assert torch.allclose(out, expected)
    assert list(trace)[0].equation == "ij,j->i"


def test_einsum_trace_out_kw_implicit_equation() -> None:
    trace = tv.EinsumTrace()
    a = torch.ones((2, 2))
    b = torch.ones((2,))
    out = torch.empty((2,))
    tv.einsum("ab,b", a, b, trace=trace, out=out)
    assert str(list(trace)[0]) == "ab,b->a"


def test_einsum_trace_rejects_out_tensor_already_on_trace() -> None:
    trace = tv.EinsumTrace()
    a0 = torch.ones((3, 2))
    x0 = torch.ones((3,))
    r0 = tv.einsum("pa,p->a", a0, x0, trace=trace)
    a1 = torch.ones((2, 3))
    with pytest.raises(ValueError, match="already on this trace"):
        tv.einsum("ap,a->p", a1, r0, trace=trace, out=r0)
    assert len(trace) == 1


def test_einsum_trace_unary_and_ternary_use_einsum_trace_step() -> None:
    trace = tv.EinsumTrace()
    m = torch.ones(4, 4)
    tv.einsum("ii->i", m, trace=trace)
    step0 = list(trace)[0]
    assert isinstance(step0, tv.einsum_trace_step)
    assert step0.equation == "ii->i"
    assert step0.operand_names == ("t0",)

    A = torch.ones(2, 3)
    B = torch.ones(3, 4)
    C = torch.ones(4, 5)
    tv.einsum("ab,bc,cd->ad", A, B, C, trace=trace)
    step1 = list(trace)[1]
    assert isinstance(step1, tv.einsum_trace_step)
    assert step1.equation == "ab,bc,cd->ad"
    assert len(step1.operand_names) == 3


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
