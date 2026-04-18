# Backend Examples

This page collects copy-paste examples for each supported backend. Install the corresponding extra
first; see [installation.md](installation.md) for setup commands.

If you are working in a notebook and want the embedded Matplotlib controls to stay interactive,
install the `jupyter` extra and enable:

```python
%pip install "tensor-network-visualization[jupyter]"
%matplotlib widget
```

If you just installed the extra in the current kernel, restart the kernel once first. The same
backend examples below can then be used unchanged inside the notebook.

## Contents

- [Base Dependency Example](#base-dependency-example)
- [TensorKrowch](#tensorkrowch)
- [TensorNetwork](#tensornetwork)
- [Quimb](#quimb)
- [TeNPy](#tenpy)
- [`einsum`](#einsum)
- [Cross-Backend Translation](#cross-backend-translation)
- [Where to Go Next](#where-to-go-next)

## Base Dependency Example

`EinsumTrace` can be used with NumPy and the base package install.

```python
import numpy as np
from tensor_network_viz import EinsumTrace, TensorElementsConfig, einsum, show_tensor_elements

trace = EinsumTrace()
a = np.arange(6, dtype=float).reshape(2, 3)
x = np.array([1.0, -0.5, 0.25], dtype=float)

trace.bind("A", a)
trace.bind("x", x)
einsum("ab,b->a", a, x, trace=trace, backend="numpy")

fig, ax = show_tensor_elements(
    trace,
    config=TensorElementsConfig(mode="auto", robust_percentiles=(1.0, 99.0)),
    show=False,
)
fig.savefig("tensor-elements.png", bbox_inches="tight")
```

## TensorKrowch

Install:

```bash
python -m pip install "tensor-network-visualization[tensorkrowch]"
```

Minimal network:

```python
import tensorkrowch as tk
from tensor_network_viz import PlotConfig, show_tensor_network

network = tk.TensorNetwork(name="demo")
left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
left["b"] ^ right["b"]

fig, ax = show_tensor_network(
    network,
    engine="tensorkrowch",
    config=PlotConfig(show_tensor_labels=True, show_index_labels=False),
)
```

Recovered contraction history:

```python
config = PlotConfig(
    show_contraction_scheme=True,
    contraction_scheme_cost_hover=True,
    contraction_tensor_inspector=True,
)

fig, ax = show_tensor_network(contracted_network, engine="tensorkrowch", config=config)
```

This works only when the contracted TensorKrowch network still preserves usable `leaf_nodes` and
`resultant_nodes` history. If the history is unavailable, the library avoids guessing.

TensorKrowch nodes can also be passed as a nested 2D/3D grid to fix their visual positions:

```python
grid = [
    [left, right],
    [None, bottom],
]

fig, ax = show_tensor_network(grid, engine="tensorkrowch", view="2d")
```

That grid is only a layout hint; bonds still come from the nodes' own edges. If you need automatic
recovered contraction playback, pass the original TensorKrowch network object instead and use
`PlotConfig(positions={id(node): (...)})` for manual placement.

## TensorNetwork

Install:

```bash
python -m pip install "tensor-network-visualization[tensornetwork]"
```

Example:

```python
import numpy as np
import tensornetwork as tn
from tensor_network_viz import PlotConfig, show_tensor_network

left = tn.Node(np.ones((2, 2)), name="L", axis_names=("a", "b"))
right = tn.Node(np.ones((2, 2)), name="R", axis_names=("b", "c"))
left["b"] ^ right["b"]

fig, ax = show_tensor_network(
    [left, right],
    engine="tensornetwork",
    config=PlotConfig(show_tensor_labels=True, hover_labels=True),
)
```

Grid layout:

```python
import numpy as np
import tensornetwork as tn

left = tn.Node(np.ones((2, 2)), name="L", axis_names=("a", "b"))
right = tn.Node(np.ones((2, 2)), name="R", axis_names=("b", "c"))
bottom = tn.Node(np.ones((2, 2)), name="B", axis_names=("c", "d"))
left["b"] ^ right["b"]
right["c"] ^ bottom["c"]

grid = [
    [left, right],
    [None, bottom],
]

fig, ax = show_tensor_network(grid, engine="tensornetwork", view="2d")
```

## Quimb

Install:

```bash
python -m pip install "tensor-network-visualization[quimb]"
```

Example:

```python
import numpy as np
import quimb.tensor as qtn
from tensor_network_viz import PlotConfig, show_tensor_network

tensors = [
    qtn.Tensor(np.ones((2, 3)), inds=("i0", "b0"), tags={"T0"}),
    qtn.Tensor(np.ones((3, 2)), inds=("b0", "i1"), tags={"T1"}),
]
network = qtn.TensorNetwork(tensors)

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    config=PlotConfig(show_tensor_labels=True),
)
```

Snapshot export:

```python
from tensor_network_viz import export_tensor_network_snapshot

snapshot = export_tensor_network_snapshot(network, engine="quimb", view="2d")
payload = snapshot.to_dict()
```

## TeNPy

Install:

```bash
python -m pip install "tensor-network-visualization[tenpy]"
```

Native finite MPS:

```python
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tensor_network_viz import PlotConfig, show_tensor_network

sites = [SpinHalfSite() for _ in range(4)]
mps = MPS.from_product_state(sites, ["up", "up", "up", "up"], bc="finite")

fig, ax = show_tensor_network(
    mps,
    engine="tenpy",
    config=PlotConfig(show_tensor_labels=True),
)
```

Static export:

```python
fig, ax = show_tensor_network(
    mps,
    engine="tenpy",
    config=PlotConfig(show_tensor_labels=True),
    show_controls=False,
    show=False,
)
fig.savefig("tenpy_mps.png", bbox_inches="tight")
```

Use `make_tenpy_tensor_network(...)` when you already have named TeNPy `npc.Array` tensors and want
to provide the bond metadata explicitly.

For momentum-style chains, the repository demo keeps a lightweight compatibility fallback:

```bash
python examples/run_demo.py tenpy excitation --view 2d --no-show
```

Direct `MomentumMPS` objects work when your installed TeNPy release is compatible with the NumPy
version in your environment. Some TeNPy releases still construct `MomentumMPS` through
`numpy.find_common_type`, which NumPy 2 removed. If that combination fails, use the `excitation`
demo above or pin a NumPy release compatible with your TeNPy build.

## `einsum`

NumPy-backed traces work with the base package. Install the `einsum` extra when you need PyTorch:

```bash
python -m pip install "tensor-network-visualization[einsum]"
```

Manual NumPy trace:

```python
import numpy as np
from tensor_network_viz import EinsumTrace, PlotConfig, einsum, show_tensor_network

trace = EinsumTrace()
a = np.ones((2, 3), dtype=float)
b = np.ones((3, 4), dtype=float)

trace.bind("A", a)
trace.bind("B", b)
einsum("ab,bc->ac", a, b, trace=trace, backend="numpy")

fig, ax = show_tensor_network(
    trace,
    engine="einsum",
    config=PlotConfig(
        show_tensor_labels=True,
        show_contraction_scheme=True,
        contraction_scheme_cost_hover=True,
    ),
)
```

Manual step list with `pair_tensor(...)`:

```python
from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network

manual_trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
    pair_tensor("r1", "x1", "out", "pb,p->b"),
]

fig, ax = show_tensor_network(
    manual_trace,
    engine="einsum",
    config=PlotConfig(
        show_tensor_labels=True,
        show_contraction_scheme=True,
        contraction_scheme_cost_hover=True,
    ),
)
```

Use this form when you want to describe a known binary contraction order explicitly without routing
the contraction through `EinsumTrace`. A manual `pair_tensor(...)` trace drives the graph and the
playback order, but it does not carry live tensor values, so tensor-value inspection still needs an
`EinsumTrace` with bound arrays. For unary or n-ary manual steps, use `einsum_trace_step(...)`.

Tensor inspection from the `EinsumTrace` example above:

```python
from tensor_network_viz import TensorElementsConfig, show_tensor_elements

fig, ax = show_tensor_elements(
    trace,
    engine="einsum",
    config=TensorElementsConfig(mode="magnitude"),
)
```

## Cross-Backend Translation

If you want code for another backend instead of drawing the current object directly, use
`translate_tensor_network(...)`.

```python
from tensor_network_viz import translate_tensor_network

code = translate_tensor_network(
    network,
    engine="quimb",
    target_engine="tensornetwork",
    path="translated_network.py",
)
print(code)
```

This is useful when you want to:

- inspect the same structure in another engine,
- generate a small reproducible script for a collaborator,
- compare the original and translated networks with the repository demo.

Current scope:

- supported targets are `tensornetwork`, `quimb`, `einsum`, and `tensorkrowch`,
- `tenpy` is currently source-only,
- `tensorkrowch` exports reject disconnected structures that would need an outer product.

For a ready-made end-to-end example, run:

```bash
python examples/translate_demo.py --source-engine quimb --target-engine tensornetwork --example mps
```

## Where to Go Next

- [api.md](api.md): exact public API names and configuration fields.
- [guide.md](guide.md): workflows, layouts, exports, tensor inspection, and performance tips.
- [troubleshooting.md](troubleshooting.md): fixes for common install, backend, and notebook issues.
- [../examples/README.md](../examples/README.md): repository demo launcher commands.
