# Backend Examples

This page collects copy-paste examples for each supported backend.

## Tensor Inspection

### Base-dependency example with `EinsumTrace`

```python
import numpy as np
from tensor_network_viz import EinsumTrace, TensorElementsConfig, einsum, show_tensor_elements

trace = EinsumTrace()
a = np.arange(6, dtype=float).reshape(2, 3)
x = np.array([1.0, -0.5, 0.25], dtype=float)
trace.bind("A", a)
trace.bind("x", x)
r0 = einsum("ab,b->a", a, x, trace=trace, backend="numpy")

fig, ax = show_tensor_elements(
    trace,
    config=TensorElementsConfig(mode="auto", robust_percentiles=(1.0, 99.0)),
    show=False,
)
fig.savefig("tensor-elements.png", bbox_inches="tight")
```

## TensorKrowch

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
    config=PlotConfig(
        show_tensor_labels=True,
        show_index_labels=False,
    ),
)
```

If you pass a TensorKrowch network **after** real contractions have been performed and the network
still keeps usable `leaf_nodes` plus `resultant_nodes` history, `PlotConfig(show_contraction_scheme=True)`
can reconstruct the contraction steps automatically. This recovery is conservative: unusual
mutations or broken history simply disable the automatic scheme instead of guessing. When the
recovered steps still expose result tensors and metric data, the same playback extras used by
`einsum` also work here: `contraction_scheme_cost_hover=True` and
`contraction_tensor_inspector=True`.

## TensorNetwork

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
    config=PlotConfig(
        show_tensor_labels=True,
        hover_labels=True,
    ),
)
```

## Quimb

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
    show_controls=False,
    show=False,
)
fig.savefig("quimb_network.png", bbox_inches="tight")
```

## TeNPy

### Native finite MPS

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

### Static export

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

## `einsum`

### Manual trace

```python
from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network

trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
]

fig, ax = show_tensor_network(
    trace,
    engine="einsum",
    config=PlotConfig(
        show_contraction_scheme=True,
        show_tensor_labels=True,
    ),
)
```

### Traced execution

```python
from tensor_network_viz import EinsumTrace, PlotConfig, einsum, show_tensor_network
import torch

trace = EinsumTrace()
a = torch.ones((2, 3))
x = torch.ones((3,))
trace.bind("A", a)
trace.bind("x", x)
einsum("ab,b->a", a, x, trace=trace, backend="torch")

fig, ax = show_tensor_network(
    trace,
    engine="einsum",
    config=PlotConfig(show_contraction_scheme=True),
)
```

## Notes

- You can often omit `engine=...` because `show_tensor_network` auto-detects the backend.
- `show_tensor_elements(...)` auto-detects the same backends, but it needs real tensor values.
- Use `show_controls=False` when you want a clean saved figure with no embedded buttons/sliders.
- Use `PlotConfig(...)` for labels, hover behavior, contraction schemes, and performance-related
  rendering choices.
- Use `TensorElementsConfig(...)` for tensor-view mode, matrix grouping, downsampling, top-k data
  summaries, and optional robust/shared scaling controls.
- When multiple tensors are present, `show_tensor_elements(...)` shows one tensor at a time and
  adds a slider to move between them. Interactive views are grouped into `basic`, `complex`, and
  `diagnostic`, including `log_magnitude`, `sparsity`, and `nan_inf`.
- TensorKrowch shape-only nodes and manual `pair_tensor(...)` lists are intentionally unsupported
  for `show_tensor_elements(...)` because they do not expose inspectable tensor values.
