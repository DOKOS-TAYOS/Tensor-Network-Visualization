# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced PyTorch/NumPy `einsum` tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## Features

- 2D and 3D plotting for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced binary `einsum` tensor networks
- Tensors rendered as nodes
- Contractions rendered as edges between tensors
- Dangling indices rendered as labeled stubs
- Self-contractions rendered as loops
- No UI; uses Matplotlib for rendering and NetworkX for graph layouts

## Installation

### As a dependency

The base package provides the rendering core (Matplotlib, NetworkX). Install at least one backend extra for the engine you use:

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = ["tensor-network-visualization[tensorkrowch]"]  # or [tensornetwork], [quimb], [tenpy]
```

Or install with pip:

```bash
pip install "tensor-network-visualization[tensorkrowch]"
```

Other backends:

```bash
pip install "tensor-network-visualization[tensornetwork]"
pip install "tensor-network-visualization[quimb]"
pip install "tensor-network-visualization[tenpy]"
pip install "tensor-network-visualization[einsum]"
```

The `einsum` backend ships with the base package. Install the `[einsum]` extra for PyTorch, or use NumPy if already installed, to execute traced `einsum(...)` calls through the convenience wrapper.

### Local development

Inside the project virtual environment:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

For runtime-only (editable install without dev tools):

```powershell
.\.venv\Scripts\python -m pip install -e .
```

## Usage

### Input formats

Supported inputs depend on the selected engine:

- `engine="tensorkrowch"`:
  - a TensorKrowch network object with `nodes` or `leaf_nodes`
  - any iterable of TensorKrowch nodes, e.g. `[node1, node2, node3]`
- `engine="tensornetwork"`:
  - any iterable of `tensornetwork.Node`, e.g. `[node1, node2, node3]` or `{node1, node2}`
- `engine="quimb"`:
  - a Quimb `TensorNetwork`
  - any iterable of Quimb `Tensor`
- `engine="tenpy"`:
  - a finite, segment, or infinite `tenpy.networks.mps.MPS`
  - a finite or infinite `tenpy.networks.mpo.MPO`
- `engine="einsum"`:
  - an ordered iterable of `pair_tensor` entries representing binary explicit `left,right->out` einsums
  - an `EinsumTrace` containing auto-recorded binary explicit `left,right->out` einsums

Each backend consumes its native tensor-network objects directly and normalizes them to the shared graph model internally.

Quimb support includes hyper-indices shared by three or more tensors, rendered through invisible internal hub nodes so the original topology is preserved without exposing extra tensor markers. Infinite TeNPy `MPS`/`MPO` networks are rendered as a single unit cell closed periodically, so the repeating structure stays visible without introducing a separate infinite-edge primitive.
The `einsum` backend reconstructs the underlying tensor network from the traced contractions and only shows the fundamental tensors, not the intermediate einsum results.

When passing a subset of nodes, edges to nodes outside the input collection are drawn as dangling legs. Disconnected components (for example nodes from different networks) are supported.

### PlotConfig options

`PlotConfig` controls styling and layout. Notable options:

- `figsize`, `node_color`, `bond_edge_color`, `dangling_edge_color` – figure and color styling
- `node_radius`, `stub_length`, `self_loop_radius` – scale of nodes and stubs
- `line_width_2d`, `line_width_3d` – line thickness
- `positions` – custom node positions (dict mapping node id to `(x, y)` or `(x, y, z)`)
- `layout_iterations` – force-directed layout iterations (default 220; reduce for small networks, increase for large ones)
- `show_tensor_labels`, `show_index_labels` – toggle labels on nodes and edges

```python
import torch

from tensor_network_viz import EinsumTrace, PlotConfig, einsum, pair_tensor, show_tensor_network

config = PlotConfig(figsize=(8, 6))

# TensorKrowch network object
fig, ax = show_tensor_network(network, engine="tensorkrowch", view="2d", config=config)

# TensorKrowch node collection
fig, ax = show_tensor_network([node1, node2, node3], engine="tensorkrowch", view="2d", config=config)

# TensorNetwork node collection
fig, ax = show_tensor_network([node1, node2, node3], engine="tensornetwork", view="2d", config=config)

# Quimb TensorNetwork or tensor collection
fig, ax = show_tensor_network(quimb_network, engine="quimb", view="2d", config=config)

# TeNPy MPS or MPO
fig, ax = show_tensor_network(tenpy_network, engine="tenpy", view="2d", config=config)

# Auto-traced torch or numpy einsum contractions
trace = EinsumTrace()
trace.bind("A0", A0)
trace.bind("x0", x0)
trace.bind("A1", A1)
r0 = einsum("pa,p->a", A0, x0, trace=trace, backend="torch")
_ = einsum("a,apb->pb", r0, A1, trace=trace, backend="torch")
fig, ax = show_tensor_network(trace, engine="einsum", view="2d", config=config)

# Manual pair_tensor + torch.einsum tracing
trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
]
r0 = torch.einsum(trace[0], A0, x0)
_ = torch.einsum(trace[1], r0, A1)
fig, ax = show_tensor_network(trace, engine="einsum", view="2d", config=config)
```

You can also use engine-specific helpers directly:

```python
from tensor_network_viz import EinsumTrace, einsum
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d
from tensor_network_viz.einsum_module import plot_einsum_network_2d, plot_einsum_network_3d

plot_tensorkrowch_network_2d(network)   # or plot_tensorkrowch_network_2d([node1, node2, ...])
plot_tensorkrowch_network_3d(network)
plot_tensornetwork_network_2d([node1, node2, node3])
plot_tensornetwork_network_3d([node1, node2, node3])
plot_quimb_network_2d(quimb_network)
plot_quimb_network_3d(quimb_network)
plot_tenpy_network_2d(tenpy_network)
plot_tenpy_network_3d(tenpy_network)
trace = EinsumTrace()
_ = einsum("ab,b->a", A, x, trace=trace, backend="numpy", optimize=True)
plot_einsum_network_2d(trace)
plot_einsum_network_3d(trace)
```

## Internal architecture

The public API is split by backend, but the render pipeline is shared:

- `tensor_network_viz._core` – normalized graph model (`_GraphData`), layout, axis directions, curve geometry, drawing, and shared renderer factory.
- `tensor_network_viz._core._nodes_edges_common` – shared graph-building logic for TensorKrowch and TensorNetwork (both use nodes with `edges` and axis names).
- `tensor_network_viz._registry` – engine registry for lazy-loaded plotters.
- `tensor_network_viz.tensorkrowch` – TensorKrowch adapter (uses `axes_names`, `nodes`/`leaf_nodes`).
- `tensor_network_viz.tensornetwork` – TensorNetwork adapter (uses `axis_names`, iterable nodes).
- `tensor_network_viz.quimb` contains the Quimb adapter that converts `TensorNetwork` objects or tensor collections into the shared graph model.
- `tensor_network_viz.tenpy` contains the TeNPy adapter that converts finite, segment, and infinite `MPS` plus finite and infinite `MPO` objects into the shared graph model.
- `tensor_network_viz.einsum_module` contains the trace adapter that converts ordered `pair_tensor` contractions or `EinsumTrace` objects into the shared graph model.

This means backends are not converted into each other. Each backend normalizes its own input to the common `_GraphData` structure and the shared core handles the rest.

## Project layout

- `examples/` - Demo scripts. Run `python examples/tensorkrowch_demo.py mps 2d`, `python examples/tensornetwork_demo.py mps 2d`, `python examples/quimb_demo.py mps 2d`, `python examples/tenpy_demo.py mps 2d`, or `python examples/einsum_demo.py mps 2d`.
- `scripts/` - Utility scripts (for example `clean.py` to remove caches and build artifacts).

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```
