# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, and TeNPy tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## Features

- 2D and 3D plotting for TensorKrowch, TensorNetwork, Quimb, and TeNPy tensor networks
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
```

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
  - a finite or segment `tenpy.networks.mps.MPS`
  - a finite `tenpy.networks.mpo.MPO`

Each backend consumes its native tensor-network objects directly and normalizes them to the shared graph model internally.

Quimb support in v1 is limited to pairwise indices: indices shared by more than two tensors raise a clear error. TeNPy support in v1 is limited to finite or segment `MPS`/`MPO`; infinite networks are rejected.

When passing a subset of nodes, edges to nodes outside the input collection are drawn as dangling legs. Disconnected components (for example nodes from different networks) are supported.

```python
from tensor_network_viz import PlotConfig, show_tensor_network

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
```

You can also use engine-specific helpers directly:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d

plot_tensorkrowch_network_2d(network)   # or plot_tensorkrowch_network_2d([node1, node2, ...])
plot_tensorkrowch_network_3d(network)
plot_tensornetwork_network_2d([node1, node2, node3])
plot_tensornetwork_network_3d([node1, node2, node3])
plot_quimb_network_2d(quimb_network)
plot_quimb_network_3d(quimb_network)
plot_tenpy_network_2d(tenpy_network)
plot_tenpy_network_3d(tenpy_network)
```

## Internal architecture

The public API is split by backend, but the render pipeline is now shared:

- `tensor_network_viz._core` contains the normalized graph model, layout, curve geometry, drawing, and shared renderer.
- `tensor_network_viz.tensorkrowch` contains the TensorKrowch adapter that converts TensorKrowch inputs into the shared graph model.
- `tensor_network_viz.tensornetwork` contains the TensorNetwork adapter that converts TensorNetwork node collections into the shared graph model.
- `tensor_network_viz.quimb` contains the Quimb adapter that converts `TensorNetwork` objects or tensor collections into the shared graph model.
- `tensor_network_viz.tenpy` contains the TeNPy adapter that converts finite/segment `MPS` and finite `MPO` objects into the shared graph model.

This means backends are not converted into each other. Each backend normalizes its own input to the common `_GraphData` structure and the shared core handles the rest.

## Project layout

- `examples/` - Demo scripts. Run `python examples/tensorkrowch_demo.py mps 2d`, `python examples/tensornetwork_demo.py mps 2d`, `python examples/quimb_demo.py mps 2d`, or `python examples/tenpy_demo.py mps 2d`.
- `scripts/` - Utility scripts (for example `clean.py` to remove caches and build artifacts).

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```
