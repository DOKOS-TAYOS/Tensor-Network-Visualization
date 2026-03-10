# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch and TensorNetwork tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## Features

- 2D and 3D plotting for TensorKrowch and TensorNetwork tensor networks
- Tensors rendered as nodes
- Contractions rendered as edges between tensors
- Dangling indices rendered as labeled stubs
- Self-contractions rendered as loops
- No UI; uses Matplotlib for rendering and NetworkX for graph layouts

## Installation

### As a dependency

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = ["tensor-network-visualization>=0.1.0"]
```

Or install with pip:

```bash
pip install tensor-network-visualization
```

If you also want to visualize `google/TensorNetwork` nodes, install `tensornetwork`
in your environment as well:

```bash
pip install tensornetwork
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

Each node must have `edges`, `axes_names` or `axis_names`, and `name`. Each edge must have `node1`, `node2`, and `name`.

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
```

You can also use engine-specific helpers directly:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d

plot_tensorkrowch_network_2d(network)   # or plot_tensorkrowch_network_2d([node1, node2, ...])
plot_tensorkrowch_network_3d(network)
plot_tensornetwork_network_2d([node1, node2, node3])
plot_tensornetwork_network_3d([node1, node2, node3])
```

## Internal architecture

The public API is split by backend, but the render pipeline is now shared:

- `tensor_network_viz._core` contains the normalized graph model, layout, curve geometry, drawing, and shared renderer.
- `tensor_network_viz.tensorkrowch` contains the TensorKrowch adapter that converts TensorKrowch inputs into the shared graph model.
- `tensor_network_viz.tensornetwork` contains the TensorNetwork adapter that converts TensorNetwork node collections into the shared graph model.

This means TensorKrowch and TensorNetwork are not converted into each other. Each backend normalizes its own input to the common `_GraphData` structure and the shared core handles the rest.

## Project layout

- `examples/` - Demo scripts. Run `python examples/tensorkrowch_demo.py mps 2d` for TensorKrowch or `python examples/tensornetwork_demo.py mps 2d` for TensorNetwork.
- `scripts/` - Utility scripts (for example `clean.py` to remove caches and build artifacts).

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```
