# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## Features

- 2D and 3D plotting for TensorKrowch tensor networks
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

Networks must expose `nodes` or `leaf_nodes` (iterable or dict). Each node must have `edges`, `axes_names`, and `name`. Each edge must have `node1`, `node2`, and `name`.

```python
from tensor_network_viz import PlotConfig, show_tensor_network

config = PlotConfig(figsize=(8, 6))
fig, ax = show_tensor_network(
    network,
    engine="tensorkrowch",
    view="2d",
    config=config,
)
```

You can also use the TensorKrowch-specific helpers directly:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
```

## Project layout

- `examples/` — Demo scripts. Run `python examples/tensor_network_demo.py mps 2d` from the project root.
- `scripts/` — Utility scripts (e.g. `clean.py` to remove caches and build artifacts).

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```
