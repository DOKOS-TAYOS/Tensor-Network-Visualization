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

Inside the project virtual environment:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.dev.txt
```

For runtime-only usage:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Usage

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

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pytest
```
