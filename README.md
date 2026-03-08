# tensor-visualization

Minimal Matplotlib visualizations for TensorKrowch tensor networks.

## Features

- 2D and 3D plotting for TensorKrowch tensor networks
- Tensors rendered as nodes
- Contractions rendered as edges between tensors
- Dangling indices rendered as labeled stubs
- Self-contractions rendered as loops
- No UI and no `networkx`

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
from tensor_visualization import plot_tensor_network_2d, plot_tensor_network_3d

fig2d, ax2d = plot_tensor_network_2d(network)
fig3d, ax3d = plot_tensor_network_3d(network)
```

Both functions accept an existing Matplotlib axis through `ax=` and return `(fig, ax)`.

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pytest
```
