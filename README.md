# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced
PyTorch/NumPy `einsum` tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## What It Does

- renders tensor networks in `2d` or `3d`
- works with multiple tensor-network ecosystems through one shared API
- returns standard Matplotlib `Figure` and `Axes` objects for further customization
- can also reconstruct and render tensor networks from traced binary `einsum` contractions

There is no custom UI layer. The package builds a normalized graph and draws it with Matplotlib.

## Supported Engines

- `tensorkrowch`
- `tensornetwork`
- `quimb`
- `tenpy`
- `einsum`

## Documentation

- Extended guide: [`docs/guide.md`](docs/guide.md)
- Examples catalog: [`examples/README.md`](examples/README.md)
- Third-party licenses: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)

## Installation

### As a dependency

The base package includes the shared rendering core. Install the extra that matches the backend you
want to visualize:

```bash
pip install "tensor-network-visualization[tensorkrowch]"
pip install "tensor-network-visualization[tensornetwork]"
pip install "tensor-network-visualization[quimb]"
pip install "tensor-network-visualization[tenpy]"
pip install "tensor-network-visualization[einsum]"
```

`[einsum]` installs PyTorch for executing traced contractions. If you already have NumPy and only
need to render an existing trace, the base package is enough.

### Local development

Inside the project virtual environment:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Runtime-only editable install:

```powershell
.\.venv\Scripts\python -m pip install -e .
```

The repository also keeps:

- `requirements.txt` -> `-e .`
- `requirements.dev.txt` -> `-e ".[dev]"`

These are thin wrappers around the editable installs above.

## Quick Start

```python
from tensor_network_viz import PlotConfig, show_tensor_network

config = PlotConfig(figsize=(8, 6))

fig, ax = show_tensor_network(
    network,
    engine="tensorkrowch",
    view="2d",
    config=config,
    show=False,
)

ax.set_title("My tensor network")
fig.savefig("network.png", bbox_inches="tight")
```

`show_tensor_network(...)` is the main dispatcher. You provide:

- `network`: the backend-native object or node/tensor collection
- `engine`: which backend adapter to use
- `view`: `2d` or `3d`
- `config`: optional `PlotConfig`
- `show=False` if you want to save or modify the figure before displaying it

The function returns `(fig, ax)`.

## Public API

From the root package:

```python
from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    einsum,
    pair_tensor,
    show_tensor_network,
)
```

Engine-specific helpers are also available:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d
from tensor_network_viz.tenpy import plot_tenpy_network_3d
from tensor_network_viz.einsum_module import plot_einsum_network_2d
```

## Accepted Inputs at a Glance

- `tensorkrowch`: a network object with `nodes` or `leaf_nodes`, or an iterable of nodes
- `tensornetwork`: an iterable of `tensornetwork.Node`
- `quimb`: a `TensorNetwork` or an iterable of `Tensor`
- `tenpy`: finite, segment, or infinite `MPS`, and finite or infinite `MPO`
- `einsum`: an `EinsumTrace` or an ordered iterable of `pair_tensor`

See the extended guide for backend-specific details and caveats.

## Plot Configuration

`PlotConfig` controls figure size, colors, line widths, label visibility, scale parameters, custom
positions, and layout iterations.

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    figsize=(10, 6),
    show_tensor_labels=True,
    show_index_labels=True,
    layout_iterations=300,
)
```

Important options:

- `positions`: custom node positions keyed by node id
- `validate_positions=True`: warn about unknown ids or wrong coordinate dimensions
- `layout_iterations`: tune the force-directed fallback layout

Automatic layout now prefers structural embeddings before falling back to forces:

- linear backbones are drawn as straight chains
- regular 2D meshes are kept on regular lattices
- trees use a deterministic hierarchical layout
- 3D views start from a principal plane and only lift nodes when the planar embedding becomes
  ambiguous
- dangling legs and other free exits in 3D use deterministic orthogonal directions instead of
  radial spreading

## Important Backend Notes

- Quimb hyper-indices shared by more than two tensors are rendered through internal virtual hubs.
- Infinite TeNPy `MPS` and `MPO` objects are drawn as one periodic unit cell.
- The `einsum` backend reconstructs the network of fundamental tensors rather than plotting
  intermediate contraction results.
- If you pass only a subset of nodes/tensors, connections to outside objects appear as dangling
  legs.
- Disconnected components are supported.

## Examples

The repository includes runnable scripts for every backend plus an extra TensorKrowch TSP example.
See [`examples/README.md`](examples/README.md) for commands and a short explanation of each script.

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```
