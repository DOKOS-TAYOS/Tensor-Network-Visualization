# Tensor-Network-Visualization

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced
PyTorch/NumPy `einsum` tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## Problem, Audience, and Scope

**Problem:** Tensor network libraries (Quimb, TeNPy, TensorNetwork, TensorKrowch) provide different APIs and limited or backend-specific visualization. Researchers often need to switch tools or write custom plotting code when working across ecosystems.

**Audience:** Researchers in condensed matter physics, quantum chemistry, and machine learning who use tensor networks and need publication-ready diagrams.

**Supported backends:** `tensorkrowch`, `tensornetwork`, `quimb`, `tenpy`, `einsum`

**Differential:** One unified API across five backends; returns standard Matplotlib `Figure` and `Axes` for full customization; 2D and 3D views; reconstructs networks from traced binary `einsum` contractions. No custom UI—builds a normalized graph and draws it with Matplotlib.

## Installation

### Base

```bash
pip install tensor-network-visualization
```

The base package depends only on `matplotlib` and `networkx`. It can render existing `einsum` traces (manual `pair_tensor` lists) without backend extras.

### By backend (extras)

Install the extra for each backend you use:

| Backend | Command |
|---------|---------|
| TensorKrowch | `pip install "tensor-network-visualization[tensorkrowch]"` |
| TensorNetwork | `pip install "tensor-network-visualization[tensornetwork]"` |
| Quimb | `pip install "tensor-network-visualization[quimb]"` |
| TeNPy | `pip install "tensor-network-visualization[tenpy]"` |
| einsum (PyTorch) | `pip install "tensor-network-visualization[einsum]"` |

`[einsum]` adds PyTorch for executing traced contractions. For rendering an existing trace with NumPy only, the base package suffices.

## Minimal example

After `pip install "tensor-network-visualization[quimb]"`:

```python
import quimb.tensor as qtn
import numpy as np
from tensor_network_viz import PlotConfig, show_tensor_network

# Build a 3-site MPS
tensors = []
for i in range(3):
    inds = ([f"b{i-1}_{i}"] if i > 0 else []) + [f"p{i}"] + ([f"b{i}_{i+1}"] if i < 2 else [])
    shape = tuple(2 for _ in inds)
    tensors.append(qtn.Tensor(np.ones(shape), inds=inds, tags=(f"A{i}",)))
network = qtn.TensorNetwork(tensors)

fig, ax = show_tensor_network(network, engine="quimb", view="2d", config=PlotConfig(figsize=(8, 6)), show=False)
fig.savefig("network.png", bbox_inches="tight")
```

The function returns `(fig, ax)`. Use `view="3d"` for 3D rendering. Replace `network` and `engine` for other backends.

## Quick verification (for reviewers)

From a fresh clone, install and run:

```bash
pip install -e ".[quimb]"
python examples/quimb_demo.py mps 2d --save quimb_mps.png --no-show
pytest
```

Verify that `quimb_mps.png` exists and contains a tensor-network diagram. All tests must pass. Alternative: `pip install -e ".[tensornetwork]"` and `python examples/tensornetwork_demo.py mps 2d --save tn_mps.png --no-show` if you prefer TensorNetwork.

## Documentation

- [Extended guide](docs/guide.md) — installation, usage, backend-specific inputs, configuration, troubleshooting
- [Examples catalog](examples/README.md) — runnable scripts for all backends
- [Third-party licenses](THIRD_PARTY_LICENSES.md)

## Quick start (API reference)

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6)),
    show=False,
)
ax.set_title("My tensor network")
fig.savefig("network.png", bbox_inches="tight")
```

- `network`: backend-native object (e.g. `quimb.TensorNetwork`, iterable of `tensornetwork.Node`, etc.)
- `engine`: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"`
- `view`: `"2d"` or `"3d"`
- Returns `(fig, ax)` — Matplotlib `Figure` and `Axes` objects

## Public API

```python
from tensor_network_viz import (
    EinsumTrace,
    PlotConfig,
    einsum,
    pair_tensor,
    show_tensor_network,
)
```

Engine-specific helpers:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d
from tensor_network_viz.tenpy import plot_tenpy_network_3d
from tensor_network_viz.einsum_module import plot_einsum_network_2d
```

## Accepted inputs (summary)

| Backend | Input |
|---------|-------|
| tensorkrowch | Network with `nodes`/`leaf_nodes`, or iterable of nodes |
| tensornetwork | Iterable of `tensornetwork.Node` |
| quimb | `TensorNetwork` or iterable of `Tensor` |
| tenpy | Finite/segment/infinite `MPS`, finite/infinite `MPO` |
| einsum | `EinsumTrace` or ordered iterable of `pair_tensor` |

See [docs/guide.md](docs/guide.md) for backend-specific details.

## Plot configuration

`PlotConfig` controls figure size, colors, line widths, label visibility, scale parameters, custom positions, and layout iterations. Important options:

- `positions`: custom node positions (dict mapping node id to `(x, y)` or `(x, y, z)`)
- `layout_iterations`: force-directed layout iterations (default 220)
- `validate_positions`: warn on unknown ids or wrong dimensions

Layout prefers structural embeddings (linear chains, 2D meshes, trees) before falling back to force-directed placement.

## Backend notes

- Quimb hyper-indices shared by more than two tensors are rendered through internal virtual hubs.
- Infinite TeNPy `MPS`/`MPO` are drawn as one periodic unit cell.
- The `einsum` backend reconstructs the network of fundamental tensors, not intermediate contraction results.
- Subsets of nodes/tensors show connections to outside objects as dangling legs. Disconnected components are supported.

## Examples

Runnable scripts for every backend plus a TensorKrowch TSP example. See [examples/README.md](examples/README.md) for commands.

## Support, issues, and contribution

- **Bug reports and feature requests:** [Issue tracker](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format .
pyright
pytest
```
