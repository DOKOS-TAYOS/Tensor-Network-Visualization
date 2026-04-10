# Tensor-Network-Visualization

[![CI](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![Python versions](https://img.shields.io/pypi/pyversions/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced
PyTorch/NumPy `einsum` tensor networks.

## Why This Exists

Tensor-network libraries expose different Python objects. This package gives them a small shared
visualization API so you can inspect structure, tensor values, contraction playback, and normalized
graph exports without rewriting plotting code for every backend.

The common entry points are:

```python
show_tensor_network(...)
show_tensor_elements(...)
show_tensor_comparison(...)
normalize_tensor_network(...)
export_tensor_network_snapshot(...)
```

## Install

- PyPI package name: `tensor-network-visualization`
- Import package: `tensor_network_viz`
- Requires Python 3.11 or newer.

```bash
python -m pip install tensor-network-visualization
```

The base install only depends on `numpy`, `matplotlib`, and `networkx`.

For interactive Jupyter figures:

```bash
python -m pip install "tensor-network-visualization[jupyter]"
```

For backend-specific packages, install the matching extra, for example:

```bash
python -m pip install "tensor-network-visualization[quimb]"
```

See [docs/installation.md](docs/installation.md) for virtual environments, all optional extras, and
local development installs.

## Basic Usage

### NumPy `einsum` trace (base install)

This example uses only base dependencies and a NumPy-backed `EinsumTrace`.

```python
import numpy as np
from tensor_network_viz import EinsumTrace, PlotConfig, einsum, show_tensor_network

trace = EinsumTrace()
a = np.ones((2, 3), dtype=float)
x = np.array([1.0, -0.5, 0.25], dtype=float)

trace.bind("A", a)
trace.bind("x", x)
einsum("ab,b->a", a, x, trace=trace, backend="numpy")

fig, ax = show_tensor_network(
    trace,
    config=PlotConfig(show_tensor_labels=True, hover_labels=True),
    show=False,
)
fig.savefig("einsum-network.png", bbox_inches="tight")
```

### TensorKrowch

Install the TensorKrowch extra (see [Installation](docs/installation.md) for details):

```bash
python -m pip install "tensor-network-visualization[tensorkrowch]"
```

```python
import tensorkrowch as tk
from tensor_network_viz import PlotConfig, show_tensor_network

network = tk.TensorNetwork(name="demo")
left = tk.Node(shape=(2, 2), axes_names=("a", "b"), name="L", network=network)
right = tk.Node(shape=(2, 2), axes_names=("b", "c"), name="R", network=network)
left["b"] ^ right["b"]

fig, ax = show_tensor_network(
    network,
    config=PlotConfig(show_tensor_labels=True, show_index_labels=False),
    show=False,
)
fig.savefig("tensorkrowch-network.png", bbox_inches="tight")
```

Use `show=False` when you want to save or customize the figure yourself. Use
`show_controls=False` when you want a clean static figure with no embedded Matplotlib controls.

In Jupyter, install `"tensor-network-visualization[jupyter]"`, run `%matplotlib widget` before
plotting, then call `show_tensor_network` as usual (default `show_controls=True`) for interactive
figures with embedded controls. See [Installation](docs/installation.md) and
[User Guide](docs/guide.md) for details.

## Documentation

- [Installation](docs/installation.md): virtual environments, optional extras, Jupyter, and local
  editable installs.
- [API Reference](docs/api.md): public functions, configuration objects, snapshots, exceptions, and
  logging.
- [User Guide](docs/guide.md): workflows, notebooks, exports, layouts, tensor inspection,
  comparisons, snapshots, and performance tips.
- [Backend Examples](docs/backends.md): copy-paste examples for TensorKrowch, TensorNetwork, Quimb,
  TeNPy, and `einsum`.
- [Troubleshooting](docs/troubleshooting.md): common install, Jupyter, Matplotlib, backend, and data
  issues.
- [Repository Examples](examples/README.md): command-line demo launcher and example catalog.

## Project Links

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues)
- [Repository](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)
