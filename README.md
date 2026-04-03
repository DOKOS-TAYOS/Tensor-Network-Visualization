# Tensor-Network-Visualization

[![CI](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![Python versions](https://img.shields.io/pypi/pyversions/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced
PyTorch/NumPy `einsum` tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## What This Library Does

Tensor-network libraries expose very different Python objects, but this package gives them one
common plotting entry point:

```python
fig, ax = show_tensor_network(...)
```

Internally, the library:

1. normalizes backend objects into one graph model,
2. computes a layout,
3. draws the graph in 2D or 3D with Matplotlib,
4. optionally adds figure controls for view/label/scheme toggles.

The goal is to keep the public API small while still being useful for notebooks, papers, debugging,
and saved figures.

## Install

PyPI package name: `tensor-network-visualization`  
Import package: `tensor_network_viz`

**Requires Python 3.11 or newer.**

### Base install

```bash
python -m pip install tensor-network-visualization
```

Base dependencies are only `numpy`, `matplotlib`, and `networkx`.

### Optional extras

| Need | Install |
| --- | --- |
| TensorKrowch support | `python -m pip install "tensor-network-visualization[tensorkrowch]"` |
| TensorNetwork support | `python -m pip install "tensor-network-visualization[tensornetwork]"` |
| Quimb support | `python -m pip install "tensor-network-visualization[quimb]"` |
| TeNPy support | `python -m pip install "tensor-network-visualization[tenpy]"` |
| Traced `einsum(...)` support | `python -m pip install "tensor-network-visualization[einsum]"` |
| Interactive Jupyter widgets | `python -m pip install "tensor-network-visualization[jupyter]"` |

### Windows and Linux quick start

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install "tensor-network-visualization[quimb]"
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "tensor-network-visualization[quimb]"
```

## The API in One Minute

### `show_tensor_network`

Use `show_tensor_network` for figure lifecycle:

```python
show_tensor_network(
    network,
    *,
    engine=None,
    view=None,
    config=None,
    ax=None,
    show_controls=True,
    show=True,
)
```

- `engine`: optional backend override. If omitted, the library auto-detects it.
- `view`: `"2d"` or `"3d"`. If omitted, it starts in `"2d"`.
- `config`: all visual behavior lives here.
- `ax`: render into an existing Matplotlib axis.
- `show_controls`: if `False`, the figure is saved/rendered without the embedded control panel.
- `show`: if `False`, nothing is displayed automatically.

### `PlotConfig`

Use `PlotConfig` for visual behavior:

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    show_tensor_labels=True,
    show_index_labels=False,
    hover_labels=True,
    show_contraction_scheme=False,
    tensor_label_refinement="auto",
)
```

This is where you configure:

- labels,
- hover tooltips,
- contraction-scheme overlays,
- styling,
- layout iterations,
- custom positions,
- label-refinement policy,
- static/export-oriented rendering choices.

## Most Common Workflows

### Interactive figure with controls

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    config=PlotConfig(
        show_tensor_labels=False,
        show_index_labels=False,
        hover_labels=True,
    ),
)
```

### Clean export with no embedded controls

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    config=PlotConfig(
        show_tensor_labels=True,
        show_index_labels=False,
    ),
    show_controls=False,
    show=False,
)
fig.savefig("network.png", bbox_inches="tight")
```

### Faster render for large graphs

```python
config = PlotConfig(
    tensor_label_refinement="never",
    layout_iterations=120,
)
```

## Copy-Paste Examples by Backend

The library supports auto-detection, but using `engine=...` in examples is often clearer.

### TensorKrowch

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
    config=PlotConfig(show_tensor_labels=True),
)
```

### TensorNetwork

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
    config=PlotConfig(show_tensor_labels=True),
)
```

### Quimb

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
)
```

### TeNPy

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
    show_controls=False,
    show=False,
)
```

### `einsum`

```python
from tensor_network_viz import PlotConfig, pair_tensor, show_tensor_network

trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
]

fig, ax = show_tensor_network(
    trace,
    engine="einsum",
    config=PlotConfig(show_contraction_scheme=True),
)
```

For fuller backend examples, see [docs/backends.md](docs/backends.md).

## `PlotConfig` Fields You Will Actually Use Often

| Field | Why it matters |
| --- | --- |
| `show_tensor_labels` | Draw tensor names on nodes. |
| `show_index_labels` | Draw index labels on bonds and dangling legs. |
| `hover_labels` | Enable hover tooltips in interactive sessions. |
| `show_contraction_scheme` | Draw contraction-step regions. |
| `contraction_playback` | Start with playback controls enabled when controls are shown. |
| `contraction_scheme_cost_hover` | Show cost tooltip on scheme regions. |
| `tensor_label_refinement` | `"auto"`, `"always"`, or `"never"` for the expensive label-fit pass. |
| `layout_iterations` | Force-layout effort. |
| `positions` | Supply custom node coordinates. |
| `figsize` | Matplotlib figure size when the figure is created internally. |

## Example Launcher

The repository ships a typed demo launcher:

```bash
python examples/run_demo.py <engine> <example> [options]
```

Useful examples:

```bash
python examples/run_demo.py quimb hyper --view 2d
python examples/run_demo.py tenpy chain --view 2d --save tenpy_chain.png --no-show
python examples/run_demo.py einsum ellipsis --view 3d --scheme
```

More details: [examples/README.md](examples/README.md)

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| Saved figure includes buttons/sliders | Use `show_controls=False`. |
| Hover tooltips do nothing | Use an interactive Matplotlib backend; hover is not useful for PNG-only runs. |
| Big graphs are slow | Set `tensor_label_refinement="never"`, reduce `layout_iterations`, or pass `positions`. |
| `Unsupported tensor network engine` | Install the matching extra or pass the correct backend object. |
| Blank / duplicate Jupyter figure | Assign `fig, ax = show_tensor_network(...)` instead of leaving the tuple as the last line. |

## Documentation Map

- [docs/guide.md](docs/guide.md): workflow-oriented guide to the public API.
- [docs/backends.md](docs/backends.md): copy-paste backend-specific examples.
- [examples/README.md](examples/README.md): demo launcher and batch-render usage.
- [CHANGELOG.md](CHANGELOG.md): release notes.

## Development

Create and use a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.dev.txt
```

Run the project checks:

```powershell
.\.venv\Scripts\python scripts\verify.py
```
