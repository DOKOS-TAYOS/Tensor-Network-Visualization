# Tensor-Network-Visualization

[![CI](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![Python versions](https://img.shields.io/pypi/pyversions/tensor-network-visualization.svg)](https://pypi.org/project/tensor-network-visualization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimal Matplotlib visualizations for TensorKrowch, TensorNetwork, Quimb, TeNPy, and traced
PyTorch/NumPy `einsum` tensor networks.

**Repository:** [https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization)

## What this project does

Tensor network libraries use different object models and rarely share a single visualization path.
This package **normalizes** each backend into one graph description, **lays out** nodes (chains,
grids, trees, planar embeddings, or force-directed fallback), and **draws** with Matplotlib in
**2D** (filled disks) or **3D** (lightweight octahedra). There is no custom GUI: you get standard
`Figure` and `Axes` objects for saving, composing subplots, and further styling.

**Audience:** Researchers who work with tensor networks and want consistent, publication-friendly
diagrams across Quimb, TeNPy, TensorNetwork, TensorKrowch, or traced `einsum` contraction graphs.

## Installation

PyPI package name: `tensor-network-visualization`. Import module: `tensor_network_viz`.

### Base install

```bash
python -m pip install tensor-network-visualization
```

Depends on `matplotlib` and `networkx` only. You can render **`einsum`** traces built from ordered
`pair_tensor` entries without installing PyTorch.

### Optional backends (extras)

| Backend        | Pip extra                     | Notes |
|----------------|-------------------------------|--------|
| TensorKrowch   | `tensorkrowch`                | `pip install "tensor-network-visualization[tensorkrowch]"` |
| TensorNetwork  | `tensornetwork`               | `pip install "tensor-network-visualization[tensornetwork]"` |
| Quimb          | `quimb`                       | `pip install "tensor-network-visualization[quimb]"` |
| TeNPy          | `tenpy`                       | Resolves to PyPI package **`physics-tenpy`**. |
| Einsum tracing | `einsum`                      | Adds **PyTorch** for the traced `einsum` helper. |
| Jupyter widgets| `jupyter`                    | `ipympl`, widgets, JupyterLab / Notebook 7+ for interactive figures. |

Combine extras, for example:

```bash
python -m pip install "tensor-network-visualization[quimb,jupyter]"
```

### Windows and Linux quick setup

From the project root (development) or any environment (end users):

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

Editable install for contributors:

```bash
python -m pip install -e ".[dev]"
```

### Jupyter and interactive figures

For pan, zoom, and (with `%matplotlib widget`) smoother interaction, install the **`jupyter`**
extra and select the ipympl backend **before** creating figures.

For **interactive** figures (e.g. rotatable 3D), install `[jupyter]` (`ipympl`, `ipywidgets`,
JupyterLab, and classic Jupyter Notebook 7+) and run **`jupyter notebook`** or **`jupyter lab`** in
the **browser**. The **Cursor / VS Code notebook tab** may not load the full widget stack; if
figures fail to show, use **`%matplotlib inline`** or set **`MPLBACKEND=inline`** before importing
`pyplot`, or open the same notebook in the browser with a normal Jupyter install.

## Modes and API knobs

Everything below maps to real parameters—there are no hidden mode switches.

| Concept | Where | Meaning |
|--------|--------|---------|
| **View mode** | `show_tensor_network(..., view=...)` | `"2d"` — disk nodes; `"3d"` — octahedra. Same normalized graph. |
| **Engine mode** | `engine=...` | Adapter: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, `"einsum"`. Invalid → `ValueError`. |
| **Display mode** | `show=True` / `False` | If `True`: Jupyter **kernel** uses `IPython.display.display(fig)`; otherwise `plt.show()`. If `False`: neither runs—use for `savefig` / batch. |
| **Label policy** | `PlotConfig` + overrides | Defaults: `show_tensor_labels`, `show_index_labels`. Per-call: `show_tensor_network(..., show_tensor_labels=..., show_index_labels=...)`. |
| **Hover labels** | `PlotConfig(hover_labels=True)` | Tensor names and bond labels appear on pointer hover (2D axes hit-test; 3D screen-space distance). Needs an **interactive** Matplotlib window. |
| **Einsum workflow** | `engine="einsum"` | **Auto:** `EinsumTrace` + `tensor_network_viz.einsum`. **Manual:** ordered list of `pair_tensor(...)`. |

## Minimal examples

### Quimb 2D, save to PNG

After `python -m pip install "tensor-network-visualization[quimb]"`:

```python
import numpy as np
import quimb.tensor as qtn
from tensor_network_viz import PlotConfig, show_tensor_network

tensors = []
for i in range(3):
    inds = ([f"b{i-1}_{i}"] if i > 0 else []) + [f"p{i}"] + ([f"b{i}_{i+1}"] if i < 2 else [])
    shape = tuple(2 for _ in inds)
    tensors.append(qtn.Tensor(np.ones(shape), inds=inds, tags=(f"A{i}",)))
network = qtn.TensorNetwork(tensors)

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6)),
    show=False,
)
fig.savefig("network.png", bbox_inches="tight")
```

### Quimb 3D, hide index labels on this call only

```python
fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="3d",
    config=PlotConfig(figsize=(9, 7)),
    show_index_labels=False,
    show=False,
)
```

### Hover labels (interactive session only)

```python
fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6), hover_labels=True),
)
```

No effect for headless `show=False` without a GUI event loop; pair with an interactive backend
(e.g. `%matplotlib widget` in Jupyter).

### TeNPy finite MPS (sketch)

Requires `python -m pip install "tensor-network-visualization[tenpy]"`. See the
[extended guide](docs/guide.md) for exact `MPS` construction; the call shape is always:

```python
from tensor_network_viz import show_tensor_network, PlotConfig

fig, ax = show_tensor_network(
    mps,
    engine="tenpy",
    view="2d",
    config=PlotConfig(figsize=(8, 6)),
    show=False,
)
```

### `show_tensor_network` reference

```text
show_tensor_network(
    network,
    *,
    engine: EngineName,
    view: ViewName,
    config: PlotConfig | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    show: bool = True,
) -> tuple[Figure, Axes | Axes3D]
```

- **`network`:** Backend-native object or iterable (see [guide — supported inputs](docs/guide.md)).
- **`engine`:** `"tensorkrowch"` \| `"tensornetwork"` \| `"quimb"` \| `"tenpy"` \| `"einsum"`.
- **`view`:** `"2d"` \| `"3d"`.
- **`config`:** Styling and layout; defaults to `PlotConfig()` if omitted.
- **`show_tensor_labels` / `show_index_labels`:** If `None`, use values from `config`; else override for this call only.
- **`show`:** Whether to push the figure to the current UI (`display`/`show`).
- **Returns:** `(fig, ax)` for titles, `savefig`, colorbars, or embedding in a subplot.

Backend-specific shortcuts (same renderer core) accept **`ax=`** and **`seed=`** for subplots and
reproducible layout; see the [guide](docs/guide.md).

### `PlotConfig` quick reference

Frozen dataclass; all fields optional beyond defaults. Values shown are **constructor defaults**.
Numeric fields with **`None`** use the corresponding `DEFAULT_*` on the class (see table footnote).

| Field | Default | Role |
|-------|---------|------|
| `node_color` | `"#E8E8E8"` | Tensor node fill. |
| `node_edge_color` | `"#2D3748"` | Tensor node outline. |
| `node_color_degree_one` | `"#E8D6D6"` | Fill for tensors with total graph degree 1. |
| `node_edge_color_degree_one` | `"#4A3436"` | Outline for degree-1 tensors. |
| `tensor_label_color` | `"#1A202C"` | Tensor name text. |
| `label_color` | `"#0C1319"` | Index / bond label text. |
| `bond_edge_color` | `"#00008B"` | Contraction edges. |
| `dangling_edge_color` | `"#8B0000"` | Dangling (open) legs. |
| `figsize` | `(8, 6)` | `inches`; `None` uses Matplotlib fallback `(14, 10)` when the renderer creates a new figure. |
| `show_tensor_labels` | `True` | Draw tensor names on nodes. |
| `show_index_labels` | `True` | Draw axis names on bonds / stubs. |
| `node_radius` | `None` | → `0.08` data units (scaled with layout; multiplies geometric radius). |
| `stub_length` | `None` | → `0.16` (dangling stub length scale). |
| `self_loop_radius` | `None` | → `0.2` (self-contraction loops). |
| `line_width_2d` | `None` | → `0.85` |
| `line_width_3d` | `None` | → `0.75` |
| `layout_iterations` | `None` | → automatic: `int(min(220, max(45, 14*√n)))` with `n = max(n_nodes, 1)` when unset; explicit int always wins. |
| `positions` | `None` | `dict[int, tuple[float, ...]]` — custom positions keyed by **normalized node id** (`id` of adapter node); partial dicts get layout for missing ids. |
| `validate_positions` | `False` | If `True`, warn on unknown keys or short coordinates vs `view`. |
| `refine_tensor_labels` | `True` | Extra passes to fit tensor names inside the node marker (2D or 3D); set `False` for speed. |
| `hover_labels` | `False` | Hide labels until hover (interactive only). |

Defaults `0.08`, `0.16`, `0.2`, `0.85`, `0.75`, `220` are also available as
`PlotConfig.DEFAULT_NODE_RADIUS`, `DEFAULT_STUB_LENGTH`, `DEFAULT_SELF_LOOP_RADIUS`,
`DEFAULT_LINE_WIDTH_2D`, `DEFAULT_LINE_WIDTH_3D`, `DEFAULT_LAYOUT_ITERATIONS`.

## Public Python API

```python
from tensor_network_viz import (
    EngineName,
    EinsumTrace,
    PlotConfig,
    ViewName,
    einsum,
    pair_tensor,
    show_tensor_network,
)
```

Per-backend plotters (optional; same as `show_tensor_network` internals):

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d
from tensor_network_viz.einsum_module import plot_einsum_network_2d, plot_einsum_network_3d
```

## Accepted inputs (summary)

| Backend | Input |
|---------|--------|
| tensorkrowch | Network with `nodes` / `leaf_nodes`, or iterable of nodes |
| tensornetwork | Iterable of `tensornetwork.Node` |
| quimb | `TensorNetwork` or iterable of `Tensor` |
| tenpy | Finite/segment/infinite `MPS`, finite/infinite `MPO` |
| einsum | `EinsumTrace` or ordered iterable of `pair_tensor` |

Details, subgraph behavior, and Quimb hyperindex hubs are in **[docs/guide.md](docs/guide.md)**.

## Example scripts

Runnable demos live under **`examples/`**. From the repo root with the right extra installed:

| Script | Purpose |
|--------|---------|
| `demo_cli.py` | Shared helpers (`--hover-labels` → `PlotConfig(hover_labels=True)`); imported by demos. |
| `tensorkrowch_demo.py` | MPS, MPO, PEPS, weird, disconnected; `--from-list` subset. |
| `tensornetwork_demo.py` | Same topologies with `tensornetwork.Node`. |
| `mera_tree_demo.py` | Large MERA + binary TTN stress test. |
| `cubic_peps_demo.py` | 3D cubic PEPS lattice. |
| `quimb_demo.py` | Includes hyper-index example; `--from-list`. |
| `tenpy_demo.py` | Finite and infinite MPS/MPO. |
| `einsum_demo.py` | Auto trace vs manual `pair_tensor`. |
| `tn_tsp.py` | Larger TensorKrowch TSP construction. |

Catalog and one-liner commands: **[examples/README.md](examples/README.md)**.

## Backend notes

- Quimb **hyper-indices** (three or more tensors) are drawn via internal virtual hubs.
- Infinite TeNPy **MPS/MPO** use one periodic unit cell.
- The **einsum** backend visualizes the **fundamental** tensor network, not each intermediate
  contraction tensor.
- Passing a **subset** of nodes/tensors shows connections outside the subset as **dangling** legs.

## Quick verification (reviewers)

```bash
python -m pip install -e ".[quimb]"
python examples/quimb_demo.py mps 2d --save quimb_mps.png --no-show
python -m pytest
```

Expect `quimb_mps.png` and all tests passing.

## Troubleshooting (short)

| Symptom | What to try |
|---------|-------------|
| `ModuleNotFoundError` for quimb, tenpy, torch, … | Install the matching extra, e.g. `"tensor-network-visualization[quimb]"`. |
| `ValueError: Unsupported tensor network engine` / `view` | Use only listed `engine` / `view` literals (see above). |
| Blank or double figure in Jupyter | Assign `fig, ax = show_tensor_network(...)`; avoid bare tuple as last line; try `%matplotlib widget` or inline. |
| Hover labels do nothing | Requires interactive backend and `show` path that runs a GUI or widget event loop; not for `--no-show` PNG only. |
| Huge graphs are slow | `PlotConfig(refine_tensor_labels=False)`; optionally set `layout_iterations` explicitly. |

**Full troubleshooting:** [docs/guide.md — Troubleshooting](docs/guide.md#troubleshooting).

## Documentation index

- **[docs/guide.md](docs/guide.md)** — Installation, backends, `PlotConfig` recipes, layout/draw
  behavior, architecture, **extended troubleshooting**.
- **[CHANGELOG.md](CHANGELOG.md)** — Release notes by version.
- **[examples/README.md](examples/README.md)** — CLI examples per script.
- **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**

## Support and contributing

- **Issues:** [GitHub Issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format .
pyright
pytest
```
