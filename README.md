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
**2D** (filled disks) or **3D** (lightweight octahedra). The interactive layer stays inside
standard Matplotlib figures: `show_tensor_network` can add a small control panel, while the return
value is still a normal `Figure` plus `Axes` object that you can save, embed, or restyle.

**Audience:** Researchers who work with tensor networks and want consistent, publication-friendly
diagrams across Quimb, TeNPy, TensorNetwork, TensorKrowch, or traced `einsum` contraction graphs.

## Installation

PyPI package name: `tensor-network-visualization`. Import module: `tensor_network_viz`.

**Requires Python 3.11 or newer.**

### Base install

```bash
python -m pip install tensor-network-visualization
```

Base runtime dependencies are `numpy`, `matplotlib`, and `networkx` only. You can **build and
render** rich **`einsum`** graphs from ordered **`pair_tensor`** / **`einsum_trace_step`** entries
(ellipsis and repeated indices need shapes in metadata); **`tensor-network-visualization[einsum]`**
(PyTorch) is only needed to **execute** `tensor_network_viz.einsum(..., trace=...)` and record
those rows automatically.

### Optional backends (extras)

| Backend        | Pip extra                     | Notes |
|----------------|-------------------------------|--------|
| TensorKrowch   | `tensorkrowch`                | `pip install "tensor-network-visualization[tensorkrowch]"` |
| TensorNetwork  | `tensornetwork`               | `pip install "tensor-network-visualization[tensornetwork]"` |
| Quimb          | `quimb`                       | `pip install "tensor-network-visualization[quimb]"` |
| TeNPy          | `tenpy`                       | Resolves to PyPI package **`physics-tenpy`**. |
| Einsum tracing | `einsum`                      | Adds **PyTorch** for auto-traced `einsum(..., trace=...)` (layout from `pair_tensor` / `einsum_trace_step` lists with metadata as needed). |
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
| **View mode** | `show_tensor_network(..., view=...)` | Default is `"2d"`; `"3d"` is available from the same figure controls when widgets are enabled. |
| **Engine mode** | `engine=...` | Optional override. When omitted, `show_tensor_network` auto-detects `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"` from the input. Invalid explicit values still raise `ValueError`. |
| **Display mode** | `show=True` / `False` | If `True`: Jupyter **kernel** uses `IPython.display.display(fig)`; otherwise `plt.show()`. If `False`: neither runs—use for `savefig` / batch. |
| **Interactive controls** | `interactive_controls=True` | Figure-level Matplotlib widgets for view/hover/label toggles. Set `False` for clean static exports or headless batch rendering. |
| **Label policy** | `PlotConfig` + overrides | Defaults: `show_tensor_labels=False`, `show_index_labels=False`. Per-call: `show_tensor_network(..., show_tensor_labels=..., show_index_labels=...)`. |
| **Hover labels** | `PlotConfig(hover_labels=True)` | Default is `True`. Hover tooltips are independent from static labels, so both can stay enabled together in an interactive Matplotlib window. |
| **Contraction scheme** | `PlotConfig(show_contraction_scheme=True)` | **Einsum:** cumulative per-step highlights from the trace. **Other engines:** set **`contraction_scheme_by_name`**. Compatible figures now add Matplotlib toggles for **Scheme**, **Playback**, and **Cost hover**; if you start with those flags off, the scheme bundle is computed lazily on first use. **2D:** rounded boxes (AABB + pad); colored borders (no fill by default). **3D:** wireframe box. See **`docs/guide.md`**. |
| **Einsum workflow** | `engine="einsum"` | **Auto:** `EinsumTrace` + `einsum` (binary `pair_tensor`, unary/ternary+ `einsum_trace_step`; implicit `->`, `out=`). **Manual:** ordered `pair_tensor` / `einsum_trace_step` lists when you want a fully explicit trace. See **`examples/run_demo.py`** and **`examples/einsum_demo.py`**. |

`show_tensor_network` now creates figure-level Matplotlib widgets by default: a `2d/3d` selector on
figures it creates itself, plus `Hover`, `Tensor labels`, and `Edge labels` checkboxes. If you
pass an external `ax`, the same-view checkboxes stay available but the `2d/3d` selector is hidden.
The default widget state is `Hover=True`, `Tensor labels=False`, `Edge labels=False`.

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
    config=PlotConfig(figsize=(8, 6)),
    interactive_controls=False,
    show=False,
)
fig.savefig("network.png", bbox_inches="tight")
```

Use `interactive_controls=False` when you want the saved figure without the Matplotlib control
panel.

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
    config=PlotConfig(figsize=(8, 6), hover_labels=True),
)
```

Hover is independent from static labels, so you can also enable tensor or edge labels from the
figure checkboxes without losing the hover tooltips. This still needs an interactive backend
(e.g. `%matplotlib widget` in Jupyter).

### TeNPy finite MPS (sketch)

Requires `python -m pip install "tensor-network-visualization[tenpy]"`. See the
[extended guide](docs/guide.md) for exact `MPS` construction; the call shape is always:

```python
from tensor_network_viz import show_tensor_network, PlotConfig

fig, ax = show_tensor_network(
    mps,
    engine="tenpy",
    config=PlotConfig(figsize=(8, 6)),
    show=False,
)
```

### `show_tensor_network` reference

```text
show_tensor_network(
    network,
    *,
    engine: EngineName | None = None,
    view: ViewName | None = None,
    config: PlotConfig | None = None,
    ax: Axes | Axes3D | None = None,
    show_tensor_labels: bool | None = None,
    show_index_labels: bool | None = None,
    interactive_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes | Axes3D]
```

- **`network`:** Backend-native object or iterable (see [guide — supported inputs](docs/guide.md)).
- **`engine`:** optional explicit override. If omitted, the backend is inferred from `network`.
- **`view`:** `"2d"` \| `"3d"`; omitted means `"2d"` unless a 3D `ax` is passed.
- **`config`:** Styling and layout; defaults to `PlotConfig()` if omitted.
- **`ax`:** Optional Matplotlib axes. When present, rendering stays on that axes and the `2d/3d` selector is hidden.
- **`show_tensor_labels` / `show_index_labels`:** If `None`, use values from `config`; else override for this call only.
- **`interactive_controls`:** If `True`, attach the view/hover/label widgets when the figure supports them. Set `False` for a static render.
- **`show`:** Whether to push the figure to the current UI (`display`/`show`).
- **Returns:** `(fig, ax)` for titles, `savefig`, colorbars, or embedding in a subplot.

Backend-specific shortcuts (same renderer core) accept **`ax=`** and **`seed=`** for subplots and
reproducible layout; they stay fixed-dimension helpers (`plot_*_network_2d` or `plot_*_network_3d`)
and do not add the `2d/3d` selector. See the [guide](docs/guide.md).

### `PlotConfig` quick reference

Frozen dataclass; all fields optional beyond defaults. Values shown are **constructor defaults**.
Numeric fields with **`None`** use the corresponding `DEFAULT_*` on the class (see table footnote).

| Field | Default | Role |
|-------|---------|------|
| `node_color` | `"#E8EEF5"` | Tensor node fill. |
| `node_edge_color` | `"#1E293B"` | Tensor node outline. |
| `node_color_degree_one` | `"#FEE2E2"` | Fill for tensors with total graph degree 1. |
| `node_edge_color_degree_one` | `"#7F1D1D"` | Outline for degree-1 tensors. |
| `tensor_label_color` | `"#0F172A"` | Tensor name text. |
| `label_color` | `"#334155"` | Index / bond label text. |
| `bond_edge_color` | `"#0369A1"` | Contraction edges. |
| `dangling_edge_color` | `"#BE123C"` | Dangling (open) legs. |
| `figsize` | `(8, 6)` | `inches`; `None` uses Matplotlib fallback `(14, 10)` when the renderer creates a new figure. |
| `show_tensor_labels` | `False` | Draw tensor names on nodes. |
| `show_index_labels` | `False` | Draw axis names on bonds / stubs. |
| `node_radius` | `None` | → `0.08` data units (scaled with layout; multiplies geometric radius). |
| `stub_length` | `None` | → `0.16` (dangling stub length scale). |
| `self_loop_radius` | `None` | → `0.2` (self-contraction loops). |
| `line_width_2d` | `None` | → `0.85` |
| `line_width_3d` | `None` | → `0.75` |
| `layout_iterations` | `None` | → automatic: `int(min(220, max(45, 14*√n)))` with `n = max(n_nodes, 1)` when unset; explicit int always wins. |
| `positions` | `None` | `dict[int, tuple[float, ...]]` — custom positions keyed by **normalized node id** (`id` of adapter node); partial dicts get layout for missing ids. |
| `validate_positions` | `False` | If `True`, warn on unknown keys or short coordinates vs `view`. |
| `refine_tensor_labels` | `True` | Extra passes to fit tensor names inside the node marker (2D or 3D); set `False` for speed. |
| `hover_labels` | `True` | Enable hover tooltips independently from static labels (interactive only). |

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
    einsum_trace_step,
    pair_tensor,
    show_tensor_network,
)
```

Per-backend plotters (optional; same as `show_tensor_network` internals):

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.tenpy import (
    make_tenpy_tensor_network,
    plot_tenpy_network_2d,
    plot_tenpy_network_3d,
)
from tensor_network_viz.einsum_module import (
    parse_einsum_equation,
    parse_equation_for_shapes,
    plot_einsum_network_2d,
    plot_einsum_network_3d,
)
```

(`parse_equation_for_shapes` — binary only; `parse_einsum_equation` — any arity, NumPy-validated.)

## Accepted inputs (summary)

| Backend | Input |
|---------|--------|
| tensorkrowch | Network with `nodes` / `leaf_nodes`, or iterable of nodes |
| tensornetwork | Iterable of `tensornetwork.Node` |
| quimb | `TensorNetwork` or iterable of `Tensor` |
| tenpy | `TenPyTensorNetwork` / `make_tenpy_tensor_network` (`npc.Array` + bonds); `MPS`, `MPO`, `MomentumMPS`-like; no stable TeNPy PEPS class (hand-built TN ok) |
| einsum | `EinsumTrace` or ordered iterable of `pair_tensor` / `einsum_trace_step` (ellipsis / hyperedges in the normalized graph) |

Details, subgraph behavior, and Quimb hyperindex hubs are in **[docs/guide.md](docs/guide.md)**.

## Example scripts

Runnable demos live under **`examples/`** and now share one public launcher:

```bash
python examples/run_demo.py <engine> <example> [options]
```

Main files:

| File | Purpose |
|--------|---------|
| `run_demo.py` | Public CLI entry point for every engine and example. |
| `demo_cli.py` | Shared typed helpers for parsing, save paths, plot config, and topology builders. |
| `tensorkrowch_demo.py` | TensorKrowch registry: MPS/TT, MPO, ladder, PEPS, cubic PEPS, MERA, MERA+TTN, weird, disconnected. |
| `tensornetwork_demo.py` | TensorNetwork registry with the same structured graph examples. |
| `quimb_demo.py` | Quimb registry with the same graph family plus a native hypergraph example. |
| `tenpy_demo.py` | TeNPy registry: native MPS/MPO/iMPS/iMPO/purification/uniform/excitation plus explicit chain/hub/hyper examples. |
| `einsum_demo.py` | Einsum registry: network-style traces plus `ellipsis`, `batch`, `trace`, `ternary`, `unary`, `nway`, `implicit_out`. |
| `run_all_examples.py` | Headless batch runner that calls `run_demo.py` and saves PNGs. |

Useful launcher examples:

```bash
python examples/run_demo.py quimb hyper --view 2d
python examples/run_demo.py tenpy chain --view 2d --save tenpy_chain.png --no-show
python examples/run_demo.py einsum ellipsis --view 3d --from-list
python examples/run_demo.py tensornetwork mera_ttn --view 2d --scheme
```

Catalog and CLI details: **[examples/README.md](examples/README.md)**.

## Backend notes

- Quimb **hyper-indices** (three or more tensors) are drawn via internal virtual hubs.
- Infinite TeNPy **MPS/MPO** use one periodic unit cell.
- The **einsum** backend visualizes the **fundamental** tensor network, not each intermediate
  contraction tensor. Pairwise summed indices are drawn as ordinary bonds; repeated or
  output-carrying indices use **virtual hubs** (layout separates colocated hubs, nudges **2D**
  hubs that attach to **one** tensor only—e.g. **`ii->i`**—off that tensor, and offsets hubs on a
  tensor–tensor chord when a **direct** bond also links that pair).
- Optional **`contraction_steps`** from **einsum**: **running union** of operand physical lineages
  (each step is a superset of the previous); **`PlotConfig`** draws per-step **AABB** highlights
  (**2D:** rounded rectangles; **3D:** wireframe), colored borders (no fill by default), extra
  padding as steps advance, and later steps underneath.
- Passing a **subset** of nodes/tensors shows connections outside the subset as **dangling** legs.

## Quick verification (reviewers)

```bash
python -m pip install -e ".[quimb]"
python examples/run_demo.py quimb mps --view 2d --save quimb_mps.png --no-show
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
| Saved figure includes the control panel | Pass `interactive_controls=False` when exporting a static figure from `show_tensor_network`. |
| Huge graphs are slow | `PlotConfig(refine_tensor_labels=False)`; lower `layout_iterations` or pass `positions`. Force layout samples repulsion when node count is large (about 72+; see [guide](docs/guide.md#layout-and-draw-scale)). |
| Einsum unary trace (`ii->i`) looks odd in 2D | Layout offsets the virtual hub off the tensor in 2D; try `view="3d"` or read [Einsum unary / same-tensor trace in 2D](docs/guide.md#toc-einsum-unary-2d-layout) in the guide. |

**Full troubleshooting:** [docs/guide.md — Troubleshooting](docs/guide.md#troubleshooting).

## Documentation index

- **[docs/guide.md](docs/guide.md)** — Installation, backends, `PlotConfig` recipes, layout/draw
  behavior, architecture, **extended troubleshooting**.
- **[CHANGELOG.md](CHANGELOG.md)** — Release notes by version.
- **[examples/README.md](examples/README.md)** — launcher usage, engine/example matrix, and batch examples.
- **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**

## Support and contributing

- **Issues:** [GitHub Issues](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Development

```bash
python -m venv .venv
python -m pip install -r requirements.dev.txt
```

Windows:

```powershell
.\.venv\Scripts\python scripts\verify.py
```

Linux/macOS (with the venv activated):

```bash
python scripts/verify.py
```
