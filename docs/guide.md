# Tensor-Network-Visualization Guide

In-depth, code-aligned manual: installation, **modes**, APIs, copy-paste recipes, backends,
`PlotConfig`, layout and drawing behavior, architecture, and troubleshooting.

## Table of contents

- [Overview](#toc-overview)
- [Installation](#toc-installation)
- [Modes and parameters](#toc-modes-and-parameters)
- [Core usage](#toc-core-usage)
  - [Generic dispatcher](#toc-generic-dispatcher)
  - [Notebook snippet](#toc-notebook-snippet)
  - [Headless script](#toc-headless-script)
  - [Subplot embedding](#toc-subplot-embedding)
  - [Backend-specific plotters](#toc-backend-specific-plotters)
- [Supported inputs by engine](#toc-supported-inputs-by-engine)
- [PlotConfig](#toc-plotconfig)
- [Layout and draw scale](#toc-layout-and-draw-scale)
- [Custom positions](#toc-custom-positions)
- [Working with the returned figure](#toc-working-with-the-returned-figure)
- [Example scripts](#toc-example-scripts)
- [Limitations](#toc-limitations)
- [Troubleshooting](#toc-troubleshooting)
- [Internal architecture](#toc-internal-architecture)
- [Development notes](#toc-development-notes)

<a id="toc-overview"></a>

## Overview

`tensor_network_viz` does four things:

1. Accept **backend-native** tensor-network objects (or traced `einsum` graphs).
2. **Normalize** them into a shared graph model (`_GraphData`).
3. **Lay out** tensor positions (structure-aware defaults + force fallback).
4. **Draw** with Matplotlib in **2D** or **3D**.

The main entry point is **`show_tensor_network`**. Each backend also exposes **`plot_*_network_2d`**
and **`plot_*_network_3d`** — thin wrappers around the same core renderer (see
[`renderer._make_plot_functions`](../src/tensor_network_viz/_core/renderer.py)).

<a id="toc-installation"></a>

## Installation

### PyPI

Install the base package plus the extra for each backend you need:

```bash
python -m pip install "tensor-network-visualization[tensorkrowch]"
python -m pip install "tensor-network-visualization[tensornetwork]"
python -m pip install "tensor-network-visualization[quimb]"
python -m pip install "tensor-network-visualization[tenpy]"
python -m pip install "tensor-network-visualization[einsum]"
python -m pip install "tensor-network-visualization[jupyter]"
```

- **`tenpy`** extra installs **`physics-tenpy`** from PyPI.
- **`einsum`** adds **PyTorch** for the traced `tensor_network_viz.einsum` helper. Rendering a
  pre-built trace of `pair_tensor` entries does not require PyTorch if you only draw.
- **`jupyter`** pulls in `ipympl`, `ipywidgets`, JupyterLab, and Notebook 7+ for interactive
  figures.

### Editable / dev install

**Windows (PowerShell):**

```powershell
cd Tensor-Network-Visualization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

**Linux / macOS:**

```bash
cd Tensor-Network-Visualization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Runtime-only editable:

```bash
python -m pip install -e .
```

The repo also ships `requirements.txt` / `requirements.dev.txt` for conventional `-e` installs.

<a id="toc-modes-and-parameters"></a>

## Modes and parameters

There are no hidden “modes”. Use this table as the user-facing map to the API (see also the
[README modes table](../README.md#modes-and-api-knobs)):

| Concept | API | Notes |
|--------|-----|--------|
| View | `view="2d"` or `"3d"` | 2D uses disk nodes; 3D uses octahedra; same graph. |
| Engine | `engine="tensorkrowch"` … `"einsum"` | Lazy-imports adapter module; invalid → `ValueError`. |
| Display | `show=True` / `False` | If `True`: Jupyter **kernel** uses `IPython.display.display(fig)`; otherwise `plt.show()`. If `False`: no display call (for `savefig` / batch). |
| Labels | `PlotConfig` + call kwargs | `show_tensor_network(..., show_tensor_labels=..., show_index_labels=...)` overrides `PlotConfig`. |
| Hover | `PlotConfig(hover_labels=True)` | Interactive only; 2D hit-testing vs 3D projected distance. |
| Einsum | trace vs list | `EinsumTrace` + `einsum()` or ordered `pair_tensor` list. |

<a id="toc-core-usage"></a>

## Core usage

<a id="toc-generic-dispatcher"></a>

### Generic dispatcher

```python
show_tensor_network(
    network,
    *,
    engine="quimb",
    view="2d",
    config=None,
    show_tensor_labels=None,
    show_index_labels=None,
    show=True,
)
```

Returns **`(fig, ax)`** with `ax` either **2D** `Axes` or **3D** `Axes3D`.

Use **`show=False`** when you want to add titles, **`savefig`**, embed in another app, or avoid
popping a window.

<a id="toc-notebook-snippet"></a>

### Notebook snippet

Install the **`jupyter`** extra. Select a **Matplotlib** backend **before** creating figures (first cell):

```python
%matplotlib widget

from tensor_network_viz import PlotConfig, show_tensor_network

# network = ...  # your backend object

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6)),
)
```

With **`show=True`** (default), `show_tensor_network` uses **`IPython.display.display(fig)`** in a
Jupyter kernel instead of `plt.show()`, which works cleanly with interactive backends.

**Avoid double display:** if the **last line** of a cell is a bare call that returns `(fig, ax)`,
some front ends render twice. Prefer **`fig, ax = show_tensor_network(...)`** or use **`show=False`**
and **`display(fig)`** yourself.

<a id="toc-headless-script"></a>

### Headless script

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensor_network_viz import PlotConfig, show_tensor_network

# network = ...  # your backend object

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6)),
    show=False,
)
fig.savefig("out.png", bbox_inches="tight")
plt.close(fig)
```

<a id="toc-subplot-embedding"></a>

### Subplot embedding

Backend plotters accept **`ax=`**. The axis dimension must match (**2D** vs **3d** projection):

```python
import matplotlib.pyplot as plt
from tensor_network_viz import PlotConfig
from tensor_network_viz.quimb import plot_quimb_network_2d

# network = ...  # your quimb.TensorNetwork or iterable of tensors

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_quimb_network_2d(network, ax=axes[0], config=PlotConfig(figsize=None))
axes[0].set_title("Left")

plot_quimb_network_2d(network, ax=axes[1], config=PlotConfig(figsize=None))
axes[1].set_title("Right")

fig.tight_layout()
fig.savefig("two_panels.png", bbox_inches="tight")
```

When **`ax` is provided**, the renderer does not create a new figure; pass **`figsize`** only when
you create **`fig`** yourself, or use **`PlotConfig(figsize=None)`** so auto figure sizing is not
applied incorrectly (the code uses `config.figsize` only when it creates the figure).

<a id="toc-backend-specific-plotters"></a>

### Backend-specific plotters

Each backend module exposes:

```text
plot_<backend>_network_2d(network, *, ax=None, config=None,
    show_tensor_labels=None, show_index_labels=None, seed=0)
plot_<backend>_network_3d(network, *, ax=None, config=None,
    show_tensor_labels=None, show_index_labels=None, seed=0)
```

**`seed`** is passed into the shared layout pipeline for stochastic / force-directed phases so you
can reproduce placements.

Imports:

```python
from tensor_network_viz.tensorkrowch import plot_tensorkrowch_network_2d, plot_tensorkrowch_network_3d
from tensor_network_viz.tensornetwork import plot_tensornetwork_network_2d, plot_tensornetwork_network_3d
from tensor_network_viz.quimb import plot_quimb_network_2d, plot_quimb_network_3d
from tensor_network_viz.tenpy import plot_tenpy_network_2d, plot_tenpy_network_3d
from tensor_network_viz.einsum_module import plot_einsum_network_2d, plot_einsum_network_3d
```

<a id="toc-supported-inputs-by-engine"></a>

## Supported inputs by engine

### TensorKrowch

- Network with **`nodes`** or **`leaf_nodes`**, or any iterable of TensorKrowch nodes.
- Adapter reads **`edges`**, **`axes_names`**, **`name`**.
- Subsets show missing neighbors as **dangling** legs; disconnected components supported.

### TensorNetwork

- Any iterable of **`tensornetwork.Node`** with **`edges`**, **`axis_names`**, **`name`**.
- Unordered collections are normalized to a **stable** order before graph construction.

### Quimb

- **`TensorNetwork`** or iterable of **`Tensor`**.
- Tags become names when useful; otherwise fallback names like **`T0`**, **`T1`**, ….
- **Hyper-indices** (three or more tensors on one index) route through invisible **hub** nodes so
  topology stays readable.

### TeNPy

- Finite, segment, or infinite **`tenpy.networks.mps.MPS`**.
- Finite or infinite **`tenpy.networks.mpo.MPO`**.
- Infinite structures render as one **periodic unit cell**.

### `einsum`

**Inputs:** `EinsumTrace`, or ordered iterable of **`pair_tensor`**.

**Auto tracing:**

```python
from tensor_network_viz import EinsumTrace, einsum, show_tensor_network

trace = EinsumTrace()
trace.bind("A0", a0)
trace.bind("x0", x0)

r0 = einsum("pa,p->a", a0, x0, trace=trace, backend="torch")

fig, ax = show_tensor_network(trace, engine="einsum", view="2d", show=False)
```

**Manual trace:**

```python
from tensor_network_viz import pair_tensor, show_tensor_network

trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
]

fig, ax = show_tensor_network(trace, engine="einsum", view="2d", show=False)
```

The diagram shows the **underlying** tensor network, not every intermediate result tensor. Trace
order must match contraction order.

<a id="toc-plotconfig"></a>

## PlotConfig

Frozen dataclass in [`src/tensor_network_viz/config.py`](../src/tensor_network_viz/config.py).

### Field reference

| Field | Default | Meaning |
|-------|---------|---------|
| `node_color` | `"#E8EEF5"` | Node fill. |
| `node_edge_color` | `"#1E293B"` | Node outline. |
| `node_color_degree_one` | `"#FEE2E2"` | Fill when tensor has total degree 1. |
| `node_edge_color_degree_one` | `"#7F1D1D"` | Outline for degree 1. |
| `tensor_label_color` | `"#0F172A"` | Tensor name color. |
| `label_color` | `"#334155"` | Index label color. |
| `bond_edge_color` | `"#0369A1"` | Contraction bonds. |
| `dangling_edge_color` | `"#BE123C"` | Open legs. |
| `figsize` | `(8, 6)` | Inches; if `None` and the renderer **creates** a figure, fallback **`(14, 10)`**. |
| `show_tensor_labels` | `True` | Draw tensor names. |
| `show_index_labels` | `True` | Draw index names. |
| `node_radius` | `None` | → **`0.08`** (`DEFAULT_NODE_RADIUS`); scales drawn radius. |
| `stub_length` | `None` | → **`0.16`** (`DEFAULT_STUB_LENGTH`). |
| `self_loop_radius` | `None` | → **`0.2`** (`DEFAULT_SELF_LOOP_RADIUS`). |
| `line_width_2d` | `None` | → **`0.85`**. |
| `line_width_3d` | `None` | → **`0.75`**. |
| `layout_iterations` | `None` | See [Layout and draw scale](#layout-and-draw-scale). |
| `positions` | `None` | Custom coords per normalized node id. |
| `validate_positions` | `False` | Warn on bad ids / short tuples. |
| `refine_tensor_labels` | `True` | Extra draw passes to shrink tensor names so they fit the node marker (2D or 3D). |
| `hover_labels` | `False` | Show labels on hover (interactive). |

### Recipe: publication palette

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    figsize=(10, 6),
    node_color="#F4F1EA",
    node_edge_color="#2D3748",
    bond_edge_color="#173F7A",
    dangling_edge_color="#8B1E1E",
    tensor_label_color="#111827",
    label_color="#111827",
    layout_iterations=300,
)
```

### Recipe: large graphs (faster)

```python
config = PlotConfig(
    figsize=(12, 10),
    refine_tensor_labels=False,
    layout_iterations=120,
)
```

Tune **`layout_iterations`** if the automatic default (when **`layout_iterations` is `None`**) is too heavy or too loose.

### Recipe: hover labels in an interactive session

```python
config = PlotConfig(figsize=(8, 6), hover_labels=True)
```

Use with a GUI or **`%matplotlib widget`**; useless for PNG-only **`--no-show`** runs.

### Recipe: custom positions with validation

```python
config = PlotConfig(
    positions={
        id(node_a): (0.0, 0.0),
        id(node_b): (1.0, 0.0),
    },
    validate_positions=True,
)
```

Use **`(x, y)`** for `view="2d"` and **`(x, y, z)`** for `view="3d"`. Missing keys still get
automatic layout, then the cloud is centered/scaled.

<a id="toc-layout-and-draw-scale"></a>

## Layout and draw scale

### Pipeline (high level)

Shared layout in **`tensor_network_viz._core.layout`**:

1. Apply **`positions`** when provided (partial → layout for the rest, then normalize).
2. Analyze contraction components for structure.
3. Prefer specialized layouts when they match: **chains**, **2D grids**, **trees**, **planar**
   embeddings.
4. **Force-directed** fallback when no structural layout fits.
5. In **3D**, start from planar structure and move nodes to parallel **layers** when needed.
6. Compute **axis directions** for stubs, bonds, and labels.

### Force iterations when `layout_iterations is None`

Implementation [`_effective_layout_iterations`](../src/tensor_network_viz/_core/renderer.py):

```text
n = max(n_nodes, 1)
iterations = int(min(220, max(45, 14 * sqrt(n))))
```

(Implemented as floats inside `int(...)` in [`_effective_layout_iterations`](../src/tensor_network_viz/_core/renderer.py).)

If you set **`layout_iterations`** to an **int**, that value is used exactly.

### Dense graphs and large node counts

When the pipeline falls back to **force-directed** layout and the graph has more than about **72**
nodes, pairwise repulsion uses **sampled** pairs instead of every \(O(n^2)\) interaction so the step
cost stays closer to \(O(n)\)–\(O(n \log n)\) in practice. Results are still stochastic (use
**`seed`**). Very dense graphs can remain expensive from attraction and iteration count; supplying
**`positions`** or lowering **`layout_iterations`** remains the main lever for speed.

### Draw scale and node size

With default **`node_radius`**, the renderer targets node **radius ≈ 0.3 × d_min** where **`d_min`**
is the shortest **contraction** edge (center-to-center). **`PlotConfig.node_radius`** multiplies that
radius. Without usable contraction geometry, a **heuristic** from node count and extent applies.

In **2D**, node patches live in **data** coordinates so zoom/pan keeps rims aligned with bonds;
line widths and fonts stay in **points** (screen space). **3D** uses octahedra with the same metric
radius idea.

<a id="toc-custom-positions"></a>

## Custom positions

**Keys** are **`int` ids of normalized graph nodes** — typically **`id(...)`** of the Python objects
the adapter used (TensorKrowch nodes, Quimb tensors, etc.). If you pass copies or rebuild objects,
ids may not match: prefer positions keyed from the **same objects** you pass into **`show_tensor_network`**.

- Unknown keys when **`validate_positions=True`** → **`UserWarning`** (and those coords ignored).
- Too-short tuples → zero-filled extra dims with a warning when validating.

<a id="toc-working-with-the-returned-figure"></a>

## Working with the returned figure

All paths return Matplotlib objects:

```python
# network = ...  # your TeNPy MPS/MPO

fig, ax = show_tensor_network(network, engine="tenpy", view="2d", show=False)
ax.set_title("Finite MPO")
fig.savefig("finite-mpo.png", bbox_inches="tight")
```

For batch jobs: **`show=False`**, **`savefig`**, then **`plt.close(fig)`** if you create many
figures.

<a id="toc-example-scripts"></a>

## Example scripts

The [`examples/`](../examples/) directory includes:

| Script | Role |
|--------|------|
| `demo_cli.py` | `--hover-labels` → `PlotConfig(hover_labels=True)`. |
| `tensorkrowch_demo.py` | MPS, MPO, PEPS, weird, disconnected. |
| `tensornetwork_demo.py` | TensorNetwork equivalents. |
| `mera_tree_demo.py` | Deep / wide MERA + TTN stress test. |
| `cubic_peps_demo.py` | Cubic PEPS (3D-friendly). |
| `quimb_demo.py` | Hyper-index demo, tensor lists. |
| `tenpy_demo.py` | Finite + infinite MPS/MPO. |
| `einsum_demo.py` | Auto vs manual trace. |
| `einsum_general.py` | Ellipsis, batch hyperedges, traces, short MPS (auto-trace). |
| `tn_tsp.py` | TensorKrowch TSP construction. |

Command catalog: [`examples/README.md`](../examples/README.md).

<a id="toc-limitations"></a>

## Limitations

### `einsum` tracing

The traced helper enforces a **binary**, explicit-output `einsum` string on each recorded step:

- Exactly **two** operands per traced call; **`->`** required; no `out=`.
- **Validation** uses NumPy’s `einsum` on the operand **shapes** (same rules as NumPy/PyTorch for
  duplicates, ellipsis, broadcasting batch indices, etc.).
- **`...`** is supported: ranks are taken from the tensors, and the graph builder expands ellipsis
  to explicit labels when rendering (metadata stores `left_shape` / `right_shape` per step).
- **Visualization:** indices that appear **twice or more** on the LHS of the equation are merged at
  **virtual hub** nodes (hyperedge spokes), including batch indices carried to the output. See
  [`examples/einsum_general.py`](../examples/einsum_general.py).
- **Graph build** still rejects **unary index disappearance** (an index appears on only one operand,
  not in the output, and is summed away) so the fundamental network stays well-formed for the
  current layout code.

Without `trace=`, `tensor_network_viz.einsum` is a thin wrapper and does not enforce tracing rules.

### Subgraphs

Omitted neighbors become **dangling** legs — intentional for inspection.

### Disconnected components

Supported; laid out **together** in one figure (with spacing heuristics).

<a id="toc-troubleshooting"></a>

## Troubleshooting

### Missing package / `ModuleNotFoundError`

Install the extra that matches **`engine=`**:

```bash
python -m pip install "tensor-network-visualization[quimb]"
```

TeNPy maps to **`physics-tenpy`**. **`einsum`** tracing needs **`torch`**.

### Invalid `engine` or `view`

Allowed engines: **`tensorkrowch`**, **`tensornetwork`**, **`quimb`**, **`tenpy`**, **`einsum`**.
Views: **`2d`**, **`3d`**. Anything else raises **`ValueError`**.

### Wrong axis type for backend plotter

Pass a **2D** axis to `plot_*_2d` and an **Axes3D** to `plot_*_3d`. Mixed use raises a clear
**`ValueError`** from [`_prepare_axes`](../src/tensor_network_viz/_core/renderer.py).

### Headless servers / CI

```bash
export MPLBACKEND=Agg   # Linux/macOS
```

```powershell
$env:MPLBACKEND = "Agg"  # Windows PowerShell
```

Always use **`show=False`** for batch saves and **`plt.close(fig)`** when generating many images.

### Jupyter quirks

- **Double figures:** assign **`fig, ax = ...`** or **`show=False`** + explicit **`display`**. 
- **`ipympl`:** needs JupyterLab/Notebook with widget extensions; start kernels after installing. 
- **VS Code / Cursor notebook:** if widgets fail, use **`%matplotlib inline`** or set
  **`MPLBACKEND=inline`** (or **`Agg`** for static files) before importing `pyplot`. This package
  does not read custom env vars—only Matplotlib/Jupyter do. Alternatively run the notebook in the
  **browser** with a full Jupyter stack and **`[jupyter]`** extras.

### `hover_labels` never appears

Requires an **interactive** figure with event handling. Does nothing for **`--no-show`** PNG-only
CLI runs or **Agg** without a GUI loop.

### Slow rendering

- Set **`refine_tensor_labels=False`**.
- Lower **`layout_iterations`** or rely on the automatic cap for large **`n`** (see formula above).

### Surprise figure size when `figsize=None`

If the renderer creates a new figure and **`PlotConfig.figsize` is `None`**, Matplotlib gets
**`(14, 10)`** inches by default in `_prepare_axes`. Set **`figsize=(w, h)`** explicitly if you need
a fixed size.

### Custom positions “ignored”

Keys must match **normalized** node ids. Wrong **`id(...)`** → coords skipped (warnings when
**`validate_positions=True`**).

<a id="toc-internal-architecture"></a>

## Internal architecture

### Public surface

Root package [**`__init__.py`**](../src/tensor_network_viz/__init__.py): `show_tensor_network`,
`PlotConfig`, `EngineName`, `ViewName`, `EinsumTrace`, `einsum`, `pair_tensor`.

### Engine registry

[**`_registry.py`**](../src/tensor_network_viz/_registry.py) maps each **`EngineName`** to a module
and two callables (`plot_*_2d`, `plot_*_3d`). `show_tensor_network` **lazy-imports** that module.

### Graph model

[**`_core/graph.py`**](../src/tensor_network_viz/_core/graph.py): `_NodeData`, `_EdgeData`, endpoints,
`_GraphData`.

### Adapters

Each backend builds `_GraphData`:

- TensorKrowch / TensorNetwork share [**`_nodes_edges_common`**](../src/tensor_network_viz/_core/_nodes_edges_common.py).
- Quimb: [**`quimb/graph.py`**](../src/tensor_network_viz/quimb/graph.py).
- TeNPy: [**`tenpy/graph.py`**](../src/tensor_network_viz/tenpy/graph.py).
- Einsum: [**`einsum_module/graph.py`**](../src/tensor_network_viz/einsum_module/graph.py).

### Drawing

[**`renderer.py`**](../src/tensor_network_viz/_core/renderer.py) resolves `PlotConfig`, positions,
scale, and calls [**`_draw_graph`**](../src/tensor_network_viz/_core/draw/graph_pipeline.py) (re-exported from [**`_draw_common.py`**](../src/tensor_network_viz/_core/_draw_common.py)).

### Layout implementation

The [**`layout` package**](../src/tensor_network_viz/_core/layout/__init__.py) together with
[**`layout_structure.py`**](../src/tensor_network_viz/_core/layout_structure.py) implements
structure detection, planar attempts, force layout, and 3D layering heuristics.

<a id="toc-development-notes"></a>

## Development notes

Verify from repo root:

**Windows:**

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```

**Linux / macOS:**

```bash
ruff check .
pyright
pytest
```

Tests cover adapters, rendering, optional backends, and `einsum` tracing.
