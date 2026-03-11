# Tensor-Network-Visualization Guide

This guide covers the current public surface of the project: installation, common usage, supported
inputs for each backend, configuration, example scripts, troubleshooting, and the important parts
of the internal architecture.

## Overview

`tensor_network_viz` is a small visualization package built around one idea:

1. accept backend-native tensor-network objects
2. normalize them into a shared graph model
3. compute positions and axis directions
4. draw the result with Matplotlib

The package supports both a generic dispatcher (`show_tensor_network(...)`) and backend-specific
helper functions such as `plot_quimb_network_2d(...)`.

## Installation

### Runtime installation

Install the base package plus the extra matching the backend you want to render:

```bash
pip install "tensor-network-visualization[tensorkrowch]"
pip install "tensor-network-visualization[tensornetwork]"
pip install "tensor-network-visualization[quimb]"
pip install "tensor-network-visualization[tenpy]"
pip install "tensor-network-visualization[einsum]"
```

### Which extra should you choose?

- `tensorkrowch`: visualize TensorKrowch networks or node subsets
- `tensornetwork`: visualize `tensornetwork.Node` collections
- `quimb`: visualize Quimb `TensorNetwork` objects or tensor lists
- `tenpy`: visualize TeNPy `MPS` and `MPO` objects
- `einsum`: execute and trace binary `einsum` calls with PyTorch

### Editable installs for development

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Runtime-only editable install:

```powershell
.\.venv\Scripts\python -m pip install -e .
```

The repository also includes thin wrapper files:

- `requirements.txt` for `-e .`
- `requirements.dev.txt` for `-e ".[dev]"`

## Core Usage

### The generic dispatcher

The main entry point is:

```python
show_tensor_network(
    network,
    engine="tensorkrowch",
    view="2d",
    config=None,
    show_tensor_labels=None,
    show_index_labels=None,
    show=True,
)
```

It returns a tuple:

```python
(fig, ax)
```

Use `show=False` when you want to:

- add a title
- save the figure
- embed the output in another application
- avoid opening an interactive Matplotlib window

### Minimal example

```python
from tensor_network_viz import PlotConfig, show_tensor_network

config = PlotConfig(figsize=(8, 6))

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=config,
    show=False,
)

ax.set_title("Network")
fig.savefig("network.png", bbox_inches="tight")
```

### Backend-specific helpers

If you already know which backend you want, you can call the engine-specific entry points
directly:

```python
from tensor_network_viz.quimb import plot_quimb_network_2d

fig, ax = plot_quimb_network_2d(quimb_network)
```

These helpers are thin wrappers around the same shared rendering core.

## Supported Inputs by Engine

### TensorKrowch

Accepted inputs:

- a TensorKrowch network object exposing `nodes` or `leaf_nodes`
- any iterable of TensorKrowch nodes

Behavior notes:

- the adapter reads node `edges`, `axes_names`, and `name`
- if you pass a subset of nodes, missing neighbors become dangling legs
- disconnected components are supported

### TensorNetwork

Accepted inputs:

- any iterable of `tensornetwork.Node`

Behavior notes:

- nodes must expose `edges`, `axis_names`, and `name`
- unordered collections are normalized to a stable order before graph construction

### Quimb

Accepted inputs:

- a Quimb `TensorNetwork`
- any iterable of Quimb `Tensor`

Behavior notes:

- tensor tags become display names when available
- if a tensor has no useful tags, the adapter assigns fallback names like `T0`, `T1`, ...
- hyper-indices shared by three or more tensors are routed through invisible internal hub nodes so
  the original topology remains visible without introducing extra tensor markers

### TeNPy

Accepted inputs:

- finite, segment, or infinite `tenpy.networks.mps.MPS`
- finite or infinite `tenpy.networks.mpo.MPO`

Behavior notes:

- finite chains render naturally as line-like tensor networks
- infinite `MPS` and `MPO` objects are rendered as one periodic unit cell
- the adapter focuses on structural connectivity rather than physical tensor values

### `einsum`

Accepted inputs:

- an `EinsumTrace`
- an ordered iterable of `pair_tensor`

There are two supported workflows.

#### Auto tracing

```python
from tensor_network_viz import EinsumTrace, einsum, show_tensor_network

trace = EinsumTrace()
trace.bind("A0", a0)
trace.bind("x0", x0)

r0 = einsum("pa,p->a", a0, x0, trace=trace, backend="torch")

fig, ax = show_tensor_network(trace, engine="einsum", view="2d", show=False)
```

#### Manual trace construction

```python
from tensor_network_viz import pair_tensor, show_tensor_network

trace = [
    pair_tensor("A0", "x0", "r0", "pa,p->a"),
    pair_tensor("r0", "A1", "r1", "a,apb->pb"),
]

fig, ax = show_tensor_network(trace, engine="einsum", view="2d", show=False)
```

Behavior notes:

- the visualization shows the underlying tensor network, not the intermediate contraction tensors
- trace entries must preserve order
- the MVP tracing API supports binary, explicit-output equations only

## `PlotConfig`

`PlotConfig` controls styling and layout. Important fields:

- `figsize`: figure size in inches
- `node_color`
- `node_edge_color`
- `tensor_label_color`
- `label_color`
- `bond_edge_color`
- `dangling_edge_color`
- `show_tensor_labels`
- `show_index_labels`
- `node_radius`
- `stub_length`
- `self_loop_radius`
- `line_width_2d`
- `line_width_3d`
- `layout_iterations`
- `positions`
- `validate_positions`

### Example

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    figsize=(10, 6),
    node_color="#F4F1EA",
    bond_edge_color="#173F7A",
    dangling_edge_color="#8B1E1E",
    show_tensor_labels=True,
    show_index_labels=True,
    layout_iterations=300,
)
```

### Custom positions

`positions` is a dictionary keyed by node id:

```python
config = PlotConfig(
    positions={
        id(node_a): (0.0, 0.0),
        id(node_b): (1.0, 0.0),
    },
    validate_positions=True,
)
```

Important details:

- use `(x, y)` for `2d` and `(x, y, z)` for `3d`
- if only some nodes have explicit positions, the remaining nodes are placed by the layout engine
- `layout_iterations` still affects that fallback placement
- `validate_positions=True` warns about unknown ids or dimension mismatches

## Working with the Returned Figure

All plotting entry points return Matplotlib objects, so normal Matplotlib customization works:

```python
fig, ax = show_tensor_network(network, engine="tenpy", view="2d", show=False)
ax.set_title("Finite MPO")
fig.savefig("finite-mpo.png", bbox_inches="tight")
```

For scripts or batch jobs, a common pattern is:

- set `show=False`
- save the figure
- close it explicitly if needed

## Example Scripts

The repository ships example scripts in [`examples/`](../examples/README.md):

- `tensorkrowch_demo.py`: regular TensorKrowch examples such as MPS, MPO, PEPS, and disconnected
  graphs
- `tensornetwork_demo.py`: the equivalent examples using `tensornetwork.Node`
- `quimb_demo.py`: includes a hypergraph example
- `tenpy_demo.py`: finite and infinite `MPS`/`MPO`
- `einsum_demo.py`: automatic and manual trace construction
- `tn_tsp.py`: a larger TensorKrowch example based on a TSP tensor-network construction

Use these scripts when you want runnable examples of accepted inputs, saving figures, and CLI
behavior.

## Limitations and Troubleshooting

### Missing backend package

If a backend import fails, install the matching extra. For example:

```bash
pip install "tensor-network-visualization[quimb]"
```

### Interactive windows in scripts or CI

Use `show=False` in Python code, or `--no-show` in the example scripts that support it. The
examples also switch Matplotlib to `Agg` when saving without showing.

### `einsum` tracing restrictions

The tracing layer currently supports only a constrained subset of `einsum`:

- exactly two operands
- explicit output with `->`
- no ellipsis
- no repeated labels within one operand or output
- no unary reductions
- no `out=` when tracing

These restrictions apply to the traced `einsum` backend, not to general `torch.einsum` or
`numpy.einsum`.

### Subset rendering

If you pass only part of a larger network, any connection to omitted nodes is rendered as a
dangling edge. This is intentional and often useful when inspecting subgraphs.

### Disconnected components

Disconnected components are supported. The visualizer will lay them out in the same figure.

### Node ids for custom positions

Custom positions are keyed by the Python object id of the normalized node. This is most natural
when you already have direct access to the input node objects and want to pin specific nodes to
known coordinates.

## Internal Architecture

This section is aimed at contributors or advanced users who want to understand how the package is
organized.

### Public surface

The root package exposes:

- `show_tensor_network`
- `PlotConfig`
- `EngineName`
- `ViewName`
- `EinsumTrace`
- `einsum`
- `pair_tensor`

Each backend package also exposes 2D and 3D helper functions.

### Engine registry

`tensor_network_viz._registry` maps each engine name to:

- the backend module path
- the 2D plot function name
- the 3D plot function name

The registry is used by `show_tensor_network(...)` to lazy-load the proper backend module only when
needed.

### Normalized graph model

All backends eventually produce the shared graph representation defined in
`tensor_network_viz._core.graph`:

- `_NodeData`
- `_EdgeData`
- `_EdgeEndpoint`
- `_GraphData`

This keeps the drawing code backend-agnostic.

### Backend adapters

Each backend is responsible for converting its native objects into `_GraphData`.

- TensorKrowch and TensorNetwork share the node/edge normalization helpers in
  `_core._nodes_edges_common`
- Quimb has its own adapter because it works in terms of tensors and indices rather than explicit
  edge objects
- TeNPy has its own adapter because it derives connectivity from `MPS` and `MPO` structure
- `einsum_module.graph` reconstructs a tensor network from an ordered trace of contractions

### Layout pipeline

The shared layout engine lives in `tensor_network_viz._core.layout`.

The current flow is:

1. use custom positions if provided
2. in `2d`, try a grid layout for grid-like graphs
3. in `2d`, try a planar layout if possible
4. otherwise fall back to a force-directed layout
5. compute per-axis directions for stubs, contractions, and labels

The layout code also includes heuristics for free-leg directions so dangling indices and labels do
not all point in the same direction.

### Drawing pipeline

The shared renderer lives in `tensor_network_viz._core.renderer`.

It:

- resolves the effective `PlotConfig`
- prepares a 2D or 3D Matplotlib axis
- computes positions and axis directions
- delegates drawing to the shared primitives in `_core._draw_common`

The actual 2D and 3D frontends are:

- `_core.draw_2d`
- `_core.draw_3d`

These use the same shared geometry helpers for curved edges, self-loops, labels, and node drawing.

### Important backend-specific internals

- Quimb hyperedges are rewritten as virtual hubs so multi-tensor indices remain legible.
- Infinite TeNPy structures are rendered as periodic unit cells instead of pretending to be finite
  chains.
- `EinsumTrace` tracks tensor identity, names, and consumption state so the trace can be converted
  back into a normalized graph later.

## Development Notes

Repository verification commands:

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python -m pytest
```

The test suite covers:

- backend normalization
- rendering behavior
- examples
- integration with real optional dependencies
- `einsum` tracing behavior
