# User Guide

This guide focuses on practical workflows: exploring tensor networks, exporting figures, inspecting
tensor values, comparing tensors, and saving backend-normalized graph data.

## Contents

- [Core Idea](#core-idea)
- [Choose the Right Entry Point](#choose-the-right-entry-point)
- [Notebook Workflow](#notebook-workflow)
- [Static Figure Exports](#static-figure-exports)
- [Reuse Existing Matplotlib Axes](#reuse-existing-matplotlib-axes)
- [Layouts and Positions](#layouts-and-positions)
- [Labels, Hover, and Controls](#labels-hover-and-controls)
- [Contraction Schemes](#contraction-schemes)
- [Tensor Value Inspection](#tensor-value-inspection)
- [Tensor Comparison](#tensor-comparison)
- [Snapshots and External Tools](#snapshots-and-external-tools)
- [Supported Inputs](#supported-inputs)
- [Performance Tips](#performance-tips)
- [Related Docs](#related-docs)

## Core Idea

Tensor-Network-Visualization separates three jobs:

1. convert backend-specific objects into one internal graph model,
2. resolve a 2D or 3D layout,
3. render a Matplotlib figure or export normalized data.

Most users only need a few imports:

```python
from tensor_network_viz import (
    PlotConfig,
    TensorElementsConfig,
    show_tensor_comparison,
    show_tensor_elements,
    show_tensor_network,
)
```

The API keeps figure display separate from figure creation:

- `show=False` means "return the figure, but do not display it automatically".
- `show_controls=False` means "render a clean static figure without embedded widgets".

These flags are independent.

## Choose the Right Entry Point

| Goal | Use |
| --- | --- |
| Draw the structure of a tensor network | `show_tensor_network(...)` |
| Inspect values inside one or more tensors | `show_tensor_elements(...)` |
| Compare one tensor against one reference tensor | `show_tensor_comparison(...)` |
| Export backend-independent graph structure | `normalize_tensor_network(...)` |
| Export graph structure plus resolved layout | `export_tensor_network_snapshot(...)` |

Use [api.md](api.md) when you need exact signatures, configuration fields, and exception names.

## Notebook Workflow

Install the Jupyter extra:

```bash
python -m pip install "tensor-network-visualization[jupyter]"
```

In the notebook, enable the widget backend before drawing interactive figures:

```python
%matplotlib widget
```

Then use the same API as in a script:

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    config=PlotConfig(show_tensor_labels=True, hover_labels=True),
)
```

If a notebook displays the same figure twice, prefer this pattern:

```python
fig, ax = show_tensor_network(network, show=False)
fig
```

For static notebook output, disable controls:

```python
fig, ax = show_tensor_network(network, show_controls=False, show=False)
fig
```

## Static Figure Exports

For clean exported images, create the figure without automatic display and without controls:

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    config=PlotConfig(theme="paper", show_tensor_labels=True),
    show_controls=False,
    show=False,
)
fig.savefig("network.png", dpi=180, bbox_inches="tight")
```

Use this pattern for papers, reports, CI-generated images, and scripts that should not open a GUI
window. Use `theme="paper"` for clean exports, or `theme="colorblind"` when maximum color
distinguishability matters.

Tensor inspection exports work the same way:

```python
from tensor_network_viz import TensorElementsConfig, show_tensor_elements

fig, ax = show_tensor_elements(
    tensor,
    config=TensorElementsConfig(mode="magnitude"),
    show_controls=False,
    show=False,
)
fig.savefig("tensor-values.png", bbox_inches="tight")
```

## Reuse Existing Matplotlib Axes

Pass `ax=` when you want to place a visualization in an existing Matplotlib layout.

```python
import matplotlib.pyplot as plt
from tensor_network_viz import show_tensor_network

fig, ax = plt.subplots(figsize=(7, 5))
show_tensor_network(network, ax=ax, view="2d", show_controls=False, show=False)
```

Rules to remember:

- A 2D axis needs `view="2d"`.
- A 3D axis needs `view="3d"`.
- Passing an axis suppresses the built-in view selector.
- `show_tensor_elements(..., ax=...)` supports an external axis only for a single tensor.

## Layouts and Positions

By default, the library computes positions for the graph. You can influence layout in three ways.

### Select 2D or 3D

```python
show_tensor_network(network, view="2d")
show_tensor_network(network, view="3d")
```

### Pass Grid Inputs

Nested 2D and 3D grids fix regular cell positions:

```python
# left, middle, right, and bottom are existing backend nodes.
grid = [
    [left, middle, right],
    [None, bottom, None],
]

fig, ax = show_tensor_network(grid, engine="tensornetwork", view="2d")
```

Each non-`None` cell must be an existing backend node or tensor. The grid is a layout hint: it is
flattened before graph extraction, while the cell coordinates become the base positions. Bonds are
still read from the backend objects, so adjacent cells are not connected automatically.

Use `None` for empty cells. In 3D, use `network[layer][row][col]`; 3D grid inputs default to
`view="3d"` when `view` is omitted. Grid inputs are supported for TensorKrowch, TensorNetwork, and
Quimb. They are rejected for TeNPy and `einsum`.

For TensorKrowch contraction playback, prefer passing the original TensorKrowch network object when
you rely on automatic history recovery. A grid of TensorKrowch nodes fixes the layout, but it does
not expose the network-level `leaf_nodes` and `resultant_nodes` history used for automatic recovered
schemes. If you need both fixed positions and recovered playback, pass the network object and set
`PlotConfig(positions={id(node): (...)})` instead.

### Pass Explicit Positions

`PlotConfig(positions=...)` accepts node ids mapped to 2D or 3D coordinates.

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    positions={
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, 0.8),
    },
    validate_positions=True,
)

fig, ax = show_tensor_network(network, config=config, view="2d")
```

When a grid and explicit positions are both used, explicit positions override only the node ids you
provide.

### Use Directional Axis Names

Dangling/free axis names can act as direction hints. Names such as `left`, `right`, `up`, `down`,
`front`, `back`, `xp`, `xm`, `yp`, `ym`, `zp`, `zm`, `north`, `south`, `east`, `west`, `in`, and
`out` are treated as hard directions for dangling indices.

## Labels, Hover, and Controls

Use static labels when the figure must remain readable after export:

```python
config = PlotConfig(
    show_tensor_labels=True,
    show_index_labels=True,
)
```

Use hover labels when you are exploring interactively:

```python
config = PlotConfig(hover_labels=True)
```

The most common controls are:

- tensor labels on/off,
- index labels on/off,
- hover labels on/off,
- 2D/3D view selector when no external axis fixes the view,
- contraction playback when the network exposes or receives contraction steps.

For dense graphs, static labels can become noisy. A practical pattern is to export with tensor
labels only, then keep index labels available through hover while exploring.

Use visual themes when you want a consistent look without setting each color manually:

```python
config = PlotConfig(theme="colorblind", hover_labels=True)
```

Manual color settings still override the selected theme, so `PlotConfig(theme="paper",
node_color="#ABCDEF")` keeps the custom node color.

## Contraction Schemes

Contraction schemes show how groups of tensors are merged step by step.

```python
from tensor_network_viz import PlotConfig, show_tensor_network

config = PlotConfig(
    show_contraction_scheme=True,
    contraction_scheme_cost_hover=True,
)

fig, ax = show_tensor_network(trace, engine="einsum", config=config)
```

You can provide explicit steps by visible tensor name:

```python
config = PlotConfig(
    show_contraction_scheme=True,
    contraction_scheme_by_name=(
        ("A", "B"),
        ("AB", "C"),
    ),
)
```

Playback details depend on the backend:

- `einsum` traces can carry contraction steps and live tensor values.
- TensorKrowch can recover contraction history for some already-contracted native networks when
  usable `leaf_nodes` and `resultant_nodes` history remains.
- Manual schemes can draw step groups even when detailed cost data is unavailable.

For linked tensor inspection during playback, set:

```python
config = PlotConfig(
    show_contraction_scheme=True,
    contraction_tensor_inspector=True,
)
```

## Tensor Value Inspection

Use `show_tensor_elements(...)` when you care about actual tensor values, not only graph structure.

```python
import numpy as np
from tensor_network_viz import TensorElementsConfig, show_tensor_elements

tensor = np.arange(24, dtype=float).reshape(2, 3, 4)

fig, ax = show_tensor_elements(
    tensor,
    config=TensorElementsConfig(
        mode="elements",
        row_axes=(0,),
        col_axes=(1, 2),
    ),
)
```

Useful modes:

| Mode | Use |
| --- | --- |
| `elements` | Raw real-valued heatmap. |
| `magnitude` | Absolute value heatmap, especially useful for complex data. |
| `log_magnitude` | Magnitude across a wide dynamic range. |
| `distribution` | Histogram of tensor values. |
| `data` | Text summary with top entries. |
| `real`, `imag`, `phase` | Complex-valued diagnostics. |
| `sign`, `signed_value`, `sparsity`, `nan_inf` | Diagnostic views. |
| `singular_values`, `eigen_real`, `eigen_imag` | Spectral views. |
| `slice`, `reduce`, `profiles` | Analytical views for higher-rank tensors. |

For rank greater than 2, choose row and column axes to control matrixization:

```python
config = TensorElementsConfig(row_axes=(0, 1), col_axes=(2,))
```

For robust color scaling:

```python
config = TensorElementsConfig(
    mode="magnitude",
    robust_percentiles=(1.0, 99.0),
    highlight_outliers=True,
)
```

## Tensor Comparison

Use `show_tensor_comparison(...)` when you have one current tensor and one reference tensor.

```python
import numpy as np
from tensor_network_viz import TensorComparisonConfig, show_tensor_comparison

current = np.array([[1.0, 2.1], [2.9, 4.0]])
reference = np.array([[1.0, 2.0], [3.0, 4.0]])

fig, ax = show_tensor_comparison(
    current,
    reference,
    comparison_config=TensorComparisonConfig(mode="relative_diff"),
)
```

Start with:

- `abs_diff` for absolute numerical changes,
- `relative_diff` for scale-aware changes,
- `ratio` when multiplicative drift matters,
- `topk_changes` when you want a compact ranked text view.

Both inputs must resolve to exactly one tensor, and their shapes must match for numeric comparison
modes.

## Snapshots and External Tools

Use normalized exports when you want data instead of a Matplotlib figure.

```python
from tensor_network_viz import export_tensor_network_snapshot, normalize_tensor_network

graph = normalize_tensor_network(network, engine="quimb")
graph_payload = graph.to_dict()

snapshot = export_tensor_network_snapshot(network, engine="quimb", view="2d")
snapshot_payload = snapshot.to_dict()
```

`normalize_tensor_network(...)` includes graph structure: nodes, edges, endpoints, shapes, dtypes,
labels, and optional contraction steps.

`export_tensor_network_snapshot(...)` adds layout information: positions, axis directions, draw
scale, and curve padding.

Use snapshots when you want to:

- feed another visualization system,
- compare graph extraction in tests,
- store lightweight diagnostics,
- inspect the resolved layout without parsing Matplotlib artists.

## Supported Inputs

### TensorKrowch

Use native `TensorNetwork` objects or supported node lists. Install the `tensorkrowch` extra first.

### TensorNetwork

Use native `tn.Node` objects or lists of connected nodes. Install the `tensornetwork` extra first.

### Quimb

Use `quimb.tensor.TensorNetwork` objects. Install the `quimb` extra first.

### TeNPy

Use supported TeNPy objects or build explicit networks with `make_tenpy_tensor_network(...)`.
Install the `tenpy` extra first.

### Direct Tensor Inputs

Use direct numeric arrays with `show_tensor_elements(...)` and `show_tensor_comparison(...)`.

### `einsum`

Use `EinsumTrace` for traced NumPy or PyTorch contraction workflows. NumPy-backed traces work with
base dependencies; PyTorch-backed traces need the `einsum` extra.

More backend-specific examples are in [backends.md](backends.md).

## Performance Tips

For large graphs:

- prefer `view="2d"` while exploring,
- use `show_index_labels=False`,
- keep `hover_labels=True` instead of drawing every label,
- reduce `layout_iterations`,
- use `show_controls=False` for final static exports,
- reuse the same network object when possible so graph extraction can be cached.

Example:

```python
config = PlotConfig(
    show_tensor_labels=False,
    show_index_labels=False,
    hover_labels=True,
    layout_iterations=80,
)

fig, ax = show_tensor_network(network, config=config, view="2d")
```

After mutating a network in place, clear the graph cache before drawing again:

```python
from tensor_network_viz import clear_tensor_network_graph_cache

clear_tensor_network_graph_cache(network)
```

For tensor inspection:

- set `max_matrix_shape` lower for very large tensors,
- use `distribution` or `data` mode when a full heatmap is not helpful,
- use `robust_percentiles` to avoid one extreme value flattening the color range.

## Related Docs

- [README](../README.md): quick overview, minimal install, and basic example.
- [Installation](installation.md): virtual environments, optional extras, Jupyter, and development setup.
- [API Reference](api.md): exact public API names and practical signatures.
- [Backend Examples](backends.md): copy-paste examples by backend.
- [Troubleshooting](troubleshooting.md): common failures and fixes.
- [Repository Examples](../examples/README.md): command-line demos in this repository.
