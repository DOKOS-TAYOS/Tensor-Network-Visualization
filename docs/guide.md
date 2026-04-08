# Tensor-Network-Visualization Guide

This guide is organized around user workflows instead of internal implementation details.
Use [`docs/backends.md`](backends.md) when you want backend-specific copy-paste examples instead of
workflow guidance.

## Core Idea

The public API is intentionally split into a few small responsibilities:

- `show_tensor_network(...)` manages the figure lifecycle.
- `show_tensor_elements(...)` manages tensor element inspection figures.
- `show_tensor_comparison(...)` manages one-current-vs-one-reference comparison figures.
- `normalize_tensor_network(...)` exports the backend-normalized structural graph.
- `export_tensor_network_snapshot(...)` exports the normalized graph plus resolved layout data.
- `PlotConfig(...)` manages how tensor networks should look and behave.
- `TensorElementsConfig(...)` manages how tensor inspection should look and behave.
- `TensorComparisonConfig(...)` manages how tensor comparison should behave.

If you remember only one thing, remember this:

```python
fig, ax = show_tensor_network(network, config=PlotConfig(...))
fig, ax = show_tensor_elements(data, config=TensorElementsConfig(...))
fig, ax = show_tensor_comparison(
    data,
    reference,
    config=TensorElementsConfig(...),
    comparison_config=TensorComparisonConfig(...),
)
```

## Public API

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

```python
show_tensor_elements(
    data,
    *,
    engine=None,
    config=None,
    ax=None,
    show_controls=True,
    show=True,
)
```

```python
show_tensor_comparison(
    data,
    reference,
    *,
    engine=None,
    config=None,
    comparison_config=None,
    ax=None,
    show_controls=True,
    show=True,
)
```

`show_controls` and `show` are independent in both public entry points:

- `show_controls=False` renders a static figure with no embedded widgets.
- `show=False` skips automatic display and just returns `(fig, ax)`.

### Parameters

| Parameter | Meaning |
| --- | --- |
| `network` | Backend-native tensor network object or supported iterable/trace. |
| `engine` | Optional explicit backend: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"`. |
| `view` | `"2d"` or `"3d"`. If omitted, the initial view is `"2d"`. |
| `config` | A `PlotConfig` instance. If omitted, `PlotConfig()` is used. |
| `ax` | Existing Matplotlib axis to draw into. |
| `show_controls` | If `True`, add the figure control panel. If `False`, render only the network. |
| `show` | If `True`, display the figure immediately. If `False`, just return `(fig, ax)`. |

| Parameter | Meaning |
| --- | --- |
| `data` | Direct numeric tensor input, direct iterable of tensors (order preserved, duplicates allowed), backend-native tensor container, or `EinsumTrace` with live tensors. |
| `engine` | Optional explicit backend: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"`. |
| `config` | A `TensorElementsConfig` instance. If omitted, `TensorElementsConfig()` is used. |
| `ax` | Existing Matplotlib axis for single-tensor rendering only. |
| `show_controls` | If `True`, add compact `group + mode` controls and, when several tensors are present, a tensor slider. |
| `show` | If `True`, display the figure immediately. If `False`, just return `(fig, ax)`. |

| Parameter | Meaning |
| --- | --- |
| `data` | The current tensor to inspect. |
| `reference` | The reference tensor used for comparison. |
| `engine` | Optional explicit backend: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"`. |
| `config` | A `TensorElementsConfig` instance for matrixization and rendering details. |
| `comparison_config` | A `TensorComparisonConfig` instance for comparison mode and zero-aware behavior. |
| `ax` | Existing Matplotlib axis for single-tensor rendering only. |
| `show_controls` | If `True`, add the compare-mode selector together with the tensor-elements controls. |
| `show` | If `True`, display the figure immediately. If `False`, just return `(fig, ax)`. |

```python
normalize_tensor_network(network, *, engine=None)
export_tensor_network_snapshot(network, *, engine=None, view="2d", config=None, seed=0)
```

Both snapshot helpers return immutable public objects with `.to_dict()` methods so external tools
can serialize or validate the backend-normalized graph and layout.

## Errors and Diagnostics

The public API raises package-specific exceptions so callers can distinguish user-input problems
from unrelated runtime failures without parsing error strings:

- `TensorNetworkVizError`: root class for package-specific failures.
- `VisualizationInputError`: unsupported or ambiguous network input.
- `AxisConfigurationError`: incompatible `ax`, `view`, or figure-control setup.
- `UnsupportedEngineError`: unknown backend name.
- `TensorDataError`: unsupported tensor values or collections for `show_tensor_elements(...)`.
- `MissingOptionalDependencyError`: backend requested but its dependency is not installed.

These classes deliberately preserve compatibility with the built-in families they refine
(`ValueError` or `ImportError`), so existing downstream handlers keep working.

For diagnostics, enable the package logger:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tensor_network_viz").setLevel(logging.DEBUG)
```

The logger name is `tensor_network_viz`, and the library installs a `NullHandler`, so imports stay
quiet unless your application opts in.

## `PlotConfig` in Practice

`PlotConfig` is where visual behavior lives.

### Labels and hover

```python
config = PlotConfig(
    show_tensor_labels=True,
    show_index_labels=False,
    hover_labels=True,
)
```

### Contraction scheme

```python
config = PlotConfig(
    show_contraction_scheme=True,
    contraction_scheme_cost_hover=True,
)
```

Important detail:

- if `show_controls=True`, the figure exposes a `Scheme` toggle plus the slider controls;
- if `contraction_scheme_cost_hover=True`, the current slider step shows a fixed detail panel;
- if `show_controls=False`, `show_contraction_scheme=True` raises `ValueError` because the scheme is dynamic-only.

### Performance-oriented rendering

```python
config = PlotConfig(
    tensor_label_refinement="never",
    layout_iterations=120,
)
```

`tensor_label_refinement` can be:

- `"auto"`: default, balanced choice.
- `"always"`: best label fitting, slowest on large figures.
- `"never"`: skip the expensive post-draw label-fit passes.

### Custom positions

```python
config = PlotConfig(
    positions={
        id(node_a): (0.0, 0.0),
        id(node_b): (1.5, 0.0),
    },
    validate_positions=True,
)
```

## `TensorElementsConfig` in Practice

`TensorElementsConfig` is where tensor inspection behavior lives.

### Useful defaults

```python
config = TensorElementsConfig(
    mode="auto",
    max_matrix_shape=(256, 256),
    histogram_bins=40,
    topk_count=8,
    robust_percentiles=(1.0, 99.0),
)
```

When multiple tensors are present, `show_tensor_elements(...)` keeps one tensor visible at a time.
The slider changes the active tensor. The controls are grouped so you first choose a family of
views and then the concrete mode inside that family:

- `basic`: `elements`, `magnitude`, `log_magnitude`, `distribution`, `data`
- `complex`: `real`, `imag`, `phase`
- `diagnostic`: `sign`, `signed_value`, `sparsity`, `nan_inf`, `singular_values`, `eigen_real`,
  `eigen_imag`

`data` mode combines the global tensor stats with a compact per-axis summary and the top-k entries
by magnitude. The `singular_values` mode renders the singular-value spectrum derived from the same
matrixized tensor used by the heatmap views. The `eigen_real` and `eigen_imag` modes render the
real and imaginary parts of the corresponding eigenvalues, ordered by eigenvalue magnitude. These
spectral modes hide themselves automatically whenever the active analysis matrix is not finite, and
the eigenvalue views also stay hidden for non-square analysis matrices. Use
`shared_color_scale=True` when you want slider-based tensor comparisons to reuse the same limits,
and `highlight_outliers=True` to overlay extreme values on continuous heatmaps.

### Rank > 2 tensors

```python
config = TensorElementsConfig(
    row_axes=("left", "phys"),
    col_axes=("right",),
)
```

If you omit `row_axes` / `col_axes`, the library chooses a deterministic balanced partition.

## Common Workflows

### 1. Explore interactively

```python
fig, ax = show_tensor_network(
    network,
    config=PlotConfig(
        hover_labels=True,
        show_tensor_labels=False,
        show_index_labels=False,
    ),
)
```

Use this when you want the embedded controls and interactive hover behavior.

When the network exposes tensor values, clicking a visible tensor node opens the auxiliary tensor
inspector directly. For playback-enabled traces, the inspector still follows the current
contraction result by default, but a manual node click pins the inspector to that tensor until you
click empty space to clear the manual selection.

### 2. Export a clean figure

```python
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

Use this for papers, reports, or batch generation.

### 3. Reuse an existing subplot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
show_tensor_network(
    network,
    config=PlotConfig(figsize=None),
    ax=ax,
    show=False,
)
```

If you pass `ax`, the plot is rendered into that axis and the figure is not recreated.
For `show_tensor_elements(...)`, an external `ax` is only supported when visualizing a single
tensor.

### 4. Inspect tensor values

```python
fig, ax = show_tensor_elements(
    trace,
    config=TensorElementsConfig(mode="auto"),
    show=False,
)
fig.savefig("tensor-elements.png", bbox_inches="tight")
```

Use this when you want one tensor at a time, quick switches between grouped views, and a `data`
summary without leaving the same figure.

### 5. Compare a tensor against a reference

```python
fig, ax = show_tensor_comparison(
    current_tensor,
    reference_tensor,
    config=TensorElementsConfig(mode="elements"),
    comparison_config=TensorComparisonConfig(mode="relative_diff"),
    show=False,
)
```

Use `TensorComparisonConfig(mode=...)` to switch between `reference`, `abs_diff`,
`relative_diff`, `ratio`, `sign_change`, `phase_change`, and `topk_changes`.

### 6. Export the normalized graph for external tooling

```python
graph = normalize_tensor_network(network)
snapshot = export_tensor_network_snapshot(network, view="3d")
```

Use these helpers when you want validators, snapshot tests, or custom tooling to consume the same
backend-normalized structure that the plotting layer already uses internally.

## Supported Inputs

### TensorKrowch

- `TensorNetwork`
- iterable of TensorKrowch nodes
- single TensorKrowch node with a materialized `tensor`

### TensorNetwork

- iterable of `tensornetwork.Node`
- single `tensornetwork.Node`

### Quimb

- `quimb.tensor.TensorNetwork`
- iterable of `quimb.tensor.Tensor`
- single `quimb.tensor.Tensor`

### TeNPy

- common native MPS/MPO-style objects
- explicit `TenPyTensorNetwork`
- single TeNPy tensor exposing `to_ndarray()` and `get_leg_labels()`

### Direct tensor inputs

- single NumPy / array-like tensor input
- direct iterables of tensors preserve order and duplicates; they are treated as inspection data,
  not as backend container objects

### `einsum`

- `EinsumTrace`
- manual `pair_tensor(...)` / `einsum_trace_step(...)` lists are only valid for
  `show_tensor_network(...)`, not for `show_tensor_elements(...)`

More detailed copy-paste examples: [backends.md](backends.md)

## Using the Returned Figure

Every path returns ordinary Matplotlib objects.

```python
fig, ax = show_tensor_network(network, show=False)
ax.set_title("My tensor network")
fig.savefig("out.png", bbox_inches="tight")
```

```python
fig, ax = show_tensor_elements(data, show=False)
fig.savefig("tensor-elements.png", bbox_inches="tight")
```

That means you can:

- add titles,
- save to disk,
- embed in larger figures,
- continue styling with standard Matplotlib.

## Troubleshooting

### Saved figure still shows controls

Use:

```python
show_tensor_network(..., show_controls=False, show=False)
```

### Hover does not work

Hover requires an interactive Matplotlib backend. It is not meaningful for pure PNG export flows.

### Large graph is slow

Try this first:

```python
config = PlotConfig(
    tensor_label_refinement="never",
    layout_iterations=120,
)
```

Then consider passing `positions` if you already know the layout you want.

### Tensor inspection fails for TensorKrowch shape-only nodes

`show_tensor_elements(...)` needs real tensor values. Nodes with only `shape` metadata are not
enough.

### Tensor inspection fails for manual `pair_tensor(...)` lists

Manual trace steps describe contractions, not tensor values. Use `EinsumTrace` and keep the traced
tensors alive until you render them.

### `AxisConfigurationError` appears immediately

This means the plotting surface and the requested behavior disagree. Typical cases:

- `show_tensor_network(..., view="3d", ax=<2D axis>)`
- `show_tensor_network(..., view="2d", ax=<3D axis>)`
- `show_tensor_elements(..., ax=...)` with more than one tensor selected
- `show_tensor_elements(..., show_controls=True, ax=...)` on a figure that already has extra axes

### Jupyter shows duplicate output

Prefer:

```python
fig, ax = show_tensor_network(...)
```

instead of leaving the returned tuple as the last expression in the cell.

## Related Docs

- [README.md](../README.md): quick overview and install guide.
- [backends.md](backends.md): backend-specific examples.
- [examples/README.md](../examples/README.md): launcher usage.
