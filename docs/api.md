# API Reference

This page summarizes the public API exposed by `tensor_network_viz`.

## Contents

- [Public Surface](#public-surface)
- [Render a Tensor Network](#render-a-tensor-network)
- [Inspect Tensor Values](#inspect-tensor-values)
- [Compare Two Tensors](#compare-two-tensors)
- [Export Normalized Data](#export-normalized-data)
- [Configuration Objects](#configuration-objects)
- [Einsum Helpers](#einsum-helpers)
- [TeNPy Helper](#tenpy-helper)
- [Cache Helper](#cache-helper)
- [Exceptions and Logging](#exceptions-and-logging)

## Public Surface

The package exports these user-facing names from `tensor_network_viz.__all__`.

| Area | Names |
| --- | --- |
| Rendering | `show_tensor_network`, `show_tensor_elements`, `show_tensor_comparison` |
| Tensor-network config | `PlotConfig`, `TensorNetworkDiagnosticsConfig`, `TensorNetworkFocus`, `PlotTheme`, `EngineName`, `ViewName` |
| Tensor inspection config | `TensorElementsConfig`, `TensorAnalysisConfig`, `TensorComparisonConfig` |
| Normalized exports | `normalize_tensor_network`, `export_tensor_network_snapshot`, `NormalizedTensorGraph`, `NormalizedTensorNode`, `NormalizedTensorEdge`, `NormalizedTensorEndpoint`, `TensorNetworkSnapshot`, `TensorNetworkLayoutSnapshot`, `NormalizedContractionStepMetrics` |
| `einsum` tracing | `EinsumTrace`, `einsum`, `einsum_trace_step`, `pair_tensor` |
| TeNPy explicit networks | `TenPyTensorNetwork`, `make_tenpy_tensor_network` |
| Cache | `clear_tensor_network_graph_cache` |
| Exceptions | `TensorNetworkVizError`, `VisualizationInputError`, `VisualizationTypeError`, `UnsupportedEngineError`, `AxisConfigurationError`, `TensorDataError`, `TensorDataTypeError`, `MissingOptionalDependencyError` |

## Render a Tensor Network

Use `show_tensor_network(...)` for structural visualizations.

```python
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tensor_network_viz import EngineName, PlotConfig, ViewName


def show_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
    view: ViewName | None = None,
    config: PlotConfig | None = None,
    ax: Axes | Axes3D | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes | Axes3D]: ...
```

Main parameters:

| Parameter | Meaning |
| --- | --- |
| `network` | Backend-native network object, supported iterable of nodes/tensors, nested 2D/3D grid, or `EinsumTrace`. |
| `engine` | Optional backend override: `"tensorkrowch"`, `"tensornetwork"`, `"quimb"`, `"tenpy"`, or `"einsum"`. |
| `view` | `"2d"` or `"3d"`. If omitted, the default is usually `"2d"` unless an external 3D axis or 3D grid makes the choice clear. |
| `config` | A `PlotConfig` instance. |
| `ax` | Existing Matplotlib axis. Passing an axis fixes the figure target. |
| `show_controls` | Add embedded controls for view, labels, hover, and contraction playback when available. |
| `show` | Display the figure immediately. In notebooks this uses IPython display; elsewhere it uses `plt.show()`. |

Example:

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(show_tensor_labels=True, hover_labels=True),
    show=False,
)
fig.savefig("network.png", bbox_inches="tight")
```

Important behavior:

- `show_controls=False` disables embedded Matplotlib controls.
- `show=False` skips automatic display and still returns `(fig, ax)`.
- Contraction playback requires `show_controls=True`.
- Repeated calls with the same regular network object can reuse a normalized graph cache.
- Nested grid inputs use grid positions as the base layout. Explicit `PlotConfig(positions=...)`
  overrides only the node ids you pass.
- A nested grid is a `list`/`tuple` layout of existing backend nodes or tensors. It is flattened for
  graph extraction; it does not add bonds between adjacent cells. Use `None` for empty cells.
- Grid inputs are supported for `"tensorkrowch"`, `"tensornetwork"`, and `"quimb"`. For
  TensorKrowch, a grid of nodes may not carry the original network's recovered contraction history;
  pass the network object plus explicit positions when automatic playback history matters.

## Inspect Tensor Values

Use `show_tensor_elements(...)` for heatmaps, distributions, sparsity, spectral views, and tensor
data summaries.

```python
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tensor_network_viz import EngineName, TensorElementsConfig


def show_tensor_elements(
    data: Any,
    *,
    engine: EngineName | None = None,
    config: TensorElementsConfig | None = None,
    ax: Axes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]: ...
```

`data` can be:

- a direct numeric tensor or array-like object,
- an iterable of tensors,
- a supported backend-native tensor container,
- an `EinsumTrace` with live tensor values.

Example:

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
        robust_percentiles=(1.0, 99.0),
    ),
    show=False,
)
```

Notes:

- With several tensors, the figure uses a slider to keep one tensor active at a time.
- An external `ax` is supported only when visualizing one tensor.
- `mode="auto"` chooses a sensible default from the tensor dtype.

## Compare Two Tensors

Use `show_tensor_comparison(...)` for one current tensor against one reference tensor.

```python
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tensor_network_viz import EngineName, TensorComparisonConfig, TensorElementsConfig


def show_tensor_comparison(
    data: Any,
    reference: Any,
    *,
    engine: EngineName | None = None,
    config: TensorElementsConfig | None = None,
    comparison_config: TensorComparisonConfig | None = None,
    ax: Axes | None = None,
    show_controls: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]: ...
```

Example:

```python
import numpy as np
from tensor_network_viz import TensorComparisonConfig, show_tensor_comparison

current = np.array([[1.0, 2.1], [2.9, 4.0]])
reference = np.array([[1.0, 2.0], [3.0, 4.0]])

fig, ax = show_tensor_comparison(
    current,
    reference,
    comparison_config=TensorComparisonConfig(mode="abs_diff"),
    show=False,
)
```

Available comparison modes are:

- `reference`
- `abs_diff`
- `relative_diff`
- `ratio`
- `sign_change`
- `phase_change`
- `topk_changes`

## Export Normalized Data

Use normalized exports when another tool needs backend-independent graph data.

```python
from typing import Any

from tensor_network_viz import (
    EngineName,
    NormalizedTensorGraph,
    PlotConfig,
    PlotTheme,
    TensorNetworkSnapshot,
    ViewName,
)


def normalize_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
) -> NormalizedTensorGraph: ...


def export_tensor_network_snapshot(
    network: Any,
    *,
    engine: EngineName | None = None,
    view: ViewName = "2d",
    config: PlotConfig | None = None,
    seed: int = 0,
) -> TensorNetworkSnapshot: ...
```

`normalize_tensor_network(...)` returns structure only: nodes, edges, endpoints, labels, shapes,
dtypes, and optional contraction-step data.

`export_tensor_network_snapshot(...)` returns the normalized graph plus resolved layout data:
positions, axis directions, draw scale, and curve padding.

Every public snapshot object has `to_dict()` for JSON-friendly export:

```python
snapshot = export_tensor_network_snapshot(network, engine="einsum", view="2d")
payload = snapshot.to_dict()
```

## Configuration Objects

### `PlotConfig`

Use `PlotConfig` with `show_tensor_network(...)`.

Common fields:

| Field | Purpose |
| --- | --- |
| `show_nodes` | Draw full tensor-node geometry instead of compact markers. |
| `show_tensor_labels` | Draw tensor names directly on nodes. |
| `show_index_labels` | Draw index labels on edges. |
| `hover_labels` | Enable hover tooltips. |
| `show_contraction_scheme` | Enable contraction playback controls. |
| `contraction_scheme_cost_hover` | Show contraction-cost information during playback. |
| `contraction_tensor_inspector` | Link playback steps to tensor inspection when tensors are available. |
| `diagnostics` | Optional `TensorNetworkDiagnosticsConfig`. |
| `focus` | Optional `TensorNetworkFocus` for neighborhood/path views. Path focus uses the fewest tensor-to-tensor hops; n-way hubs are collapsed, and cost, memory, bond dimension, and layout distance are not part of the metric. |
| `theme` | Visual preset: `default`, `paper`, or `colorblind`. Manual color and line-width overrides still win. |
| `figsize` | Matplotlib figure size. |
| `positions` | Explicit node-id positions. |
| `layout_iterations` | Force-layout iteration override. |
| `tensor_label_fontsize`, `edge_label_fontsize` | Preferred label sizes. |
| `node_color`, `bond_edge_color`, `dangling_edge_color` | Main styling colors. |
| `contraction_scheme_by_name` | Explicit contraction steps by visible tensor name. |

For clean exported figures, start with the paper theme and disable controls:

```python
fig, ax = show_tensor_network(
    network,
    config=PlotConfig(theme="paper", show_tensor_labels=True),
    show_controls=False,
    show=False,
)
fig.savefig("network-paper.png", dpi=180, bbox_inches="tight")
```

### `TensorElementsConfig`

Use `TensorElementsConfig` with `show_tensor_elements(...)` and as the base rendering config for
`show_tensor_comparison(...)`.

Common fields:

| Field | Purpose |
| --- | --- |
| `mode` | Initial mode: `auto`, `elements`, `magnitude`, `log_magnitude`, `distribution`, `data`, `real`, `imag`, `phase`, `sign`, `signed_value`, `sparsity`, `nan_inf`, `singular_values`, `eigen_real`, `eigen_imag`, `slice`, `reduce`, or `profiles`. |
| `row_axes`, `col_axes` | Matrixization for rank greater than 2. |
| `analysis` | Optional `TensorAnalysisConfig` for slice/reduce/profile modes. |
| `max_matrix_shape` | Downsampled heatmap size limit. |
| `shared_color_scale` | Reuse compatible color scales across the tensor slider. |
| `robust_percentiles` | Percentile-based color scaling. |
| `highlight_outliers` | Overlay outlier markers on heatmaps. |
| `zero_threshold` | Floor for zero-aware modes. |
| `histogram_bins`, `histogram_max_samples` | Distribution mode controls. |
| `topk_count` | Text summary count. |

### `TensorComparisonConfig`

Use `TensorComparisonConfig` with `show_tensor_comparison(...)`.

| Field | Purpose |
| --- | --- |
| `mode` | Comparison transform to display. |
| `zero_threshold` | Positive floor for relative-difference and ratio denominators. |
| `topk_count` | Number of entries in `topk_changes`. |

## Einsum Helpers

`EinsumTrace` records symbolic tensor names and contraction steps. The helper `einsum(...)` can
trace NumPy-backed or PyTorch-backed calls.

```python
import numpy as np
from tensor_network_viz import EinsumTrace, einsum

trace = EinsumTrace()
a = np.ones((2, 3), dtype=float)
b = np.ones((3, 4), dtype=float)

trace.bind("A", a)
trace.bind("B", b)
result = einsum("ab,bc->ac", a, b, trace=trace, backend="numpy")
```

`pair_tensor(...)` and `einsum_trace_step(...)` are public helpers for manual traced steps when you
need tighter control over the trace.

## TeNPy Helper

Use `make_tenpy_tensor_network(...)` when you want to build an explicit TeNPy-compatible network
from named tensors and axis metadata instead of relying on TeNPy-native objects.

## Cache Helper

`clear_tensor_network_graph_cache(network)` clears cached graph extraction for a network object.
Call it after mutating a network in place and before drawing it again.

## Exceptions and Logging

All package-specific exceptions inherit from `TensorNetworkVizError`.

| Exception | When it is useful |
| --- | --- |
| `VisualizationInputError` | Invalid or ambiguous visualization input. |
| `VisualizationTypeError` | Invalid runtime type. |
| `UnsupportedEngineError` | Unknown backend name. |
| `AxisConfigurationError` | Incompatible `ax`, `view`, or figure layout. |
| `TensorDataError` | Missing or unsupported tensor values. |
| `TensorDataTypeError` | Incompatible tensor data type. |
| `MissingOptionalDependencyError` | Backend extra is not installed. |

The package logger is named `tensor_network_viz` and installs a `logging.NullHandler()` by default.
Enable logs in your application when debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tensor_network_viz").setLevel(logging.DEBUG)
```
