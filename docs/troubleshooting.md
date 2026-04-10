# Troubleshooting

This page lists common problems and the shortest practical fix.

## Contents

- [Missing Optional Dependency](#missing-optional-dependency)
- [Jupyter Shows Duplicate Output](#jupyter-shows-duplicate-output)
- [Jupyter Controls Do Not Respond](#jupyter-controls-do-not-respond)
- [Hover Labels Do Not Work](#hover-labels-do-not-work)
- [Saved Figure Still Shows Controls](#saved-figure-still-shows-controls)
- [Large Graphs Are Slow](#large-graphs-are-slow)
- [Axis Configuration Errors](#axis-configuration-errors)
- [Tensor Inspection Cannot Read Data](#tensor-inspection-cannot-read-data)
- [Comparison Shapes Do Not Match](#comparison-shapes-do-not-match)
- [Backend-Specific Notes](#backend-specific-notes)
- [Enable Debug Logging](#enable-debug-logging)

## Missing Optional Dependency

If you see `MissingOptionalDependencyError`, install the extra for the backend you are using:

```bash
python -m pip install "tensor-network-visualization[quimb]"
```

Available extras:

- `tensorkrowch`
- `tensornetwork`
- `quimb`
- `tenpy`
- `einsum`
- `jupyter`

You can combine extras:

```bash
python -m pip install "tensor-network-visualization[jupyter,quimb,einsum]"
```

## Jupyter Shows Duplicate Output

If a notebook displays the same figure twice, create the figure without automatic display and then
return the figure as the last cell value:

```python
fig, ax = show_tensor_network(network, show=False)
fig
```

For static output:

```python
fig, ax = show_tensor_network(network, show_controls=False, show=False)
fig
```

## Jupyter Controls Do Not Respond

Install the Jupyter extra:

```bash
python -m pip install "tensor-network-visualization[jupyter]"
```

Then enable the widget backend before drawing:

```python
%matplotlib widget
```

If you already created figures before switching backend, restart the kernel and run the notebook
again from the top.

## Hover Labels Do Not Work

Check these points:

- Use an interactive Matplotlib backend.
- Keep the figure window active.
- Use `PlotConfig(hover_labels=True)`.
- In Jupyter, use `%matplotlib widget`.
- For saved PNG/SVG/PDF output, hover is not available because the exported file is static.

## Saved Figure Still Shows Controls

Use both `show_controls=False` and `show=False`:

```python
fig, ax = show_tensor_network(
    network,
    show_controls=False,
    show=False,
)
fig.savefig("network.png", bbox_inches="tight")
```

`show=False` only controls automatic display. It does not remove controls by itself.

## Large Graphs Are Slow

Start with a simpler interactive configuration:

```python
from tensor_network_viz import PlotConfig

config = PlotConfig(
    show_tensor_labels=False,
    show_index_labels=False,
    hover_labels=True,
    layout_iterations=80,
)
```

Other useful options:

- prefer `view="2d"` while exploring,
- disable static index labels,
- use hover labels instead of drawing every label,
- use `show_controls=False` for final static exports,
- reuse the same network object when drawing repeatedly,
- call `clear_tensor_network_graph_cache(network)` after mutating a network in place.

## Axis Configuration Errors

`AxisConfigurationError` usually means the Matplotlib axis and requested view do not match.

Examples:

- A normal 2D axis needs `view="2d"`.
- A 3D axis needs `view="3d"`.
- `show_tensor_elements(..., ax=...)` only supports an external axis for one tensor.
- External axes with existing complex layouts may not have room for interactive controls.

For simple exports, avoid external axes first:

```python
fig, ax = show_tensor_network(network, view="2d", show_controls=False, show=False)
```

## Tensor Inspection Cannot Read Data

`TensorDataError` means the input does not expose supported tensor values.

Try one of these:

- pass a direct NumPy/PyTorch array,
- pass an iterable of actual tensors,
- pass an `EinsumTrace` that still has live tensor values,
- pass `engine=` explicitly if auto-detection is ambiguous,
- for TensorKrowch, avoid shape-only nodes when you need value inspection.

Manual contraction examples can draw graph structure without carrying the live tensor values needed
by the tensor inspector.

## Comparison Shapes Do Not Match

`show_tensor_comparison(...)` compares one current tensor against one reference tensor. Both inputs
must resolve to exactly one tensor.

For numeric comparison modes, shapes must match. If shapes differ, inspect each tensor separately
first:

```python
show_tensor_elements(current)
show_tensor_elements(reference)
```

## Backend-Specific Notes

### TensorKrowch

If contraction playback is missing after a real contraction, the original network may no longer
preserve enough `leaf_nodes` / `resultant_nodes` history for safe recovery. Provide an explicit
`contraction_scheme_by_name` or inspect an uncontracted network.

### TensorNetwork

For lists of nodes, make sure the list contains the connected nodes you want to draw. If you pass a
single disconnected object by accident, only that object can be rendered.

### Quimb

Install the `quimb` extra and pass `engine="quimb"` if auto-detection is ambiguous.

### TeNPy

If native object extraction is not enough for your case, build an explicit network with
`make_tenpy_tensor_network(...)`.

### `einsum`

NumPy-backed traces work with base dependencies. PyTorch-backed traces need:

```bash
python -m pip install "tensor-network-visualization[einsum]"
```

Use auto-traced examples when you need linked tensor inspection; manual traces may not keep live
result tensors.

## Enable Debug Logging

The package logger is named `tensor_network_viz`.

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tensor_network_viz").setLevel(logging.DEBUG)
```

Debug logs are useful when checking engine detection, rendering options, or unsupported input paths.
