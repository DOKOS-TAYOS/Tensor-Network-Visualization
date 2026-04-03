# Tensor-Network-Visualization Guide

This guide is organized around user workflows instead of internal implementation details.

## Core Idea

The public API is intentionally split into two responsibilities:

- `show_tensor_network(...)` manages the figure lifecycle.
- `PlotConfig(...)` manages how the network should look and behave.

If you remember only one thing, remember this:

```python
fig, ax = show_tensor_network(network, config=PlotConfig(...))
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
    contraction_playback=True,
    contraction_scheme_cost_hover=True,
)
```

Important detail:

- if `show_controls=True`, playback and scheme toggles are available on the figure;
- if `show_controls=False`, the scheme can still be drawn statically, but no playback widgets are added.

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

## Supported Inputs

### TensorKrowch

- `TensorNetwork`
- iterable of TensorKrowch nodes

### TensorNetwork

- iterable of `tensornetwork.Node`

### Quimb

- `quimb.tensor.TensorNetwork`
- iterable of `quimb.tensor.Tensor`

### TeNPy

- common native MPS/MPO-style objects
- explicit `TenPyTensorNetwork`

### `einsum`

- `EinsumTrace`
- ordered iterable of `pair_tensor(...)` / `einsum_trace_step(...)`

More detailed copy-paste examples: [backends.md](backends.md)

## Using the Returned Figure

Every path returns ordinary Matplotlib objects.

```python
fig, ax = show_tensor_network(network, show=False)
ax.set_title("My tensor network")
fig.savefig("out.png", bbox_inches="tight")
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
