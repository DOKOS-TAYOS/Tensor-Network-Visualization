# Show Nodes Compact View Design

## Goal

Add a new `PlotConfig.show_nodes` option that lets users switch between the current node rendering and a compact node rendering that does not dominate the image when zooming or rescaling.

## Scope

This feature affects both static plotting and the interactive `show_tensor_network` controls.

- Default behavior stays unchanged with `show_nodes=True`.
- The new compact mode activates with `show_nodes=False`.
- The feature applies only to visible tensor nodes.
- Edge rendering, layout, and contraction logic stay unchanged.

## User-Facing Behavior

### New PlotConfig Flag

Add a new field to `PlotConfig`:

- `show_nodes: bool = True`

Behavior:

- `True`: existing rendering
  - 2D uses circles
  - 3D uses octahedra
- `False`: compact rendering
  - 2D uses small points
  - 3D uses 2D circles rendered as screen-facing markers

### Compact Node Mode

The compact mode should be intentionally lightweight and intuitive:

- in 2D, nodes become point markers with fixed screen size
- in 3D, nodes become small circular screen-space markers with fixed screen size
- degree-one styling still applies through color so the user keeps that information

The compact markers should not scale with `node_radius`. Their visual size must stay fixed on screen so they remain unobtrusive during zooming or rescaling.

## Interactive Controls

Add a new checkbox in the interactive control tray:

- `Nodes`

The checkbox state reflects `config.show_nodes`.

When toggled:

- it switches only the node representation
- it does not rebuild layout
- it does not redraw edges
- it does not recreate tensor labels or edge labels unless already required for another toggle

## Recommended Architecture

### 1. PlotConfig

Extend `src/tensor_network_viz/config.py` with:

- the new `show_nodes` field
- short docstring text describing the normal and compact render modes

### 2. Plotter-Level Dual Node Rendering

Extend the draw layer so each scene can hold two node artist variants:

- normal nodes
- compact nodes

The compact variants should be dimension-specific:

- 2D: a `scatter`-style point collection
- 3D: a 2D circular marker representation, likely also via `scatter`, not a 3D solid

This requires small additions around:

- `src/tensor_network_viz/_core/draw/plotter.py`
- `src/tensor_network_viz/_core/draw/tensors.py`
- `src/tensor_network_viz/_core/draw/scene_state.py`

### 3. Lazy Dual Cache Per View

To keep toggling cheap, use lazy creation:

1. Build only the node mode requested by the initial `PlotConfig.show_nodes`.
2. If the user toggles `Nodes` for the first time in a given view, create the missing node artist set from the cached positions.
3. On later toggles in that same view, switch visibility only.

This means:

- first toggle in 2D may build compact or normal artists once
- first toggle in 3D may build compact or normal artists once
- subsequent toggles are constant-time artist visibility flips

### 4. Scene State

The interactive scene state should store enough information to support both node modes without re-rendering the whole view. The simplest extension is:

- keep current visible-node ordering and positions
- store node artists for normal mode
- store node artists for compact mode
- store which mode is currently active

This should be independent from edge geometry and label descriptors.

### 5. Controller Toggle

Extend `src/tensor_network_viz/_interaction/controller.py`:

- add `nodes_on: bool`
- include `Nodes` in the base checkbox labels
- add `set_nodes_enabled(self, enabled: bool) -> None`
- update `_on_toggle_clicked()` and `_sync_checkbuttons()`
- apply node visibility changes through the current scene without forcing a full re-render

## Data Flow

1. `PlotConfig.show_nodes` sets the initial node mode.
2. The renderer builds the initial scene as usual.
3. Node artists are created in the active mode only.
4. The scene caches positions and node artist references.
5. If `Nodes` is toggled, the controller asks the scene to switch node mode.
6. Missing node artists are created once from cached positions.
7. Existing node artists are shown/hidden without redoing layout or edge rendering.

## Performance Notes

The feature should avoid expensive work when toggling:

- do not rerun backend graph extraction
- do not recompute layout
- do not rebuild contraction controls
- do not redraw edge collections
- do not duplicate label descriptor work unnecessarily

Only node artists should be created or toggled.

## Testing Strategy

Add focused tests for:

- `PlotConfig.show_nodes` default is `True`
- static 2D rendering with `show_nodes=False` creates point-like node artists instead of circle patches
- static 3D rendering with `show_nodes=False` creates marker-like artists instead of octahedra
- interactive controls expose a `Nodes` checkbox
- toggling `Nodes` does not rebuild cached views
- repeated toggles reuse already-created artist sets instead of creating fresh ones each time

## Non-Goals

- changing edge rendering
- changing label placement
- changing layout spacing
- introducing a third node mode beyond normal vs compact

## Recommended Implementation Direction

Implement this as a node-artist toggle feature layered on top of the existing cached interactive scene system. Keep the current rendering path intact for `show_nodes=True`, add compact node artist builders for `show_nodes=False`, and make the controller switch between them lazily per view.
