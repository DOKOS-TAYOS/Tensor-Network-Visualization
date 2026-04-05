# Show Nodes Compact View Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PlotConfig.show_nodes` plus a lazy interactive `Nodes` toggle that swaps the current node glyphs with compact fixed-size markers without rebuilding layout, edges, or cached views.

**Architecture:** Keep graph extraction, layout, edge drawing, and label descriptors unchanged. Extend the draw pipeline so each scene can render nodes in `normal` or `compact` mode, store the node artists needed for both modes, and lazily build the missing mode once per view. Reuse Matplotlib collections so 2D hover keeps working with compact points and 3D compact mode uses screen-space circular markers instead of octahedra.

**Tech Stack:** Python, Matplotlib (`PatchCollection`, `PathCollection`, 3D scatter collections), pytest, monkeypatch.

---

## File Map

- `src/tensor_network_viz/config.py`
  Adds the public `show_nodes: bool = True` flag and its user-facing docs.
- `src/tensor_network_viz/_core/draw/plotter.py`
  Builds the actual node artists for normal and compact modes in 2D and 3D.
- `src/tensor_network_viz/_core/draw/tensors.py`
  Keeps visible-node ordering and passes the chosen node mode through the tensor draw path.
- `src/tensor_network_viz/_core/draw/render_prep.py`
  Seeds the interactive scene with the initially rendered node artist set and the active hover target.
- `src/tensor_network_viz/_core/draw/scene_state.py`
  Stores normal/compact node artist references plus the active node mode for each cached view.
- `src/tensor_network_viz/_interactive_scene.py`
  Lazily creates the missing node artist set, flips visibility, and refreshes the scene hover target without rebuilding the whole scene.
- `src/tensor_network_viz/_interaction/controller.py`
  Adds the `Nodes` checkbox, controller state, and `set_nodes_enabled(self, enabled: bool) -> None`.
- `src/tensor_network_viz/_core/draw/hover.py`
  Broadens 2D node hit-testing so compact point collections work the same way as circle collections.
- `tests/plotting_helpers.py`
  Adds small helpers to count 2D scatter markers and 3D marker collections.
- `tests/test_plotting.py`
  Covers public config default, static compact rendering, interactive menu wiring, and lazy cached toggling.

## Chunk 1: Static Compact Node Rendering

### Task 1: Add the public `show_nodes` flag

**Files:**
- Modify: `src/tensor_network_viz/config.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
def test_plot_config_show_nodes_defaults_to_true() -> None:
    assert PlotConfig().show_nodes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k show_nodes_defaults -v`
Expected: FAIL because `PlotConfig` does not expose `show_nodes` yet.

- [ ] **Step 3: Write minimal implementation**

Add the new dataclass field and docstring entry:

```python
show_nodes: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k show_nodes_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_viz/config.py tests/test_plotting.py
git commit -m "feat: add show nodes plot config flag"
```

### Task 2: Render compact nodes in static 2D and 3D plots

**Files:**
- Modify: `src/tensor_network_viz/_core/draw/plotter.py`
- Modify: `src/tensor_network_viz/_core/draw/tensors.py`
- Modify: `src/tensor_network_viz/_core/draw/render_prep.py`
- Modify: `src/tensor_network_viz/_core/draw/hover.py`
- Modify: `tests/plotting_helpers.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_plot_tensorkrowch_network_2d_show_nodes_false_draws_points() -> None:
    _, ax = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False),
    )
    assert patch_collection_circle_count(ax) == 0
    assert path_collection_point_count(ax) == 2


def test_plot_tensorkrowch_network_3d_show_nodes_false_draws_marker_nodes() -> None:
    _, ax = plot_tensorkrowch_network_3d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False),
    )
    assert poly3d_node_collection_count(ax) == 0
    assert path3d_collection_point_count(ax) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "show_nodes_false_draws" -v`
Expected: FAIL because both renderers still draw normal node geometry.

- [ ] **Step 3: Write minimal implementation**

Implement a small node-mode branch in the draw layer:

```python
NodeRenderMode = Literal["normal", "compact"]


def draw_tensor_nodes(
    self,
    coords: np.ndarray,
    *,
    config: PlotConfig,
    p: _DrawScaleParams,
    degree_one_mask: np.ndarray,
    mode: NodeRenderMode,
) -> NodeArtistBundle: ...
```

Implementation notes:
- Keep the current circle/octahedron path for `mode == "normal"`.
- For 2D compact mode, use one `scatter`/`PathCollection` with fixed marker size in points.
- For 3D compact mode, use `ax.scatter(..., s=..., marker="o", depthshade=False)` so the glyph stays screen-facing.
- Preserve degree-one styling through face/edge color.
- Return the node artist bundle so later interactive code can hide/show it without redrawing edges.
- Broaden 2D hover handling from `PatchCollection`-only to a generic artist collection that supports `.contains(event)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "show_nodes_false_draws" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_viz/_core/draw/plotter.py src/tensor_network_viz/_core/draw/tensors.py src/tensor_network_viz/_core/draw/render_prep.py src/tensor_network_viz/_core/draw/hover.py tests/plotting_helpers.py tests/test_plotting.py
git commit -m "feat: add compact node rendering for static plots"
```

### Task 3: Lock the compact marker size to screen space

**Files:**
- Modify: `src/tensor_network_viz/_core/draw/plotter.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_plot_tensorkrowch_network_2d_show_nodes_false_ignores_node_radius() -> None:
    _, ax_small = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.04),
    )
    _, ax_large = plot_tensorkrowch_network_2d(
        DummyNetwork(nodes=[left, right]),
        config=PlotConfig(show_nodes=False, node_radius=0.5),
    )
    assert point_collection_sizes(ax_small) == point_collection_sizes(ax_large)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "ignores_node_radius" -v`
Expected: FAIL because compact marker size still depends on data-space node radius or draw scale.

- [ ] **Step 3: Write minimal implementation**

Use fixed marker-size constants in points for compact mode, for example:

```python
_COMPACT_NODE_MARKER_AREA_2D_PT2 = ...
_COMPACT_NODE_MARKER_AREA_3D_PT2 = ...
```

Do not derive those values from `node_radius`, draw scale, or bond length.

- [ ] **Step 4: Run test to verify it passes**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "ignores_node_radius" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_viz/_core/draw/plotter.py tests/test_plotting.py
git commit -m "test: lock compact node markers to fixed screen size"
```

## Chunk 2: Interactive Lazy Toggle and Cache Reuse

### Task 4: Store both node modes in the interactive scene and lazily build the missing one

**Files:**
- Modify: `src/tensor_network_viz/_core/draw/scene_state.py`
- Modify: `src/tensor_network_viz/_core/draw/render_prep.py`
- Modify: `src/tensor_network_viz/_interactive_scene.py`
- Modify: `src/tensor_network_viz/_core/draw/plotter.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_show_tensor_network_builds_compact_node_artists_once_per_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    calls = {"count": 0}

    def counting_builder(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interactive_scene_module,
        "_ensure_scene_node_artists",
        counting_builder,
    )

    controls.set_nodes_enabled(False)
    controls.set_nodes_enabled(True)
    controls.set_nodes_enabled(False)

    assert calls["count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "builds_compact_node_artists_once_per_view" -v`
Expected: FAIL because scenes do not yet cache both node modes.

- [ ] **Step 3: Write minimal implementation**

Add scene-level storage and a lazy switch helper:

```python
@dataclass
class _InteractiveSceneState:
    ...
    node_artist_sets: dict[str, NodeArtistBundle] = field(default_factory=dict)
    active_node_mode: str = "normal"


def _ensure_scene_node_artists(
    scene: _InteractiveSceneState,
    *,
    mode: str,
) -> NodeArtistBundle: ...


def _set_scene_node_mode(
    scene: _InteractiveSceneState,
    *,
    mode: str,
) -> None: ...
```

Implementation notes:
- Seed the initial mode from `config.show_nodes` during render prep.
- Build the missing node artist set from cached `scene.visible_node_ids`, `scene.positions`, `scene.params`, and `scene.config`.
- Hide the inactive node artists instead of removing them.
- Refresh `scene.node_patch_coll` and hover state so 2D compact points remain hoverable.

- [ ] **Step 4: Run test to verify it passes**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "builds_compact_node_artists_once_per_view" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_viz/_core/draw/scene_state.py src/tensor_network_viz/_core/draw/render_prep.py src/tensor_network_viz/_interactive_scene.py src/tensor_network_viz/_core/draw/plotter.py tests/test_plotting.py
git commit -m "feat: cache compact and normal node artists per view"
```

### Task 5: Add the `Nodes` checkbox and controller setter

**Files:**
- Modify: `src/tensor_network_viz/_interaction/controller.py`
- Modify: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_show_tensor_network_interactive_controls_include_nodes_toggle() -> None:
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None
    assert [label.get_text() for label in controls._checkbuttons.labels][:4] == [
        "Hover",
        "Nodes",
        "Tensor labels",
        "Edge labels",
    ]


def test_show_tensor_network_nodes_toggle_reuses_cached_view_without_rerender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    def _unexpected_render(*args: object, **kwargs: object) -> object:
        raise AssertionError("node toggle should not rerender the view")

    monkeypatch.setattr(controls, "_render_view", _unexpected_render)

    controls.set_nodes_enabled(False)
    controls.set_nodes_enabled(True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "nodes_toggle" -v`
Expected: FAIL because the controller does not expose the new toggle yet.

- [ ] **Step 3: Write minimal implementation**

Update controller state and menu wiring:

```python
_BASE_TOGGLE_LABELS: tuple[str, str, str, str] = (
    "Hover",
    "Nodes",
    "Tensor labels",
    "Edge labels",
)


def set_nodes_enabled(self, enabled: bool) -> None:
    self.nodes_on = bool(enabled)
    self._sync_checkbuttons()
    self._apply_scene_state(self.current_scene)
```

Implementation notes:
- Initialize `self.nodes_on` from `config.show_nodes`.
- Update `_build_controls()`, `_sync_checkbuttons()`, and `_on_toggle_clicked()`.
- In `_apply_scene_state()`, switch scene node mode before refreshing labels and hover.
- Keep every other checkbox behavior unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py -k "nodes_toggle" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tensor_network_viz/_interaction/controller.py tests/test_plotting.py
git commit -m "feat: add interactive nodes toggle"
```

### Task 6: Verify cross-view reuse and regression safety

**Files:**
- Modify: `tests/test_plotting.py`
- Test: `tests/test_plotting.py`
- Test: `tests/test_interaction_state.py`

- [ ] **Step 1: Write the final regression tests**

```python
def test_show_tensor_network_nodes_toggle_persists_across_2d_and_3d_views() -> None:
    controls = getattr(fig, "_tensor_network_viz_interactive_controls", None)
    assert controls is not None

    controls.set_nodes_enabled(False)
    controls.set_view("3d")
    assert controls.current_scene.active_node_mode == "compact"

    controls.set_view("2d")
    assert controls.current_scene.active_node_mode == "compact"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py tests/test_interaction_state.py -k "nodes_toggle or active_node_mode" -v`
Expected: FAIL until both view caches keep consistent node mode state.

- [ ] **Step 3: Write minimal implementation**

Finish any missing controller/scene synchronization so:
- the active mode follows `controls.nodes_on`
- each view lazily builds its missing artist set once
- switching between views preserves the chosen node mode

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py tests/test_interaction_state.py -k "nodes_toggle or active_node_mode" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_plotting.py tests/test_interaction_state.py src/tensor_network_viz/_interaction/controller.py src/tensor_network_viz/_interactive_scene.py
git commit -m "test: cover cross-view compact node toggling"
```

## Chunk 3: Full Verification

### Task 7: Run the focused verification suite

**Files:**
- No code changes required unless a regression is found

- [ ] **Step 1: Run the focused plotting and interaction suite**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_plotting.py tests/test_interaction_state.py -v`
Expected: PASS with 0 failures

- [ ] **Step 2: Run backend smoke coverage for the plotting entry points**

Run: `& ..\..\.venv\Scripts\python.exe -m pytest tests/test_einsum_backend.py tests/test_quimb_backend.py tests/test_tenpy_backend.py -k "plot or show_tensor_network" -v`
Expected: PASS with 0 failures

- [ ] **Step 3: Run lint on touched Python files**

Run: `& ..\..\.venv\Scripts\python.exe -m ruff check src/tensor_network_viz/config.py src/tensor_network_viz/_core/draw/plotter.py src/tensor_network_viz/_core/draw/tensors.py src/tensor_network_viz/_core/draw/render_prep.py src/tensor_network_viz/_core/draw/scene_state.py src/tensor_network_viz/_core/draw/hover.py src/tensor_network_viz/_interactive_scene.py src/tensor_network_viz/_interaction/controller.py tests/plotting_helpers.py tests/test_plotting.py tests/test_interaction_state.py`
Expected: `All checks passed!`

- [ ] **Step 4: Create the final implementation commit if verification required fixes**

```bash
git add src/tensor_network_viz/config.py src/tensor_network_viz/_core/draw/plotter.py src/tensor_network_viz/_core/draw/tensors.py src/tensor_network_viz/_core/draw/render_prep.py src/tensor_network_viz/_core/draw/scene_state.py src/tensor_network_viz/_core/draw/hover.py src/tensor_network_viz/_interactive_scene.py src/tensor_network_viz/_interaction/controller.py tests/plotting_helpers.py tests/test_plotting.py tests/test_interaction_state.py
git commit -m "feat: add compact node view toggle"
```
