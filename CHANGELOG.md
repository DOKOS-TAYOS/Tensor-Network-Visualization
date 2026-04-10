# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.4] — 2026-04-10

### Added

- **Network diagnostics overlays:** `PlotConfig` now accepts `TensorNetworkDiagnosticsConfig(...)` so network figures and normalized snapshots can expose backend-normalized `shape`, `dtype`, `element_count`, estimated memory, and per-edge `bond_dimension` consistently across engines.
- **Focused subnetwork views:** `PlotConfig.focus` now supports reproducible `neighborhood` and `path` filtering in both interactive rendering and `export_tensor_network_snapshot(...)`, while preserving the coordinates from the full-graph layout.
- **Analytical tensor-element views:** `show_tensor_elements(...)` now adds the `analysis` control group with `slice`, `reduce`, and `profiles` modes, plus `TensorAnalysisConfig(...)` for programmatic axis selection, reductions, and 1D profiling.

### Changed

- **Interactive controls:** the network viewer now exposes `Dimensions` and focus controls in the Matplotlib tray, uses a compact `2D/3D` toggle button, and keeps the bottom control trays more tightly aligned when scheme costs are shown.
- **Tensor inspector controls:** the linked inspector now uses a more compact comparison/header strip (`Current/Reference`, capture, clear), adds button hover tooltips, and opens with a wider layout so comparison and analysis controls do not collide.
- **Tensor-element analysis layout:** tensor-element figures now place the tensor slider lower, give analysis axis selectors more room, and let inspector-linked figures use a dedicated controls layout instead of the default compact arrangement.

### Fixed

- **Tensor inspector activation:** clicking a tensor no longer opens the tensor inspector unless the `Tensor inspector` toggle is enabled.
- **Matplotlib slider interactions:** playback, tensor, and analysis sliders now release stale mouse grabs first, avoiding `Another Axes already grabs mouse input` errors after 3D navigation or view refreshes.
- **Inspector button tooltips:** the `Capture reference` and `Clear reference` hover labels are now anchored to their buttons instead of appearing noticeably offset.

## [1.5.3] — 2026-04-07

### Added

- **Hide nodes in the viewer:** Option to exclude selected nodes from the interactive visualization so dense or branching tensor networks stay readable without changing the underlying graph data.
- **Public diagnostics contract:** Added package-specific exceptions (`TensorNetworkVizError`, `VisualizationInputError`, `AxisConfigurationError`, `UnsupportedEngineError`, `TensorDataError`, `MissingOptionalDependencyError`) and a documented `tensor_network_viz` logger with a default `NullHandler`.
- **Spectral tensor diagnostics:** `show_tensor_elements` now exposes `singular_values`, `eigen_real`, and `eigen_imag` diagnostic modes for the current matrixization, with graceful hiding when the analysis is not applicable.

### Changed

- **Contraction scheme visualization:** Major improvements to how contraction steps and scheme highlights are rendered and read in the viewer, so dense or branching schedules stay much easier to follow.
- **Public API parameters:** Public entry points keep the same overall behavior; argument lists are reorganized for a clearer, more consistent order in signatures, help, and documentation.
- **Typing and controller structure:** Fixed the interactive tensor-inspector typing issue reported by `pyright`, extracted the linked tensor-inspector controller, and split tensor-element rendering/color-scaling helpers into a dedicated module to keep responsibilities narrower.
- **Verification workflow docs:** README, guide, and contribution docs now document the `.venv`-first verification flow (`quality`, `tests`, `smoke`, `package`) used before release and CI troubleshooting.
- **Refactor:** Great refactor to clean the code, and new logs and exceptions.

## [1.5.2] — 2026-04-05

### Added
- **Tensor element diagnostics:** `show_tensor_elements` now supports `log_magnitude`, `sparsity`, and `nan_inf` views, richer `data` summaries with per-axis stats plus top-k coordinates, optional robust/shared color scaling, and outlier overlays for continuous heatmaps.
- **Contraction-progress visualization:** Tensors can be inspected visually as contraction steps unfold (step-by-step alongside the network or scheme view).

### Changed
- **Free-index geometry & performance:** More reliable layout for free (dangling) index directions in 2D, with shorter runtimes; internal structure simplified and several bottlenecks removed in the layout/render path.
- **Interactive UI:** Polished control menus and drawing regions; clearer, more useful detail in the computational complexity panel.

### Documentation
- **Tensor inspection docs:** Refreshed the README, guide, backend examples, and demo configuration to cover the new tensor-element diagnostics and scaling controls.

## [1.5.1] — 2026-04-04

### Added
- **Tensor elements visualizer:** `show_tensor_elements` inspects tensor values in Matplotlib with optional group/mode controls (heatmap, magnitude, distribution, text summary, and complex/diagnostic views), `TensorElementsConfig` for modes and axis grouping, multi-tensor selection when applicable, and the same engine auto-detection as the network viewer.

### Changed
- **API cleanup:** Streamlined and modernized the API by removing redundant and obsolete arguments, leading to a simpler and more intuitive interface.
- **Codebase optimization:** Optimized internal logic and removed dead code for improved performance and maintainability.

### Documentation
- **Improved docs:** Updated and clarified documentation throughout the project to better reflect the current API and usage patterns.

## [1.5.0] — 2026-04-03

### Added
- **Visualization improvements:** Enhanced the rules for determining node location and bond direction, resulting in clearer, more informative layouts for complex tensor networks.
- **Geometry computation optimization:** Major performance improvement in layout and rendering by optimizing the calculation, storage, and update of node and edge positions across both 2D and 3D modes.
- **Contribution tools:** Improved and streamlined contribution tools for developers, including enhanced linting, type-checking, and developer setup scripts.
- **Interactive viewer controls:** `show_tensor_network` now opens with a Matplotlib control panel
  by default: `2d/3d` selector (when the figure owns the axes), plus `Hover`, `Tensor labels`, and
  `Edge labels` checkboxes.
- **Lazy interactive caching:** normalized graph extraction is still shared, while 2D and 3D views
  now keep their own cached geometry, label artists, hover payloads, and contraction-scheme state
  so switching modes or toggles does not rebuild everything from scratch.

### Changed

- **Python support:** the package now requires **Python 3.11+**; Python 3.10 is no longer supported.
- **Viewer defaults:** `show_tensor_network` now defaults to `view="2d"` (when omitted),
  `PlotConfig(hover_labels=True, show_tensor_labels=False, show_index_labels=False)`, and exposes
  `interactive_controls=False` for static exports or headless runs.
- **Engine selection:** `show_tensor_network` now accepts `engine` as an optional override. When
  omitted, the function auto-detects the backend from supported TensorKrowch, TensorNetwork,
  Quimb, TeNPy, or einsum inputs.
- **Hover semantics:** hover is now independent from static tensor and edge labels, so both can be
  enabled at the same time.
- **Documentation:** refreshed `README.md`, `docs/guide.md`, `examples/README.md`, and
  `CONTRIBUTING.md` to match the current API, example behavior, and interactive workflow.

## [1.4.2] — 2026-03-31

### Added

- **Contraction scheme visualization:** normalized graphs from **einsum** traces carry optional **`contraction_steps`** ([`_GraphData`](src/tensor_network_viz/_core/graph.py)). **`PlotConfig(show_contraction_scheme=True)`** draws per-step highlights **under** the graph (**2D:** rounded **`FancyBboxPatch`** on each step’s bounding box; **3D:** axis-aligned wireframe box). Later steps are drawn first so earlier steps stack on top. **`contraction_scheme_by_name`** overrides or supplies a schedule for any engine. Tests in **`tests/test_contraction_scheme.py`**; docs in **`docs/guide.md`**.
- **TeNPy engine:** **`TenPyTensorNetwork`** and **`make_tenpy_tensor_network`** ([`tenpy/explicit.py`](src/tensor_network_viz/tenpy/explicit.py)) for hand-made **`npc.Array`** networks with explicit bonds; **n-way** bonds use a virtual hub. **`MPS` / `MPO` / momentum-style** inputs are built through this path internally. Root exports **`TenPyTensorNetwork`**, **`make_tenpy_tensor_network`**. Example **`examples/tenpy_explicit_tn_demo.py`** (TeNPy-only). Docs: **`docs/guide.md`**, **`README.md`**, **`examples/README.md`**, **`examples/total_tests.bat`**, tests in **`tests/test_tenpy_backend.py`** and **`tests/test_examples.py`**.
- **Einsum backend (`einsum_module`):** richer traced-equation support for the built graph: ellipsis (`...`) expanded with NumPy-validated shapes; repeated indices and batch-style outputs (`ab,ab->ab`, traces, etc.) via **virtual hyperedge hubs**; pairwise summation indices between two tensors stay **single bonds** (no hub). Public helper **`parse_equation_for_shapes`** on the einsum submodule. Example script **`examples/einsum_general.py`** (ellipsis batch matmul, Hadamard batch, `ii,i->i`-style trace, short MPS chain). Documentation updates in **`docs/guide.md`** and **`examples/README.md`**.
- **Traced einsum:** binary **implicit** subscripts without `->` (no `...`); optional **`out=`** (shape-checked; rejected if `out` is already on the trace); **unary** and **ternary+** traced steps via **`parse_einsum_equation`** / **`einsum_trace_step`** in the graph. PyTorch builds without `einsum(..., out=...)` fall back to **`copy_`**. Examples **`implicit_out`**, **`ternary`**, **`unary`** in **`examples/einsum_general.py`**; docs in **`README.md`**, **`CONTRIBUTING.md`**, **`docs/guide.md`**.
- **Layout:** virtual hyperedge hubs that share the same tensor neighbors are **spread** perpendicular to the bond between those tensors; hubs sitting on a tensor–tensor chord while a **direct** contraction also links that pair are **offset** so batch hyperedges do not overlap matmul-style bonds (e.g. ellipsis + `j`).

### Changed

- **Einsum `contraction_steps`:** **running union** of operand physical lineages along the trace (each step is a superset of the previous). Nested padding bonus still separates nested boxes.
- **Contraction scheme:** default **`PlotConfig.contraction_scheme_alpha`** is **0.0** (border-only 2D); increase for a tinted fill. **Einsum** steps use a **running union** of operand lineages. No corner step numbers; padding grows with step order. **`disconnected`** demo schedule ends with all tensors. **`examples/demo_cli.py`** builds **full** cumulative **`contraction_scheme_by_name`** for **`mps`**, **`mpo`**, **`peps`** (TensorNetwork / TensorKrowch / Quimb) and for **`cubic_peps_demo`** at any **`lx`×`ly`×`lz`**.
- **`PlotConfig`:** default plot colors match the previous **showcase** demo palette (slate nodes, sky bonds, rose dangling legs).
- **Examples:** shared **showcase** styling (moderate ``figsize``, thicker lines, extra layout iterations), subtitles that explain what each demo stresses, **`ladder`** topology for TensorNetwork / Quimb / TensorKrowch, larger default MPS/MPO/PEPS sizes, **`--compact`** uses library defaults on a smaller figure, **`einsum_general.py`** addition **`nway`** (three tensors via two traced binary steps), **`tensorkrowch_demo`**: **`--save` / `--no-show`**, cubic PEPS defaults **4×4×4**, TSP demo captions + save path.

### Fixed

- **Contraction scheme (2D):** `FancyBboxPatch` with a **point**-based `rounding_size` inflated the patch far beyond the intended data bbox; use the string **`round,pad=…`** style instead so highlights match the original rectangle size (~same padding as before).
- **2D draw:** dangling legs incident only on **virtual** nodes were skipped by the layered edge pass; they are now drawn in a follow-up pass. Dangling stubs from virtual hubs anchor at the **node center** (not the tensor rim) in 2D, matching 3D.
- **Einsum graph:** open indices on batch/hyper hubs use the **equation letter** for labels (no `__out`-style suffix on dangling legs).
- **2D layout:** virtual hubs with **only one** physical neighbor (e.g. traced einsum **`ii->i`**) are **offset** from that tensor after the barycenter snap so the hub does not sit coincident with the disk; **3D** already separated such pairs via **z-layer** promotion. Documented in **`docs/guide.md`**.

## [1.4.1] — 2026-03-29

### Added

- `py.typed` marker (PEP 561) so type checkers treat the installed package as typed.
- Normalized graph caching when redrawing the **same** network object (preview + export, 2D then 3D, etc.); `clear_tensor_network_graph_cache` invalidates after in-place edits.
- Quimb graph builder: single pass over tags/inds (fewer redundant string conversions) when sorting tensors and building nodes.

### Fixed

- `show_tensor_network` package export: `TYPE_CHECKING` import preserves lazy runtime import while exposing the real signature, annotations, and docstring to IDEs.

### Changed

- **Performance (layout & draw):** cache contraction records, pair weights, and ``_group_contractions`` output per ``_GraphData`` instance; force-directed layout uses subgraph-local pair weights, vectorized attraction, and a reused displacement buffer; layout structure builds visible/proxy graphs from each component subgraph instead of rescanning all contractions; one ``_analyze_layout_components`` pass per plot feeds layout and 3D axis directions; 2D free-axis obstacle lists use a shared coordinate matrix; draw scale and bond-curve viewport padding share two passes over cached contractions; extent heuristic subsamples nodes when computing median nearest-neighbor distance above an internal cap (deterministic RNG). Optional bench scripts: ``scripts/bench_layout_draw.py`` and ``scripts/bench_layout_compare.py`` (see ``scripts/BENCH_LAYOUT_COMPARE_RESULTS.txt`` for example before/after timings).
- Tensor label metrics: LRU cache on ``_textpath_width_pts`` (text + fontsize); optional fast 3D tensor-disk pixel radius via ``PlotConfig.approximate_3d_tensor_disk_px`` (default True) using nominal px/data-unit from axis spans; expanded ``refine_tensor_labels`` docs on cost.
- Refactor of the `_core` module into tiny pieces.
- Removed unused code.
- Changed the drawing ordering, to improve the visualization when parts of the tensor network are in the same place.
- Plot pipeline computes ``_group_contractions`` once per figure, shared between 2D axis-direction planning (bond segments) and drawing.
- Precomputed ``_node_edge_degrees`` per draw: degree-1 node styling and masks are O(|E|+|V|) instead of O(|V|·|E|); single visible-node order reused for labels and hover.

## [1.4.0] — 2026-03-28

### Added

- This changelog as the baseline for future releases.
- PyPI project URLs for documentation and this changelog.
- README badges for CI, PyPI version, Python support, and license.

[1.5.4]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.2...v1.5.0
[1.4.2]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.3.3...v1.4.0
