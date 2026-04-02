# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Interactive viewer controls:** `show_tensor_network` now opens with a Matplotlib control panel
  by default: `2d/3d` selector (when the figure owns the axes), plus `Hover`, `Tensor labels`, and
  `Edge labels` checkboxes.
- **Lazy interactive caching:** normalized graph extraction is still shared, while 2D and 3D views
  now keep their own cached geometry, label artists, hover payloads, and contraction-scheme state
  so switching modes or toggles does not rebuild everything from scratch.

### Changed

- **Viewer defaults:** `show_tensor_network` now defaults to `view="2d"` (when omitted),
  `PlotConfig(hover_labels=True, show_tensor_labels=False, show_index_labels=False)`, and exposes
  `interactive_controls=False` for static exports or headless runs.
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

[1.4.2]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.3.3...v1.4.0
