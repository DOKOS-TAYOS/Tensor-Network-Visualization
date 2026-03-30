# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Einsum backend (`einsum_module`):** richer traced-equation support for the built graph: ellipsis (`...`) expanded with NumPy-validated shapes; repeated indices and batch-style outputs (`ab,ab->ab`, traces, etc.) via **virtual hyperedge hubs**; pairwise summation indices between two tensors stay **single bonds** (no hub). Public helper **`parse_equation_for_shapes`** on the einsum submodule. Example script **`examples/einsum_general.py`** (ellipsis batch matmul, Hadamard batch, `ii,i->i`-style trace, short MPS chain). Documentation updates in **`docs/guide.md`** and **`examples/README.md`**.
- **Layout:** virtual hyperedge hubs that share the same tensor neighbors are **spread** perpendicular to the bond between those tensors; hubs sitting on a tensor–tensor chord while a **direct** contraction also links that pair are **offset** so batch hyperedges do not overlap matmul-style bonds (e.g. ellipsis + `j`).

### Fixed

- **2D draw:** dangling legs incident only on **virtual** nodes were skipped by the layered edge pass; they are now drawn in a follow-up pass. Dangling stubs from virtual hubs anchor at the **node center** (not the tensor rim) in 2D, matching 3D.
- **Einsum graph:** open indices on batch/hyper hubs use the **equation letter** for labels (no `__out`-style suffix on dangling legs).

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

[1.4.1]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.3.3...v1.4.0
