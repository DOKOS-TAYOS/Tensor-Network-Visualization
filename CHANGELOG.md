# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `py.typed` marker (PEP 561) so type checkers treat the installed package as typed.
- Normalized graph caching when redrawing the **same** network object (preview + export, 2D then 3D, etc.); `clear_tensor_network_graph_cache` invalidates after in-place edits.
- Quimb graph builder: single pass over tags/inds (fewer redundant string conversions) when sorting tensors and building nodes.

### Fixed

- `show_tensor_network` package export: `TYPE_CHECKING` import preserves lazy runtime import while exposing the real signature, annotations, and docstring to IDEs.

### Changed
- Refactor of the `_core` module into tiny pieces.
- Removed unused code.
- Changed the drawing ordering, to improve the visualization when parts of the tensor network are in the same place.
- Plot pipeline computes ``_group_contractions`` once per figure, shared between 2D axis-direction planning (bond segments) and drawing.

## [1.4.0] — 2026-03-28

### Added

- This changelog as the baseline for future releases.
- PyPI project URLs for documentation and this changelog.
- README badges for CI, PyPI version, Python support, and license.

[1.4.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.3.3...v1.4.0
