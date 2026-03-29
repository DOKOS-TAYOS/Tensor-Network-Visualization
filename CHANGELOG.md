# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `py.typed` marker (PEP 561) so type checkers treat the installed package as typed.

### Fixed

- `show_tensor_network` package export: `TYPE_CHECKING` import preserves lazy runtime import while exposing the real signature, annotations, and docstring to IDEs.

### Changed
- Refactor of the `_core` module into tiny pieces.

## [1.4.0] — 2026-03-28

### Added

- This changelog as the baseline for future releases.
- PyPI project URLs for documentation and this changelog.
- README badges for CI, PyPI version, Python support, and license.

[1.4.0]: https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/compare/v1.3.3...v1.4.0
