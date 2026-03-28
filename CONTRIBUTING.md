# Contributing to Tensor-Network-Visualization

Contributions are welcome: bug fixes, documentation improvements, tests, and support for new backends. This document describes how to set up your environment and submit changes.

## Types of Contributions

We accept:

- **Bug reports and fixes** — reproducible issues with clear steps
- **Documentation** — clarifications, examples, guide updates
- **Tests** — coverage for new behavior or edge cases
- **New backends** — adapters for additional tensor-network libraries (see below)
- **Small enhancements** — focused improvements that fit the existing design

For larger changes, open an issue first to discuss scope and approach.

## Development Setup

1. Clone and enter the repository:
   ```bash
   git clone https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization.git
   cd Tensor-Network-Visualization
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   - Windows: `.\.venv\Scripts\Activate.ps1` or `.\.venv\Scripts\activate.bat`
   - Linux/macOS: `source .venv/bin/activate`

3. Install in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   This installs pytest, ruff, pyright, and the optional backends (tensorkrowch, tensornetwork, quimb, physics-tenpy) required for the full test suite.

## Running Tests

From the project root:

```bash
pytest
```

With the project venv (Windows):

```powershell
.\.venv\Scripts\python -m pytest
```

Add tests for new features or bug fixes. All tests must pass before opening a PR.

## Lint and Type Checks

**Ruff** (lint and format):

```bash
ruff check .
ruff format .
```

**Pyright** (type checking):

```bash
pyright
```

Configuration lives in `pyproject.toml` (ruff) and `pyrightconfig.json` (pyright).

## Code Style and Expectations

- **Line length:** 100 characters
- **Target:** Python 3.10+
- **Ruff rules:** E, F, I, B, UP, C4, SIM
- **Typing:** Use type hints on public functions and modules; the codebase is `py.typed`

Run `ruff check .` and `ruff format .` before committing. Fix any pyright errors.

## Opening Useful Issues

Use the [issue tracker](https://github.com/DOKOS-TAYOS/Tensor-Network-Visualization/issues).

**Bug reports** should include:

- Clear description of what went wrong and what you expected
- Environment: OS, Python version, `pip list` (or relevant packages)
- Minimal steps or code to reproduce
- Backend used (tensorkrowch, tensornetwork, quimb, tenpy, einsum)

**Feature requests** should describe the use case and how it fits the existing API.

## Submitting Pull Requests

1. Fork the repo and create a branch from `main`.
2. Implement changes following the code style above.
3. Run `ruff check .`, `ruff format .`, `pyright`, and `pytest` before submitting.
4. Open a PR with a clear title and description.
5. Link any related issues.

Keep PRs focused. For larger work, split into smaller PRs or discuss in an issue first.

## Proposing Support for New Backends

Adding a new engine (e.g. a new tensor-network library) requires:

1. **Adapter module** under `src/tensor_network_viz/<engine>/`:
   - `graph.py` — convert backend-native objects to `_GraphData` (see `_core/graph.py`)
   - `renderer.py` — implement `plot_<engine>_network_2d` and `plot_<engine>_network_3d` using the shared `_core` drawing layer
   - `__init__.py` — export the two plot functions

2. **Registration** in `config.py` (add to `EngineName`) and `_registry.py` (add to `_ENGINE_CONFIG`).

3. **Optional dependency** in `pyproject.toml` under `[project.optional-dependencies]`.

4. **Tests** in `tests/test_integration_<engine>.py` and optional `tests/test_<engine>_backend.py`.

5. **Example script** in `examples/<engine>_demo.py` and an entry in `examples/README.md`.

Open an issue to discuss the backend and its API before implementing.

## Updating Documentation and Examples

- **README.md** — high-level overview, installation, modes, `show_tensor_network` / `PlotConfig`, quick troubleshooting
- **docs/guide.md** — full manual: backends, recipes, layout/draw behavior, architecture, extended troubleshooting
- **CHANGELOG.md** — user-facing release notes; add an entry when cutting a PyPI release
- **examples/** — runnable scripts and **examples/README.md**; update **both** when adding CLI flags (e.g. `--hover-labels`) or new demos
- **Docstrings** — document public functions with Args, Returns, and a short Example where helpful

**When you change the public API or defaults**, update **README.md**, **docs/guide.md**, and any affected **examples** text so PyPI and the repo stay aligned with `_core` behavior.

## Pre-PR Checklist

Before opening a pull request, confirm:

- [ ] `ruff check .` and `ruff format .` pass
- [ ] `pyright` passes
- [ ] `pytest` passes
- [ ] New code has type hints and tests where appropriate
- [ ] Documentation and examples are updated if behavior changed
- [ ] PR description explains the change and links related issues
