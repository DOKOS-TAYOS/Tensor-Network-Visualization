# Installation

This page covers the common installation paths for Tensor-Network-Visualization.

## Contents

- [Requirements](#requirements)
- [Create a Virtual Environment](#create-a-virtual-environment)
- [Install the Package](#install-the-package)
- [Optional Extras](#optional-extras)
- [Jupyter Setup](#jupyter-setup)
- [Local Development Install](#local-development-install)
- [Quick Checks](#quick-checks)

## Requirements

- Python 3.11 or newer.
- A virtual environment is strongly recommended.
- Base dependencies are `numpy`, `matplotlib`, and `networkx`.

The package name on PyPI is `tensor-network-visualization`. The import name in Python is
`tensor_network_viz`.

## Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

If PowerShell blocks activation scripts, either activate from a terminal that allows local venv
scripts, or run Python through the venv path:

```powershell
.\.venv\Scripts\python.exe -m pip install -U pip
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

After the environment is active, the rest of the `python -m pip ...` commands are the same on
Windows and Linux.

## Install the Package

Base install:

```bash
python -m pip install tensor-network-visualization
```

This is enough for:

- direct NumPy tensor inspection,
- NumPy-backed `EinsumTrace` examples,
- normalized graph and snapshot exports once you pass supported inputs.

## Optional Extras

Install an extra when you want support for a specific backend or environment.

| Need | Command |
| --- | --- |
| Interactive Jupyter widgets | `python -m pip install "tensor-network-visualization[jupyter]"` |
| TensorKrowch support | `python -m pip install "tensor-network-visualization[tensorkrowch]"` |
| TensorNetwork support | `python -m pip install "tensor-network-visualization[tensornetwork]"` |
| Quimb support | `python -m pip install "tensor-network-visualization[quimb]"` |
| TeNPy support | `python -m pip install "tensor-network-visualization[tenpy]"` |
| Traced PyTorch `einsum(...)` support | `python -m pip install "tensor-network-visualization[einsum]"` |
| Development tools and optional backends | `python -m pip install "tensor-network-visualization[dev]"` |

You can combine extras:

```bash
python -m pip install "tensor-network-visualization[jupyter,quimb,einsum]"
```

## Jupyter Setup

Install the Jupyter extra:

```bash
python -m pip install "tensor-network-visualization[jupyter]"
```

In a notebook, use the widget backend before creating interactive figures:

```python
%matplotlib widget
```

Then call the normal API:

```python
fig, ax = show_tensor_network(network, show=True)
```

For static notebook output, use:

```python
fig, ax = show_tensor_network(network, show_controls=False, show=False)
fig
```

## Local Development Install

From the repository root, with `.venv` active:

```bash
python -m pip install -e ".[dev]"
```

If you only need base editable install:

```bash
python -m pip install -e .
```

Useful development commands:

```bash
python -m pytest
ruff check .
ruff format .
pyright
```

## Quick Checks

Confirm the package imports:

```bash
python -c "import tensor_network_viz as tnv; print(tnv.__all__[0])"
```

Run a base-dependency example from the repository:

```bash
python examples/tensor_elements_demo.py --demo structured
```

Run one backend demo after installing its extra:

```bash
python examples/run_demo.py quimb hyper --view 2d
```
