# THIRD_PARTY_LICENSES

Runtime dependencies relevant to `tensor-network-visualization`.

## Direct dependencies

These match `[project] dependencies` in `pyproject.toml` (versions align with `requirements.dev.txt` where pinned).

| Package | Version | License | Home page |
| --- | --- | --- | --- |
| matplotlib | 3.10.8 | Python Software Foundation License | https://matplotlib.org |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| numpy | 2.4.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |

## Runtime transitive dependencies

Includes transitives from the direct dependencies (for example drawing and date utilities for `matplotlib`) **and**, when extras such as `tensorkrowch` or `[einsum]` are installed, libraries from the PyTorch-related stack (`torch`, `sympy`, `opt_einsum`, `Jinja2`, `MarkupSafe`, `mpmath`, `filelock`, `fsspec`, and similar).

| Package | Version | License | Home page |
| --- | --- | --- | --- |
| contourpy | 1.3.3 | BSD License | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | BSD License | https://matplotlib.org/cycler/ |
| filelock | 3.25.1 | MIT | https://github.com/tox-dev/py-filelock |
| fonttools | 4.62.0 | MIT | http://github.com/fonttools/fonttools |
| fsspec | 2026.2.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| Jinja2 | 3.1.6 | BSD License | https://github.com/pallets/jinja/ |
| kiwisolver | 1.5.0 | BSD License | https://github.com/nucleic/kiwi |
| MarkupSafe | 3.0.3 | BSD-3-Clause | https://github.com/pallets/markupsafe/ |
| mpmath | 1.3.0 | BSD License | http://mpmath.org/ |
| opt_einsum | 3.4.0 | MIT | https://github.com/dgasmith/opt_einsum |
| packaging | 26.0 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pillow | 12.1.1 | MIT-CMU | https://python-pillow.github.io |
| pyparsing | 3.3.2 | MIT | https://github.com/pyparsing/pyparsing/ |
| python-dateutil | 2.9.0.post0 | Apache-2.0; BSD License | https://github.com/dateutil/dateutil |
| six | 1.17.0 | MIT License | https://github.com/benjaminp/six |
| sympy | 1.14.0 | BSD License | https://sympy.org |
| torch | 2.10.0 | BSD-3-Clause | https://pytorch.org |
| typing_extensions | 4.15.0 | PSF-2.0 | https://github.com/python/typing_extensions |

## Optional dependencies (dev)

When installing with `[dev]`, these runtime libraries are in addition to the direct dependencies (plus their own transitives, e.g. `graphviz`, `h5py`, `scipy`). Versions align with `requirements.dev.txt` where pinned.

| Package | Version | License | Home page |
| --- | --- | --- | --- |
| tensornetwork | 0.4.6 | Apache-2.0 | https://github.com/google/TensorNetwork |
| quimb | 1.13.0 | Apache-2.0 | https://github.com/jcmgray/quimb/ |
| physics-tenpy | 1.1.0 | Apache-2.0 | https://github.com/tenpy/tenpy |
| tensorkrowch | 1.1.6 | MIT License | https://github.com/joserapa98/tensorkrowch |
| graphviz | 0.21 | MIT | https://github.com/xflr6/graphviz |
| h5py | 3.16.0 | BSD-3-Clause | https://www.h5py.org/ |
| scipy | 1.17.1 | BSD License | https://scipy.org/ |

Development-only tooling such as `build`, `pytest`, `ruff`, `pyright`, `ipython`, `pip`, and `setuptools` is intentionally excluded.

## Optional dependencies (Jupyter)

When installing with `[jupyter]`, `pyproject.toml` declares the following (each pulls in further dependencies, e.g. `ipython`, `jupyter-server`, and related stack):

| Package | Version | License | Home page |
| --- | --- | --- | --- |
| ipympl | ≥0.9.0 | BSD-3-Clause | https://github.com/matplotlib/ipympl |
| ipywidgets | ≥8.0 | BSD-3-Clause | https://github.com/jupyter-widgets/ipywidgets |
| jupyterlab | ≥4.0 | BSD-3-Clause | https://github.com/jupyterlab/jupyterlab |
| notebook | ≥7.0 | BSD-3-Clause | https://github.com/jupyter/notebook |
