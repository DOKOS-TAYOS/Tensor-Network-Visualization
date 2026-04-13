# THIRD_PARTY_LICENSES

This file documents the third-party packages declared directly by
`tensor-network-visualization`.

## Scope

- It covers packages declared in `pyproject.toml` under `[project] dependencies`
  and the user-facing optional extras.
- It does not attempt to list every transitive dependency for every platform,
  Python version, or wheel build. Those resolved dependency graphs vary over time.
- This project does not vendor third-party source code under `src/`; dependencies
  are installed separately from PyPI or upstream projects.
- This file is an informational inventory, not legal advice.

For a release-specific and fully exhaustive notice bundle, generate the inventory
from the exact locked environment used to build or test that release.

## Base Runtime Dependencies

These are the packages declared directly in `[project] dependencies`.

| Package | Declared spec | Primary license | Notes | Official source |
| --- | --- | --- | --- | --- |
| matplotlib | `>=3.7` | Matplotlib license (PSF-style, based on the PSF license) | Matplotlib distributions also include bundled third-party components, fonts, and data with additional notices. | https://matplotlib.org/stable/project/license.html |
| networkx | `>=3.0` | BSD-3-Clause | Permissive license. | https://github.com/networkx/networkx/blob/main/LICENSE.txt |
| numpy | unpinned | BSD-3-Clause (NumPy core) | NumPy distributions also include bundled components under additional permissive licenses; see the official NumPy license page. | https://numpy.org/doc/stable/license.html |

## Optional User-Facing Extras

These packages are declared directly by optional extras in `pyproject.toml`.

| Package | Declared by extra | Primary license | Notes | Official source |
| --- | --- | --- | --- | --- |
| ipympl | `jupyter` | BSD-3-Clause | Interactive Matplotlib widget backend for notebooks. | https://pypi.org/project/ipympl/ |
| ipywidgets | `jupyter` | BSD-3-Clause | Notebook widget stack. | https://pypi.org/project/ipywidgets/ |
| jupyterlab | `jupyter` | BSD-3-Clause | JupyterLab frontend. | https://pypi.org/project/jupyterlab/ |
| notebook | `jupyter` | BSD-3-Clause | Classic Jupyter Notebook frontend. | https://pypi.org/project/notebook/ |
| tensorkrowch | `tensorkrowch`, `dev` | MIT | Upstream depends on `torch` and `opt_einsum`; see upstream metadata for full transitive details. | https://github.com/joserapa98/tensorkrowch |
| tensornetwork | `tensornetwork`, `dev` | Apache-2.0 | Apache-2.0 is permissive, but keep upstream LICENSE/NOTICE material if redistributing or modifying upstream code. | https://github.com/google/TensorNetwork |
| quimb | `quimb`, `dev` | Apache-2.0 | Apache-2.0 is permissive, but keep upstream LICENSE/NOTICE material if redistributing or modifying upstream code. | https://github.com/jcmgray/quimb/blob/main/LICENSE.txt |
| physics-tenpy | `tenpy`, `dev` | Apache-2.0 | Apache-2.0 is permissive, but keep upstream LICENSE/NOTICE material if redistributing or modifying upstream code. | https://raw.githubusercontent.com/tenpy/tenpy/main/LICENSE |
| torch | `einsum` | BSD-3-Clause | PyTorch distributions also ship a `NOTICE` file and additional third-party material; consult upstream release artifacts when preparing release-specific notices. | https://github.com/pytorch/pytorch/blob/main/LICENSE |

## Development-Only Tooling

The `dev` extra in `pyproject.toml` also includes `build`, `pytest`, `ruff`,
`pyright`, and `ipython`, along with some of the optional backend packages listed
above. Those tools are used for repository development and CI, not for the base
runtime distribution of the published library, so they are not repeated in the
tables above.

## Compatibility Notes

- No direct dependency or user-facing extra declared by this project currently uses
  a copyleft license such as GPL or AGPL.
- MIT, BSD, Apache-2.0, and the Matplotlib/PSF-style license family are generally
  compatible with publishing this project itself under MIT.
- Apache-2.0 remains permissive, but it carries attribution, NOTICE, and patent
  terms that matter if you redistribute or modify upstream code.
- Matplotlib, NumPy, and PyTorch distributions include bundled third-party
  materials with their own notices. When preparing binary redistribution or a
  release-specific notice bundle, consult the exact upstream distribution artifacts
  used for that release.
