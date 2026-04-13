# Repository Examples

All commands on this page are meant to be run from the repository root.

Use these examples when you want to see the package behavior quickly. For API explanations, use
[../docs/api.md](../docs/api.md). For backend-specific code snippets, use
[../docs/backends.md](../docs/backends.md). For a copy-paste command list for every demo, use
[../commands.md](../commands.md).

## Contents

- [Activate the Environment](#activate-the-environment)
- [Main Launcher](#main-launcher)
- [Standalone Tensor Inspection](#standalone-tensor-inspection)
- [Common Commands](#common-commands)
- [Useful Options](#useful-options)
- [Engine Catalog](#engine-catalog)
- [Batch Rendering](#batch-rendering)

## Activate the Environment

If you are working locally, activate the project `.venv` first.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

After the virtual environment is active, the `python ...` commands below are the same on Windows
and Linux.

## Main Launcher

```bash
python examples/run_demo.py <engine> <example> [options]
```

The launcher is useful for:

- opening an interactive figure,
- saving a clean static export,
- checking 2D and 3D layouts,
- trying labels, hover, contraction playback, and linked tensor inspection.

Examples:

```bash
python examples/run_demo.py quimb hyper --view 2d
python examples/run_demo.py tensornetwork weird --view 3d
python examples/run_demo.py themes overview --view 2d
python examples/run_demo.py placements manual_scheme --view 2d
python examples/run_demo.py geometry disconnected_irregular --view 2d
python examples/run_demo.py geometry decorated_sparse_grid2d --view 2d
```

## Standalone Tensor Inspection

`tensor_elements_demo.py` only needs base dependencies.

```bash
python examples/tensor_elements_demo.py
python examples/tensor_elements_demo.py --demo batch
python examples/tensor_elements_demo.py --demo structured
```

Use the structured demo to inspect diagnostic-friendly tensors:

- `Lattice`: useful for spectral views.
- `SparseMask`: useful for `sparsity`.
- `Specials`: useful for `nan_inf`.

## Common Commands

Open an interactive figure:

```bash
python examples/run_demo.py quimb hyper --view 2d
python examples/run_demo.py tensornetwork weird --view 3d
python examples/run_demo.py themes overview --view 2d
```

Save a clean static export:

```bash
python examples/run_demo.py tenpy chain --view 2d --save tenpy_chain.png --no-show
python examples/run_demo.py quimb mps --view 2d --save quimb_mps.png --no-show
```

Visualize contraction schemes:

```bash
python examples/run_demo.py einsum mps --view 2d --scheme
python examples/run_demo.py tenpy chain --view 2d --scheme
python examples/run_demo.py tensorkrowch mps --view 2d --n-sites 6 --scheme
python examples/run_demo.py placements manual_scheme --view 2d
```

Open linked tensor inspection during playback:

```bash
python examples/run_demo.py einsum mps --view 2d --tensor-inspector
python examples/run_demo.py tensorkrowch mps --view 2d --n-sites 6 --hover-cost --tensor-inspector
```

For linked tensor inspection, prefer auto-traced `einsum` examples such as `mps`, `mpo`, `peps`,
`ellipsis`, and `nway`. Manual `--from-scratch` / `--from-list` variants may not carry the live
tensor values needed by the inspector.

For TensorKrowch, the contracted demo path is intentionally limited to small native `mps` / `mpo`
examples with `--n-sites 6`, where contraction history recovery is safe enough for demos.

## Useful Options

| Option | Meaning |
| --- | --- |
| `--view {2d,3d}` | Initial view. |
| `--theme {default,paper,colorblind,dark,midnight,forest,slate}` | Visual theme preset. Use `paper` for clean exports, `colorblind` for accessible colors, `dark` or `midnight` for dark canvases, and `forest` or `slate` for softer light palettes. |
| `--labels-nodes` | Show tensor labels. By default this stays in auto mode and turns on below 25 visible tensors. |
| `--labels-edges` | Show index labels. |
| `--labels` | Override both label flags at once. |
| `--hover-labels` | Enable hover tooltips. |
| `--scheme` | Draw contraction-scheme overlays when available. |
| `--hover-cost` | Show contraction-cost details in the playback panel. |
| `--tensor-inspector` | Open linked tensor inspection for supported playback data. |
| `--contracted` | For small TensorKrowch demos, contract first and show recovered history. |
| `--from-scratch` | Use the manual builder when the example supports it. |
| `--from-list` | Pass list/iterable input when supported. |
| `--save [PATH]` | Save the figure. If omitted, use the auto-generated path. |
| `--no-show` | Do not open the Matplotlib window. |

## Engine Catalog

### `tensorkrowch`

`mps`, `tt`, `mpo`, `ladder`, `peps`, `cubic_peps`, `mera`, `mera_ttn`, `weird`,
`disconnected`

### `tensornetwork`

`mps`, `tt`, `peps`, `weird`

### `quimb`

`mps`, `tt`, `peps`, `hyper`

### `tenpy`

`mps`, `mpo`, `imps`, `impo`, `purification`, `uniform`, `excitation`, `chain`, `hub`,
`hyper`

If direct `MomentumMPS` construction is unavailable with your installed NumPy version, use
`excitation` as the momentum-style fallback demo.

### `einsum`

`mps`, `mpo`, `peps`, `disconnected`, `ellipsis`, `batch`, `trace`, `ternary`, `unary`,
`nway`, `implicit_out`

### `themes`

`overview`

### `placements`

`object`, `list`, `grid2d`, `grid3d`, `manual_positions`, `manual_scheme`, `named_indices`

### `geometry`

`partial_grid2d`, `decorated_sparse_grid2d`, `upper_triangle2d`, `partial_grid3d`,
`upper_pyramid3d`, `random_irregular`, `circular_ring`, `circular_chords`,
`tubular_grid`, `disconnected_irregular`

Uso de layout en estas demos: todas se pasan como listas planas de tensores para comprobar
la deteccion automatica. Las parciales prueban grids con huecos, `decorated_sparse_grid2d`
añade hojas observadas alrededor de una sparse grid 2D con huecos triangulares, `circular_ring`
y `circular_chords` prueban estructuras circulares, `tubular_grid` prueba una grid envuelta en
una direccion, y `random_irregular` y `disconnected_irregular` siguen usando layout automatico.

## Batch Rendering

List selected commands without running them:

```bash
python examples/run_all_examples.py --group engines --views 2d --list
```

Render a group headlessly:

```bash
python examples/run_all_examples.py --group geometry --views 2d --output-dir .tmp/examples
```

Useful groups include:

- `engines`
- `themes`
- `placements`
- `geometry`
- `default`
- `hover`
- `contraction`
- `all`

Start with `engines` before running `all`, because `all` is intentionally broader and slower.
`default`, `hover`, and `contraction` remain as compatibility groups for older demo workflows.
