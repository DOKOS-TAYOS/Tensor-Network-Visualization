# Examples

All example commands are meant to be run from the repository root.

If you are working locally, activate the project `.venv` first.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

## Main Launcher

```bash
python examples/run_demo.py <engine> <example> [options]
```

The launcher is useful for three common cases:

1. open an interactive figure,
2. save a clean static export,
3. inspect contraction schemes and playback behavior.

## Tensor Inspection Example

There is also a small standalone example for `show_tensor_elements(...)` that only uses base
dependencies:

```bash
python examples/tensor_elements_demo.py
python examples/tensor_elements_demo.py --demo batch
python examples/tensor_elements_demo.py --demo structured
```

It uses fairly large NumPy tensors, keeps one tensor active at a time with a slider, and exposes a
grouped control flow: `basic` (`elements`, `magnitude`, `log_magnitude`, `distribution`, `data`),
`complex` (`real`, `imag`, `phase`), and `diagnostic` (`sign`, `signed_value`, `sparsity`,
`nan_inf`, `singular_values`, `eigen_real`, `eigen_imag`). The optional demos are: `matvec` (default, traced matrix-vector), `batch` (traced
batched matmul), and `structured` (3D complex, dense, sparse, and non-finite TensorNetwork-style
nodes).

If you want to inspect the new diagnostic-friendly tensors specifically, launch the structured demo
and move the tensor slider until you reach `Lattice`, `SparseMask`, or `Specials`:

```bash
python examples/tensor_elements_demo.py --demo structured
```

- `Lattice`: useful for `singular_values`, `eigen_real`, and `eigen_imag`.
- `SparseMask`: useful for the `sparsity` mode.
- `Specials`: useful for the `nan_inf` mode and for confirming that spectral modes are hidden when
  non-finite values are present.

## Common Workflows

### Open an interactive figure

```bash
python examples/run_demo.py quimb hyper --view 2d
python examples/run_demo.py tensornetwork mera_ttn --view 3d
```

### Save a clean static export

```bash
python examples/run_demo.py tenpy chain --view 2d --save tenpy_chain.png --no-show
python examples/run_demo.py quimb mps --view 2d --save quimb_mps.png --no-show
```

When `--no-show` or `--save` is used, the launcher calls
`show_tensor_network(..., show_controls=False, show=False)` internally.

### Visualize contraction schemes

```bash
python examples/run_demo.py einsum ellipsis --view 2d --scheme
python examples/run_demo.py einsum mps --view 2d --scheme --playback
python examples/run_demo.py einsum mps --view 2d --tensor-inspector
python examples/run_demo.py tenpy chain --view 2d --scheme
python examples/run_demo.py tensorkrowch mps --view 2d --n-sites 6 --scheme
python examples/run_demo.py tensorkrowch mps --view 2d --n-sites 6 --hover-cost --tensor-inspector
```

For the linked tensor inspector, use an auto-traced `EinsumTrace` example such as `mps`, `mpo`,
`peps`, `ellipsis`, or `nway`. Manual `--from-scratch` / `--from-list` variants do not carry the
live tensor values needed by the inspector. Contracted TensorKrowch demos can also drive the
linked inspector and cost panel when the native network still preserves recoverable result nodes.

For TensorKrowch, `--contracted` is intentionally limited to small native demos so the example can
perform a real contraction first and then let the library recover the contraction history
automatically. Right now the safe documented path is `mps` / `mpo` with `--n-sites 6`, and those
small demos enable the contracted path by default. Use `--no-contracted` if you want to inspect
the uncontracted native network instead.

## Useful Options

| Option | Meaning |
| --- | --- |
| `--view {2d,3d}` | Initial view. |
| `--labels-nodes` | Show tensor labels. |
| `--labels-edges` | Show index labels. |
| `--labels` | Override both label flags at once. |
| `--hover-labels` | Enable hover tooltips. |
| `--scheme` | Draw contraction-scheme overlays when available. |
| `--playback` | Start with contraction playback enabled. |
| `--hover-cost` | Show contraction-cost details in the playback panel. |
| `--tensor-inspector` | Open the linked tensor inspector for `EinsumTrace` playback or contracted TensorKrowch playback with recoverable step tensors. |
| `--contracted` | For small TensorKrowch demos, contract the native network first and show the recovered scheme. |
| `--from-scratch` | Use the manual builder when that example supports it. |
| `--from-list` | Pass list/iterable input when supported. |
| `--save [PATH]` | Save the figure. If omitted, use the auto-generated path. |
| `--no-show` | Do not open the Matplotlib window. |

## Engine Catalog

### `tensorkrowch`

- `mps` / `tt`
- `mpo`
- `ladder`
- `peps`
- `cubic_peps`
- `mera`
- `mera_ttn`
- `weird`
- `disconnected`

### `tensornetwork`

- `mps` / `tt`
- `mpo`
- `ladder`
- `peps`
- `cubic_peps`
- `mera`
- `mera_ttn`
- `weird`
- `disconnected`

### `quimb`

- `mps` / `tt`
- `mpo`
- `ladder`
- `peps`
- `cubic_peps`
- `mera`
- `mera_ttn`
- `weird`
- `disconnected`
- `hyper`

### `tenpy`

- `mps`
- `mpo`
- `imps`
- `impo`
- `purification`
- `uniform`
- `excitation`
- `chain`
- `hub`
- `hyper`

### `einsum`

- `mps`
- `mpo`
- `peps`
- `disconnected`
- `ellipsis`
- `batch`
- `trace`
- `ternary`
- `unary`
- `nway`
- `implicit_out`

## Exhaustive Copy-Paste Command List

This section lists every example command currently available from the command line.

### Standalone tensor inspection demos

```bash
python examples/tensor_elements_demo.py
python examples/tensor_elements_demo.py --demo batch
python examples/tensor_elements_demo.py --demo structured
```

```bash
# Open the structured inspection demo, then use the slider to reach SparseMask / Specials
python examples/tensor_elements_demo.py --demo structured
```

### `tensorkrowch`

```bash
python examples/run_demo.py tensorkrowch mps --view 2d
python examples/run_demo.py tensorkrowch tt --view 2d
python examples/run_demo.py tensorkrowch mpo --view 2d
python examples/run_demo.py tensorkrowch ladder --view 2d
python examples/run_demo.py tensorkrowch peps --view 2d
python examples/run_demo.py tensorkrowch cubic_peps --view 2d
python examples/run_demo.py tensorkrowch mera --view 2d
python examples/run_demo.py tensorkrowch mera_ttn --view 2d
python examples/run_demo.py tensorkrowch weird --view 2d
python examples/run_demo.py tensorkrowch disconnected --view 2d
```

### `tensornetwork`

```bash
python examples/run_demo.py tensornetwork mps --view 2d
python examples/run_demo.py tensornetwork tt --view 2d
python examples/run_demo.py tensornetwork mpo --view 2d
python examples/run_demo.py tensornetwork ladder --view 2d
python examples/run_demo.py tensornetwork peps --view 2d
python examples/run_demo.py tensornetwork cubic_peps --view 2d
python examples/run_demo.py tensornetwork mera --view 2d
python examples/run_demo.py tensornetwork mera_ttn --view 2d
python examples/run_demo.py tensornetwork weird --view 2d
python examples/run_demo.py tensornetwork disconnected --view 2d
```

### `quimb`

```bash
python examples/run_demo.py quimb mps --view 2d
python examples/run_demo.py quimb tt --view 2d
python examples/run_demo.py quimb mpo --view 2d
python examples/run_demo.py quimb ladder --view 2d
python examples/run_demo.py quimb peps --view 2d
python examples/run_demo.py quimb cubic_peps --view 2d
python examples/run_demo.py quimb mera --view 2d
python examples/run_demo.py quimb mera_ttn --view 2d
python examples/run_demo.py quimb weird --view 2d
python examples/run_demo.py quimb disconnected --view 2d
python examples/run_demo.py quimb hyper --view 2d
```

### `tenpy`

```bash
python examples/run_demo.py tenpy mps --view 2d
python examples/run_demo.py tenpy mpo --view 2d
python examples/run_demo.py tenpy imps --view 2d
python examples/run_demo.py tenpy impo --view 2d
python examples/run_demo.py tenpy purification --view 2d
python examples/run_demo.py tenpy uniform --view 2d
python examples/run_demo.py tenpy excitation --view 2d
python examples/run_demo.py tenpy chain --view 2d
python examples/run_demo.py tenpy hub --view 2d
python examples/run_demo.py tenpy hyper --view 2d
```

### `einsum`

```bash
python examples/run_demo.py einsum mps --view 2d
python examples/run_demo.py einsum mpo --view 2d
python examples/run_demo.py einsum peps --view 2d
python examples/run_demo.py einsum disconnected --view 2d
python examples/run_demo.py einsum ellipsis --view 2d
python examples/run_demo.py einsum batch --view 2d
python examples/run_demo.py einsum trace --view 2d
python examples/run_demo.py einsum ternary --view 2d
python examples/run_demo.py einsum unary --view 2d
python examples/run_demo.py einsum nway --view 2d
python examples/run_demo.py einsum implicit_out --view 2d
```

## Batch Rendering

Use `run_all_examples.py` to save many examples headlessly:

```bash
python examples/run_all_examples.py --group default
python examples/run_all_examples.py --group contraction --views 2d
python examples/run_all_examples.py --group hover --list
```

Outputs are written to `.tmp/run-all-examples/` by default.

## A Few Good Starting Commands

```bash
python examples/run_demo.py tensorkrowch mps --view 2d
python examples/run_demo.py tensorkrowch mps --view 2d --n-sites 6 --scheme
python examples/run_demo.py quimb hyper --view 2d --save
python examples/run_demo.py tenpy imps --view 2d --save tenpy_imps.png --no-show
python examples/run_demo.py einsum ellipsis --view 3d
```
