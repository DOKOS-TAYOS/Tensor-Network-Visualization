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

When `--no-show` or `--save` is used, the launcher calls `show_tensor_network(..., show_controls=False, show=False)` internally.

### Visualize contraction schemes

```bash
python examples/run_demo.py einsum ellipsis --view 2d --scheme
python examples/run_demo.py einsum mps --view 2d --scheme --playback
python examples/run_demo.py tenpy chain --view 2d --scheme
```

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
| `--hover-cost` | Show contraction-cost hover on scheme regions. |
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
python examples/run_demo.py quimb hyper --view 2d --save
python examples/run_demo.py tenpy imps --view 2d --save tenpy_imps.png --no-show
python examples/run_demo.py einsum ellipsis --view 3d
```
