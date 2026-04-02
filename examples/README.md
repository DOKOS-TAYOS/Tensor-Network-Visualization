# Examples

Run demos from the repository root with the project `.venv` and the matching optional extra
installed.

Main launcher:

```bash
python examples/run_demo.py <engine> <example> [options]
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python examples\run_demo.py quimb hyper --view 2d
```

Linux / macOS:

```bash
source .venv/bin/activate
python examples/run_demo.py quimb hyper --view 2d
```

## Common options

- `--view {2d,3d}` chooses the initial view.
- `--labels-nodes` / `--labels-edges` control static labels.
- `--labels` overrides both static label flags at once.
- `--hover-labels` controls hover tooltips.
- `--scheme`, `--playback`, and `--hover-cost` enable contraction overlays.
- `--from-scratch` forces the manual builder when the example supports it.
- `--from-list` passes an iterable/list form to `show_tensor_network(...)` when supported.
- `--save [PATH]` saves the figure; without a path it writes to `.tmp/examples/<engine>/<example>.png`.
- `--no-show` renders headlessly without opening the Matplotlib window.

## Engines and examples

- `tensorkrowch`: `mps`/`tt`, `mpo`, `ladder`, `peps`, `cubic_peps`, `mera`, `mera_ttn`, `weird`, `disconnected`
- `tensornetwork`: `mps`/`tt`, `mpo`, `ladder`, `peps`, `cubic_peps`, `mera`, `mera_ttn`, `weird`, `disconnected`
- `quimb`: `mps`/`tt`, `mpo`, `ladder`, `peps`, `cubic_peps`, `mera`, `mera_ttn`, `weird`, `disconnected`, `hyper`
- `tenpy`: `mps`, `mpo`, `imps`, `impo`, `purification`, `uniform`, `excitation`, `chain`, `hub`, `hyper`
- `einsum`: `mps`, `mpo`, `peps`, `disconnected`, `ellipsis`, `batch`, `trace`, `ternary`, `unary`, `nway`, `implicit_out`

## Copy-paste commands

```bash
python examples/run_demo.py tensorkrowch mps --view 2d --from-list
python examples/run_demo.py tensornetwork mera_ttn --view 3d
python examples/run_demo.py quimb hyper --view 2d --save
python examples/run_demo.py tenpy imps --view 2d --save tenpy_imps.png --no-show
python examples/run_demo.py tenpy chain --view 2d --scheme
python examples/run_demo.py einsum mps --view 2d --from-scratch
python examples/run_demo.py einsum ellipsis --view 3d --from-list
```

## Run every example with defaults

Activate the `.venv` first, then launch any command from the repository root. Each command uses
the default options, so it opens the figure directly. Close one window before running the next.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### `tensorkrowch`

```powershell
python examples\run_demo.py tensorkrowch mps
python examples\run_demo.py tensorkrowch mpo
python examples\run_demo.py tensorkrowch ladder
python examples\run_demo.py tensorkrowch peps
python examples\run_demo.py tensorkrowch cubic_peps
python examples\run_demo.py tensorkrowch mera
python examples\run_demo.py tensorkrowch mera_ttn
python examples\run_demo.py tensorkrowch weird
python examples\run_demo.py tensorkrowch disconnected
```

### `tensornetwork`

```powershell
python examples\run_demo.py tensornetwork mps
python examples\run_demo.py tensornetwork mpo
python examples\run_demo.py tensornetwork ladder
python examples\run_demo.py tensornetwork peps
python examples\run_demo.py tensornetwork cubic_peps
python examples\run_demo.py tensornetwork mera
python examples\run_demo.py tensornetwork mera_ttn
python examples\run_demo.py tensornetwork weird
python examples\run_demo.py tensornetwork disconnected
```

### `quimb`

```powershell
python examples\run_demo.py quimb mps
python examples\run_demo.py quimb mpo
python examples\run_demo.py quimb ladder
python examples\run_demo.py quimb peps
python examples\run_demo.py quimb cubic_peps
python examples\run_demo.py quimb mera
python examples\run_demo.py quimb mera_ttn
python examples\run_demo.py quimb weird
python examples\run_demo.py quimb disconnected
python examples\run_demo.py quimb hyper
```

### `tenpy`

```powershell
python examples\run_demo.py tenpy mps
python examples\run_demo.py tenpy mpo
python examples\run_demo.py tenpy imps
python examples\run_demo.py tenpy impo
python examples\run_demo.py tenpy purification
python examples\run_demo.py tenpy uniform
python examples\run_demo.py tenpy excitation
python examples\run_demo.py tenpy chain
python examples\run_demo.py tenpy hub
python examples\run_demo.py tenpy hyper
```

### `einsum`

```powershell
python examples\run_demo.py einsum mps
python examples\run_demo.py einsum mpo
python examples\run_demo.py einsum peps
python examples\run_demo.py einsum disconnected
python examples\run_demo.py einsum ellipsis
python examples\run_demo.py einsum batch
python examples\run_demo.py einsum trace
python examples\run_demo.py einsum ternary
python examples\run_demo.py einsum unary
python examples\run_demo.py einsum nway
python examples\run_demo.py einsum implicit_out
```

`tt` is an alias of `mps`, so it does not need a separate command.

## Batch runner

Use `run_all_examples.py` to save a matrix of PNGs headlessly:

```bash
python examples/run_all_examples.py --group default
python examples/run_all_examples.py --group contraction --views 2d
python examples/run_all_examples.py --group hover --list
python examples/run_all_examples.py --group all --fail-fast
```

Outputs go to `.tmp/run-all-examples/` by default.
