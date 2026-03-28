# Examples

Runnable scripts for every supported backend plus larger stress demos. For **installation**, API
**modes**, and **`PlotConfig`**, see the [user README](../README.md) and the
[in-depth guide](../docs/guide.md).

Run commands from the **repository root** with a virtual environment that has the matching optional
dependency (or `pip install -e ".[dev]"` for everything needed by tests).

## Copy-paste runs

**Windows (PowerShell)** — from repo root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[quimb]"
python examples\quimb_demo.py mps 2d
```

**Linux / macOS:**

```bash
source .venv/bin/activate
python -m pip install -e ".[quimb]"
python examples/quimb_demo.py mps 2d
```

Swap `.[quimb]` for `.[tensorkrowch]`, `.[tensornetwork]`, `.[tenpy]`, `.[einsum]`, etc.

Shared CLI helpers live in [`demo_cli.py`](demo_cli.py): **`add_hover_labels_argument`** adds
`--hover-labels`, and **`demo_plot_config(args)`** returns **`None`** or **`PlotConfig(hover_labels=True)`**
for the `config=` argument to **`show_tensor_network`**.

## Interactive labels (`--hover-labels`)

Most example scripts accept **`--hover-labels`**. That sets **`PlotConfig(hover_labels=True)`** so
tensor names and bond indices appear when you **hover** over a node or edge:

- **2D:** axes pick event hit-testing.
- **3D:** screen-space distance after projecting the geometry.

Requires a **real interactive** Matplotlib window (or **`%matplotlib widget`** in Jupyter). The
flag has **no visible effect** with **`--no-show`** or PNG-only / **Agg** batch runs, because there
is no pointer event loop.

To use the same behavior directly in code:

```python
from tensor_network_viz import PlotConfig, show_tensor_network

fig, ax = show_tensor_network(
    network,
    engine="quimb",
    view="2d",
    config=PlotConfig(figsize=(8, 6), hover_labels=True),
)
```

## `tensorkrowch_demo.py`

```bash
python examples/tensorkrowch_demo.py mps 2d
python examples/tensorkrowch_demo.py weird 3d
python examples/tensorkrowch_demo.py disconnected 2d
python examples/tensorkrowch_demo.py mps 2d --from-list
```

Covers `mps`, `mpo`, `peps`, `weird`, `disconnected`; full network vs **subset** via `--from-list`.

## `tensornetwork_demo.py`

```bash
python examples/tensornetwork_demo.py mps 2d
python examples/tensornetwork_demo.py weird 3d
python examples/tensornetwork_demo.py disconnected 2d
python examples/tensornetwork_demo.py mps 2d --save mps.png --no-show
```

## `mera_tree_demo.py`

Large binary MERA + TTN (layout stress test).

```bash
python examples/mera_tree_demo.py 2d
python examples/mera_tree_demo.py 3d --mera-log2 5 --tree-depth 4
python examples/mera_tree_demo.py 2d --save mera_tree.png --no-show
```

## `cubic_peps_demo.py`

Volumetric cubic PEPS on an `Lx × Ly × Lz` grid; **3D** view is the default.

```bash
python examples/cubic_peps_demo.py
python examples/cubic_peps_demo.py 3d --lx 3 --ly 3 --lz 4
python examples/cubic_peps_demo.py 2d --save cubic_peps.png --no-show
```

## `quimb_demo.py`

Includes a **hyper-index** example.

```bash
python examples/quimb_demo.py hyper 2d
python examples/quimb_demo.py mps 2d
python examples/quimb_demo.py weird 3d
python examples/quimb_demo.py mps 2d --from-list --save quimb.png --no-show
```

## `tenpy_demo.py`

Finite and infinite TeNPy structures.

```bash
python examples/tenpy_demo.py mps 2d
python examples/tenpy_demo.py mpo 3d --save tenpy.png --no-show
python examples/tenpy_demo.py imps 2d
python examples/tenpy_demo.py impo 3d --save tenpy-infinite.png --no-show
```

## `einsum_demo.py`

Auto-traced vs manual `pair_tensor` lists.

```bash
python examples/einsum_demo.py mps 2d
python examples/einsum_demo.py mps 2d --mode manual
python examples/einsum_demo.py peps 3d
python examples/einsum_demo.py disconnected 3d
python examples/einsum_demo.py mps 2d --save einsum.png --no-show
```

Install **`tensor-network-visualization[einsum]`** (PyTorch) if you execute contractions in the demo.

## `tn_tsp.py`

TensorKrowch TSP tensor network (larger than toy demos).

```bash
python examples/tn_tsp.py -n 4 --view 2d
python examples/tn_tsp.py -n 5 --view 3d
python examples/tn_tsp.py --cities 6 --view 2d
```

## `demo_cli.py`

Not a top-level entry point: **`add_hover_labels_argument`** and **`demo_plot_config`** for the
other scripts.

## Run-all cheat sheet

```bash
python examples/tensorkrowch_demo.py mps 2d
python examples/tensorkrowch_demo.py mps 3d
python examples/tensorkrowch_demo.py mpo 2d
python examples/tensorkrowch_demo.py mpo 3d
python examples/tensorkrowch_demo.py peps 2d
python examples/tensorkrowch_demo.py peps 3d
python examples/tensorkrowch_demo.py weird 2d
python examples/tensorkrowch_demo.py weird 3d
python examples/tensorkrowch_demo.py disconnected 2d
python examples/tensorkrowch_demo.py disconnected 3d
python examples/tensorkrowch_demo.py mps 2d --from-list
python examples/tensornetwork_demo.py mps 2d
python examples/tensornetwork_demo.py mps 3d
python examples/tensornetwork_demo.py mpo 2d
python examples/tensornetwork_demo.py mpo 3d
python examples/tensornetwork_demo.py peps 2d
python examples/tensornetwork_demo.py peps 3d
python examples/tensornetwork_demo.py weird 2d
python examples/tensornetwork_demo.py weird 3d
python examples/tensornetwork_demo.py disconnected 2d
python examples/tensornetwork_demo.py disconnected 3d
python examples/quimb_demo.py mps 2d
python examples/quimb_demo.py mps 3d
python examples/quimb_demo.py hyper 2d
python examples/quimb_demo.py hyper 3d
python examples/quimb_demo.py mpo 2d
python examples/quimb_demo.py mpo 3d
python examples/quimb_demo.py peps 2d
python examples/quimb_demo.py peps 3d
python examples/quimb_demo.py weird 2d
python examples/quimb_demo.py weird 3d
python examples/quimb_demo.py disconnected 2d
python examples/quimb_demo.py disconnected 3d
python examples/quimb_demo.py mps 2d --from-list
python examples/tenpy_demo.py mps 2d
python examples/tenpy_demo.py mps 3d
python examples/tenpy_demo.py mpo 2d
python examples/tenpy_demo.py mpo 3d
python examples/tenpy_demo.py imps 2d
python examples/tenpy_demo.py imps 3d
python examples/tenpy_demo.py impo 2d
python examples/tenpy_demo.py impo 3d
python examples/einsum_demo.py mps 2d
python examples/einsum_demo.py mps 3d
python examples/einsum_demo.py mps 2d --mode manual
python examples/einsum_demo.py mps 3d --mode manual
python examples/einsum_demo.py peps 2d
python examples/einsum_demo.py peps 3d
python examples/einsum_demo.py disconnected 2d
python examples/einsum_demo.py disconnected 3d
python examples/tn_tsp.py -n 4 --view 2d
python examples/tn_tsp.py -n 4 --view 3d
python examples/tn_tsp.py -n 5 --view 2d
python examples/tn_tsp.py -n 5 --view 3d
python examples/tn_tsp.py -n 6 --view 2d
python examples/tn_tsp.py -n 6 --view 3d
```

## See also

- [README.md](../README.md) — **Modes**, **`show_tensor_network`**, **`PlotConfig`** table, short troubleshooting.
- [docs/guide.md](../docs/guide.md) — Full manual, recipes, and **Troubleshooting** section.
