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

## Batch runs

To run the same grouped smoke checks documented in `CONTRIBUTING.md`, use
[`run_all_examples.py`](run_all_examples.py) from the repository root.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python examples\run_all_examples.py --group default
.\.venv\Scripts\python examples\run_all_examples.py --group contraction --views 2d
.\.venv\Scripts\python examples\run_all_examples.py --group hover --list
.\.venv\Scripts\python examples\run_all_examples.py --group all --fail-fast
```

**Linux / macOS:**

```bash
python examples/run_all_examples.py --group default
python examples/run_all_examples.py --group contraction --views 2d
python examples/run_all_examples.py --group hover --list
python examples/run_all_examples.py --group all --fail-fast
```

The runner saves PNGs into `.tmp/run-all-examples/` by default and appends `--save` plus `--no-show`
to every demo invocation. Because those runs are headless, the demos automatically disable
`interactive_controls`, so the exported images do not include the widget panel. The `hover` batch
group is still useful to check that the `--hover-labels` CLI path works, but it does **not** test
real pointer interaction; for that you still need an interactive Matplotlib session.

Shared helpers in [`demo_cli.py`](demo_cli.py): **`add_hover_labels_argument`** (`--hover-labels`),
**`add_compact_argument`** (`--compact` = smaller figure, default line widths and auto layout
iterations), **`apply_demo_caption`** for titles/subtitles, **`demo_runs_headless(args)`** to detect
export-only runs, and **`render_demo_tensor_network(...)`** which routes everything through
`show_tensor_network(..., interactive_controls=...)`. `demo_plot_config(args)` still merges
hover/compact into a **showcase** `PlotConfig` (slightly larger canvas, thicker lines, more layout
iterations) unless `--compact` is set.

All example scripts call `show_tensor_network`, so when you run them interactively you also get the
figure-level Matplotlib controls from the library: `2d/3d`, `Hover`, `Tensor labels`, and
`Edge labels`. The positional `2d` / `3d` CLI argument chooses the **initial** view only; you can
still switch after the window opens.

## Example defaults vs library defaults

The library default is `PlotConfig(hover_labels=True)`, but the example scripts intentionally start
with hover **off** unless you pass `--hover-labels`. That keeps screenshots and smoke checks more
predictable while still letting you turn hover back on from the CLI or the figure checkbox.

## Interactive labels (`--hover-labels`)

Most example scripts accept **`--hover-labels`**. That sets **`PlotConfig(hover_labels=True)`** so
tensor names and bond indices appear when you **hover** over a node or edge:

- **2D:** axes pick event hit-testing.
- **3D:** screen-space distance after projecting the geometry.

Requires a **real interactive** Matplotlib window (or **`%matplotlib widget`** in Jupyter). The
flag has **no visible effect** with **`--no-show`** or PNG-only / **Agg** batch runs, because there
is no pointer event loop.

Static tensor and edge labels are still off at startup in the demos. In an interactive window you
can turn them on from the `Tensor labels` and `Edge labels` checkboxes without rerunning the
script.

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

If you want a plain export from your own code without the widget panel, add
`interactive_controls=False` to `show_tensor_network(...)`.

## `tensorkrowch_demo.py`

```bash
python examples/tensorkrowch_demo.py mps 2d
python examples/tensorkrowch_demo.py ladder 3d
python examples/tensorkrowch_demo.py weird 3d
python examples/tensorkrowch_demo.py disconnected 2d
python examples/tensorkrowch_demo.py mps 2d --from-list
python examples/tensorkrowch_demo.py mps 2d --save mps.png --no-show
```

Covers `mps`, `mpo`, `peps`, `ladder`, `weird`, `disconnected`; full network vs **subset** via `--from-list`.

## `tensornetwork_demo.py`

```bash
python examples/tensornetwork_demo.py mps 2d
python examples/tensornetwork_demo.py ladder 2d
python examples/tensornetwork_demo.py weird 3d
python examples/tensornetwork_demo.py disconnected 2d
python examples/tensornetwork_demo.py peps 2d --compact
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

Finite / infinite MPS/MPO, **PurificationMPS**, **UniformMPS**, and an **excitation** chain with the
same duck-typed API as **MomentumMPS** (`get_X` + `uMPS_GS`).

```bash
python examples/tenpy_demo.py mps 2d
python examples/tenpy_demo.py mpo 3d --save tenpy.png --no-show
python examples/tenpy_demo.py imps 2d
python examples/tenpy_demo.py impo 3d --save tenpy-infinite.png --no-show
python examples/tenpy_demo.py purification 2d
python examples/tenpy_demo.py uniform 3d
python examples/tenpy_demo.py excitation 2d --save tenpy-excitation.png --no-show
```

## `tenpy_explicit_tn_demo.py`

Hand-made **`TenPyTensorNetwork`** / **`make_tenpy_tensor_network`**: open **chain** (binary bonds) or
**hub** (three tensors on one index, drawn with a virtual hub). Uses **only** the **`[tenpy]`**
extra.

```bash
python examples/tenpy_explicit_tn_demo.py chain 2d
python examples/tenpy_explicit_tn_demo.py hub 3d --save tenpy-explicit.png --no-show
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

## `einsum_general.py`

Ellipsis (batched matmul), batch Hadamard-style bonds (`ab,ab->ab`), diagonal/trace-style equations
(`ii,i->i`), a short MPS-style chain, **implicit binary** subscripts with optional **`out=`**, and
**single-step** **unary** / **ternary** traced einsums. Uses the same **auto-trace** workflow as
`einsum_demo.py` (`EinsumTrace` + `tensor_network_viz.einsum`). The renderer expands `...` using
ranks stored in **binary** `pair_tensor` metadata; **n-ary** steps are stored as **`einsum_trace_step`**
(see [`einsum_module/trace.py`](../src/tensor_network_viz/einsum_module/trace.py)). Virtual **hub**
nodes follow [`einsum_module/graph.py`](../src/tensor_network_viz/einsum_module/graph.py); **2D** layout
offsets single-neighbor trace hubs (e.g. **`unary`**) off the physical tensor so the figure stays readable.

```bash
python examples/einsum_general.py ellipsis 2d
python examples/einsum_general.py batch 3d
python examples/einsum_general.py nway 2d
python examples/einsum_general.py trace 2d
python examples/einsum_general.py mps_short 2d
python examples/einsum_general.py implicit_out 2d
python examples/einsum_general.py ternary 3d
python examples/einsum_general.py unary 2d
python examples/einsum_general.py ellipsis 2d --save einsum_general.png --no-show
```

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

For a **short pre-PR smoke checklist** (2D / 3D, default demo settings, hover flag coverage, and
contraction overlays), see [CONTRIBUTING.md](../CONTRIBUTING.md#optional-manual-example-smoke-checks).

```bash
python examples/run_all_examples.py --group default
python examples/run_all_examples.py --group contraction
python examples/run_all_examples.py --group hover --views 2d
python examples/run_all_examples.py --group all --fail-fast
```

## See also

- [README.md](../README.md) — **Modes**, **`show_tensor_network`**, **`PlotConfig`** table, short troubleshooting.
- [docs/guide.md](../docs/guide.md) — Full manual, recipes, and **Troubleshooting** section.
