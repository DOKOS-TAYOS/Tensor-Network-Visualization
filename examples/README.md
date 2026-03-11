# Examples

Run from the project root (with venv activated or `python` from `.venv`).

## tensorkrowch_demo.py

```bash
python examples/tensorkrowch_demo.py mps 2d
python examples/tensorkrowch_demo.py weird 3d
python examples/tensorkrowch_demo.py disconnected 2d
python examples/tensorkrowch_demo.py mps 2d --from-list
```

- **Networks:** `disconnected`, `mps`, `mpo`, `peps`, `weird`
- **Views:** `2d`, `3d`
- **`--from-list`:** Pass the network as a list of nodes instead of the TensorNetwork object

## tensornetwork_demo.py

```bash
python examples/tensornetwork_demo.py mps 2d
python examples/tensornetwork_demo.py weird 3d
python examples/tensornetwork_demo.py disconnected 2d
python examples/tensornetwork_demo.py mps 2d --save mps.png --no-show
```

- **Networks:** `disconnected`, `mps`, `mpo`, `peps`, `weird`
- **Views:** `2d`, `3d`
- **Input:** Always passes a list of `tensornetwork.Node`
- **`--save PATH`:** Save the rendered figure
- **`--no-show`:** Do not open the interactive Matplotlib window

## quimb_demo.py

```bash
python examples/quimb_demo.py hyper 2d
python examples/quimb_demo.py mps 2d
python examples/quimb_demo.py weird 3d
python examples/quimb_demo.py disconnected 2d
python examples/quimb_demo.py mps 2d --from-list --save quimb.png --no-show
```

- **Networks:** `disconnected`, `hyper`, `mps`, `mpo`, `peps`, `weird`
- **Views:** `2d`, `3d`
- **Input:** Quimb `TensorNetwork` by default, or list of tensors with `--from-list`
- **`--save PATH`:** Save the rendered figure
- **`--no-show`:** Do not open the interactive Matplotlib window

## tenpy_demo.py

```bash
python examples/tenpy_demo.py imps 2d
python examples/tenpy_demo.py impo 3d --save tenpy-infinite.png --no-show
python examples/tenpy_demo.py mps 2d
python examples/tenpy_demo.py mpo 3d --save tenpy.png --no-show
```

- **Networks:** `impo`, `imps`, `mps`, `mpo`
- **Views:** `2d`, `3d`
- **Input:** Finite, segment, or infinite TeNPy `MPS`, and finite or infinite `MPO`
- **`--save PATH`:** Save the rendered figure
- **`--no-show`:** Do not open the interactive Matplotlib window

## einsum_demo.py

```bash
python examples/einsum_demo.py mps 2d
python examples/einsum_demo.py peps 3d
python examples/einsum_demo.py disconnected 3d
python examples/einsum_demo.py mps 2d --save einsum.png --no-show
```

- **Networks:** `disconnected`, `mps`, `peps`
- **Views:** `2d`, `3d`
- **Input:** Ordered list of `pair_tensor` entries traced from binary explicit `torch.einsum`
- **Note:** PyTorch is only required to execute the demo contractions, not to render a previously built trace
- **`--save PATH`:** Save the rendered figure
- **`--no-show`:** Do not open the interactive Matplotlib window

## Architecture note

- `tensor_network_viz._core`: Common graph, layout, drawing, and rendering pipeline
- `tensor_network_viz.tensorkrowch`: TensorKrowch-specific input adapter plus public helpers
- `tensor_network_viz.tensornetwork`: TensorNetwork-specific input adapter plus public helpers
- `tensor_network_viz.quimb`: Quimb-specific input adapter plus public helpers
- `tensor_network_viz.tenpy`: TeNPy-specific input adapter plus public helpers
- `tensor_network_viz.einsum`: Ordered `pair_tensor` trace adapter plus public helpers

## tn_tsp.py

TSP tensor network visualization (step 0, before contraction).

```bash
python examples/tn_tsp.py -n 4 --view 2d
python examples/tn_tsp.py -n 5 --view 3d
python examples/tn_tsp.py --cities 6 --view 2d
```

- `-n`, `--cities`: Number of cities (default: 4). Use 4–6 for readable plots.
- `--view`: `2d` or `3d` (default: 2d).
