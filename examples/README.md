# Examples

Run from the project root (with venv activated or `python` from `.venv`).

## tensor_network_demo.py

```bash
python examples/tensor_network_demo.py mps 2d
python examples/tensor_network_demo.py weird 3d
python examples/tensor_network_demo.py disconnected 2d
python examples/tensor_network_demo.py mps 2d --from-list
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

## tn_tsp.py

TSP tensor network visualization (step 0, before contraction).

```bash
python examples/tn_tsp.py -n 4 --view 2d
python examples/tn_tsp.py -n 5 --view 3d
python examples/tn_tsp.py --cities 6 --view 2d
```

- `-n`, `--cities`: Number of cities (default: 4). Use 4–6 for readable plots.
- `--view`: `2d` or `3d` (default: 2d).
