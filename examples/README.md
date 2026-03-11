# Examples

This folder contains runnable scripts for every supported backend plus one larger TensorKrowch
example. For the full package guide, see [`../docs/guide.md`](../docs/guide.md).

Run examples from the project root, using the project virtual environment or an equivalent Python
environment with the needed optional dependencies installed.

## `tensorkrowch_demo.py`

Demonstrates the TensorKrowch backend with regular toy topologies.

```bash
python examples/tensorkrowch_demo.py mps 2d
python examples/tensorkrowch_demo.py weird 3d
python examples/tensorkrowch_demo.py disconnected 2d
python examples/tensorkrowch_demo.py mps 2d --from-list
```

Shows:

- `mps`, `mpo`, `peps`, `weird`, and `disconnected` examples
- object-level dispatch with a full TensorKrowch network
- subset-style dispatch by passing a list of nodes with `--from-list`

## `tensornetwork_demo.py`

Demonstrates the TensorNetwork backend using `tensornetwork.Node` collections.

```bash
python examples/tensornetwork_demo.py mps 2d
python examples/tensornetwork_demo.py weird 3d
python examples/tensornetwork_demo.py disconnected 2d
python examples/tensornetwork_demo.py mps 2d --save mps.png --no-show
```

Shows:

- `mps`, `mpo`, `peps`, `weird`, and `disconnected` examples
- non-interactive rendering with `--save` and `--no-show`
- the iterable-of-nodes input style used by the TensorNetwork adapter

## `quimb_demo.py`

Demonstrates the Quimb backend, including a hypergraph-style example.

```bash
python examples/quimb_demo.py hyper 2d
python examples/quimb_demo.py mps 2d
python examples/quimb_demo.py weird 3d
python examples/quimb_demo.py mps 2d --from-list --save quimb.png --no-show
```

Shows:

- `hyper`, `mps`, `mpo`, `peps`, `weird`, and `disconnected` examples
- dispatch with a full `TensorNetwork`
- dispatch with a list of tensors via `--from-list`

## `tenpy_demo.py`

Demonstrates the TeNPy backend for both finite and infinite structures.

```bash
python examples/tenpy_demo.py mps 2d
python examples/tenpy_demo.py mpo 3d --save tenpy.png --no-show
python examples/tenpy_demo.py imps 2d
python examples/tenpy_demo.py impo 3d --save tenpy-infinite.png --no-show
```

Shows:

- finite `mps` and `mpo`
- infinite `imps` and `impo`
- headless rendering and saving from the command line

## `einsum_demo.py`

Demonstrates the `einsum` backend in both auto-traced and manual-trace modes.

```bash
python examples/einsum_demo.py mps 2d
python examples/einsum_demo.py mps 2d --mode manual
python examples/einsum_demo.py peps 3d
python examples/einsum_demo.py disconnected 3d
python examples/einsum_demo.py mps 2d --save einsum.png --no-show
```

Shows:

- `mps`, `peps`, and `disconnected` traces
- `--mode auto` using `EinsumTrace` + `tensor_network_viz.einsum(...)`
- `--mode manual` using ordered `pair_tensor` entries + `torch.einsum(...)`

Note:

- install `tensor-network-visualization[einsum]` if you want to execute the examples with PyTorch

## `tn_tsp.py`

Builds and visualizes the TensorKrowch tensor network used for a TSP formulation before
contraction.

```bash
python examples/tn_tsp.py -n 4 --view 2d
python examples/tn_tsp.py -n 5 --view 3d
python examples/tn_tsp.py --cities 6 --view 2d
```

Useful when you want:

- a larger TensorKrowch example than the toy demos
- a concrete grid-like layout use case
- an example tied to a real tensor-network modeling workflow
