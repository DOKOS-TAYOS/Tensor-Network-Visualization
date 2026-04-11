# Demo Commands

Run these from the repository root after activating the project virtual environment.

## TensorKrowch

```bash
python examples/run_demo.py tensorkrowch mps
python examples/run_demo.py tensorkrowch mpo
python examples/run_demo.py tensorkrowch ladder
python examples/run_demo.py tensorkrowch peps
python examples/run_demo.py tensorkrowch cubic_peps
python examples/run_demo.py tensorkrowch mera
python examples/run_demo.py tensorkrowch mera_ttn
python examples/run_demo.py tensorkrowch weird
python examples/run_demo.py tensorkrowch disconnected
```

## TensorNetwork

```bash
python examples/run_demo.py tensornetwork mps
python examples/run_demo.py tensornetwork peps
python examples/run_demo.py tensornetwork weird
```

## Quimb

```bash
python examples/run_demo.py quimb mps
python examples/run_demo.py quimb peps
python examples/run_demo.py quimb hyper
```

## TeNPy

```bash
python examples/run_demo.py tenpy mps
python examples/run_demo.py tenpy mpo
python examples/run_demo.py tenpy imps
python examples/run_demo.py tenpy impo
python examples/run_demo.py tenpy purification
python examples/run_demo.py tenpy uniform
python examples/run_demo.py tenpy excitation
python examples/run_demo.py tenpy chain
python examples/run_demo.py tenpy hub
python examples/run_demo.py tenpy hyper
```

## Einsum

```bash
python examples/run_demo.py einsum mps
python examples/run_demo.py einsum mpo
python examples/run_demo.py einsum peps
python examples/run_demo.py einsum disconnected
python examples/run_demo.py einsum ellipsis
python examples/run_demo.py einsum batch
python examples/run_demo.py einsum trace
python examples/run_demo.py einsum ternary
python examples/run_demo.py einsum unary
python examples/run_demo.py einsum nway
python examples/run_demo.py einsum implicit_out
```

## Themes

```bash
python examples/run_demo.py themes overview
python examples/run_demo.py themes overview --view 3d
python examples/run_demo.py themes overview --save .tmp/examples/themes_overview.png --no-show
```

## Placements

```bash
python examples/run_demo.py placements object
python examples/run_demo.py placements list
python examples/run_demo.py placements grid2d
python examples/run_demo.py placements grid3d --view 3d
python examples/run_demo.py placements manual_positions
python examples/run_demo.py placements manual_scheme
python examples/run_demo.py placements named_indices
```

## Geometry

```bash
python examples/run_demo.py geometry partial_grid2d
python examples/run_demo.py geometry upper_triangle2d
python examples/run_demo.py geometry partial_grid3d --view 2d
python examples/run_demo.py geometry partial_grid3d --view 3d
python examples/run_demo.py geometry upper_pyramid3d --view 2d
python examples/run_demo.py geometry upper_pyramid3d --view 3d
python examples/run_demo.py geometry random_irregular
python examples/run_demo.py geometry circular_ring
python examples/run_demo.py geometry circular_chords
python examples/run_demo.py geometry tubular_grid --view 2d
python examples/run_demo.py geometry tubular_grid --view 3d
python examples/run_demo.py geometry disconnected_irregular
```

## Batch Checks

```bash
python examples/run_all_examples.py --group engines --views 2d --list
python examples/run_all_examples.py --group themes --views 2d --output-dir .tmp/examples
python examples/run_all_examples.py --group placements --views 2d --output-dir .tmp/examples
python examples/run_all_examples.py --group geometry --views 2d --output-dir .tmp/examples
python examples/run_all_examples.py --group all --views 2d --output-dir .tmp/examples
```
