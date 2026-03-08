"""
TSP tensor network visualization.

Creates the TSP tensor network at step 0 (before contraction) for a random instance
and visualizes it using the tensor network visualizer.

Based on: https://github.com/DOKOS-TAYOS/Traveling_Salesman_Problem_with_Tensor_Networks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import tensorkrowch as tk

try:
    from tensor_network_viz import PlotConfig, show_tensor_network
except ImportError:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from tensor_network_viz import PlotConfig, show_tensor_network


def generate_superposition_layer(tn: tk.TensorNetwork, n_nodes: int) -> list[tk.Node]:
    """Creates uniform superposition vectors of cities."""
    uniform_node = tk.Node(
        tensor=torch.ones(n_nodes),
        name="uniform",
        axes_names=["right"],
        network=tn,
        virtual=True,
    )
    layer: list[tk.Node] = []
    for i in range(n_nodes):
        layer.append(
            tk.Node(
                shape=(n_nodes,),
                name=f"initial_({i},{0})",
                axes_names=["right"],
                network=tn,
            )
        )
        layer[i].set_tensor_from(uniform_node)
    return layer


def generate_evolution_layer(
    tn: tk.TensorNetwork,
    n_nodes: int,
    distances: torch.Tensor,
    tau: float,
) -> list[tk.Node]:
    """Creates the MPO tensors responsible for imaginary time evolution."""
    layer: list[tk.Node] = []

    # Initial node
    initial_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes))
    for current in range(n_nodes):
        initial_tensor[current, current, current] = torch.exp(
            -tau * distances[n_nodes, current]
        )
    initial_node = tk.Node(
        tensor=initial_tensor,
        name="evolution_(0,1)",
        network=tn,
        axes_names=["left", "right", "down"],
    )
    layer.append(initial_node)

    # Intermediate nodes
    intermediate_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes, n_nodes))
    for previous in range(n_nodes):
        for current in range(n_nodes):
            intermediate_tensor[current, current, previous, current] = torch.exp(
                -tau * distances[previous, current]
            )
    intermediate_node = tk.Node(
        tensor=intermediate_tensor,
        name="evolution_(uniform)",
        network=tn,
        virtual=True,
        axes_names=["left", "right", "up", "down"],
    )
    for node in range(1, n_nodes - 1):
        layer.append(
            tk.Node(
                shape=(n_nodes, n_nodes, n_nodes, n_nodes),
                name=f"evolution_({node},1)",
                network=tn,
                axes_names=["left", "right", "up", "down"],
            )
        )
        layer[node].set_tensor_from(intermediate_node)
        layer[node]["up"] ^ layer[node - 1]["down"]

    # Final node
    final_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes))
    for previous in range(n_nodes):
        for current in range(n_nodes):
            final_tensor[current, current, previous] = torch.exp(
                -tau
                * (
                    distances[previous, current]
                    + distances[current, n_nodes + 1]
                )
            )
    layer.append(
        tk.Node(
            tensor=final_tensor,
            name=f"evolution_({n_nodes-1},1)",
            network=tn,
            axes_names=["left", "right", "up"],
        )
    )
    layer[-1]["up"] ^ layer[-2]["down"]

    return layer


def generate_restriction_layer(
    tn: tk.TensorNetwork,
    n_nodes: int,
    target_node: int,
) -> list[tk.Node]:
    """Creates the MPO tensors responsible for enforcing city visit restrictions."""
    layer: list[tk.Node] = []

    # Initial node
    initial_tensor = torch.zeros(size=(n_nodes, n_nodes, 2))
    for current in range(n_nodes):
        initial_tensor[current, current, int(current == target_node)] = 1
    layer.append(
        tk.Node(
            tensor=initial_tensor,
            name=f"restr_({target_node},{0})",
            network=tn,
            axes_names=["left", "right", "down"],
        )
    )

    # Intermediate nodes
    intermediate_tensor = torch.zeros(size=(n_nodes, n_nodes, 2, 2))
    for current in range(n_nodes):
        if current == target_node:
            intermediate_tensor[current, current, 0, 1] = 1
        else:
            intermediate_tensor[current, current, 0, 0] = 1
            intermediate_tensor[current, current, 1, 1] = 1
    intermediate_node = tk.Node(
        tensor=intermediate_tensor,
        name=f"restr_({target_node},uniform)",
        network=tn,
        virtual=True,
        axes_names=["left", "right", "up", "down"],
    )
    for i_node in range(1, n_nodes - 1):
        layer.append(
            tk.Node(
                shape=(n_nodes, n_nodes, 2, 2),
                name=f"restr_({target_node},{i_node})",
                network=tn,
                axes_names=["left", "right", "up", "down"],
            )
        )
        layer[i_node].set_tensor_from(intermediate_node)
        layer[i_node]["up"] ^ layer[i_node - 1]["down"]

    # Final node
    final_tensor = torch.zeros(size=(n_nodes, n_nodes, 2))
    for current in range(n_nodes):
        final_tensor[current, current, int(current != target_node)] = 1
    layer.append(
        tk.Node(
            tensor=final_tensor,
            name=f"restr_({target_node},{n_nodes-1})",
            network=tn,
            axes_names=["left", "right", "up"],
        )
    )
    layer[-1]["up"] ^ layer[-2]["down"]

    return layer


def generate_trace_layer(tn: tk.TensorNetwork, n_nodes: int) -> list[tk.Node]:
    """Creates a layer of trace tensors for the final contraction."""
    uniform_node = tk.Node(
        tensor=torch.ones(n_nodes),
        name="trace_template",
        axes_names=["left"],
        network=tn,
        virtual=True,
    )
    layer: list[tk.Node] = []
    for i in range(n_nodes):
        layer.append(
            tk.Node(
                shape=(n_nodes,),
                name=f"trace_({i})",
                axes_names=["left"],
                network=tn,
            )
        )
        layer[i].set_tensor_from(uniform_node)
    return layer


def create_tsp_tensor_network(
    tn: tk.TensorNetwork,
    distances: torch.Tensor,
    tau: float,
    n_layers: int | None,
) -> list[list[tk.Node]]:
    """Creates the tensors of the tensor network for the TSP problem (no contraction)."""
    n_nodes = len(distances) - 2

    superp_layer = generate_superposition_layer(tn, n_nodes)
    evol_layer = generate_evolution_layer(tn, n_nodes, distances, tau)
    restr_layer: list[list[tk.Node]] = []

    if n_layers is None or n_layers >= n_nodes - 1:
        target_nodes = range(n_nodes - 1)
    else:
        target_nodes = torch.randperm(n_nodes - 1)[:n_layers]

    for target_node in target_nodes:
        restr_layer.append(generate_restriction_layer(tn, n_nodes, int(target_node)))

    trace_layer = generate_trace_layer(tn, n_nodes)

    # Connect the tensors
    for node in range(n_nodes):
        superp_layer[node]["right"] ^ evol_layer[node]["left"]

        if not restr_layer:
            evol_layer[node]["right"] ^ trace_layer[node]["left"]
        else:
            evol_layer[node]["right"] ^ restr_layer[0][node]["left"]
            for depth in range(len(restr_layer) - 1):
                restr_layer[depth][node]["right"] ^ restr_layer[depth + 1][node]["left"]
            restr_layer[-1][node]["right"] ^ trace_layer[node]["left"]

    layers = [superp_layer, evol_layer]
    layers.extend(restr_layer)
    layers.append(trace_layer)

    return layers


def _grid_positions_from_layers(
    layers: list[list[tk.Node]],
) -> dict[int, tuple[float, float, float]]:
    """Build PEPS-style grid positions: rows = layers, cols = positions along path.
    Returns 3D coords (x, y, z) for compatibility with both 2D and 3D views."""
    positions: dict[int, tuple[float, float, float]] = {}
    for row, layer in enumerate(layers):
        for col, node in enumerate(layer):
            positions[id(node)] = (float(col), -float(row), 0.0)
    return positions


def _network_view_for_grid(layers: list[list[tk.Node]]) -> Any:
    """Wrapper exposing only grid nodes (excludes virtual templates) for PEPS layout."""

    class _View:
        def __init__(self, ls: list[list[tk.Node]]) -> None:
            self._nodes = [node for layer in ls for node in layer]

        @property
        def nodes(self) -> list[tk.Node]:
            return self._nodes

    return _View(layers)


def random_distance_matrix(n: int, seed: int | None = None) -> torch.Tensor:
    """Creates a random symmetric distance matrix for n nodes (including start/end)."""
    if seed is not None:
        torch.manual_seed(seed)
    d = torch.rand(n, n) * 10 + 1
    d = (d + d.T) / 2
    d.fill_diagonal_(0)
    return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSP tensor network visualization (step 0, before contraction).",
    )
    parser.add_argument(
        "-n",
        "--cities",
        type=int,
        default=4,
        help="Number of cities (default: 4). Use small values (4–6) for readable plots.",
    )
    parser.add_argument(
        "--view",
        choices=("2d", "3d"),
        default="2d",
        help="Visualization mode (default: 2d).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_cities = args.cities
    tau = 1.0
    n_layers: int | None = None  # All restriction layers

    distances = random_distance_matrix(n_cities + 1, seed=42)

    tn = tk.TensorNetwork(name="TSP")
    layers = create_tsp_tensor_network(tn, distances, tau, n_layers)

    network_view = _network_view_for_grid(layers)
    grid_positions = _grid_positions_from_layers(layers)

    print(f"TSP tensor network: {len(network_view.nodes)} nodes (grid)")
    print(f"Instance: {n_cities} cities, tau={tau}, view={args.view}")

    config = PlotConfig(
        figsize=(12, 8),
        show_tensor_labels=args.view == "2d",
        show_index_labels=args.view == "2d",
        positions=grid_positions,
    )
    fig, ax = show_tensor_network(
        network_view,
        engine="tensorkrowch",
        view=args.view,
        config=config,
        show=False,
    )
    fig.suptitle(f"TSP Tensor Network ({n_cities} cities, step 0)", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
