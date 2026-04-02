from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

BatchGroup = Literal["default", "hover", "contraction", "all"]
BatchViews = Literal["2d", "3d", "both"]

_DEFAULT_OUTPUT_DIR = Path(".tmp") / "run-all-examples"


class ExampleCommand(NamedTuple):
    slug: str
    argv: tuple[str, ...]


def _command(*argv: str, slug: str) -> ExampleCommand:
    return ExampleCommand(slug=slug, argv=tuple(argv))


def _network_demo_commands(
    script: str,
    names: Sequence[str],
    view: Literal["2d", "3d"],
) -> tuple[ExampleCommand, ...]:
    stem = Path(script).stem.replace("_demo", "")
    return tuple(_command(script, name, view, slug=f"{stem}_{name}_{view}") for name in names)


def _hover_commands(commands: Sequence[ExampleCommand]) -> tuple[ExampleCommand, ...]:
    return tuple(
        ExampleCommand(
            slug=f"{command.slug}_hover",
            argv=command.argv + ("--hover-labels",),
        )
        for command in commands
    )


def _TENSORKROWCH_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return _network_demo_commands(
        "examples/tensorkrowch_demo.py",
        ("disconnected", "ladder", "mps", "mpo", "peps", "weird"),
        view,
    )


def _TENSORNETWORK_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return _network_demo_commands(
        "examples/tensornetwork_demo.py",
        ("disconnected", "ladder", "mps", "mpo", "peps", "weird"),
        view,
    )


def _QUIMB_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return _network_demo_commands(
        "examples/quimb_demo.py",
        ("disconnected", "hyper", "ladder", "mps", "mpo", "peps", "weird"),
        view,
    )


def _TENPY_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return _network_demo_commands(
        "examples/tenpy_demo.py",
        ("impo", "imps", "mpo", "mps", "purification", "uniform", "excitation"),
        view,
    )


def _TENPY_EXPLICIT_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return _network_demo_commands(
        "examples/tenpy_explicit_tn_demo.py",
        ("chain", "hub"),
        view,
    )


def _EINSUM_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return (
        *_network_demo_commands(
            "examples/einsum_demo.py",
            ("disconnected", "mps", "peps"),
            view,
        ),
        _command(
            "examples/einsum_demo.py",
            "mps",
            view,
            "--mode",
            "manual",
            slug=f"einsum_mps_{view}_manual",
        ),
        _command(
            "examples/einsum_demo.py",
            "peps",
            view,
            "--mode",
            "manual",
            slug=f"einsum_peps_{view}_manual",
        ),
    )


def _EINSUM_GENERAL_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    stem = "einsum_general"
    return tuple(
        _command(
            "examples/einsum_general.py",
            name,
            view,
            slug=f"{stem}_{name}_{view}",
        )
        for name in (
            "batch",
            "ellipsis",
            "mps_short",
            "nway",
            "trace",
            "implicit_out",
            "ternary",
            "unary",
        )
    )


def _SPECIAL_DEFAULT(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    if view == "2d":
        return (
            _command("examples/mera_tree_demo.py", "2d", slug="mera_tree_2d"),
            _command("examples/cubic_peps_demo.py", "2d", slug="cubic_peps_2d"),
            _command("examples/tn_tsp.py", "-n", "4", "--view", "2d", slug="tn_tsp_2d"),
        )
    return (
        _command(
            "examples/mera_tree_demo.py",
            "3d",
            "--mera-log2",
            "5",
            "--tree-depth",
            "4",
            slug="mera_tree_3d",
        ),
        _command(
            "examples/cubic_peps_demo.py",
            "3d",
            "--lx",
            "3",
            "--ly",
            "3",
            "--lz",
            "4",
            slug="cubic_peps_3d_lx3_ly3_lz4",
        ),
        _command("examples/tn_tsp.py", "-n", "5", "--view", "3d", slug="tn_tsp_3d"),
    )


def _DEFAULT_COMMANDS(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return (
        *_TENSORKROWCH_DEFAULT(view),
        *_TENSORNETWORK_DEFAULT(view),
        *_QUIMB_DEFAULT(view),
        *_TENPY_DEFAULT(view),
        *_TENPY_EXPLICIT_DEFAULT(view),
        *_EINSUM_DEFAULT(view),
        *_EINSUM_GENERAL_DEFAULT(view),
        *_SPECIAL_DEFAULT(view),
    )


def _CONTRACTION_COMMANDS(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    if view == "2d":
        return (
            _command(
                "examples/tensorkrowch_demo.py",
                "mps",
                "2d",
                "--contraction-scheme",
                slug="tensorkrowch_mps_2d_contraction",
            ),
            _command(
                "examples/tensornetwork_demo.py",
                "peps",
                "2d",
                "--contraction-scheme",
                slug="tensornetwork_peps_2d_contraction",
            ),
            _command(
                "examples/quimb_demo.py",
                "hyper",
                "2d",
                "--contraction-scheme",
                slug="quimb_hyper_2d_contraction",
            ),
            _command(
                "examples/tenpy_explicit_tn_demo.py",
                "hub",
                "2d",
                "--contraction-scheme",
                slug="tenpy_explicit_hub_2d_contraction",
            ),
            _command(
                "examples/einsum_demo.py",
                "peps",
                "2d",
                "--contraction-scheme",
                slug="einsum_peps_2d_contraction",
            ),
            _command(
                "examples/einsum_general.py",
                "mps_short",
                "2d",
                "--contraction-scheme",
                slug="einsum_general_mps_short_2d_contraction",
            ),
            _command(
                "examples/cubic_peps_demo.py",
                "2d",
                "--contraction-scheme",
                slug="cubic_peps_2d_contraction",
            ),
        )
    return (
        _command(
            "examples/tensorkrowch_demo.py",
            "mps",
            "3d",
            "--contraction-scheme",
            slug="tensorkrowch_mps_3d_contraction",
        ),
        _command(
            "examples/tensornetwork_demo.py",
            "mpo",
            "3d",
            "--contraction-scheme",
            slug="tensornetwork_mpo_3d_contraction",
        ),
        _command(
            "examples/quimb_demo.py",
            "mps",
            "3d",
            "--contraction-scheme",
            slug="quimb_mps_3d_contraction",
        ),
        _command(
            "examples/cubic_peps_demo.py",
            "3d",
            "--lx",
            "3",
            "--ly",
            "3",
            "--lz",
            "3",
            "--contraction-scheme",
            slug="cubic_peps_3d_lx3_ly3_lz3_contraction",
        ),
        _command(
            "examples/einsum_demo.py",
            "mps",
            "3d",
            "--contraction-scheme",
            slug="einsum_mps_3d_contraction",
        ),
    )


def _views_to_run(views: BatchViews) -> tuple[Literal["2d", "3d"], ...]:
    if views == "both":
        return ("2d", "3d")
    return (views,)


def select_example_commands(
    *,
    group: BatchGroup,
    views: BatchViews,
) -> tuple[ExampleCommand, ...]:
    selected_views = _views_to_run(views)
    if group == "default":
        return tuple(command for view in selected_views for command in _DEFAULT_COMMANDS(view))
    if group == "hover":
        return tuple(
            command
            for view in selected_views
            for command in _hover_commands(_DEFAULT_COMMANDS(view))
        )
    if group == "contraction":
        return tuple(command for view in selected_views for command in _CONTRACTION_COMMANDS(view))
    if group == "all":
        return (
            *select_example_commands(group="default", views=views),
            *select_example_commands(group="hover", views=views),
            *select_example_commands(group="contraction", views=views),
        )
    raise ValueError(f"Unsupported example group: {group}")


def build_subprocess_command(
    command: ExampleCommand,
    *,
    output_dir: Path,
    python_executable: str | None = None,
) -> list[str]:
    save_path = output_dir / f"{command.slug}.png"
    return [
        python_executable or sys.executable,
        *command.argv,
        "--save",
        str(save_path),
        "--no-show",
    ]


def format_example_command(command: ExampleCommand) -> str:
    return " ".join(command.argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the example demos in batch from the repository root and save PNGs headlessly. "
            "The command matrix mirrors the grouped examples documented in CONTRIBUTING.md."
        ),
    )
    parser.add_argument(
        "--group",
        choices=("default", "hover", "contraction", "all"),
        default="default",
        help="Which command set to run (default: default).",
    )
    parser.add_argument(
        "--views",
        choices=("2d", "3d", "both"),
        default="both",
        help="Which views to include (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for saved PNG outputs (default: .tmp/run-all-examples).",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable to use for subprocesses (default: current interpreter).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected example commands without running them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing example command.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = select_example_commands(
        group=args.group,
        views=args.views,
    )
    if args.list:
        for command in commands:
            print(format_example_command(command))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[ExampleCommand] = []
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{total}] {format_example_command(command)}")
        completed = subprocess.run(
            build_subprocess_command(
                command,
                output_dir=args.output_dir,
                python_executable=args.python_executable,
            ),
            check=False,
        )
        if completed.returncode == 0:
            continue
        failures.append(command)
        print(
            "Command failed with exit code "
            f"{completed.returncode}: {format_example_command(command)}"
        )
        if args.fail_fast:
            break

    if failures:
        print("\nFailed example commands:")
        for command in failures:
            print(f"- {format_example_command(command)}")
        return 1

    print(f"Saved {total} example render(s) to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
