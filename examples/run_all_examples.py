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


def _command(
    engine: str, example: str, view: Literal["2d", "3d"], *extra: str, slug: str
) -> ExampleCommand:
    return ExampleCommand(
        slug=slug,
        argv=("examples/run_demo.py", engine, example, "--view", view, *extra),
    )


def _default_commands(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return (
        _command("tensorkrowch", "disconnected", view, slug=f"tensorkrowch_disconnected_{view}"),
        _command("tensorkrowch", "mps", view, slug=f"tensorkrowch_mps_{view}"),
        _command("tensornetwork", "mera_ttn", view, slug=f"tensornetwork_mera_ttn_{view}"),
        _command("tensornetwork", "peps", view, slug=f"tensornetwork_peps_{view}"),
        _command("quimb", "hyper", view, slug=f"quimb_hyper_{view}"),
        _command("quimb", "ladder", view, slug=f"quimb_ladder_{view}"),
        _command("tenpy", "mps", view, slug=f"tenpy_mps_{view}"),
        _command("tenpy", "imps", view, slug=f"tenpy_imps_{view}"),
        _command("tenpy", "chain", view, slug=f"tenpy_chain_{view}"),
        _command("einsum", "mps", view, slug=f"einsum_mps_{view}"),
        _command("einsum", "ellipsis", view, slug=f"einsum_ellipsis_{view}"),
    )


def _hover_commands(commands: Sequence[ExampleCommand]) -> tuple[ExampleCommand, ...]:
    return tuple(
        ExampleCommand(
            slug=f"{command.slug}_hover",
            argv=command.argv + ("--hover-labels",),
        )
        for command in commands
    )


def _contraction_commands(view: Literal["2d", "3d"]) -> tuple[ExampleCommand, ...]:
    return (
        _command(
            "tensornetwork",
            "mera_ttn",
            view,
            "--scheme",
            slug=f"tensornetwork_mera_ttn_{view}_scheme",
        ),
        _command("quimb", "hyper", view, "--scheme", slug=f"quimb_hyper_{view}_scheme"),
        _command("tenpy", "chain", view, "--scheme", slug=f"tenpy_chain_{view}_scheme"),
        _command("einsum", "mps", view, "--scheme", slug=f"einsum_mps_{view}_scheme"),
    )


def _views_to_run(views: BatchViews) -> tuple[Literal["2d", "3d"], ...]:
    if views == "both":
        return ("2d", "3d")
    return (views,)


def select_example_commands(*, group: BatchGroup, views: BatchViews) -> tuple[ExampleCommand, ...]:
    selected_views = _views_to_run(views)
    if group == "default":
        return tuple(command for view in selected_views for command in _default_commands(view))
    if group == "hover":
        return tuple(
            command
            for view in selected_views
            for command in _hover_commands(_default_commands(view))
        )
    if group == "contraction":
        return tuple(command for view in selected_views for command in _contraction_commands(view))
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
            "Run the example demos in batch from the repository root and save PNGs headlessly."
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
        help="Print the selected commands without running them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing command.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = select_example_commands(group=args.group, views=args.views)
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
