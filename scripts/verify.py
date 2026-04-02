from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class VerificationStep:
    label: str
    command: tuple[str, ...]


VerificationGroup: TypeAlias = tuple[VerificationStep, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _command_groups() -> dict[str, VerificationGroup]:
    python = sys.executable
    return {
        "quality": (
            VerificationStep("ruff-check", (python, "-m", "ruff", "check", ".")),
            VerificationStep("ruff-format", (python, "-m", "ruff", "format", "--check", ".")),
            VerificationStep("pyright", (python, "-m", "pyright")),
        ),
        "tests": (VerificationStep("pytest", (python, "-m", "pytest", "-q")),),
        "smoke": (
            VerificationStep(
                "quimb-smoke",
                (
                    python,
                    "examples/quimb_demo.py",
                    "mps",
                    "2d",
                    "--save",
                    "smoke.png",
                    "--no-show",
                ),
            ),
        ),
        "wheel": (VerificationStep("build-wheel", (python, "-m", "build", "--wheel")),),
    }


def _ordered_steps(mode: str) -> VerificationGroup:
    command_groups = _command_groups()
    if mode == "all":
        return (
            *command_groups["quality"],
            *command_groups["tests"],
            *command_groups["smoke"],
            *command_groups["wheel"],
        )
    return command_groups[mode]


def _format_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def _run_step(step: VerificationStep, repo_root: Path) -> None:
    print(f"[verify] {step.label}")
    print(f"[verify] $ {_format_command(step.command)}")
    subprocess.run(step.command, cwd=repo_root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the repository verification steps used before merging.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("all", "quality", "tests", "smoke", "wheel"),
        default="all",
        help="Verification slice to run. Defaults to the full pre-merge suite.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()

    try:
        for step in _ordered_steps(args.mode):
            _run_step(step, repo_root)
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
