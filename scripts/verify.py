from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class VerificationStep:
    label: str
    command: tuple[str, ...]
    expand_globs: bool = False


VerificationGroup: TypeAlias = tuple[VerificationStep, ...]
LOGGER = logging.getLogger("tensor_network_viz.verify")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _command_groups() -> dict[str, VerificationGroup]:
    python = sys.executable
    pytest_base = (python, "-m", "pytest", "-q")
    return {
        "quality": (
            VerificationStep("ruff-check", (python, "-m", "ruff", "check", ".")),
            VerificationStep("ruff-format", (python, "-m", "ruff", "format", "--check", ".")),
            VerificationStep("pyright", (python, "-m", "pyright")),
        ),
        "tests": (VerificationStep("pytest", pytest_base),),
        "perf": (
            VerificationStep(
                "pytest-perf",
                (*pytest_base, "-p", "no:cacheprovider", "--override-ini=addopts=", "-m", "perf"),
            ),
        ),
        "security": (
            VerificationStep("pip-check", (python, "-m", "pip", "check")),
            VerificationStep(
                "pip-audit",
                (python, "-m", "pip_audit", "--skip-editable", "--local"),
            ),
            VerificationStep(
                "bandit",
                (
                    python,
                    "-m",
                    "bandit",
                    "-r",
                    "src",
                    "scripts",
                    "examples",
                    "--severity-level",
                    "medium",
                ),
            ),
        ),
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
        "package": (
            VerificationStep(
                "build-dist",
                (
                    python,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    ".tmp/package-dist",
                    "--no-isolation",
                ),
            ),
            VerificationStep(
                "twine-check",
                (python, "-m", "twine", "check", "--strict", ".tmp/package-dist/*"),
                expand_globs=True,
            ),
        ),
    }


def _ordered_steps(mode: str) -> VerificationGroup:
    command_groups = _command_groups()
    if mode == "all":
        return (
            *command_groups["quality"],
            *command_groups["tests"],
            *command_groups["smoke"],
            *command_groups["package"],
        )
    return command_groups[mode]


def _format_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def _expand_command_globs(command: tuple[str, ...], repo_root: Path) -> tuple[str, ...]:
    expanded: list[str] = []
    for part in command:
        if "*" not in part and "?" not in part:
            expanded.append(part)
            continue
        matches = sorted(repo_root.glob(part))
        if matches:
            expanded.extend(str(path) for path in matches)
        else:
            expanded.append(part)
    return tuple(expanded)


def _run_step(step: VerificationStep, repo_root: Path) -> None:
    LOGGER.debug("Running verification step '%s' in %s.", step.label, repo_root)
    command = _expand_command_globs(step.command, repo_root) if step.expand_globs else step.command
    print(f"[verify] {step.label}")
    print(f"[verify] $ {_format_command(command)}")
    subprocess.run(command, cwd=repo_root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the repository verification steps used before merging.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("all", "quality", "tests", "perf", "security", "smoke", "package"),
        default="all",
        help=(
            "Verification slice to run. Defaults to quality, tests, smoke, and package; "
            "security and perf are explicit modes."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    LOGGER.debug("Starting verification mode='%s' in repo_root=%s.", args.mode, repo_root)

    try:
        for step in _ordered_steps(args.mode):
            _run_step(step, repo_root)
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
