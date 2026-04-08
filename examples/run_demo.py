from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import einsum_demo
import quimb_demo
import tenpy_demo
import tensorkrowch_demo
import tensornetwork_demo
from demo_cli import (
    AUTO_SAVE_SENTINEL,
    ExampleCliArgs,
    ExampleDefinition,
    auto_save_path,
    build_run_demo_parser,
    format_joined_names,
    namespace_to_cli_args,
    resolve_example_definition,
)
from demo_cli import (
    available_examples as available_example_names,
)

EngineModule: dict[str, Any] = {
    "einsum": einsum_demo,
    "quimb": quimb_demo,
    "tenpy": tenpy_demo,
    "tensorkrowch": tensorkrowch_demo,
    "tensornetwork": tensornetwork_demo,
}
_SMALL_CONTRACTED_TENSORKROWCH_EXAMPLES = frozenset({"mps", "mpo"})
_MAX_SMALL_CONTRACTED_N_SITES = 6


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(EngineModule))


def _definitions_for_engine(engine: str) -> tuple[ExampleDefinition, ...] | None:
    module = EngineModule.get(engine)
    if module is None:
        return None
    return module.EXAMPLES


def available_examples(engine: str) -> tuple[str, ...]:
    definitions = _definitions_for_engine(engine)
    if definitions is None:
        return ()
    return available_example_names(definitions)


def resolve_requested_example(*, engine: str, example: str) -> str:
    definitions = _definitions_for_engine(engine)
    if definitions is None:
        raise ValueError(f"Unknown engine: {engine}")
    definition = resolve_example_definition(definitions, example)
    if definition is None:
        raise ValueError(f"Unknown example: {example}")
    return definition.name


def _validate_and_normalize_args(
    parser: Any,
    args: ExampleCliArgs,
) -> tuple[ExampleCliArgs, ExampleDefinition]:
    engine = args.engine.lower()
    definitions = _definitions_for_engine(engine)
    if definitions is None:
        available = format_joined_names(available_engines())
        parser.error(f"Unknown engine {args.engine!r}. Available engines: {available}.")
    definition = resolve_example_definition(definitions, args.example)
    if definition is None:
        parser.error(
            f"Unknown example {args.example!r} for engine {engine!r}. "
            f"Available examples: {format_joined_names(available_example_names(definitions))}."
        )
    requested_contracted = args.contracted
    resolved_contracted = bool(requested_contracted)
    if requested_contracted is None:
        resolved_contracted = bool(
            engine == "tensorkrowch" and definition.name in _SMALL_CONTRACTED_TENSORKROWCH_EXAMPLES
        )
    normalized = replace(
        args,
        engine=engine,
        example=definition.name,
        contracted=resolved_contracted,
    )
    if normalized.from_list and not definition.supports_list:
        parser.error(
            f"Example {definition.name!r} for engine {engine!r} does not support --from-list."
        )
    if normalized.from_scratch and not definition.supports_from_scratch:
        parser.error(
            f"Example {definition.name!r} for engine {engine!r} does not support --from-scratch."
        )
    if normalized.contracted:
        if engine != "tensorkrowch":
            parser.error("The launcher only supports --contracted for engine 'tensorkrowch'.")
        if normalized.from_list:
            parser.error(
                "--contracted requires the native TensorKrowch network object; remove --from-list."
            )
        if definition.name not in _SMALL_CONTRACTED_TENSORKROWCH_EXAMPLES:
            parser.error(
                "--contracted is limited to small TensorKrowch demos (currently: mps, mpo)."
            )
        if normalized.n_sites > _MAX_SMALL_CONTRACTED_N_SITES:
            parser.error(
                "--contracted is limited to small TensorKrowch demos; use --n-sites 6 or less."
            )
    if normalized.save == AUTO_SAVE_SENTINEL:
        normalized = replace(
            normalized,
            save=auto_save_path(engine=engine, example=definition.name),
        )
    for name in ("n_sites", "lx", "ly", "lz", "mera_log2", "tree_depth"):
        value = getattr(normalized, name)
        if value < 1:
            parser.error(f"{name.replace('_', '-')} must be >= 1.")
    return normalized, definition


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> ExampleCliArgs:
    parser = build_run_demo_parser()
    namespace = parser.parse_args(argv)
    parsed = namespace_to_cli_args(namespace)
    normalized, _definition = _validate_and_normalize_args(parser, parsed)
    return normalized


def dispatch_run(args: ExampleCliArgs) -> tuple[Any, Path | None]:
    module = EngineModule[args.engine]
    return module.run_example(args)


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    args = parse_args(argv)
    dispatch_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
