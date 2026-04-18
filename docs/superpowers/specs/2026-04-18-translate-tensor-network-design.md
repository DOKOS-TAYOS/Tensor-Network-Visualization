# Design: `translate_tensor_network(...)`

## Summary

Add a new public function, `translate_tensor_network(...)`, that accepts the same source inputs as
`show_tensor_network(...)` and generates readable Python code for another supported tensor-network
engine.

This feature is aimed at practical export rather than exact object-to-object reconstruction. The
main user experience is:

1. pass an existing tensor network from a supported engine,
2. choose a target engine,
3. receive Python code as a string,
4. optionally save that code to a `.py` file,
5. run the generated code and visualize the translated network with `show_tensor_network(...)`.

The first version supports every current engine as a source and supports every engine except
`tenpy` as a target. `tensorkrowch` remains a restricted target when the translated structure would
require an outer product across disconnected components.

## Goals

- Add a user-facing export API that feels as natural as `show_tensor_network(...)`.
- Reuse the current engine detection and normalization flow as much as possible.
- Preserve real tensor values when they can be extracted safely.
- Fall back to valid placeholder tensors when values are unavailable.
- Generate code that is readable, deterministic enough for testing, and directly usable from Python.
- Support `einsum` as a target both when a contraction order exists and when only connectivity is
  known.

## Non-Goals

- Reproduce every backend-specific internal object detail.
- Support `tenpy` as a translation target in the first version.
- Guarantee byte-for-byte or pixel-perfect equivalence between original and translated renders.
- Introduce a large general-purpose serialization layer beyond what this feature needs.

## Public API

```python
from __future__ import annotations

from os import PathLike
from typing import Any, Literal, TypeAlias

from tensor_network_viz import EngineName

TranslationTargetName: TypeAlias = Literal[
    "tensorkrowch",
    "tensornetwork",
    "quimb",
    "einsum",
]


def translate_tensor_network(
    network: Any,
    *,
    engine: EngineName | None = None,
    target_engine: TranslationTargetName,
    path: str | PathLike[str] | None = None,
) -> str: ...
```

### API behavior

- `network` accepts the same public source inputs as `show_tensor_network(...)`.
- `engine` is optional and follows the same meaning as in `show_tensor_network(...)`.
- `target_engine` is required.
- the function always returns the generated Python code as `str`,
- when `path` is provided, the function also writes the same code to disk,
- `target_engine="tenpy"` is rejected with a clear error because TeNPy is not a target in v1.

This API intentionally stays small. It does not return a custom result object in v1.

## Generated Code Contract

Every generated translation file should expose the same simple contract:

```python
from typing import Any


def build_tensor_network() -> Any:
    ...


network = build_tensor_network()
```

This keeps the generated output easy to understand and easy to test:

- users can import `build_tensor_network()` when they want explicit control,
- users can use `network` directly for quick scripts,
- tests can `exec(...)` the generated source and then call `show_tensor_network(...)` on the result.

For `einsum` targets, `network` should be the final trace object that `show_tensor_network(...)`
already knows how to visualize, typically an `EinsumTrace` or a manual trace list depending on the
generated form.

## Translation Strategy

The implementation should follow the current public pipeline instead of inventing a separate input
system.

### Step 1: resolve source engine and input

- prepare the input with the same helpers used by `show_tensor_network(...)`,
- auto-detect the source engine when `engine is None`,
- validate unsupported grid and malformed input cases through the existing public rules.

### Step 2: recover normalized structure

Use `normalize_tensor_network(...)` as the canonical backend-independent graph for:

- tensor names,
- axis names,
- node ids,
- connectivity,
- open edges,
- contraction-step metadata when available.

This normalized graph should be the main structural source of truth for translation.

### Step 3: recover tensor values when available

Use the tensor extraction path already present in `_extract_tensor_records(...)` to recover real
arrays when the source engine exposes them. This gives v1 a realistic path to preserve values for:

- `tensornetwork`,
- `tensorkrowch` with materialized node tensors,
- `quimb`,
- `tenpy`,
- `einsum` traces with live bound arrays.

If real arrays are not available, the export should still succeed by generating placeholders with
the right shapes whenever shape information is available. Placeholder generation should be explicit
in code comments so the user can see what happened.

## Internal Export Model

The feature should introduce a small internal translation model instead of driving code generation
directly from the public snapshot objects. That model should stay private and be shaped for codegen.

Suggested contents:

- translated tensor records with stable variable names,
- axis names,
- shapes,
- optional concrete arrays,
- edge connectivity grouped by shared index labels,
- open indices,
- disconnected components,
- optional contraction steps reconstructed from the normalized graph.

This model should be deterministic enough to keep generated code stable across runs for the same
input.

## Target-Specific Rules

### `tensornetwork`

Generate readable code that:

- imports `numpy` and `tensornetwork`,
- creates one `tn.Node(...)` per tensor,
- supplies tensor values when available, otherwise placeholders,
- preserves `name` and `axis_names`,
- connects shared axes with edge connections,
- returns a list of nodes or another directly accepted public input shape.

### `quimb`

Generate readable code that:

- imports `numpy` and `quimb.tensor as qtn`,
- creates one `qtn.Tensor(...)` per tensor,
- maps shared edges to shared `inds`,
- uses tags derived from tensor names,
- returns a `qtn.TensorNetwork(...)`.

### `tensorkrowch`

Generate readable code that:

- imports `torch` and `tensorkrowch`,
- creates a `tk.TensorNetwork(...)`,
- creates one `tk.Node(...)` per tensor,
- materializes tensor values when available and otherwise uses valid placeholders,
- connects matching axes with the standard edge connection syntax.

Restriction:

- if the translated graph contains disconnected components and the target result would require an
  outer product to represent the whole structure as one TensorKrowch network expression, the export
  should fail with a clear error.

This is stricter than the `einsum` target by design.

### `einsum`

This target has two modes.

#### Ordered mode

If the source graph exposes contraction steps with enough information to reconstruct an ordered
trace, generate a sequential traced form using:

- `EinsumTrace`,
- bound operands,
- `einsum(...)`,
- `pair_tensor(...)` and/or `einsum_trace_step(...)` when a manual step list is the cleaner
  representation.

This mode should preserve playback order when that information already exists.

#### Connectivity-only mode

If no contraction order is available, generate a single `einsum` expression that reflects all
connections at once:

- each tensor receives labels derived from normalized connectivity,
- shared labels represent shared indices,
- labels that appear only once become output labels,
- disconnected components naturally become an implicit outer product inside the global expression.

This behavior is important because it still gives the user a valid and visualizable `einsum`
translation even when stepwise contraction history is missing.

## Error Handling

The API should fail loudly and clearly for the cases that would otherwise produce misleading code.

Important errors:

- unsupported target engine,
- unsupported source input under the existing public rules,
- duplicate or unusable names when no stable readable code can be produced safely,
- `tensorkrowch` target with disconnected components requiring an outer product,
- malformed or incomplete contraction metadata that cannot support the chosen export path.

Where possible, the error text should tell the user what alternative is likely to work, for example
using `quimb` or `einsum` instead of `tensorkrowch`.

## Demo

Add a focused example script:

- `examples/translate_demo.py`

The demo should:

1. build a small deterministic source network,
2. translate it to a chosen target engine,
3. optionally save the generated code,
4. execute the generated code in memory,
5. render the original and translated networks for visual comparison.

Suggested CLI options:

- `--source-engine`,
- `--target-engine`,
- `--save-code`,
- `--no-show`.

This demo should stay intentionally small and deterministic so it is useful both for humans and
tests.

## Tests

Add a dedicated test module:

- `tests/test_translation_api.py`

### API tests

- returns `str`,
- writes the same string to `path` when requested,
- auto-detects source engines correctly,
- rejects `tenpy` as a target,
- rejects invalid target names clearly.

### Structural round-trip tests

For representative small examples:

- translate from each supported source family into each supported target family where valid,
- execute the generated Python,
- recover the translated `network`,
- compare `normalize_tensor_network(...)` output between original and translated forms.

The structural comparison should focus on:

- tensor names,
- axis names,
- connectivity,
- open indices,
- number of connected components.

### Value preservation tests

- when the source exposes materialized tensors, confirm that translated tensors preserve numeric
  values in reasonable representative examples,
- when values are unavailable, confirm that placeholder-based output still executes and renders.

### Render compatibility tests

The translated result should be rendered with `show_tensor_network(..., show=False)` and checked
against the original in a robust way:

- both renders succeed,
- both produce compatible normalized structure,
- optional layout comparisons should use fixed seeds or shared manual positions where needed,
- tests should avoid pixel-perfect image matching because geometry and drawing are not fully
  deterministic.

For the visual equivalence check, compare what is stable and meaningful rather than exact pixels.

## Documentation Updates

Keep docs additions small and practical:

- add the new function to `docs/api.md`,
- add one usage example to `docs/backends.md` or `docs/guide.md`,
- mention the demo in `examples/README.md`,
- update top-level exports documentation if `__all__` changes.

No extra standalone markdown documents are needed beyond this design spec.

## Recommended Implementation Order

1. add the public API and export surface,
2. add the private translation model,
3. implement `tensornetwork` and `quimb` generators first,
4. implement `einsum` ordered and connectivity-only generation,
5. implement `tensorkrowch` generation plus outer-product guardrails,
6. add demo and docs,
7. add round-trip and rendering tests.

This order reduces risk because it starts with the simplest structural targets and leaves the most
constrained target for later.
