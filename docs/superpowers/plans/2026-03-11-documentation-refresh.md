# Documentation Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the project docs so users and contributors can install, use, and understand the package from the current repository state.

**Architecture:** Keep `README.md` concise as the landing page, add one extended guide in `docs/`, and refocus `examples/README.md` into an examples catalog. Reuse existing repo facts and public APIs instead of introducing new terminology or unsupported behavior.

**Tech Stack:** Markdown, Python package metadata from `pyproject.toml`, existing examples and source code

---

## Chunk 1: Documentation Structure

### Task 1: Create the documentation scaffolding

**Files:**
- Create: `docs/guide.md`
- Create: `docs/superpowers/specs/2026-03-11-documentation-refresh-design.md`
- Create: `docs/superpowers/plans/2026-03-11-documentation-refresh.md`

- [ ] Create the `docs/` and `docs/superpowers/` folders if missing.
- [ ] Add the design spec file with the approved documentation structure and content boundaries.
- [ ] Add this implementation plan file.
- [ ] Review the created files for path accuracy and consistency with the repo layout.

### Task 2: Define the guide scope from the current codebase

**Files:**
- Modify: `docs/guide.md`
- Reference: `README.md`
- Reference: `examples/README.md`
- Reference: `src/tensor_network_viz/__init__.py`
- Reference: `src/tensor_network_viz/config.py`
- Reference: `src/tensor_network_viz/viewer.py`

- [ ] Extract the currently supported public API and engines from source and README.
- [ ] Write the guide outline so it covers installation, usage, backend inputs, configuration, examples, troubleshooting, and architecture.
- [ ] Ensure every documented capability maps to current code or examples.

## Chunk 2: User-Facing Documentation

### Task 3: Rewrite the landing README

**Files:**
- Modify: `README.md`
- Reference: `pyproject.toml`
- Reference: `examples/README.md`
- Reference: `docs/guide.md`

- [ ] Replace the current long-form README with a tighter landing-page structure.
- [ ] Keep essential installation commands, quick-start examples, and public API overview.
- [ ] Add links to `docs/guide.md` and `examples/README.md`.
- [ ] Keep wording aligned with supported extras and editable-install wrappers.

### Task 4: Write the extended guide

**Files:**
- Modify: `docs/guide.md`
- Reference: `examples/*.py`
- Reference: `src/tensor_network_viz/_registry.py`
- Reference: `src/tensor_network_viz/_core/*`
- Reference: `src/tensor_network_viz/*/graph.py`
- Reference: `src/tensor_network_viz/einsum_module/trace.py`

- [ ] Document installation options and backend extras.
- [ ] Document the generic dispatcher flow with `show_tensor_network(...)`.
- [ ] Document backend-specific accepted inputs and important behavior differences.
- [ ] Document `PlotConfig` and practical customization patterns.
- [ ] Document `einsum` auto and manual tracing workflows.
- [ ] Add a compact architecture section covering normalization, layout, drawing, and lazy engine dispatch.

### Task 5: Refocus the examples catalog

**Files:**
- Modify: `examples/README.md`
- Reference: `examples/tensorkrowch_demo.py`
- Reference: `examples/tensornetwork_demo.py`
- Reference: `examples/quimb_demo.py`
- Reference: `examples/tenpy_demo.py`
- Reference: `examples/einsum_demo.py`
- Reference: `examples/tn_tsp.py`

- [ ] Keep the examples README practical and CLI-oriented.
- [ ] For each script, document what it demonstrates and the most relevant options.
- [ ] Remove duplicated architecture explanation that belongs in the main guide.
- [ ] Add a pointer back to the extended guide for full conceptual documentation.

## Chunk 3: Verification

### Task 6: Review and verify the documentation update

**Files:**
- Modify: `README.md`
- Modify: `docs/guide.md`
- Modify: `examples/README.md`

- [ ] Proofread all new Markdown for consistency, broken headings, and duplicated claims.
- [ ] Run `pytest` from the worktree to verify the repository baseline still holds.
- [ ] Run `git diff --stat` and review the final scope for accidental unrelated edits.
- [ ] Summarize the updated documentation structure and any remaining caveats.
