# Documentation Refresh Design

## Summary

Refresh the project documentation so that a new user can understand what the package does, how to
install it, how to use each supported backend, what configuration knobs exist, and how the
internal rendering pipeline is organized.

The documentation should stay compact enough to scan quickly, but deep enough to answer the
questions a user or contributor is likely to have without reading the source first.

## Goals

- Make `README.md` a clear landing page for first contact with the project.
- Add a deeper guide under `docs/` that explains installation, usage, examples, limitations, and
  internal architecture.
- Keep `examples/README.md` focused on running example scripts rather than duplicating the entire
  product description.

## Chosen Structure

### README

Use `README.md` as the short, entry-level document:

- project overview and supported engines
- installation paths
- quick-start examples
- public API overview
- links to the extended guide and examples catalog

### Extended Guide

Add a single guide file under `docs/`:

- installation and extras
- common usage patterns
- backend-specific accepted inputs
- `PlotConfig` behavior and customization
- `einsum` tracing workflow
- example scripts overview
- important limitations and troubleshooting
- internal architecture overview

### Examples Catalog

Keep `examples/README.md` as a practical catalog:

- how to run each script
- available CLI options that matter to users
- what each example demonstrates
- link back to the main guide for conceptual material

## Content Boundaries

- Do not invent unsupported features or APIs.
- Do not document internal details that are irrelevant to users.
- Do include the important shared internals so contributors can orient themselves quickly:
  registry, graph normalization, layout, and rendering.
- Keep the `tn_tsp.py` example documented as an extra example, not as part of the main public API.

## Review Notes

Subagent-based spec review is not available in this harness, so this spec will be self-reviewed
against the current repository state before implementation.
