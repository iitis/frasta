# Agent Notes

This file collects maintenance instructions for automated coding agents working
on FRASTA-toolbox. User-facing documentation remains in `README.md`, `docs/`,
and `examples/README.md`.

## Start Here

Before editing code, read:

1. `ARCHITECTURE.md` for the module layout and dependency rules.
2. `agent-docs/conventions/README.md` for coding conventions.
3. The module-specific convention file for the area being changed.

## Documentation Rules

- Keep user-facing documentation free of agent-specific notes.
- Use UTF-8 without BOM and Windows CRLF line endings.
- Avoid decorative Unicode characters in Markdown. Prefer plain text labels such
  as `OK`, `BAD`, `NOTE`, and `WARNING`.
- When behavior, public APIs, file formats, or workflow rules change, update the
  relevant Markdown documentation in the same change.

## Code Rules

- Processing functions belong in `frasta/processing/` and should be pure
  functions over NumPy arrays.
- GUI code may call processing and I/O code, but processing code must not import
  GUI modules.
- Keep function and method docstrings suitable for automatic documentation
  generation.