# FRASTA-toolbox Documentation

This directory contains the main user-facing and developer-facing guides.

## Start here

- [Quick Start Guide](QUICK_START_GUI.md) - first launch, demo datasets, and the main GUI workflow
- [Methods Overview](METHODS.md) - computational assumptions, data model, and reproducibility notes
- [Quick Reference](QUICK_REFERENCE.md) - concise cheat sheet for advanced-processing functions

## User guides

- [Advanced Processing Guide](ADVANCED_PROCESSING.md) - when to use filtering, leveling, transforms, and registration helpers
- [GUI Integration Guide](GUI_INTEGRATION.md) - where the advanced-processing tools appear in the GUI and how they affect scans
- `crack_path_analysis_pl.tex` - szczegolowy polskojezyczny dokument LaTeX opisujacy aktualny modul crack-path analysis

## Related material outside this folder

- [Repository README](../README.md) - installation, requirements, troubleshooting, and project overview
- [Examples](../examples/README.md) - runnable scripts and demo material
- [Architecture Guide](../ARCHITECTURE.md) - internal structure and contributor guidance

## Suggested reading paths

For first-time GUI users:

1. [Quick Start Guide](QUICK_START_GUI.md)
2. [Repository README](../README.md)
3. [Methods Overview](METHODS.md)

For users who want more control over preprocessing:

1. [Quick Reference](QUICK_REFERENCE.md)
2. [Advanced Processing Guide](ADVANCED_PROCESSING.md)
3. [GUI Integration Guide](GUI_INTEGRATION.md)

For contributors:

1. [Repository README](../README.md)
2. [Architecture Guide](../ARCHITECTURE.md)
3. Inspect the neighboring modules in `frasta/` to follow existing implementation patterns.
4. For device-native imports, start from `frasta/io/parsers/` or the shared `surface_io/parsers/` package and keep the parser independent from GUI code.
