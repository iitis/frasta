# FRASTA-toolbox Coding Conventions

This directory contains detailed coding standards, patterns, and best practices for FRASTA-toolbox development organized by module structure.

**Purpose:** Ensure consistency, maintainability, and effective collaboration (human-to-human and human-to-AI).

---

## 📂 Organization

Conventions are organized to mirror the project structure:

```
conventions/
├── processing/     # Algorithm implementation standards
├── io/            # File I/O conventions and format specs
├── gui/           # GUI development patterns and widgets
├── core/          # Data structure conventions
└── general/       # Cross-cutting concerns (naming, imports, logging)
```

---

## 📋 Quick Index

### For Algorithm Development
- **[Processing Algorithms](processing/algorithms.md)** ⭐ ESSENTIAL
  - Function signatures, NaN handling, masking, units, testing
  - Templates and complete examples
  - Start here when adding new filters/transforms/analysis functions

### For I/O Development
- **[File Format Specifications](io/file_formats.md)**
  - NPZ, HDF5, CSV, STL format details
  - Unit conventions and metadata standards
- **[Loader/Exporter Patterns](io/loaders.md)**
  - How to implement new format support
  - Error handling and validation

### For GUI Development
- **[GUI Development Guide](gui/development.md)**
  - Dialog patterns, workers, signal/slot best practices
  - How to connect menu items to processing functions
- **[Widget Development](gui/widgets.md)**
  - Custom widget patterns and integration

### For Core Data Structures
- **[Data Structures](core/data_structures.md)**
  - Surface usage and extension
  - When to create new data structures

### General Standards
- **[Naming Conventions](general/naming.md)**
  - Variables, functions, classes, modules
- **[Import Organization](general/imports.md)**
  - Import order, lazy imports, dependency management
- **[Logging Standards](general/logging.md)**
  - When to use DEBUG/INFO/WARNING/ERROR levels

---

## 🎯 When to Use Which Document

| **You Want To...** | **Read This** |
|-------------------|---------------|
| Add a new filter (bilateral, median, etc.) | [processing/algorithms.md](processing/algorithms.md) |
| Add a new transformation (rotate, scale) | [processing/algorithms.md](processing/algorithms.md) |
| Implement morphology operation (leveling) | [processing/algorithms.md](processing/algorithms.md) |
| Support a new file format | [io/file_formats.md](io/file_formats.md) + [io/loaders.md](io/loaders.md) |
| Create a parameter dialog | [gui/development.md](gui/development.md) |
| Add a custom widget | [gui/widgets.md](gui/widgets.md) |
| Extend Surface or create new structure | [core/data_structures.md](core/data_structures.md) |
| Understand project-wide naming | [general/naming.md](general/naming.md) |
| Set up logging in new module | [general/logging.md](general/logging.md) |

---

## 🚀 Getting Started

**New to the project?** Read in this order:
1. [ARCHITECTURE.md](../../ARCHITECTURE.md) - Understand overall structure
2. [processing/algorithms.md](processing/algorithms.md) - Core algorithm patterns (most code lives here)
3. [general/naming.md](general/naming.md) - Naming standards
4. Module-specific conventions as needed

**Adding a feature?** 
1. Check [ARCHITECTURE.md](../../ARCHITECTURE.md) to see which module it belongs to
2. Read the relevant convention document for that module
3. Copy templates, follow patterns, verify checklist

**Working with AI assistants?**
- Share relevant convention files to ensure consistent code generation
- AI assistants should consult these before making changes
- Conventions contain templates specifically designed for AI code generation

---

## 📝 Contributing to Conventions

These documents are living references. When you:
- Discover a new pattern that works well
- Find an edge case not covered
- Identify a common mistake
- Create a reusable template

**Update the relevant convention file!** Keep examples concrete and actionable.

---

## 🔗 Related Documentation

- **[ARCHITECTURE.md](../../ARCHITECTURE.md)** - System architecture and design principles
- **[ADVANCED_PROCESSING.md](../ADVANCED_PROCESSING.md)** - API reference for processing functions
- **[GUI_INTEGRATION.md](../GUI_INTEGRATION.md)** - Using processing functions in GUI
- **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Function cheat sheet
- **[README.md](../../README.md)** - Project overview and quick start

---

**Last Updated:** 2026-02-18

**Maintained By:** Project contributors (human and AI)
