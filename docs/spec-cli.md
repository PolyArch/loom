# FCC CLI Specification

## Overview

FCC supports three main invocation modes:

1. Full compilation flow from source files
2. DFG-direct mapping flow using `--dfg` plus `--adg`
3. Visualization-only flow using `--viz-only`

## Invocation Modes

### Full Flow

```
fcc <sources> --adg <fabric.mlir> -o <output_dir> [options]
```

This mode lowers source input, builds a DFG, maps against the ADG, emits
reports, and generates visualization.

### DFG-Direct Mapping

```
fcc --dfg <dfg.mlir> --adg <fabric.mlir> -o <output_dir> [options]
```

This mode skips frontend lowering and maps a pre-built DFG onto a pre-built ADG.

### Visualization Only

```
fcc --viz-only [--dfg <dfg.mlir>] [--adg <fabric.mlir>] -o <output_dir>
```

This mode emits HTML visualization without running mapping.

## Core Options

| Option | Meaning |
|--------|---------|
| `-o <dir>` | Output directory |
| `-I <path>` | Frontend include path |
| `--adg <path>` | Fabric MLIR ADG input |
| `--dfg <path>` | Pre-built DFG input |
| `--viz-only` | Visualization-only mode |
| `--simulate` | Run standalone simulation after mapping |
| `--sim-max-cycles <n>` | Simulation cycle budget |
| `--mapper-budget <s>` | Mapper time budget in seconds |
| `--mapper-seed <n>` | Deterministic mapper seed |

## Output Artifact Family

Let `<base>` be the derived stem for the current invocation. FCC emits
artifacts under the requested output directory.

Current output families include:

- `<base>.map.json`
- `<base>.map.txt`
- `<base>.viz.html`

The full target artifact family may also include intermediate lowering results,
configuration binaries, host-side files, simulation traces, and statistics.

## Naming Rules

- If source files are provided, `<base>` is derived from the first source file.
- If only `--dfg` is provided, `<base>` is derived from the DFG file stem.
- If only `--adg` is provided in visualization-only mode, `<base>` is derived
  from the ADG file stem.

## CLI Responsibilities

The CLI is responsible for:

- loading MLIR inputs
- selecting the correct pipeline mode
- creating the output directory
- invoking verification where required
- deriving the base artifact name

The CLI is not the authority for mapping semantics, route legality, or Fabric
configuration encoding. Those are defined by the corresponding subsystem specs.
