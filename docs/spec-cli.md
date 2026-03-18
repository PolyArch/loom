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

FCC emits artifacts under the requested output directory using three naming
families:

- software-only: `<dfg>.*`
- hardware-only: `<adg>.*`
- mixed software-plus-hardware: `<dfg>.<adg>.*`

Representative outputs include:

- software-only:
  - `<dfg>.ll`
  - `<dfg>.llvm.mlir`
  - `<dfg>.cf.mlir`
  - `<dfg>.scf.mlir`
  - `<dfg>.dfg.mlir`
- hardware-only:
  - `<adg>.fabric.mlir`
  - `<adg>.fabric.viz.json`
- mixed:
  - `<dfg>.<adg>.map.json`
  - `<dfg>.<adg>.map.txt`
  - `<dfg>.<adg>.config.json`
  - `<dfg>.<adg>.config.bin`
  - `<dfg>.<adg>.config.h`
  - `<dfg>.<adg>.viz.html`

The full target artifact family may also include host-side files, simulation
traces, and statistics.

## Naming Rules

- If source files are provided, `<dfg>` is derived from the first source file.
- If only `--dfg` is provided, `<dfg>` is derived from the DFG file stem.
- If `--adg` is provided, `<adg>` is derived from the ADG file stem after
  dropping the trailing `.fabric` stem component when present.
- Mixed artifacts must use both names when both software and hardware are part
  of the artifact's meaning.
- Visualization-only mode follows the same rule:
  - with both DFG and ADG: `<dfg>.<adg>.viz.html`
  - with only DFG: `<dfg>.viz.html`
  - with only ADG: `<adg>.viz.html`

## CLI Responsibilities

The CLI is responsible for:

- loading MLIR inputs
- selecting the correct pipeline mode
- creating the output directory
- invoking verification where required
- deriving the software, hardware, and mixed artifact stems

The CLI is not the authority for mapping semantics, route legality, or Fabric
configuration encoding. Those are defined by the corresponding subsystem specs.
