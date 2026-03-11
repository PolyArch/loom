# Loom Command Line Interface (CLI) Specification

## Overview

The `loom` binary provides a clang-compatible front-end wrapper that compiles
C/C++ sources to LLVM IR and then emits MLIR for the Loom pipeline. The CLI is
similar to `clang++`, but it is not a full drop-in replacement.

Key differences:

- Linker flags are ignored.
- The tool always emits LLVM IR and MLIR outputs.
- Additional MLIR outputs (`.mlir`, `.scf.mlir`, `.handshake.mlir`) are always
  generated based on the chosen `-o` path.

## Invocation

```
loom [options] <sources...> -o <output.llvm.ll>
loom --adg <file.fabric.mlir>
loom --adg <fabric.mlir> [sources...|-dfgs <f1[,f2,...]>] -o <output> [mapper-options]
loom --gen-adg --dfgs <f1[,f2,...]> -o <output.fabric.mlir> [gen-options]
loom --dfg-analyze --dfgs <f1[,f2,...]> -o <output.mlir> [analysis-options]
loom --as-clang [clang-options...] [sources...]
```

`-o` is required in all modes except ADG validation-only (`--adg` without
sources/DFGs) and `--help`/`--version`.

## Outputs

The tool produces four output files:

- LLVM IR (text): the exact `-o` output path.
- LLVM dialect MLIR: derived from `-o`.
- SCF MLIR: derived from `-o`.
- Handshake MLIR: derived from `-o`.

Output path derivation:

- If `-o` ends with `.llvm.ll`, then:
  - MLIR output: replace `.ll` with `.mlir` (e.g., `foo.llvm.ll` -> `foo.llvm.mlir`)
  - SCF output: replace `.llvm.ll` with `.scf.mlir`
  - Handshake output: replace `.llvm.ll` with `.handshake.mlir`
- If `-o` ends with `.ll`, then:
  - MLIR output: replace `.ll` with `.mlir`
  - SCF output: replace `.ll` with `.scf.mlir`
  - Handshake output: replace `.ll` with `.handshake.mlir`
- Otherwise:
  - MLIR output: append `.mlir`
  - SCF output: append `.scf.mlir`
  - Handshake output: append `.handshake.mlir`

## Supported Options

The CLI recognizes and handles these options directly:

- `-h`, `--help`: print usage and exit.
- `--version`: print the tool version and exit.
- `-o <path>` or `-o<path>`: select output path.
- `--adg <file.fabric.mlir>`: validate a fabric MLIR file or specify ADG for mapping (see below).
- `--as-clang`: operate as a standard C++ compiler (see below).
- `--dfgs <f1.handshake.mlir[,f2,...]>`: use pre-compiled Handshake MLIR files (see below).
- `--gen-adg`: enable ADG generation mode from DFG analysis (see below).
- `--dfg-analyze`: enable standalone DFG analysis pass (see below).

`--` terminates option parsing. All subsequent arguments are treated as input
files, even if they begin with `-`.

## Loom-Specific Options

### `--adg`

`--adg <path>` has two modes depending on whether sources or
`--dfgs` are provided:

**Mode 1: Validation only** (`--adg` without sources and without
`--dfgs`):

- Parses the given fabric MLIR file.
- Registers Fabric, Dataflow, Handshake, Arith, Math, MemRef, and Func
  dialects.
- Runs MLIR parse and semantic verification.
- Exits 0 if the file is valid, 1 if any errors are detected.
- Error codes are emitted to stderr with `[CPL_XXX]` prefix
  (see [spec-fabric-error.md](./spec-fabric-error.md)).
- No MLIR/SCF/Handshake outputs are generated.
- Source files and `-o` are ignored in this mode.

**Mode 2: Mapper invocation** (`--adg` with sources + `-o`, or
`--adg` with `--dfgs` + `-o`):

- The fabric MLIR is loaded as the ADG (hardware graph).
- The DFG is extracted from the Handshake MLIR (compiled from sources
  via Stage A, or directly from `--dfgs`).
- The mapper pipeline runs (place, route, temporal, validate).
- Configuration and mapping artifacts are emitted.
- See **Stage C: Mapper Invocation** below for full details.

**Incompatibilities:**

`--adg` is incompatible with `--as-clang`. If both are specified, a
usage error is reported.

**Examples:**

```bash
# Mode 1: Validate an ADG fabric MLIR file
loom --adg my_cgra.fabric.mlir
echo $?  # 0 = valid, 1 = errors

# Mode 2: Compile + map
loom --adg my_cgra.fabric.mlir app.cpp -o app.config.bin

# Mode 2: Map from pre-compiled handshake (single or multiple DFGs)
loom --adg my_cgra.fabric.mlir \
     --dfgs app.handshake.mlir -o app.config.bin
loom --adg my_cgra.fabric.mlir \
     --dfgs a.handshake.mlir,b.handshake.mlir -o app.config.bin
```

### `--as-clang`

When `--as-clang` is specified, `loom` operates as a standard C++ compiler
without performing Loom-specific analysis or MLIR transformations.

**Behavior:**

- The tool behaves like `clang++` with the same defaults
- No MLIR outputs are generated
- Linking is enabled (linker flags are processed, not ignored)
- The Loom ADG library (`libloom-adg`) is automatically linked
- Include paths for Loom headers (`<loom/adg.h>` and `<loom/loom.h>`) are
  automatically added

**Use case:**

This mode is intended for compiling ADG construction programs. An ADG program
is a standard C++ executable that uses the ADGBuilder API to construct hardware
descriptions and export them to MLIR, DOT, or SystemVerilog.

**Example:**

```bash
# Compile an ADG construction program
loom --as-clang my_cgra.cpp -o my_cgra

# Run the program to generate hardware outputs
./my_cgra
```

**Output with `--as-clang`:**

When `--as-clang` is specified:

- If `-c` is present: produces object file only
- If `-S` is present: produces assembly only
- Otherwise: produces linked executable
- `-o` is still required.

**Differences from standard mode:**

| Aspect | Standard mode | `--as-clang` mode |
|--------|---------------|-------------------|
| MLIR generation | Yes | No |
| Linker flags | Ignored | Processed |
| Output naming | `-o` required | `-o` required |
| ADG library | Not linked | Auto-linked |

### Header Include Path Behavior

- Standard mode: no mode-specific guarantee is made for implicit Loom include
  paths; users should pass explicit include flags (`-I`, `-isystem`) when needed.
- `--as-clang` mode: Loom include paths are added automatically for both
  `<loom/adg.h>` and `<loom/loom.h>`.

**Combining with other options:**

`--as-clang` is incompatible with options that require MLIR processing. If
both `--as-clang` and MLIR-related options are specified, an error is reported.

## Forwarded Compile Options

All non-linker flags (arguments beginning with `-`) are forwarded to the clang
driver. Options that consume a value are recognized by a fixed list and the
next argument is forwarded as part of the option. Examples of commonly used
forwarded options include:

- Include and preprocessor: `-I`, `-isystem`, `-D`, `-U`, `-include`, `-imacros`
- Language mode: `-std`, `-x`
- Targeting: `-target`, `-isysroot`, `-sysroot`, `-resource-dir`
- Debug/optimization: `-g`, `-O0`..`-O3`

Linker options are ignored entirely:

- `-l`, `-L`, `-Wl,...`, `-shared`, `-static`

## Default Arguments

If not provided by the user, the following defaults are added:

- `-x c++` if no `-x` is specified.
- `-c` if neither `-c`, `-S`, nor `-E` is present.
- `-O1` if no optimization level is specified.
- `-emit-llvm`
- `-g`
- `-gno-column-info`
- `-fno-discard-value-names`

## Resource Directory

If `-resource-dir` is not provided, `loom` sets it to:

```
<loom-install>/lib/clang
```

where `<loom-install>` is derived from the directory of the `loom` executable.

## Compilation and Linking Behavior

- Each input is compiled using clang's cc1 pipeline (in-process).
- All compiled inputs are linked into a single LLVM module.
- If target triples or data layouts do not match across inputs, the compile
  fails.
- The linked module is verified before translation to MLIR.

## MLIR Pipelines

After generating LLVM dialect MLIR, the tool runs two pass sequences:

SCF conversion sequence:

- `loom::createConvertLLVMToSCFPass()`
- `canonicalize`
- `cse`
- `mem2reg`
- `canonicalize`
- `cse`
- `lift-control-flow-to-scf`
- `loop-invariant-code-motion`
- `canonicalize`
- `cse`
- `loom::createUpliftWhileToForPass()`
- `canonicalize`
- `cse`
- `loom::createAttachLoopAnnotationsPass()`
- `loom::createMarkWhileStreamablePass()`

Handshake conversion sequence:

- `loom::createSCFToHandshakeDataflowPass()`
- `canonicalize`
- `cse`

The SCF and Handshake MLIR outputs are written after their respective pass
sequences complete successfully.

### Loom-Specific Passes

- `loom::createConvertLLVMToSCFPass()`: Converts LLVM dialect to structured
  control flow (SCF) using pattern-based rewriting.
- `loom::createUpliftWhileToForPass()`: Converts `scf.while` loops that match
  canonical for-loop patterns into `scf.for` operations.
- `loom::createAttachLoopAnnotationsPass()`: Extracts loop pragma metadata from
  marker calls and attaches them as MLIR attributes on loop operations. This
  pass handles annotations from all loop-level pragmas including
  `LOOM_PARALLEL`, `LOOM_UNROLL`, `LOOM_TRIPCOUNT`, `LOOM_REDUCE`, and
  `LOOM_MEMORY_BANK`. See [spec-pragma.md](./spec-pragma.md).
- `loom::createMarkWhileStreamablePass()`: Analyzes `scf.while` loops to
  determine if they can be converted to streaming dataflow and marks them
  accordingly.
- `loom::createSCFToHandshakeDataflowPass()`: Converts SCF operations to
  Handshake dialect for dataflow execution on hardware. This pass consumes
  the `LOOM_REDUCE` annotation to generate reduction trees and the
  `LOOM_MEMORY_BANK` annotation to generate banked memory interfaces.

### Pragma Handling Summary

| Pragma | Handling Pass |
|--------|--------------|
| `LOOM_ACCEL` | Processed during function selection (pre-pipeline) |
| `LOOM_NO_ACCEL` | Processed during function selection (pre-pipeline) |
| `LOOM_TARGET` | Processed during function selection (pre-pipeline) |
| `LOOM_STREAM` | Processed during function selection (pre-pipeline) |
| `LOOM_PARALLEL` | `createAttachLoopAnnotationsPass()` + `createSCFToHandshakeDataflowPass()` |
| `LOOM_NO_PARALLEL` | `createAttachLoopAnnotationsPass()` |
| `LOOM_UNROLL` | `createAttachLoopAnnotationsPass()` + SCF canonicalization |
| `LOOM_NO_UNROLL` | `createAttachLoopAnnotationsPass()` |
| `LOOM_TRIPCOUNT` | `createAttachLoopAnnotationsPass()` |
| `LOOM_REDUCE` | `createAttachLoopAnnotationsPass()` + `createSCFToHandshakeDataflowPass()` |
| `LOOM_MEMORY_BANK` | `createAttachLoopAnnotationsPass()` + `createSCFToHandshakeDataflowPass()` |

### `--dfgs`

When `--dfgs <path1[,path2,...]>` is specified alongside `--adg` and `-o`,
`loom` skips compilation (Stages A-B) and feeds the given Handshake MLIR
files directly to the mapper.

**Syntax:** Comma-separated file paths, no spaces between entries.

**Behavior:**

- Parses each `.handshake.mlir` file and registers all required
  dialects (Handshake, Dataflow, Arith, Math, MemRef, Func).
- Runs MLIR parse and semantic verification on each file.
- Merges all `handshake.func` operations into a single module.
- Proceeds directly to the mapper pipeline (Stage C).
- No LLVM IR, LLVM MLIR, SCF MLIR, or Handshake MLIR outputs are
  generated (the inputs already are Handshake MLIR).

**Use case:**

This mode enables decoupled compilation and mapping workflows, and
allows combining DFGs from multiple source files into a single mapping
session.

**Example:**

```bash
# Single DFG file
loom --adg my_cgra.fabric.mlir \
     --dfgs app.handshake.mlir \
     -o app.config.bin --mapper-budget 30

# Multiple DFG files
loom --adg my_cgra.fabric.mlir \
     --dfgs kernel_a.handshake.mlir,kernel_b.handshake.mlir \
     -o app.config.bin
```

**Incompatibilities:**

`--dfgs` requires `--adg` and `-o`. It is mutually exclusive with
source file arguments. If source files are provided alongside `--dfgs`,
a usage error is reported.

### `--gen-adg`

When `--gen-adg` is specified alongside `--dfgs` and `-o`, `loom` analyzes the
given DFGs and generates a matching ADG (Architecture Description Graph) that
contains the hardware resources required to map those DFGs.

**Behavior:**

- Parses each `.handshake.mlir` file from `--dfgs`.
- Analyzes PE operation requirements, data widths, and memory interfaces.
- Generates a lattice-based ADG with switches, PEs, and memory modules.
- Exports the generated ADG to the `-o` path as fabric MLIR.

**ADG generation options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--gen-topology <mesh\|cube>` | Lattice topology: 2D mesh or 3D cube | `mesh` |
| `--gen-track <n>` | Switch routing track count (more tracks = more routing resources) | 1 |
| `--gen-fifo-mode <none\|single\|dual>` | FIFO insertion mode on inter-switch edges | `none` |
| `--gen-fifo-depth <n>` | FIFO depth when FIFOs are enabled | 2 |
| `--gen-fifo-bypassable` | Make inserted FIFOs bypassable (combinational path available) | off |
| `--gen-temporal` | Generate temporal domain (dual-mesh with tag bridges for time-multiplexed execution) | off |

**Combinable with DFG analysis:**

`--gen-adg` can be combined with `--dfg-analyze` and `--dump-analysis` to
enable analysis-guided generation. When `--dfg-analyze` is active during
generation, the analysis results (loop depth, execution frequency, temporal
score) drive spatial/temporal PE partitioning.

**Examples:**

```bash
# Basic ADG generation from one DFG
loom --gen-adg --dfgs app.handshake.mlir -o app.fabric.mlir

# Multi-DFG generation with extra routing tracks
loom --gen-adg --dfgs a.handshake.mlir,b.handshake.mlir \
     -o combined.fabric.mlir --gen-track 2

# Generation with Cube3D topology and FIFOs
loom --gen-adg --dfgs app.handshake.mlir -o app.fabric.mlir \
     --gen-topology cube --gen-fifo-mode single --gen-fifo-depth 4

# Analysis-guided temporal generation
loom --gen-adg --dfg-analyze --dump-analysis \
     --dfgs app.handshake.mlir -o app.fabric.mlir \
     --gen-temporal --temporal-threshold 0.3
```

**Incompatibilities:**

`--gen-adg` is mutually exclusive with `--adg`. `--gen-adg` requires `--dfgs`
and `-o`. Generation-specific options (`--gen-topology`, `--gen-track`,
`--gen-fifo-*`, `--gen-temporal`) are only meaningful with `--gen-adg`.

### `--dfg-analyze`

When `--dfg-analyze` is specified alongside `--dfgs`, `loom` runs the DFG
analysis pass on the given Handshake MLIR files. The analysis computes
per-operation metrics and annotates the MLIR with `loom.analysis` attributes.

**Behavior:**

- Parses each `.handshake.mlir` file from `--dfgs`.
- Runs two-level analysis:
  - Level A (MLIR-level): loop depth, execution frequency
  - Level B (graph-level): recurrence detection, critical path, temporal score
- Annotates operations with `loom.analysis` attributes containing computed
  metrics.
- Outputs annotated MLIR to the `-o` path.

**Analysis options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dfg-analyze` | Enable DFG analysis pass | off |
| `--temporal-threshold <f>` | Temporal score threshold for classifying ops as temporal candidates (0.0-1.0) | 0.5 |
| `--dump-analysis` | Print analysis summary to stdout (operation counts, temporal scores, partitioning) | off |

**Standalone vs combined usage:**

- **Standalone**: `--dfg-analyze --dfgs ... -o ...` runs analysis only and
  outputs annotated MLIR.
- **With `--gen-adg`**: analysis results guide ADG generation (spatial/temporal
  PE partitioning based on temporal scores).
- **With `--adg` (mapper)**: if the DFG has pre-annotated `loom.analysis`
  attributes, the mapper can use them for placement heuristics.

**Examples:**

```bash
# Standalone analysis with summary output
loom --dfg-analyze --dump-analysis \
     --dfgs app.handshake.mlir -o app.analyzed.mlir

# Analysis with custom temporal threshold
loom --dfg-analyze --temporal-threshold 0.3 \
     --dfgs app.handshake.mlir -o app.analyzed.mlir
```

## Stage C: Mapper Invocation

When `--adg` is specified alongside source files (or `--dfgs`)
and `-o`, the mapper place-and-route pipeline runs after Handshake
conversion (or directly from the provided DFG files).

**Behavior:**

- Parses `<fabric.mlir>` into the ADG (hardware graph).
- Extracts the DFG from the Handshake MLIR produced by Stage A.
- Runs the mapper pipeline (place, route, temporal assign, validate).
- Emits configuration and mapping artifacts.

**Mapper-specific options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--mapper-budget <seconds>` | Search time limit for CP-SAT solver | 60 |
| `--mapper-seed <int>` | Deterministic seed for tie-breaking | 0 (deterministic) |
| `--mapper-profile <name>` | Weight profile: `balanced`, `cpsat_full`, `heuristic_only`, `throughput_first`, `area_power_first`, `deterministic_debug` | `balanced` |
| `--mapper-verbose` | Enable verbose mapper logging | off |

**Mapper outputs** (in addition to Stage A outputs):

| File | Description |
|------|-------------|
| `<name>.config.bin` | Binary config_mem image |
| `<name>_addr.h` | C header with per-node addresses and masks |
| `<name>.map.json` | Machine-readable mapping metadata (always emitted) |
| `<name>.map.txt` | Human-readable mapping report (always emitted) |
| `<name>.fabric.mlir` | Configured fabric MLIR with route tables (always emitted) |

Output path derivation follows the same `-o` base name convention as
Stage A.

**Exit codes:**

- `0`: Mapping succeeded.
- `1`: Mapping failed (no valid mapping under constraints). Diagnostics
  are emitted to stderr per
  [spec-mapper-algorithm.md](./spec-mapper-algorithm.md).

**Incompatibilities:**

`--adg` with source files requires `-o`. `--adg` without source files
operates in validation-only mode (existing behavior).

Forward references:

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)

## Stage D Placeholder: Backend Invocation

Stage D backend realization (SystemC/SystemVerilog generation and config_mem
artifact flow) is specified in ADG/backend documents.

Forward references:

- [spec-adg.md](./spec-adg.md)
- [spec-adg-sysc.md](./spec-adg-sysc.md)
- [spec-adg-sv.md](./spec-adg-sv.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)

## Stage E Placeholder: Co-Simulation Invocation

Co-simulation orchestration (host runtime, ESI transport, backend session
lifecycle, trace/perf collection, and end-to-end compare flow) is specified in
the `spec-cosim` document family.

Forward references:

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)

## Exit Codes

- `0`: Success.
- `1`: Usage error, compilation failure, MLIR conversion failure, verification
  failure, or output/write failure.

## Complete CLI Reference

All user-visible command-line options in one table:

### General Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h`, `--help` | flag | - | Print usage and exit |
| `--version` | flag | - | Print version and exit |
| `-o <path>` | string | (required) | Output file path |
| `--` | separator | - | Treat all following arguments as input files |

### ADG and DFG Input Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--adg <path>` | string | (none) | Load fabric MLIR as ADG (validation or mapping) |
| `--dfgs <f1[,f2,...]>` | string | (none) | Comma-separated pre-compiled Handshake MLIR files |
| `--as-clang` | flag | off | Invoke clang driver mode (compile ADG programs) |

### Mapper Options (used with `--adg` + sources/DFGs)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mapper-budget <seconds>` | double | 60 | CP-SAT solver search time limit |
| `--mapper-seed <int>` | int | 0 | Deterministic seed for reproducible mapping |
| `--mapper-profile <name>` | string | `balanced` | Weight profile (`balanced`, `cpsat_full`, `heuristic_only`, `throughput_first`, `area_power_first`, `deterministic_debug`) |
| `--mapper-verbose` | flag | off | Enable verbose mapper logging |

### ADG Generation Options (used with `--gen-adg`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gen-adg` | flag | off | Enable ADG generation from DFG analysis |
| `--gen-topology <mesh\|cube>` | string | `mesh` | Lattice topology (2D mesh or 3D cube) |
| `--gen-track <n>` | unsigned | 1 | Switch routing track count |
| `--gen-fifo-mode <none\|single\|dual>` | string | `none` | FIFO insertion mode on inter-switch edges |
| `--gen-fifo-depth <n>` | unsigned | 2 | FIFO depth when FIFOs are enabled |
| `--gen-fifo-bypassable` | flag | off | Make inserted FIFOs bypassable |
| `--gen-temporal` | flag | off | Generate temporal domain (dual-mesh with tag bridges) |

### DFG Analysis Options (used with `--dfg-analyze`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dfg-analyze` | flag | off | Enable DFG analysis pass |
| `--temporal-threshold <f>` | double | 0.5 | Temporal score threshold for op classification |
| `--dump-analysis` | flag | off | Print analysis summary to stdout |

### Operating Modes

| Mode | Invocation | Description |
|------|-----------|-------------|
| Compilation | `loom <sources> -o <out>` | Compile C++ to LLVM IR, then to MLIR (LLVM -> SCF -> Handshake) |
| ADG Validation | `loom --adg <fabric.mlir>` | Parse and verify fabric MLIR, exit 0/1 |
| Mapping | `loom --adg <fabric.mlir> --dfgs <dfgs> -o <out>` | Place-and-route DFGs onto ADG |
| ADG Generation | `loom --gen-adg --dfgs <dfgs> -o <out.fabric.mlir>` | Generate ADG from DFG requirements |
| DFG Analysis | `loom --dfg-analyze --dfgs <dfgs> -o <out.mlir>` | Analyze DFGs and annotate with metrics |
| Clang Driver | `loom --as-clang [args...]` | Standard C++ compiler with ADG lib linking |

## Related Documents

- [spec-loom.md](./spec-loom.md): End-to-end Loom compilation and mapping pipeline
- [spec-pragma.md](./spec-pragma.md): Loom pragma system specification
- [spec-dataflow.md](./spec-dataflow.md): Dataflow dialect (conversion target)
- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview
- [spec-mapper.md](./spec-mapper.md): Mapper place-and-route bridge from Handshake to Fabric
- [spec-mapper-model.md](./spec-mapper-model.md): Mapper data model and hard constraints
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md): Mapper algorithm contract
- [spec-mapper-output.md](./spec-mapper-output.md): Mapper output formats (map.json, map.txt, config.bin, addr.h)
- [spec-fabric-mem.md](./spec-fabric-mem.md): Memory specification (fabric.memory, fabric.extmemory)
- [spec-adg.md](./spec-adg.md): ADG stage definition and outputs
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC backend artifacts
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog backend artifacts
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md): Runtime configuration memory definition
- [spec-cosim.md](./spec-cosim.md): Co-simulation stage overview and authority map
- [spec-cosim-runtime.md](./spec-cosim-runtime.md): Host runtime and threading contract
- [spec-cosim-validation.md](./spec-cosim-validation.md): End-to-end validation requirements
- [spec-viz.md](./spec-viz.md): Visualization specification
- [spec-viz-mapped.md](./spec-viz-mapped.md): Mapped visualization conventions
- [spec-viz-gui.md](./spec-viz-gui.md): Browser viewer specification
