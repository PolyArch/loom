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
```

`-o` is required.

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
- `-o <path>` or `-o<path>`: select the LLVM IR output path.
- `--adg <file.fabric.mlir>`: validate a fabric MLIR file (see below).
- `--as-clang`: operate as a standard C++ compiler (see below).

`--` terminates option parsing. All subsequent arguments are treated as input
files, even if they begin with `-`.

## Loom-Specific Options

### `--adg`

When `--adg <path>` is specified, `loom` operates in ADG validation mode.

**Behavior:**

- Parses the given fabric MLIR file
- Registers Fabric, Dataflow, Handshake, Arith, Math, MemRef, and Func dialects
- Runs MLIR parse and semantic verification
- Exits 0 if the file is valid, 1 if any errors are detected
- Error codes are emitted to stderr with `[COMP_XXX]` prefix
  (see [spec-fabric-error.md](./spec-fabric-error.md))
- No MLIR/SCF/Handshake outputs are generated
- Source files and `-o` are ignored when `--adg` is set

**Incompatibilities:**

`--adg` is incompatible with `--as-clang`. If both are specified, behavior is
undefined.

**Example:**

```bash
# Validate an ADG fabric MLIR file
loom --adg my_cgra.fabric.mlir

# Check exit code
echo $?  # 0 = valid, 1 = errors
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

## Stage C Placeholder: Mapper Invocation

Stage C (mapper place-and-route) CLI surface is specified in mapper documents.
This document currently specifies Stage A frontend behavior and `--as-clang`
behavior used to compile Stage B ADG programs.

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

## Related Documents

- [spec-loom.md](./spec-loom.md): End-to-end Loom compilation and mapping pipeline
- [spec-pragma.md](./spec-pragma.md): Loom pragma system specification
- [spec-dataflow.md](./spec-dataflow.md): Dataflow dialect (conversion target)
- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview
- [spec-mapper.md](./spec-mapper.md): Mapper place-and-route bridge from Handshake to Fabric
- [spec-mapper-model.md](./spec-mapper-model.md): Mapper data model and hard constraints
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md): Mapper algorithm contract
- [spec-adg.md](./spec-adg.md): ADG stage definition and outputs
- [spec-adg-sysc.md](./spec-adg-sysc.md): SystemC backend artifacts
- [spec-adg-sv.md](./spec-adg-sv.md): SystemVerilog backend artifacts
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md): Runtime configuration memory definition
- [spec-cosim.md](./spec-cosim.md): Co-simulation stage overview and authority map
- [spec-cosim-runtime.md](./spec-cosim-runtime.md): Host runtime and threading contract
- [spec-cosim-validation.md](./spec-cosim-validation.md): End-to-end validation requirements
