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
- `--as-clang`: operate as a standard C++ compiler (see below).

`--` terminates option parsing. All subsequent arguments are treated as input
files, even if they begin with `-`.

## Loom-Specific Options

### `--as-clang`

When `--as-clang` is specified, `loom` operates as a standard C++ compiler
without performing Loom-specific analysis or MLIR transformations.

**Behavior:**

- The tool behaves like `clang++` with the same defaults
- No MLIR outputs are generated
- Linking is enabled (linker flags are processed, not ignored)
- The Loom ADG library (`libloom-adg`) is automatically linked
- Include paths for Loom headers (`<loom/adg.h>`) are automatically added

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

**Differences from standard mode:**

| Aspect | Standard mode | `--as-clang` mode |
|--------|---------------|-------------------|
| MLIR generation | Yes | No |
| Linker flags | Ignored | Processed |
| Output naming | `-o` required | `<stem>` or `a.out` when `-o` is omitted |
| ADG library | Not linked | Auto-linked |

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

SCF lowering sequence:

- `loom::createLowerLLVMToSCFPass()`
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

Handshake lowering sequence:

- `loom::createSCFToHandshakeDataflowPass()`
- `canonicalize`
- `cse`

The SCF and Handshake MLIR outputs are written after their respective pass
sequences complete successfully.

### Loom-Specific Passes

- `loom::createLowerLLVMToSCFPass()`: Converts LLVM dialect to structured
  control flow (SCF) using pattern-based rewriting.
- `loom::createUpliftWhileToForPass()`: Converts `scf.while` loops that match
  canonical for-loop patterns into `scf.for` operations.
- `loom::createAttachLoopAnnotationsPass()`: Extracts loop pragma metadata from
  marker calls and attaches them as MLIR attributes on loop operations. This
  pass handles annotations from all loop-level pragmas including
  `LOOM_PARALLEL`, `LOOM_UNROLL`, `LOOM_TRIPCOUNT`, `LOOM_REDUCE`, and
  `LOOM_MEMORY_BANK`. See [spec-pragma.md](./spec-pragma.md).
- `loom::createMarkWhileStreamablePass()`: Analyzes `scf.while` loops to
  determine if they can be lowered to streaming dataflow and marks them
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

## Related Documents

- [spec-loom.md](./spec-loom.md): End-to-end Loom compilation and mapping pipeline
- [spec-pragma.md](./spec-pragma.md): Loom pragma system specification
- [spec-dataflow.md](./spec-dataflow.md): Dataflow dialect (lowering target)
- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview
- [spec-mapper.md](./spec-mapper.md): Mapper place-and-route bridge from Handshake to Fabric
