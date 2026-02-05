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

`-o` is required for explicit output naming, but if omitted a default output
path is chosen (see below).

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

Default output path:

- If there is exactly one input, the default is `<stem>.llvm.ll`.
- Otherwise, the default is `a.llvm.ll`.

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
| Default output | `<stem>.llvm.ll` | `<stem>` or `a.out` |
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
  marker calls and attaches them as MLIR attributes on loop operations.
- `loom::createMarkWhileStreamablePass()`: Analyzes `scf.while` loops to
  determine if they can be lowered to streaming dataflow and marks them
  accordingly.
- `loom::createSCFToHandshakeDataflowPass()`: Converts SCF operations to
  Handshake dialect for dataflow execution on hardware.
