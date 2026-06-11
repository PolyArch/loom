# CMSIS Drop-In Compiler

This document specifies Loom's target CMSIS drop-in compiler contract.
The target product behavior is that CMSIS-DSP and CMSIS-NN projects can
use Loom where they would otherwise use a C or C++ compiler, while Loom
can also emit its internal compiler artifacts and acceleration reports
when explicitly enabled.

## Purpose

The CMSIS drop-in compiler answers this question:

```text
Can a CMSIS-DSP or CMSIS-NN user replace the ordinary compiler driver
with Loom and keep the build working, while gaining access to Loom's
dataflow, mapping, simulation, and estimation flow?
```

The repository-owned app drop-in corpus is specified separately in
`docs/spec-app-dropin-test-corpus.md`. It validates the same drop-in
compiler principle on self-contained C/C++ programs, while this
document owns the external CMSIS source-tree contract.

The compiler consumes:

* C or C++ source files;
* ordinary compiler command-line options;
* CMSIS include paths and preprocessor definitions;
* optional Loom-specific options;
* optional hardware, mapping, simulator, and estimation profiles.

It produces:

* the ordinary requested compiler output;
* optional LLVM IR;
* optional raised MLIR;
* optional dataflow IR;
* optional mapping artifacts;
* optional simulation reports;
* optional RTL or FPA reports;
* structured diagnostics.

## Drop-In Rule

Without explicit Loom acceleration or artifact options, `loom-cc` and
`loom-c++` must behave as ordinary C and C++ compiler drivers for the
same source, flags, target, and output request.

Drop-in compatibility includes:

* preprocessing options;
* include path options;
* macro definition and undefinition options;
* language standard options;
* warning and diagnostic options;
* optimization options;
* target triple and CPU options;
* ABI, floating-point ABI, and target feature options;
* compile-only, assemble-only, and link-oriented driver modes;
* dependency-file generation;
* response files;
* build-system use through `CC` and `CXX` overrides.

Loom-specific behavior must be additive and namespaced. It must not
silently reinterpret ordinary compiler flags as acceleration requests.

## Driver Surface

The public driver surface consists of:

* `loom-cc`: C-mode compiler driver;
* `loom-c++`: C++-mode compiler driver;
* optional aliases whose behavior is selected by the executable name;
* optional Loom-specific flags for artifact emission and acceleration.

The driver must preserve ordinary compiler behavior for commonly used
queries such as version reporting, target reporting, resource paths,
search paths, preprocessing, and dependency generation.

Loom-specific flags should be explicit. Baseline option classes are:

* emit intermediate artifacts;
* select acceleration policy;
* select hardware or ADG profile;
* select mapping policy;
* select simulator or estimation policy;
* select artifact output directory;
* require acceleration or allow fallback;
* control diagnostic verbosity.

The exact spelling of options may evolve, but the option classes above
must remain separable. Artifact emission must be possible without
requiring hardware mapping. Acceleration must be possible without
exposing all intermediate artifacts to the user.

## Compatibility Mode

Compatibility mode is the default mode. It compiles, assembles, and
links according to ordinary compiler semantics. It may internally
produce transient LLVM IR, but those transients are not user-visible
unless an artifact option requests them.

If compatibility mode cannot support an ordinary compiler invocation,
it must report the unsupported option or missing tool capability
directly. It must not fail because Loom acceleration analysis is
missing.

Compatibility mode is allowed to use the embedded clang provider
described in `docs/spec-compiler-part-1-source.md`.

## Artifact Mode

Artifact mode emits Loom compiler artifacts while preserving the
ordinary output requested by the build when possible.

Artifact classes include:

* LLVM IR plus Loom metadata;
* raised SCF-shaped MLIR;
* dataflow IR;
* mapping artifacts;
* DFG-sim reports;
* CGRA-sim reports;
* simulation comparison reports;
* RTL manifests;
* FPA reports.

Artifact mode must write artifacts to an explicit output location or to
a deterministic side-output location. It must not overwrite ordinary
compiler outputs unexpectedly.

## Acceleration Mode

Acceleration mode enables Loom's compiler and architecture flow for
selected candidate regions. It may use:

* source integration from `docs/spec-compiler-part-1-source.md`;
* LLVM-to-SCF raising from `docs/spec-compiler-part-2-scf.md`;
* SCF-to-dataflow lowering from `docs/spec-compiler-part-3-dfg.md`;
* Fabric ADG from `docs/spec-fabric-system-adg.md`;
* PnR from `docs/spec-pnr.md`;
* mapping artifacts from `docs/spec-mapping-artifact.md`;
* simulation from `docs/spec-sim-dfg.md` and
  `docs/spec-sim-cgra.md`;
* RTL/FPA flows from `docs/spec-rtl-lowering.md` and
  `docs/spec-fpa-estimation.md`.

Acceleration mode must have an explicit fallback policy:

* `allow_fallback`: keep unsupported regions on the host or ScalarCore
  and still produce the ordinary compiler output when possible;
* `require_acceleration`: fail if requested acceleration cannot be
  represented, mapped, or validated;
* `report_only`: run analysis and emit reports without changing the
  ordinary compiler output.

Fallback decisions must be visible in diagnostics and reports.

## CMSIS Source Policy

CMSIS external source trees are treated as immutable user or vendored
inputs. Loom must not require editing CMSIS-DSP, CMSIS-NN, or
CMSIS-Core source files to make the drop-in path work.

Per-source or per-library adaptation belongs in:

* ordinary compiler flags;
* include path configuration;
* target feature selection;
* Loom metadata or analysis;
* target manifests used by tests;
* explicit compatibility shims supplied by the user or runtime.

The compiler must preserve CMSIS public API names, symbol visibility,
target triples, data layout, ABI decisions, and ordinary diagnostics.

## CMSIS Target Scope

The target CMSIS scope includes:

* CMSIS-DSP C sources;
* CMSIS-NN C sources;
* CMSIS-Core headers used by those libraries;
* Cortex-M style target triples;
* CPU selection through ordinary compiler flags;
* floating-point ABI options;
* fixed-point and floating-point kernels;
* scalar paths and target-feature-gated intrinsic paths.

The global workload universe is owned by `docs/spec-loom-stack.md`.
This document owns only the CMSIS-DSP and CMSIS-NN portion of that
universe. Every tracked CMSIS source row must eventually have an
explicit status for compile/run, IR emission, raise, dataflow lowering,
DFG-sim, PnR/CGRA-sim, and RTL/FPA evidence. Rows that cannot reach a
tier must carry structured unsupported-scope, failed, or blocked
records.

Unsupported target intrinsics, missing sysroots, unavailable target
backends, and unsupported library configurations must produce
structured diagnostics. In compatibility mode, unsupported acceleration
must not prevent ordinary compilation when the underlying compiler can
compile the source.

## Runtime And Linking

The drop-in compiler must support ordinary object generation and
link-oriented driver flows. Acceleration mode may require additional
runtime support for:

* host-to-accelerator launch;
* data movement;
* synchronization;
* memory allocation or binding;
* generated configuration data;
* simulator or profiling hooks.

The target runtime ABI is specified in `docs/spec-runtime-abi.md`.

Runtime requirements must be explicit. If an invocation requests an
accelerated binary but the required runtime is unavailable, the driver
must diagnose the missing runtime rather than silently producing a
host-only binary unless fallback is allowed and reported.

## Diagnostics

Diagnostics must distinguish:

* ordinary compiler failures;
* unsupported ordinary compiler options;
* missing CMSIS headers or source files;
* target triple or CPU incompatibility;
* unsupported intrinsic or target feature;
* unsupported LLVM-to-SCF raising;
* unsupported dataflow lowering;
* unmappable accelerator regions;
* simulator or report generation failures;
* runtime or linking failures.

Diagnostics should preserve source locations when available and should
identify the relevant pipeline component.

Unsupported-scope diagnostics must satisfy the Unsupported Scope Policy
in `docs/spec-loom-stack.md`. CMSIS-specific diagnostics may add
source-tree, target-triple, intrinsic, or source-row details.

## Reports

When artifact or acceleration mode is enabled, reports should identify:

* input source and target identity;
* ordinary compiler command identity;
* selected target triple and CPU;
* emitted artifact paths or ids;
* selected acceleration policy;
* selected hardware or ADG profile;
* fallback decisions;
* DFG-sim, CGRA-sim, comparison, and FPA report identities when
  present;
* diagnostics.

Reports must not mutate source trees or vendored CMSIS inputs.

## Testing And Acceptance

The target is complete when:

* every tracked CMSIS-DSP and CMSIS-NN source row has an explicit status
  for every validation tier, with either passing evidence or a
  structured unsupported-scope, failed, or blocked record;
* C and C++ build-system overrides can use `loom-cc` and `loom-c++`
  without editing source code;
* IR emission preserves target triples, data layout, and expected
  public symbols;
* raising and dataflow lowering produce deterministic artifacts or
  structured diagnostics;
* skip lists are empty or every skip is justified by a structured
  unsupported-scope diagnostic;
* negative tests prove runner and per-row gate failures are not masked;
* acceleration-required mode fails when acceleration is impossible;
* fallback-allowed mode reports host or ScalarCore fallback explicitly;
* representative CMSIS-DSP and CMSIS-NN gates may be used as
  intermediate regression checks, but they do not replace full
  per-source target tracking across the validation ladder.

## Non-Goals

The CMSIS drop-in compiler spec is not a replacement for the dataflow,
fabric, PnR, simulator, RTL, or FPA specs. It composes those contracts
into a source-facing product flow.

The CMSIS target does not limit Loom to CMSIS forever. It is the first
product-quality C/C++ target because it stresses embedded compiler
compatibility and practical accelerator mapping.
