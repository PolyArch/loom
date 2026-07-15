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
* sparse smoke metadata used by tests;
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
universe. Its canonical inventory is the tracked C source set in the
pinned CMSIS submodules. Explicit smoke targets exercise real compiler paths
while support expands, but they do not redefine that inventory or act as
per-source status records.

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

## Canonical Source Inventory

Canonical CMSIS membership comes from every tracked `.c` file
under `Source` at the submodule commits pinned by the parent repository.
`test/corpus_inventory.py` derives these rows directly from each verified
submodule commit tree, reports their counts mechanically, and verifies that the
checked-out submodule revisions match the parent repository gitlinks.
LoomBench membership is defined by its own manifest rather than by this CMSIS
contract.

The CMSIS-DSP and CMSIS-NN DFG smoke tables select replaceable sources. Each
row identifies a `Source`-relative translation unit, target triple, CPU, public
source symbol, and optional compiler flags. The tables must validate as strict
subsets of their canonical inventories. They neither define suite membership
nor require adjacent status, binding, or provenance files. The parent
repository's pinned submodule commit and the source-relative path identify the
source used by a smoke run.

## Testing And Acceptance

Core CMSIS regression coverage requires:

* the inventory count comes directly from tracked `.c` files in each pinned
  `Source` tree;
* smoke-selected CMSIS-DSP and CMSIS-NN sources compile through LLVM IR,
  raised MLIR, and dataflow MLIR without modifying the external source trees;
* generated MLIR reparses, preserves the selected public source symbol, and
  contains a dataflow definition associated with that symbol;
* validation rejects a dataflow artifact whose definitions are unrelated to
  the selected source symbol;
* ordinary compiler compatibility and acceleration-specific behavior are
  tested at their owning driver and pipeline boundaries.

Smoke targets are deliberately replaceable as compiler coverage changes.
Expanding coverage toward the complete inventory does not require a parallel
status ledger or one wrapper test per source file. Inventory enumeration alone
does not claim compilation, lowering, simulation, mapping, or support success;
an attempted source keeps its own honest pipeline outcome.

## Non-Goals

The CMSIS drop-in compiler spec is not a replacement for the dataflow,
fabric, PnR, simulator, RTL, or FPA specs. It composes those contracts
into a source-facing product flow.

The CMSIS target does not limit Loom to CMSIS forever. It is the first
product-quality C/C++ target because it stresses embedded compiler
compatibility and practical accelerator mapping.
