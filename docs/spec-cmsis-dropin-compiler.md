# CMSIS Drop-In Compiler

This document specifies Loom's target CMSIS drop-in compiler contract.
The target product behavior is that CMSIS-DSP and CMSIS-NN projects can
use Loom where they would otherwise use a C or C++ compiler, while Loom
can also emit its internal compiler artifacts and acceleration evidence
when explicitly enabled.

## Purpose

The CMSIS drop-in compiler answers this question:

```text
Can a CMSIS-DSP or CMSIS-NN user replace the ordinary compiler driver
with Loom and keep the build working, while gaining access to Loom's
dataflow, mapping, simulation, and estimation flow?
```

The repository-owned LoomBench corpus is specified separately in
`docs/spec-loombench.md`. It validates the same drop-in
compiler principle on self-contained C/C++ programs, while this
document owns the external CMSIS source-tree contract.

The compiler consumes:

* C or C++ source files;
* ordinary compiler command-line options;
* CMSIS include paths and preprocessor definitions;
* optional Loom-specific options;
* an optional hardware selection, otherwise the designated builtin default;
* optional resolved compiler, Mapping, and Evaluation policy inputs; and
* optional Mapping constraints and Evaluation requests.

It produces:

* the ordinary requested compiler output;
* optional LLVM IR;
* optional raised MLIR;
* optional dataflow IR;
* optional mapping artifacts;
* optional SimulationExecution and EvaluationEvidence artifacts;
* optional RTL HardwareImplementation and EDA Evaluation artifacts;
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

Source annotation is optional. The only initial public spelling is the
nonbinding `#pragma loom candidate` defined by
`docs/spec-compiler-part-1-source.md`. It marks the immediately following
function or loop as a candidate but neither enables acceleration nor requires
that candidate to be selected. Unannotated programs remain fully eligible for
profile-driven discovery and ordinary drop-in compilation.

## Driver Surface

The public driver surface consists of:

* `loom-cc`: C-mode compiler driver;
* `loom-c++`: C++-mode compiler driver.

They orchestrate Loom through the same in-process libraries used by developer
tools. A public driver must not shell out to stage binaries or reimplement a
stage pipeline, schema, verifier, or default.

The driver must preserve ordinary compiler behavior for commonly used
queries such as version reporting, target reporting, resource paths,
search paths, preprocessing, and dependency generation.

The stable Loom-specific options already owned by the full-stack contract are:

```text
--loom-accel-profile=<builtin-preset-or-config-path>
--loom-hardware=<fabric.mlir>
--loom-viz-export=<directory>
--loom-deploy-output=<path>
```

The profile is the only public semantic policy selector. The external Fabric,
visualization, and Deployment paths are invocation bindings. Hardware import
finalizes the exact Fabric Artifact that becomes the semantic target. Mapping
constraints, Evaluation requests, and focused artifact inputs remain typed
shared-library or developer-tool interfaces until a public driver contract
explicitly assigns their spelling. Components must not independently expose ad
hoc public flags for them.

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

## Artifact Access

Intermediate artifacts are available through shared library APIs, developer
tools, and `--loom-viz-export` projections. This does not create a separate
public Artifact Mode, implicit side-output directory, or parallel artifact
schema. Ordinary compiler output is never replaced or overwritten by an
intermediate projection.

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

Acceleration mode must make ownership disposition explicit. Unsupported
SpatialCore work either remains under HostCore or InstructionCore ownership in
a different Structured Program Candidate, or the requested acceleration
fails. This is a compile-time candidate decision, not a runtime fallback from
an invalid Mapping. Report-only analysis may emit diagnostics and Evaluation
requests without changing ordinary compiler output. The exact flow is selected
by the resolved acceleration profile; no component adds a second public policy
flag.

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

CMSIS names and arities are never compiler semantics. Loom obtains callable
bodies through ordinary source compilation, linker-selected object and archive
members, LLVM Linker, and LTO. It does not maintain a CMSIS symbol-rewrite table
or a Dataflow-level library linker.

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
universe. Its canonical inventory is the independently invocable tracked C
translation-unit set in the pinned CMSIS submodules. Explicit smoke targets
exercise real compiler paths while support expands, but they do not redefine
that inventory or act as per-source status records.

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
Configuration images, Deployment closure, and package publication are owned by
`docs/spec-configuration-deployment.md`.

Drop-in acceleration requires ordinary separate compilation:

```text
source -> object with frontend-owned relocatable accelerator payload
       -> final link -> unified LLVM boundary -> Loom flow -> Deployment
```

Compile-only output remains an ordinary object. Its accelerator payload must
not contain Fabric, Mapping, ConfigurationABI, or HardwareConfigurationImage
artifacts. Objects without such a payload remain legal InstructionCore or
external-code inputs. `docs/spec-compiler-part-1-source.md` owns the exact
`loom.relocatable_accelerator_payload 1.0` root, LLVM-owned symbol semantics,
config-view compatibility, carrier-independent identity, and final-link merge.
This CMSIS contract does not redefine that encoding.

Runtime requirements must be explicit. If an invocation requests an
accelerated binary but the required runtime is unavailable, the driver
must diagnose the missing runtime rather than silently producing a host-only
binary. A host or InstructionCore alternative is a separately selected
Structured Program Candidate and deployment disposition, not an error-path
substitution.

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
* simulator or Evaluation/projection generation failures;
* runtime or linking failures.

Diagnostics should preserve source locations when available and should
identify the relevant pipeline component.

Hard compiler failures use the clang diagnostic engine. Candidate selection,
missed acceleration, and analysis explanations use LLVM optimization remarks
and the standard optimization-record projection. Their facts come from the
owning typed failure or Evaluation records; Loom does not define a global
diagnostic Artifact, duplicate error taxonomy, or text-snapshot contract.

Unsupported work must use the typed unsupported or incomplete boundaries in
`docs/spec-loom-stack.md` and the Evaluation outcome contract in
`docs/spec-dse-feedback.md` when Evaluation has been requested. Diagnostics
are projections of those typed facts and compiler-owned failure facts.
CMSIS-specific projections may add source-tree, target-triple, intrinsic, or
source-row details without becoming a second outcome authority.

## Artifacts And Projections

`docs/spec-dse-feedback.md` section `Invocation and Recovery Records` is
the sole owner of InvocationManifest and ExecutionJournal fields. CMSIS
projections reference those records and exact artifacts; they do not repeat
the schema, mutate source trees or vendored inputs, or become DSE inputs.

## Canonical Source Inventory

Canonical CMSIS membership comes from every independently invocable tracked C
translation unit under `Source` at the submodule commits pinned by the parent
repository. CMSIS reserves a leading underscore on a source basename for
private implementation fragments that require an including translation unit's
macro environment; those fragments are not independent compiler invocations.
They remain covered through their including translation units and the
target-feature configurations that select them. `test/corpus_inventory.py`
derives these rows directly from each verified submodule commit tree, reports
their counts mechanically, and verifies that the checked-out submodule
revisions match the parent repository gitlinks.
LoomBench membership is defined by its own manifest rather than by this CMSIS
contract.

The CMSIS-DSP and CMSIS-NN fast smoke tables select replaceable sources. Each
row identifies a `Source`-relative translation unit, target triple, CPU, public
source symbol, and optional compiler flags. The tables must validate as strict
subsets of their canonical inventories. They neither define suite membership
nor define a shallower compiler capability for unselected members, and they do
not require adjacent status, binding, or provenance files. The parent
repository's pinned submodule commit and the source-relative path identify the
source used by a smoke run.

## Testing And Acceptance

Core CMSIS regression coverage requires:

* the inventory count comes directly from independently invocable tracked C
  translation units in each pinned `Source` tree;
* smoke-selected CMSIS-DSP and CMSIS-NN sources compile through LLVM IR,
  raised MLIR, and dataflow MLIR without modifying the external source trees;
* generated MLIR reparses, preserves the selected public source symbol, and
  contains a dataflow definition associated with that symbol;
* validation rejects a dataflow artifact whose definitions are unrelated to
  the selected source symbol;
* ordinary compiler compatibility and acceleration-specific behavior are
  tested at their owning driver and pipeline boundaries.

A complete-suite stage invocation attempts every requested translation unit
through the public driver and the same in-process stage libraries used for
LoomBench. Every unit must satisfy ordinary drop-in compilation when its
underlying compiler target is supported. When Canonical Dataflow is requested,
every unit must publish a valid whole-program Canonical Dataflow Artifact or
report a real compiler/provider limitation. A valid Artifact may be graph-free
when the exact profile selects no SpatialCore-owned region; this is not a
CMSIS-specific exception and cannot be represented by an empty placeholder.
When Mapping, simulation, or a later stage is requested, the same owner
contracts apply without a CMSIS-specific stopping rule.

Smoke targets are deliberately replaceable as compiler coverage changes.
Expanding coverage toward the complete inventory does not require a parallel
status ledger or one wrapper test per source file. Inventory enumeration alone
does not claim compilation, lowering, simulation, mapping, or support success;
an attempted source keeps its own honest pipeline outcome.

## Non-Goals

The CMSIS drop-in compiler spec is not a replacement for the dataflow,
fabric, PnR, simulator, RTL, or FPA specs. It composes those contracts
into a source-facing product flow.

The CMSIS target does not limit Loom to CMSIS. It exercises embedded compiler
compatibility and practical accelerator mapping through ordinary C/C++ source
interfaces.
