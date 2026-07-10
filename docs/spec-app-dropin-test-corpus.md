# LoomBench Drop-In Test Corpus

This document specifies the target testing contract for Loom's
self-contained LoomBench corpus under `test/app`. The corpus is the
general-purpose companion to the CMSIS drop-in compiler tests: each app
is a small independent C or C++ program that must build and run when
`loom-cc` or `loom-c++` replaces an ordinary compiler driver.

## Purpose

LoomBench answers this question:

```text
Can Loom act as a drop-in compiler for ordinary standalone C/C++
programs while also producing the artifacts needed by the Loom
dataflow, mapping, simulation, and hardware-evaluation stack?
```

The corpus is not a microbenchmark-only suite. It is a broad functional
compatibility and compiler-pipeline suite. It should cover scalar code,
array kernels, reductions, scans, stencils, sorting, bit operations,
signal processing, graph kernels, sparse kernels, geometric kernels,
string-like kernels, and small neural-network style kernels.

## Relationship To CMSIS Tests

The CMSIS drop-in compiler contract is specified in
`docs/spec-cmsis-dropin-compiler.md`. CMSIS tests validate Loom against
real external library source trees and public CMSIS APIs.

LoomBench validates Loom against small, self-contained programs
owned by this repository. It exercises ordinary compiler behavior,
runtime behavior, artifact generation, and simulator integration
without requiring external library layout, CMSIS headers, or CMSIS
build-system conventions.

The global workload universe is owned by `docs/spec-loom-stack.md`.
This document owns the repository `loombench` portion of that universe.
A case remains in that universe until it is deliberately removed from the
authoritative manifest.

Both suites share the same compiler-product principle: replacing
`gcc` or `g++` with `loom-cc` or `loom-c++` must preserve ordinary
program behavior in compatibility mode.

## Source Policy

Each accepted app case is a standalone source package under
`test/app/<case>/`. The case must not depend on Loom libraries,
top-level Loom build targets, private local paths, network access, wall
clock time, random devices, host-specific files, or mutable global
state outside the case directory.

Every case must provide:

* source files needed to build the app;
* a deterministic entry point;
* a reference-oracle path or expected-output oracle;
* a build descriptor or manifest entry;
* runner metadata that identifies source language, expected compiler
  mode, expected outputs, and supported pipeline stages.

The LoomBench source may contain both a reference implementation and a
candidate accelerator-facing implementation. The harness must compare
observable results and exit nonzero on mismatch. Naming conventions
such as reference and candidate function suffixes are test-harness
conventions only; they do not define dataflow or mapping semantics.

Changes to source formatting, build descriptors, and runner metadata
must not change algorithmic behavior or silently weaken result checks.

## Corpus Manifest

The target corpus is manifest-driven. A manifest entry for each case
records:

* case name;
* source files;
* language mode;
* compiler flags;
* link flags;
* expected executable names;
* expected stdout or oracle mode;
* supported validation tiers;
* required feature tags;
* optional grouping tags such as reduction, scan, stencil, sort, graph,
  sparse, signal, bit, geometry, string, or neural.

Runners must discover cases from the manifest rather than from a
hard-coded shell array. Targeted integration tiers are explicit manifest
subsets and do not redefine the complete corpus.

## Validation Tiers

LoomBench uses ordered validation tiers. A case may support a
later tier only if all earlier tiers that are relevant to that case are
well-defined.

### Tier 0: Baseline Native Build And Run

The case builds and runs with ordinary `gcc` or `g++` using the same
flags that Loom compatibility mode receives. The output oracle passes.

This tier proves the test itself is valid before Loom-specific behavior
is evaluated.

### Tier 1: Loom Drop-In Build And Run

The same case builds and runs with `loom-cc` or `loom-c++` replacing the
ordinary compiler driver. Compatibility mode must preserve observable
program behavior. The output oracle must pass.

Loom-specific artifact emission must be opt-in and namespaced. Ordinary
compile and link behavior must not fail because acceleration analysis is
incomplete.

### Tier 2: LLVM IR Emission

The case emits LLVM IR through the Loom driver. The emitted IR must
round-trip through the selected LLVM parser and preserve expected public
symbols, target triple, data layout, and source-language mode.

### Tier 3: Raise To MLIR

The case raises from LLVM IR to Loom's supported MLIR representation.
The raised artifact must parse and must retain enough structure for the
selected lowering path or produce a structured unsupported-scope
diagnostic.

### Tier 4: Dataflow Lowering

The case lowers to dataflow IR when the selected region and lowering
policy support it. The emitted dataflow artifact must parse, verify,
and satisfy the target contracts for `dataflow.thread`,
`dataflow.graph`, and `dataflow.subgraph` in the dataflow specs.

Unsupported regions must be diagnosed explicitly. A compatibility-mode
run must not be marked failed solely because an optional dataflow
artifact is unsupported.

### Tier 5: DFG-sim

The case runs under DFG-sim for supported dataflow artifacts and
concrete inputs. Functional outputs, memory effects, and diagnostics
must agree with the case oracle according to `docs/spec-sim-dfg.md`.

### Tier 6: PnR And CGRA-sim

The case maps through PnR and runs under CGRA-sim when a compatible
Fabric ADG and mapping policy are selected. CGRA-sim results must
preserve functional behavior and emit hardware-aware metrics according
to `docs/spec-pnr.md`, `docs/spec-mapping-artifact.md`, and
`docs/spec-sim-cgra.md`.

### Tier 7: RTL And FPA Evidence

The case feeds mapped-workload RTL checks and FPA estimation when
the selected hardware profile supports those flows. Reports must follow
`docs/spec-rtl-lowering.md`, `docs/spec-fpa-estimation.md`, and
`docs/spec-eda-tooling.md`.

## Runner Requirements

The app runner stack must provide:

* one all-cases native and Loom drop-in runner;
* representative source-to-IR integration tiers;
* a single-case runner for debugging;
* a manifest validation command;
* deterministic output directories;
* nonzero exit status for real failures;
* zero exit status only when every selected case passes;
* validation that rejects empty selections, missing outputs, and unrelated
  source symbols.

Diagnostics for an attempted unsupported path must satisfy the Unsupported
Scope Policy in `docs/spec-loom-stack.md` and may add component-specific
case, tier, runner, or expected-output fields.

## Artifact Requirements

Generated artifacts must have stable names and ownership when a tier is run:

* native executables and stdout files belong to the build/run tier;
* LLVM IR files belong to the IR tier;
* raised MLIR files belong to the raise tier;
* dataflow MLIR files belong to the dataflow tier;
* DFG-sim reports belong to the simulator tier;
* mapping artifacts belong to the PnR tier;
* CGRA-sim reports belong to the hardware-aware simulator tier;
* RTL and FPA reports belong to their respective hardware-evaluation
  tiers.

Generated artifacts must not be committed unless a specific checked-in
golden artifact is required by a test. Golden artifacts must be small,
stable, and justified by the test contract.

## Acceptance Criteria

The app drop-in corpus target is complete when:

* every accepted case under `test/app` is described by the manifest;
* committed `test/app` case directories do not contain generated build
  outputs unless they are small checked-in golden artifacts justified by
  the test contract;
* all cases pass the baseline native build-and-run tier;
* all required cases pass the Loom drop-in build-and-run tier;
* supported cases can proceed through LLVM IR, raise, and dataflow
  lowering tiers with structured artifacts;
* representative downstream tiers remain explicit manifest subsets while
  the complete manifest remains the suite inventory;
* runner failures, missing cases, stale expected outputs, and invalid
  manifest records are diagnosed by tests;
* compatibility-mode failures are separated from optional
  acceleration-artifact unsupported-scope diagnostics.
