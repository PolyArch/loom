# Loom Full-Stack Architecture

This document owns Loom's top-level product and component boundaries. Detailed
semantics belong to the referenced subsystem specifications. This document
does not duplicate their schemas, defaults, algorithms, or implementation
status.

## Documentation Authority

Tracked `spec-*.md` files are the sole normative design authority. They own
WHAT implementations and verifiers must do. Tracked files under
`docs/rationales/` explain WHY those contracts were selected, including
rejected alternatives, but cannot add fields, semantics, defaults, or
exceptions. Source code owns HOW and must conform to the specifications.

A rationale disagreement with a specification is a documentation defect, not
an alternate contract. A changed decision updates the one normative owner and
its rationale together; superseded implementable contracts are not retained.
Temporary meeting notes, implementation status, and work queues are neither
normative inputs nor tracked dependencies. The navigation entry point is
[Loom Design Documentation](README.md).

## Product Scope

Loom is a full-stack compiler and architecture-exploration framework for
multi-core heterogeneous spatial accelerators. Its two primary inputs are:

* software represented at Loom's compiler boundary as LLVM IR, initially from
  C and C++; and
* hardware represented as fully elaborated Fabric MLIR, either emitted through
  the public C++ ADG Builder API, selected from a built-in Fabric target, or
  supplied directly by the user.

Its three primary output families are:

* Evaluation results, including analysis, DFG simulation, CGRA simulation,
  external system simulation, RTL execution, and EDA evidence;
* a Deployment closure containing the exact configuration images and runtime
  bindings required to execute mapped software; and
* a HardwareImplementation containing generated RTL and the inputs needed for
  an explicit RTL-to-GDSII tool flow.

The only public end-user compiler drivers are `loom-cc` and `loom-c++`.
Independent development and test executables may expose individual passes or
component pipelines, but their library functionality is also integrated
in-process into the public drivers and does not define alternate semantics.

`loom-opt` is the canonical developer-only MLIR pass runner. Other focused
developer executables are justified only by a real input protocol,
multi-artifact orchestration, simulator, or performance-sensitive component
boundary. This permits tools such as a PnR replay driver without requiring one
binary per pass. Every such executable owns only its CLI and presentation; the
shared library remains the sole owner of passes, pipelines, schemas,
verification, defaults, and algorithms.

An external Fabric description is selected with:

```text
--loom-hardware=<fabric.mlir>
```

The path is an invocation binding. Import and finalization produce the exact
Fabric Artifact consumed by the flow; that identity, rather than the source
path, is the semantic hardware target. Builtin targets are selected through
the resolved acceleration profile, and omission selects its designated
canonical default. An invocation cannot silently combine or override external
and builtin hardware sources.

## Machine Model

The target machine is:

```text
System = HostCore + heterogeneous AccCores + memory/services + transport
AccCore = InstructionCore + SpatialCore
SpatialCore = arbitrary-topology CGRA described by fabric.module
```

An InstructionCore is a PC-based von Neumann core used for the selected
program regions that do not execute spatially. It may be scalar, vector,
in-order, or out-of-order. A SpatialCore executes canonical Dataflow graphs on
the configured Fabric resources. AccCores may differ in both components.

`fabric.system` is architecture-only and owns the typed system hardware
description: AccCore occurrences, InstructionCore facts, SpatialCore
attachments, memories and services, Transport Architecture, external
boundaries, and hardware domains. Interconnect Implementation is a separate
sibling Fabric-family refinement object, even when serialized near the
architecture it refines. Topology is explicit and directed. Meshes,
coordinates, and Manhattan distance are optional construction or presentation
hints, never semantic assumptions.

The first-version architecture is single-tenant. Multi-tenancy,
virtualization, preemption, migration, complex QoS, and runtime remapping are
deferred and have no placeholder fields or mechanisms in current schemas.

The core software/hardware ownership boundary is specified by
[Core Dialect Boundary](spec-core-dialect-boundary.md).

## Full-Stack Components

Loom has six semantic components:

* the compiler frontend derives and optimizes Structured and Canonical
  Dataflow program candidates;
* the hardware frontend uses the public C++ ADG Builder or builtin templates
  to produce exact Fabric artifacts;
* Mapping produces TechMapping, SpatialMapping, and SystemMapping artifacts;
* simulation backends provide DFG, CGRA, and external gem5-backed system
  execution models;
* the hardware backend lowers Fabric to HardwareImplementation artifacts and
  drives requested RTL and EDA flows; and
* the central Evaluation and DSE component owns requests, evidence,
  objectives, candidate generation, ranking, and promotion.

These owners can be viewed operationally as optimization/exploration,
hardware construction, and evaluation, but those three navigation groups do
not replace the six ownership boundaries. Compiler, Mapping, and hardware
exploration all consume the same Evaluation and DSE infrastructure rather than
creating private metric, finding, objective, or evidence systems. Reports and
visualizations are removable projections.

The shared Evaluation and DSE contract is specified by
[Evaluation and DSE](spec-dse-feedback.md). Metric definitions are owned by
[Evaluation Metrics](spec-evaluation-metrics.md).

## Compiler Pipeline

The compiler pipeline has four semantic boundaries:

```text
LLVM IR
  -> raised standard MLIR and Structured Program Candidate
  -> optimized Structured Program Candidate
  -> canonical Dataflow Program
  -> optimized canonical Dataflow Program
```

Translation into the structured MLIR stage is mechanical. The Structured
Program Candidate is the primary optimization surface for loop, memory,
parallelism, vectorization, partitioning, and InstructionCore/SpatialCore
ownership decisions. It may query fast central Evaluation models with an exact
Fabric target, but it does not invoke Mapping.

Selected SpatialCore regions use compiler-internal `loom.spatial_region`
staging. Mechanical SCF-to-Dataflow lowering must remove the staging operation
and all residual imperative control before publishing a canonical
`dataflow.graph`. Dataflow-only rewrites then optimize facts that are available
or naturally expressed only in the canonical graph. TechMapping begins only
after that boundary.

The invocation/driver resolves one exact immutable Fabric target before
frontend work begins. A user may provide Fabric MLIR through
`--loom-hardware`, select a built-in target through the resolved profile, or
omit both and receive the canonical default built-in target. Builtins are
ordinary fully elaborated Fabric artifacts, not abstract capability summaries.
The frontend consumes that target directly for mechanical target facts or
through derived capability views and central Evaluation; it does not own
target selection or invoke Mapping during candidate pruning.

Ordinary separate compilation embeds the carrier-independent
`loom.relocatable_accelerator_payload 1.0` owned by Compiler Part 1. Final link
collects payloads only from linker-selected object and archive members and uses
LLVM Linker/LTO semantics before the pipeline above. Object section and
compression choices are non-semantic; no relocatable payload contains Fabric,
Mapping, configuration, or Deployment choices.

Frontend contracts are specified by the compiler, Dataflow, and vectorization
specifications, including [Structured Compiler IR](spec-compiler-part-2-scf.md),
[Canonical Dataflow Lowering](spec-compiler-part-3-dfg.md), and
[Dataflow Vectorization](spec-dataflow-vectorization.md).

## Hardware Pipeline

Fabric MLIR is the hardware semantic SSOT. Builder objects, backend-native
models, RTL, configuration images, and reports are derived from or bound to an
exact Fabric artifact; none may silently become a competing architecture
description.

`fabric.module` owns one SpatialCore. Its graph-region body uses explicit
typed connectivity among `fabric.pe`, `fabric.switch`, `fabric.mem`, FIFO,
boundary, and instantiation resources. `fabric.fu` is a configurable physical
subgraph inside a PE. Hardware parameters describe immutable capability;
software configuration selects one supported effective function. Physical
configuration encoding belongs to ConfigurationABI rather than Fabric.

The ADG Builder contract is specified by
[ADG Builder](spec-adg-builder.md). Fabric-to-RTL ownership is specified by
[RTL Lowering](spec-rtl-lowering.md), and external tool execution by
[EDA Tooling](spec-eda-tooling.md). Persistent hardware roots and shared
resource atoms are specified by [Fabric Artifact](spec-fabric-artifact.md),
[Fabric Resource Contract](spec-fabric-resource-contract.md),
[Hardware Implementation](spec-hardware-implementation.md), and
[Implementation Platform](spec-implementation-platform.md).

## Mapping

Loom has one Dataflow-to-Fabric Mapping artifact family with three cumulative
profiles:

* TechMapping selects target-specific Compute and Memory Realizations and the
  exact configured Fabric capability relations that support them;
* SpatialMapping binds those realizations, residual logical nets, memory
  services, resources, tags, and refinements inside a SpatialCore; and
* SystemMapping binds thread and graph execution, channels, services, and
  transport across the complete heterogeneous system without carrying
  protocol implementation identity.

The closed Spatial PnR contract has exact `D/T/F/C/K` inputs: Canonical
Dataflow Program, TechMapping, Fabric, the mechanically derived Resolved PnR
config view, and one MappingConstraintSet. System PnR has the corresponding
six-input `D/F/R/H/C/K` contract. `docs/spec-pnr.md` owns the fixed
`loom.mapping_constraints 1.0` family and its Spatial and System roots.
Placement, routing, and resource
allocation are coupled through one Action and MoveTransaction model.
Persistent Mapping MLIR is the wire-schema SSOT; C++ hot structures are
removable projections optimized for search.

Mapping owns structural legality and domain-independent PnR costs. Evaluation
owns accelerator- and workload-aware observations. Central resolved policy is
the only owner that combines them for candidate ranking or acceptance.

Mapping persistence is specified by [Mapping Artifact](spec-mapping-artifact.md)
and [Mapping Identity](spec-mapping-identity.md). Fabric-local persistent
targets are owned by
[Fabric Persistent Identity And References](spec-fabric-identity.md). The PnR
algorithm and native state are owned by [Place And Route](spec-pnr.md).
Focused Mapping specs are derived views of those owners, not parallel schemas.

## Simulation And Backend Evidence

DFG-sim executes canonical Dataflow semantics without Fabric resource limits.
CGRA-sim executes mapped SpatialCore behavior using exact Dataflow, Fabric,
and Mapping inputs. Both are event-driven and may emit diagnostic ordered
cycle-coordinate traces to attempt or scratch storage. Persistent traces require
the future Simulation Artifacts schema minor and raw detailed-bundle owner.

HostCore, InstructionCore, cache, coherence, and system-interconnect execution
belong to an external system simulator integrated through Loom's bridge. Loom
does not rebuild a CPU or full-system simulator. A system simulation is an
ordinary Evaluation model over an exact Deployment, Gem5SimulationBinding,
workload, and runtime input.

Architecture-only RTL or EDA evaluation may produce EvaluationEvidence without
claiming workload execution. Raw reports remain owner-attempt or scratch
material until their exact Artifact owner is defined. Mapped RTL execution
produces SimulationExecution only when it actually runs the Deployment and
observes the requested values, streams, memories, and completion behavior.

The simulator contracts are
[Simulation Artifacts](spec-simulation-artifacts.md),
[DFG Simulation](spec-sim-dfg.md),
[CGRA Simulation](spec-sim-cgra.md), and
[Simulation Comparison](spec-sim-comparison.md). Reporting and visualization
must derive from exact artifacts and executions rather than define another
result authority.

## Configuration, Artifacts, And Determinism

One fully elaborated ResolvedConfig is the configuration SSOT. Component views
are versioned mechanical projections. Semantic values, defaults, and schema
versions are each owned once; invocation paths, output directories, host
parallelism, licenses, and wall-clock limits remain nonsemantic execution
bindings.

Every persistent semantic object uses the common Artifact identity contract.
One codebase semantic/build identity plus one exact resolved semantic closure
must produce the same formal result. Randomized algorithms use explicit,
versioned deterministic protocols and stable logical work ordinals.

The global contracts are [Configuration SSOT](spec-config-ssot.md),
[Configuration And Deployment](spec-configuration-deployment.md),
[Executable Closure](spec-executable-closure.md),
[Intermediate Reports And Projections](spec-intermediate-artifacts.md), and
[Runtime ABI](spec-runtime-abi.md).

## Deliberate First-Version Boundaries

The first version closes one complete single-tenant compilation, Mapping,
simulation, configuration, runtime, and hardware-generation path. The
following subjects are deliberately absent rather than represented by empty
schemas, dormant variants, compatibility flags, or placeholder Artifacts:

* DynamicWork channel endpoints, generic EOS or channel sessions,
  spawn-then-feed, and device-side runtime spawn;
* graph-visible fault propagation and recovery;
* DFT, ATPG, multi-power-state intent, retention, partial reconfiguration,
  fault injection, and silicon bringup;
* a stable hand-written host launch API, accelerated shared-object loading,
  remote deployment service, and distribution or installation packaging;
* direct ONNX, TOSA, Linalg, or framework-graph product boundaries;
* a generic profiler, benchmark-report Artifact, diagnostic Artifact, typed
  environment schedule, or hardware action language; and
* multi-tenancy, virtualization, migration, and runtime remapping.

Ordinary LLVM PGO, optimization remarks, debug locations, content-addressed
packages, Evaluation projections, developer tools, and provider adapters cover
their existing domains without creating new semantic owners. A deferred item
is reopened only by a concrete required behavior that cannot be composed from
the current contracts.

## Corpus

The canonical high-level-language corpus consists of the repository-owned
LoomBench applications and the complete pinned CMSIS-DSP and CMSIS-NN source
suites. Membership is derived from their manifests or pinned source trees; a
smoke subset is never an alternate corpus definition. SPEC CPU 2026 is a
separate external conformance corpus rather than part of repository-owned
membership.

The representative frontend set is owned only by
[End-To-End Conformance Anchors](spec-end-to-end-demonstrators.md). Anchor
selection does not redefine corpus membership or the product boundary.

Missing capability is reported honestly as a typed unsupported or incomplete
outcome. Scaffolds, empty artifacts, skipped work, generated wrappers, or
inventory counts cannot stand in for completed semantics.

Corpus contracts are specified by
[CMSIS Compiler Contract](spec-cmsis-dropin-compiler.md), and
[LoomBench](spec-loombench.md).

## External Dependency Pinning

CIRCT is an unmodified upstream submodule at `externals/circt`. Loom builds
exactly one LLVM source tree at `externals/llvm`; the nested CIRCT LLVM
submodule remains uninitialized. A dependency upgrade selects an exact stable
`firtool-*` release commit and atomically pins top-level CIRCT and the LLVM
commit referenced by that CIRCT revision.

## Verification Boundary

Tests protect stable semantic anchors: canonical schema and identity,
cross-artifact coupling, deterministic replay, verifier acceptance and
rejection boundaries, and representative end-to-end behavior. They do not
preserve container layout, printer whitespace, diagnostics text, current code
organization, mock infrastructure, or speculative compatibility paths.

Subsystem specs own their exact conformance anchors. Tool availability,
timeouts, host resources, and license limits are execution controls. Exhausting
them yields a typed incomplete execution outcome; it does not change the
semantic plan, prove infeasibility, or authorize best-so-far publication.
