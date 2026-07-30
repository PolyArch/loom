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

The same entities form two weakly coupled execution domains:

```text
host side = HostCore + host runtime and memory services
accelerator side = heterogeneous AccCore cluster + accelerator NoC
                   + accelerator memory/services
host/accelerator interface = typed service endpoints + system transport
```

This partition expresses coupling and ownership, not physical packaging. It
does not add an accelerator-subsystem artifact or another topology owner. The
logical endpoints, services, connectivity, and guarantees remain in
`fabric.system`; PCIe, CXL, an SoC interconnect, or a custom link is an exact
Interconnect Implementation choice. Accelerator memory outside a
SpatialCore's local memories remains an explicit physical memory service in
the accelerator-side Fabric system description.

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

These are software Artifact boundaries, not a claim that optimization is a
one-way dependency chain. The exact Fabric target is resolved before the first
software candidate is produced, and central Evaluation may be queried at both
optimization surfaces:

```text
ADG Builder, builtin target, or user Fabric -> exact Fabric F

source -> LLVM -> initial Structured Program S0
  -> Generate/Promote(S, F, analysis or Evidence) -> optimized S
  -> mechanical lowering -> initial canonical Dataflow D0
  -> Generate/Promote(D, F, analysis or Evidence) -> optimized D
  -> Mapping(D, F) -> Mapping M
```

Evaluation is a typed side relation over exact immutable subjects rather than
a pipeline stage. Exact registered case signatures may evaluate `S`, `(S,F)`,
`D`, `(D,F)`, `F`, or `(D,F,M)`. Workload, runtime input, conditions, and model
binding remain part of the exact case contract, so analytically similar
subject sets do not collapse distinct questions.

Evidence returns through the central DSE plan. A high-fidelity iteration may
therefore follow `S_i -> D_i -> M_i -> Evidence -> S_j`, where `S_j` is a new
immutable candidate. `M_i` remains bound to the exact `D_i` and `F`; no
feedback mutates an existing software Artifact or rebinds a Mapping.

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

Fabric is compilation context, but it is not automatically a semantic
dependency of every Structured or Dataflow Artifact. Target facts that change
software semantics, including target triple, ABI, DataLayout, and address
width, are materialized in the software Artifact and therefore affect its own
identity. Resource inventory, topology, bandwidth, physical capability,
placement, and routing remain owned by Fabric, Evaluation, and Mapping. A
software-only Evaluation signature omits Fabric because that model does not
consume hardware facts, not because the compiler invocation lacks an exact
Fabric target.

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

The host-retargeted source oracle used by DFG-sim is not a sys-sim execution
path and proves no target timing or architecture property. Sys-sim executes
the exact target binaries selected by Deployment in gem5, including RISC-V
InstructionCore binaries, and invokes Loom only through the typed Spatial
Launch bridge.

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

Repository conformance has two orthogonal derived inventories. Neither is a
product Artifact or a second program authority:

```text
SourceTranslationUnitInventory
  independently invocable source translation units

OperatorWorkloadInventory
  exact typed operator protocol
  + exact target profile
  + exact producer/build variant
  + ordered WorkloadVector[]

SourceCoverageEdge
  exact link selection
  + object or archive-member owner
  + executable call or data-use provenance
```

The source inventory consists of the LoomBench source rows and the complete
pinned CMSIS-DSP and CMSIS-NN translation-unit inventories. Membership is
derived from repository manifests or pinned source trees. Private C
implementation fragments that require an including translation unit's macro
environment are not themselves compiler invocations. Every source row must
pass ordinary drop-in compilation and, when enabled for separate compilation,
produce a valid frontend-owned relocatable accelerator payload.

An operator identity is one independently meaningful typed call protocol, not
one source file, test function, or input-size combination. A stateful protocol
contains its ordered initialization, execution, and observation calls. Query,
invalid-parameter, and compute protocols remain distinct when their callable
contracts differ. A profile group is one operator identity under one exact
target profile. An operator workload is one profile group under one exact
producer/build variant and owns an ordered, nonempty vector list. Each vector
provides deterministic runtime input and a native or official reference
oracle. Vectors receive independent execution outcomes and limits but do not
repeat final link, compilation, candidate generation, or top-level DSE.

The workload inventory is derived from real program entries and public API
invocations. Its exact linker-selected objects and archive members form one
LLVM module through the final-link contract before S0, Sn, D0, D*, simulation,
or Mapping is judged. Every source row has at least one SourceCoverageEdge
under an applicable exact target profile, and every operator workload resolves
at least one complete producer. A data-only translation unit is covered
through a real consumer that selects its definitions. Alternative aggregate
and individual-source library builds are distinct producer variants and must
not be linked together.

The checked-in representative semantic gate selects one real producer, one
applicable target profile, and one deterministic vector for each typed operator
identity. `test/data/corpus-operator-gate-v1.jsonl` records that selection and
its pinned-input provenance; `test/corpus_inventory.py` strictly imports it and
rejects stale revisions, duplicate identities, or malformed rows. For the
currently pinned inputs the gate is:

| Suite | Representative operator executions |
| --- | ---: |
| LoomBench | 132 |
| CMSIS-DSP | 571 |
| CMSIS-NN | 186 |
| Total | 889 |

Profiles, producer aliases, and additional vectors do not multiply this total.
They remain extended coverage and may reuse the same final link and DSE result.
A pinned manifest, descriptor, build, or submodule change must update the
selection and review its semantic delta before the strict importer accepts it.

A target profile is a general typed compiler and provider configuration, not a
suite exception. The repository conformance profiles cover portable scalar
code, standard floating-point extensions used by the corpus, and any
target-specific path for which Loom has an exact semantic provider. Defining a
feature macro without the corresponding compiler and provider semantics does
not establish coverage.

A smoke subset is never an alternate inventory definition. SPEC CPU 2026 is a
separate external conformance corpus rather than part of repository-owned
membership.

The representative frontend set is owned only by
[End-To-End Conformance Anchors](spec-end-to-end-demonstrators.md). Anchor
selection does not redefine corpus membership or the product boundary.

Missing capability is reported honestly as a typed unsupported or incomplete
outcome. Scaffolds, empty artifacts, skipped work, generated wrappers, or
inventory counts cannot stand in for completed semantics.

All three repository corpus owners exercise one compiler product contract.
LoomBench, CMSIS-DSP, and CMSIS-NN may use different source manifests,
toolchain flags, runtime oracles, and fast regression selections, but no suite
has a shallower permanent compiler boundary. Source-stage requests apply to
selected source rows. Whole-program and later-stage requests apply to selected
operator workload rows after final link, with each owned vector evaluated
independently. Both use the same public driver and in-process stage libraries,
and the same validity and failure contracts apply.

Stage checkpoints are diagnostic and regression boundaries, not suite
capabilities. A checkpoint can stop after ordinary execution, Structured
Program finalization, Canonical Dataflow finalization, Mapping, simulation, or
hardware implementation to localize a failure. It cannot justify treating a
deeper stage as unnecessary for one suite. A complete-stage corpus gate applies
that stage uniformly to the requested inventory; a failure identifies a tool,
provider, target, or program-semantics limitation rather than a suite-specific
success rule.

Canonical Dataflow publication is a whole-program result and is never required
from an isolated compile-only translation unit. It may contain no Spatial graph
only when the exact linked program, target profile, Fabric, and workload admit
no selected profitable Spatial region. Such a graph-free program is different
from an empty placeholder: it retains the complete stored-program semantics,
passes the same finalizer and importer, and has a complete candidate-domain
accounting. Non-finalizable and exact-Fabric-inadmissible candidates retain
their typed compiler dispositions in invocation provenance. A legal HostCore
or InstructionCore selection is a `CandidateDecision`; a profitability choice
is backed by workload-aware `EvaluationEvidence`. Missing candidate
dispositions or unknown workload behavior cannot prove graph-free legality.
These existing owners remain authoritative; there is no graph-free Artifact,
status ledger, or duplicate diagnostic schema.

Corpus contracts are specified by
[CMSIS Compiler Contract](spec-cmsis-dropin-compiler.md), and
[LoomBench](spec-loombench.md).

## Initial Integration Gates

The first hardware gate requires the public ADG Builder to finalize both
user-authored regular or irregular heterogeneous multi-AccCore designs and all
initial builtin presets through the same Fabric path. The exact API,
Small/Default/Large catalog, backend-qualification rule, and Artifact
publication contract remain owned by [ADG Builder](spec-adg-builder.md) and
[Fabric Artifact](spec-fabric-artifact.md). Passing this gate produces exact
Fabric Artifacts plus their human-readable MLIR and HTML projections without
requiring a software input or an RTL provider.

The next product gate has two jointly required parts. The source gate compiles
every selected source row through the ordinary object and relocatable-payload
boundary. The pre-Mapping workload gate final-links every selected operator
workload once, then uses the same `loom-cc` or `loom-c++` contract to produce
LLVM IR, Structured candidates, and a finalized Canonical Dataflow Program
while resolving one exact Fabric target and using central Evaluation where
required. The representative vector then executes through the semantic gate
with its own limit and oracle; extended invocations may execute further owned
vectors without redefining operator membership. Mapping is outside this gate.
The harness verifies both coverage anti-joins separately from stage success. A
typed unsupported outcome remains a tool or target limitation; it is not
success and cannot be made suite-specific.

Frontend and non-Mapping Evaluation capabilities advance together after at
least one exact builtin Fabric is available. Hardware-aware compiler decisions
must consume the shared Evaluation contracts defined by this specification;
the frontend cannot substitute a private cost model or an abstract target
summary for the exact Fabric.

## External Dependency Pinning

CIRCT is an unmodified upstream submodule at `externals/circt`. Loom builds
exactly one LLVM source tree at `externals/llvm`; the nested CIRCT LLVM
submodule remains uninitialized. A dependency upgrade selects an exact stable
`firtool-*` release commit and atomically pins top-level CIRCT and the LLVM
commit referenced by that CIRCT revision.

Unity is an unmodified test-workload dependency at `externals/unity`, pinned to
the revision selected by the CMSIS-NN validation owner. It supplies the real
CMSIS-NN assertion runtime and runner ABI only; it is not a Loom product
runtime, compiler semantic registry, or distributed dependency of generated
accelerator programs.

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
