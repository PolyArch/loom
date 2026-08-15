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

## Invocation Diagnostics

Loom has one process-wide invocation-local diagnostic verbosity binding:

```text
LOOM_VERBOSE_LEVEL = nonnegative decimal integer
```

Common owns its parsing and exposes the resulting closed level to every
in-process consumer. An unset, empty, non-decimal, or zero value selects level
zero; values above three select level three. Level `N` includes every lower
nonzero level. No subsystem independently parses this environment binding or
defines a second verbosity variable.

The level controls presentation only. It is not ResolvedConfig, semantic work,
an Artifact field, an Evaluation observation, a cache-key component, or a
persistent trace. Enabling diagnostics cannot change semantic inputs,
candidate order, random draws, external-tool inputs, termination, normalized
results, Artifact bytes, or identity. A subsystem specification owns its event
vocabulary and the detail emitted at each shared level.

Any diagnostic useful across inputs, production runs, performance analysis, or
future investigations is maintained infrastructure and uses this binding.
Narrow probes introduced only to inspect one exceptional input are not part of
the diagnostic interface and are removed when that investigation closes.

An external process or generated simulation harness cannot parse an
independent binding or accept a provider-specific verbosity option. Its owner
mechanically projects the numeric value already parsed by Common through the
external command using the same `LOOM_VERBOSE_LEVEL` spelling. Level zero is
omitted. This presentation-only projection is excluded from semantic execution
configuration and result-cache identity. An external option that can change
results is not diagnostic verbosity and must instead have a distinct typed
owner in the exact semantic execution configuration.

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

This model denotes a modern heterogeneous spatial-accelerator SoC rather than
a uniform array with a host wrapper. A conforming complete System provides
all of these architectural roles through explicit Fabric owners:

* stored-program control in the HostCore and each AccCore InstructionCore;
* heterogeneous spatial and time-multiplexed compute tiles with locally
  configured operation capability;
* distributed local memories and explicit data-movement or memory-service
  paths, so external memory is not the implicit operand store of every tile;
* programmable local interconnect inside each SpatialCore and explicit
  accelerator-side System transport among cores, services, and memory; and
* exact configuration, runtime arbitration, and progress contracts for every
  shared physical resource.

The roles are architectural, not a mandated floorplan or a requirement that
every SpatialCore contain both scheduling styles. A heterogeneous System may
combine different Module templates, and each Module may specialize its
compute, memory, and transport mixture. Hardware DSE searches compositions of
these existing typed roles; it cannot replace them with an application name,
an opaque accelerator class, or an unconstrained property bag.

Independent physical facts remain independent typed parameters. In
particular, tag width, Temporal PE instruction residency, Temporal switch
route-table depth, Temporal memory operation residency, operand-buffer depth,
register-FIFO shape, link width and lane count, and topology dimensions are
not aliases for one generic temporal scale. A template may deliberately
derive two values from one higher-level authoring choice only when its
versioned schema owns that derivation. ResolvedConfig contains the resulting
complete values, and generated Fabric remains the final hardware truth.

The same entities form two weakly coupled execution domains:

```text
host side = HostCore + host runtime and memory services
accelerator side = heterogeneous AccCore cluster + accelerator NoC
                   + accelerator memory/services
host/accelerator interface = typed service endpoints + system transport
```

For a System containing `N` AccCores, execution therefore contains `N + 1`
stored-program engines: one HostCore and one InstructionCore in each AccCore.
The HostCore is not an additional AccCore and owns no SpatialCore occurrence
binding or endpoint attachment.
It executes the residual program, runtime, and fallback work and dispatches
the exact thread and Spatial launches selected for the AccCore cluster.

The `loom.fabric 5.0` System contract requires the HostCore and every AccCore
InstructionCore in one System execution closure to belong to one compatible
RISC-V ISA and ABI cohort. They may have different Microarchitectural
Realizations, capacities,
runtime-service sets, cache attachments, and performance. Compatibility is
proved from their Fabric-owned Architectural Contracts and the selected
Compiler Target Bindings; neither a common processor name nor a gem5 model
name is an architecture authority.

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

## Joint Design Exploration

The optimization subject is a complete software workload set together with one
complete heterogeneous `fabric.system`. SpatialCore `fabric.module` candidates
are valid intermediate hardware-design inputs, but they are not releasable
System candidates by themselves. Hardware DSE may change Spatial topology and
microarchitecture, AccCore occurrence count and type, Spatial attachments,
InstructionCore realization selection, transport resources and connections,
and system memory and service attachments. Software DSE may change compiler
decisions, graph and thread partitioning, channel-versus-memory communication,
and multicast structure. SystemMapping alone owns the realized physical
AccCore targets, imported SpatialMappings, routes, services, multicast, and
ResourceUse.

Detailed portable RTL and EDA implementation remain scoped to each
SpatialCore Module. InstructionCore, host, transport, and system-memory effects
participate through exact analytic or parameter-backed Evaluation models and
gem5-backed system execution. A report must label predicted and measured
observations by their exact model descriptors; it cannot present a modeled
non-Spatial component as synthesized or physically measured.

Fast evaluation uses an explicit typed cascade: exact admission and sound
bounds, analytic estimates, parameter-backed prediction, Mapping/PnR,
functional or cycle simulation, then selected physical ground truth. Only
exact admission or a sound bound may prove infeasibility. An estimate may
rank, promote, or select samples but cannot reject a candidate as impossible.
Software and hardware candidate batches alternate in a finite resolved DSE
plan. Cross-pair evaluation occurs only through an explicit bounded frontier
join; there is no implicit Cartesian product, mutable joint candidate, or
runtime-owned best design.

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

A Temporal switch exposes three distinct facts. Its route table owns bounded
resident configuration rows selected by Physical Tag. Its input and output
service states own per-cycle transfer capacity. Its GrantPolicy owns which
eligible runtime requester receives a contended service. SpatialMapping must
fit the configured tag-route rows within resident capacity and preserve each
row's static crosspoint legality; it must not sum mutually exclusive runtime
requests as permanently concurrent occupancy merely because their routes are
simultaneously resident. Simulation and RTL apply the service capacity and
grant policy to the actual execution trace.

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
[Fabric Persistent Identity And References](spec-fabric-identity.md).
Provider-independent PnR semantics are owned by
[Place And Route](spec-pnr.md), while observable bounded-search behavior of the
in-tree implementation is owned by
[Builtin Mapping PnR Replay Profile](spec-pnr-provider-builtin.md). Native
state layout remains an implementation detail. The production
semantic-realization generator is owned by
[TechMapping Generation](spec-tech-mapping.md).
Focused Mapping specs are derived views of those owners, not parallel schemas.

## Simulation And Backend Evidence

Simulation composition has two orthogonal implementation choices but no
persistent two-dimensional mode field:

```text
SpatialEngine  = DFG | CGRA | RTL
Environment    = SpatialOnly | Gem5System
```

Each concrete combination is one registered Evaluation model descriptor with
one exact case signature and subject closure. The descriptor selects the
Spatial engine; the workload root and case signature select whether execution
is Spatial-only or System. `SimulationExecution` repeats neither choice.

The subject closures are:

```text
SpatialOnly + DFG  : Canonical Dataflow Program
SpatialOnly + CGRA : Dataflow + Fabric + complete SpatialMapping
SpatialOnly + RTL  : HardwareImplementation + Deployment
Gem5System + any engine : Deployment + Gem5SimulationBinding
```

The System variants always execute a complete Deployment and SystemMapping
closure. System + DFG idealizes only the selected SpatialCore execution; it
does not omit or idealize HostCore, InstructionCore, NoC, cache, external
memory, dispatch, or runtime execution. Adding another environment or Spatial
engine requires a registered model descriptor, not another execution schema.

DFG-sim executes canonical Dataflow semantics without Fabric resource limits.
CGRA-sim executes mapped SpatialCore behavior using exact Dataflow, Fabric,
and Mapping inputs. Both are event-driven and may emit diagnostic ordered
cycle-coordinate `SpatialDiagnosticTrace` values to attempt or scratch
storage. They have no persistent Artifact or Evidence form.

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

The initial Spatial-only RTL path is the registered
`mapped_rtl_simulation` case and `mapped_rtl_simulator` model. Deployment is
present only to select the exact SpatialCore occurrence, Mapping,
HardwareConfigurationImage, and ABI closure; the environment remains Spatial-only
because no processor, system transport, cache, or external-memory behavior is
executed.

RTL and EDA providers materialize independently executable bundles containing
exact inputs, generated drivers, frozen local tool/runtime bindings, declared
outputs, and an importer. Loom may invoke the generated top-level script, but
it does not implement the tool's environment, process-tree resource control,
container lifecycle, scheduler, or license service.

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

Machine-local tool, runtime, and external-file bindings use the separate
explicit `--loom-local-config=<path>` input. It is never implicitly loaded and
never enters ResolvedConfig or semantic Artifact identity. Exact expected file
fingerprints or tool-bundled resource identities belong to the consuming
provider binding. Tool and input resolution freezes one local projection before
a generated script runs; scripts do not rediscover tools or inputs.

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
  + object, archive-member, or inline-definition owner
  + executable call or data-use provenance
```

The source inventory consists of the LoomBench source rows and the complete
pinned CMSIS-DSP and CMSIS-NN translation-unit inventories. Membership is
derived from repository manifests or pinned source trees. Private C
implementation fragments that require an including translation unit's macro
environment are not themselves compiler invocations. Every source row must
pass ordinary drop-in compilation and, when enabled for separate compilation,
produce a valid frontend-owned relocatable accelerator payload.

A header-defined inline operator remains distinct from a source translation
unit. Its workload producer resolves one exact pinned definition file for the
typed public protocol. A selected Spatial graph covers that operator only when
the compiler-derived operation provenance names that same definition file.
The generated caller, an included umbrella header, a declaration-only file,
or a provider-supplied path alias cannot substitute for missing definition
provenance. This rule extends operator ownership without adding headers to the
translation-unit inventory.

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

The checked-in representative conformance gate selects one real producer, one
exact target profile, and one deterministic vector for each typed operator
identity. `test/data/corpus-operator-gate-v1.jsonl` records that selection and
its pinned-input provenance; `test/corpus_inventory.py` strictly imports it and
rejects stale revisions, duplicate identities, or malformed rows. For the
currently pinned inputs the gate is:

| Suite | Representative operator executions |
| --- | ---: |
| LoomBench | 135 |
| CMSIS-DSP | 571 |
| CMSIS-NN | 186 |
| Total | 892 |

Profiles, producer aliases, and additional vectors do not multiply this total.
They remain extended coverage and may reuse the same final link and DSE result.
A pinned manifest, descriptor, build, or submodule change must update the
selection and review its semantic delta before the strict importer accepts it.

A target profile is a general typed compiler and provider configuration, not a
suite exception. The repository conformance profiles cover portable scalar
code, standard floating-point extensions used by the corpus, and selected
target-specific paths whose exact compatibility remains visible. Defining a
feature macro without the corresponding compiler and provider semantics does
not establish execution coverage.

One conformance invocation selects one exact executable ISA/ABI cohort. A
workload whose target profile is compatible with that cohort must execute the
complete semantic gate and can only report pass or failure. A profile that
requires a different instruction-set family reports typed
`Unsupported(TargetProfileInstructionSetIncompatible)` before provider setup.
It is neither retargeted to a scalar implementation nor counted as a semantic
pass. An unknown profile or a missing provider within a compatible ISA family
is a conformance failure rather than this incompatibility outcome.

The gate report has three disjoint case outcomes: `pass`, `unsupported`, and
`fail`. A complete conformance invocation succeeds only when every row is a
semantic pass or has the exact profile/cohort incompatibility above, and no row
fails. The report preserves all three counts and reasons so conformance success
cannot be misread as execution of an incompatible profile. A semantic-only
claim uses the `pass` count, never the conformance total.

A smoke subset is never an alternate inventory definition. SPEC CPU 2026 is a
separate external conformance corpus rather than part of repository-owned
membership.

The representative frontend set is owned only by
[End-To-End Conformance Anchors](spec-end-to-end-demonstrators.md). Anchor
selection does not redefine corpus membership or the product boundary.

Complete multi-operation and multi-stage application conformance is a separate
derived inventory owned by the
[Real Application Portfolio](spec-application-portfolio.md). It consumes the
same compiler, Mapping, simulation, hardware, Evaluation, and failure
contracts; it does not enlarge the operator inventory or make an application
manifest into a program Artifact.

Product validation advances through one workload ladder without creating a
second workload authority:

1. the manifest-derived operator corpus supplies broad protocol coverage;
2. the ten source-backed workflows in
   [End-To-End Conformance Anchors](spec-end-to-end-demonstrators.md) supply
   bounded vertical-stack anchors;
3. the five complete programs in
   [Real Application Portfolio](spec-application-portfolio.md) supply sustained
   multi-stage and heterogeneous-system use; and
4. exact selected subsets of those owned workloads supply reproducible
   application-specific and domain-specific release claims, while the declared
   complete supported cross-domain set supplies the general release claim.

The numbers in this ladder are derived from their named owners. A runner,
dashboard, or plan cannot copy a competing inventory or treat a smoke subset
as product membership. Each later rung reuses the same source, workload,
Mapping, Deployment, Simulation, HardwareImplementation, Evaluation, and
oracle contracts rather than gaining a shallower success definition.

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

Operator-corpus contracts are specified by
[CMSIS Compiler Contract](spec-cmsis-dropin-compiler.md) and
[LoomBench](spec-loombench.md). Complete application conformance is specified
separately by the
[Real Application Portfolio](spec-application-portfolio.md).

## Initial Integration Gates

The first hardware gate requires the public ADG Builder to finalize both
user-authored regular or irregular heterogeneous multi-AccCore designs and all
initial builtin scales through the same Fabric path. The exact API,
single general-purpose template, Small/Default/Large default-scale catalog,
backend-qualification rule, and Artifact
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
required. Each profile-compatible representative vector then executes through
the semantic gate with its own limit and oracle; extended invocations may
execute further owned vectors without redefining operator membership. Mapping
is outside this gate. The harness verifies both coverage anti-joins separately
from stage success. A typed unsupported outcome remains a tool or target
limitation and is never a semantic pass.

Frontend and non-Mapping Evaluation capabilities advance together after at
least one exact builtin Fabric is available. Hardware-aware compiler decisions
must consume the shared Evaluation contracts defined by this specification;
the frontend cannot substitute a private cost model or an abstract target
summary for the exact Fabric.

## Evidence-Driven Evolution

After the first complete vertical stack is available, Loom evolves from exact
evidence rather than speculative compatibility machinery. The protected public
semantic surface consists of:

* the `loom-cc` and `loom-c++` driver contracts;
* Artifact schemas, canonical encodings, and cross-Artifact coupling;
* ResolvedConfig and ConfigurationABI semantics;
* the public ADG Builder interface;
* cross-layer Dataflow, Fabric, Mapping, HardwareImplementation, Deployment,
  Simulation, and Evaluation behavior; and
* the Spatial-only/System-with-gem5 by DFG/CGRA/RTL execution matrix.

Private C++ organization, helper APIs, internal caches, and developer-tool CLI
presentation are not public compatibility promises merely because they exist.
They may change while the protected semantic surface remains exact.

A new product feature or semantic extension requires at least one concrete
input:

* a real application with a typed unsupported outcome;
* a correctness or approved numerical-accuracy failure;
* a quality-of-result regression in exact compatible Evidence;
* a deterministic-work scaling defect;
* a wall-time or peak-resident-memory budget failure in a declared compatible
  execution context; or
* a target implementation that cannot be expressed by the existing
  Fabric/provider recipe contracts.

The first response is composition from existing owners. If composition cannot
express the required behavior, the one normative owner changes with an
appropriate schema-version change and updated rationale. A compatibility
layer, duplicate registry, local override, or benchmark-specific exception
cannot substitute for that change. The new behavior is admitted through
anchor-level TDD and the exact affected application or conformance Evidence is
rerun.

A release baseline is an exact tuple of existing identities, not a new
Artifact:

```text
source revision and external pins
+ exact application, workload, and runtime-input set
+ complete Fabric System and ResolvedConfig
+ selected software candidates, TechMappings, SpatialMappings, and SystemMappings
+ Deployment and ConfigurationABI roots
+ selected model-parameter roots and exact model bindings
+ tool and ImplementationPlatform identities
+ EvaluationEvidence and InvocationManifest references
```

There is no `ReleaseArtifact`, benchmark database, telemetry model, mutable
latest-best authority, or compatibility registry. Correctness uses exact
equality or an approved typed precision relation. Cross-machine performance
claims use deterministic work as their primary comparable quantity; wall time
and resident memory are compared only under declared compatible execution
contexts. Physical quality remains ordinary EvaluationEvidence, and direct EDA
material remains local under its disclosure boundary.

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

OR-Tools is an unmodified upstream submodule at `externals/or-tools`, pinned to
v9.15 commit `551ad10d94835c99e5e1e684500d3db398c0e345`. Loom links its
in-process C++ CP-SAT library only through the bounded exact-repair adapter
owned by [Place And Route](spec-pnr.md). It is not a plugin, external solver
binary, Python runtime dependency, or TechMapping search authority.

gem5 is an unmodified upstream submodule at `externals/gem5`, pinned to commit
`c8222cc67a399bfc01e8658dd14b30d5bfd634f9`. Loom-owned system integration
uses gem5's supported out-of-tree component and extension build mechanism.
Bridge source, typed bindings, generated configuration projections, and build
identity remain Loom-owned; Loom does not maintain a gem5 patch stack or edit
the pinned submodule source. A gem5 upgrade is a separate exact dependency
change with Runtime ABI and System simulation conformance.

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
