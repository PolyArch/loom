# Loom Full-Stack Architecture

This document records the accepted top-level target architecture for
Loom. Subsystem documents own the detailed contracts for individual
dialects and tools. This document is intentionally limited to invariants
that have been accepted across the whole stack.

## Scope

Loom is a compiler and architecture exploration stack for heterogeneous
spatial acceleration. Its source-facing contract starts from LLVM IR.
The first production language surface is C and C++ through `loom-cc` and
`loom-c++`, with the long-term requirement that any language capable of
emitting the supported LLVM IR contract can participate.

Loom's near-term product target is a drop-in CMSIS compiler path. For
CMSIS-DSP and CMSIS-NN, Loom should eventually be usable where a user
would otherwise use `gcc` or `g++`, while still producing Loom's internal
dataflow representation, hardware mapping artifacts, and performance
reports when acceleration is enabled. The target drop-in compiler
contract is specified in `docs/spec-cmsis-dropin-compiler.md`.

Loom also has a self-contained application test corpus under
`test/app`. These apps are repository-owned C/C++ programs that validate
the same drop-in compiler principle without depending on external CMSIS
source trees. The target corpus contract is specified in
`docs/spec-app-dropin-test-corpus.md`.

## Design Principles

The IR design follows a RISC-style principle. Prefer simple, atomic
operations whose behavior is easy to simulate, verify, and map. A complex
instruction should exist only when it expresses a first-principles concept
that cannot be cleanly composed from simpler operations.

This simplicity is semantic simplicity, not reduced product scope. It
means that the core abstractions should be first-principles concepts with
clear composition rules. It does not mean the implementation should be a
small prototype or a minimal feature subset. A complete Loom stack may be
large and systematic while still preserving a small set of coherent
semantic building blocks.

Accepted stable dataflow primitives include stream shaping, control
routing, explicit memory access, and synchronization primitives such as
`dataflow.stream`, `dataflow.carry`, `dataflow.invariant`,
`dataflow.gate`, `dataflow.constant`, `dataflow.load`,
`dataflow.store`, `dataflow.sync`, `dataflow.mux`, and
`dataflow.demux`. The vector boundary primitives
`dataflow.parallelize`, `dataflow.pack`, `dataflow.unpack`, and
`dataflow.serialize` adapt scalar streams to canonical fixed-size rank-1
MLIR vectors, explicit vector masks, and packed bit representations; their
SSOT is `docs/spec-dataflow-vectorization.md`. These are the semantic base
for DFG simulation and for later hardware mapping.

## CIRCT And LLVM Dependency Pinning

CIRCT is an unmodified upstream submodule at `externals/circt`. Loom
maintains and builds exactly one LLVM source tree at `externals/llvm`;
the nested `externals/circt/llvm` submodule must remain uninitialized.

An explicit dependency upgrade resolves the latest stable, non-draft,
non-prerelease `firtool-*` release to an exact CIRCT commit, reads that
commit's `llvm` gitlink, and updates the top-level CIRCT and LLVM
gitlinks atomically. Ordinary checkout, configure, doctor, and build
operations consume those recorded commits without resolving floating
tags or branches and without network access.

For every invoking superproject worktree, the shared CIRCT and LLVM
checkouts under the main worktree must match that invoking worktree's
top-level `HEAD` gitlinks. Unmerged dependency gitlinks in the invoking
worktree index require explicit manual resolution. The expected CIRCT
commit's `llvm` gitlink must equal the top-level LLVM gitlink. Both shared
dependency repositories must have clean tracked index and worktree state;
untracked build outputs do not affect the dependency contract. Global
recursive submodule updates are not permitted because they initialize the
nested LLVM checkout. Doctor and build entry points must reject checkout
drift, inconsistent parent gitlinks, tracked dependency modifications, or
an initialized nested LLVM checkout with diagnostics that identify the
violated invariant and the corresponding repair.

CIRCT configuration uses `-DMLIR_DIR` and `-DLLVM_DIR` from the shared
top-level LLVM build. That build has one deterministic identity containing
the validated CIRCT commit, LLVM commit, exact LLVM C and C++ compiler
identities, and the canonical semantic LLVM CMake arguments. The same
canonical argument list configures LLVM and contributes to the identity.
Dependency state and compiler identities are derived while holding the
shared LLVM build lock. An absent, legacy, malformed, or changed identity
invalidates the stamp and requires removal and reconstruction of the LLVM
build directory before reuse.

## Worktree External Ownership

The primary Git worktree is the sole owner of initialized top-level
submodule checkouts. A linked worktree must leave every submodule path
uninitialized and must not retain worktree-local submodule administrative
state. It consumes external sources through the primary worktree instead.
This rule applies uniformly to LLVM, CIRCT, CMSIS, gem5, and future external
dependencies; dependency-specific build code must not create a competing
source checkout policy.

`scripts/make-worktree.py` is the SSOT for resolving the primary worktree and
its `externals` directory. Build and test consumers that need external source
paths query that dispatcher rather than assuming that `externals` belongs to
the invoking worktree. When a linked branch pins a different submodule commit,
the consumer must reject the mismatch with the shared checkout instead of
silently using the wrong source revision. A consumer must also reject tracked
modifications in each shared submodule it reads; untracked build products do
not change the source identity.

External build products are shared only when their owning build dispatcher
defines a shared location under the primary worktree. LLVM currently follows
this model. Loom's own build directory remains local to each worktree because
it contains outputs derived from that worktree's source revision. Linked
worktree cleanup must preserve shared external sources and build products.
Doctor and build entry points reject accidental linked-worktree submodule
initialization before consuming external sources.

## Core Dialect Boundary

The target ownership boundary between dataflow software semantics,
fabric hardware structure, mapping artifacts, runtime execution, and
simulation evidence is specified in
`docs/spec-core-dialect-boundary.md`. That boundary is authoritative
when a fact could otherwise be assigned to more than one subsystem.

## Contract Map

Loom specs are the natural-language target corpus for the project. They
describe final behavior, evidence, and verification surfaces. They may
lead the current implementation. Local ignored execution guides may describe
current repository work, but they are not target specs.

The highest-priority shared contracts are:

* `dataflow` MLIR for software dataflow semantics;
* `fabric` MLIR for SpatialCore and system hardware semantics.

Every compiler, mapping, simulator, runtime, RTL, FPA, reporting, or DSE
component must preserve those contracts instead of creating a private
side channel. Tool logs, temporary CSV files, generated summaries, and
builder-only state are not semantic sources of truth.

The second-layer stable artifact contracts are:

* complete Mapping artifacts;
* normalized FPA JSON reports;
* full-stack report bundles;
* full-stack artifact manifests.

CSV files, tables, dashboards, and visualization files are projections
or exports. They may help humans inspect a run, but program-to-program
exchange should use MLIR or JSON contracts unless a spec explicitly
declares another structured format for that boundary.

The spec corpus itself follows the same source-of-truth rule. A common
policy should be defined once in its owning spec and referenced by other
specs. Component specs may repeat a short local consequence only when
the local consequence is component-specific and the owning spec remains
the authority.

Configuration follows the same source-of-truth rule.
`docs/spec-config-ssot.md` owns the global contract for loadable
configuration, centralized defaults, strict resolution, early-fail
behavior, ResolvedConfig artifact identity, and the boundary between
configurable parameters and semantic verifier constants. Component specs
may reference configuration views, but they must not define independent
defaults for the same canonical parameter.

## Target Universe

The target Loom product is a complete, robust, modular, flexible,
high-performance full-stack framework for multi-core heterogeneous
spatial accelerators. The target universe includes:

* C and C++ drop-in compilation through `loom-cc` and `loom-c++`;
* the canonical `loombench`, `cmsis-dsp`, and `cmsis-nn` suites;
* architecture-aware compiler strategies including graph partitioning,
  fusion, tiling, memory placement, and operator specialization;
* SpatialCore construction through `fabric.module`;
* system-level SoC construction through `fabric.system`;
* mapping artifacts, PnR legality, routing, scheduling, memory binding,
  and mapping search;
* DFG-sim, CGRA-sim, and simulation comparison;
* architecture RTL, mapped-workload RTL metadata, EDA profiles, and FPA
  evidence;
* runtime ABI, report bundles, artifact manifests, and DSE feedback;
* a configuration SSOT that exposes DSE weights, preset profiles,
  simulator model constants, mapping search policies, tech-mapping
  parameters, ADG Builder recipe defaults, and backend profile selection
  through one resolved configuration model.

Membership in the three source suites is derived by
`test/corpus_inventory.py`: LoomBench from `test/app/manifest.json`, and both
CMSIS suites from tracked `Source/**/*.c` files in their pinned submodules.
Small compiler or hardware integration selections are explicit smoke subsets,
not alternate suite definitions.

All workload families target the full validation ladder from native or
drop-in compile/run through RTL/FPA evidence. A missing intermediate or
backend capability must be represented by a structured unsupported-scope
or diagnostic record rather than a silent skip.

The default development method is balanced advancement around the common
contracts. A narrow change may touch one component, but it must provide
the producer, consumer, and audit evidence needed to prevent local-only
success. Larger milestones must advance hardware, compiler, mapping,
simulation, RTL/FPA, and DSE/reporting evidence together.

## Required Evidence

Evidence that counts toward target completion must be explicit, finalized
with an ArtifactIdentity when persistent, and consumable by downstream
tools without reading private state. Required evidence classes include:

* source, LLVM IR, raised MLIR, and dataflow artifacts for supported
  workloads;
* Fabric ADG artifacts for SpatialCore modules and system SoCs;
* complete Mapping artifacts for mapped candidates;
* DFG-sim, CGRA-sim, and comparison reports with functional output and
  memory-diff evidence for passing runs;
* runtime packages and runtime reports when the runtime boundary is
  exercised;
* RTL manifests, EDA reports, normalized FPA JSON reports, and derived
  metric records;
* full-stack report bundles and artifact manifests for traceability;
* DSE candidate, objective, selection, and rejection records;
* the exact resolved configuration ArtifactIdentity for every configured
  artifact.

Fake or stub artifacts must not satisfy any target requirement. Scaffold
artifacts may exist while a component is being built, but they must be
explicitly marked and must not be counted as target completion. Analytic
FPA evidence must remain labeled as analytic and must not be presented as
backend evidence.

## Objective Verification

Each component spec owns its detailed verification surface. The global
verification policy is:

* each attempted workload row must report its own outcome; inventory
  enumeration alone does not assign pipeline status;
* each target boundary must have positive and negative tests or audit
  checks;
* every machine-consumed report must identify schema version, producer,
  input artifact identities, and diagnostics;
* each DSE candidate must be immutable and must carry exact artifact
  identities for its inputs and outputs;
* every configured artifact must carry the resolved configuration
  ArtifactIdentity required by `docs/spec-config-ssot.md`;
* cross-artifact contradictions must block acceptance rather than being
  reduced to warnings.

Verification commands in execution guides must declare timeouts. Default
timeouts are 120 seconds for focused or unit checks, 900 seconds for
integration and artifact-chain checks, and 7200 seconds for EDA or other
long-running checks. A timeout is a failure or blocked condition, not an
implicit skip.

## Unsupported Scope Policy

Unsupported scope is allowed only when it is explicit, structured, and
auditable. A record must name the case, component, missing capability,
owner category, selected profile, and diagnostic class. Optional stages
may be unsupported for a candidate, but a required stage cannot be
treated as passing through a missing file, empty corpus, default zero
metric, or uninspected scaffold output.

## Relationships To Other Contracts

Detailed contracts live in the component specs. This document is the
index that ties them together. When two specs disagree, resolve the
conflict by preserving the ownership boundary in
`docs/spec-core-dialect-boundary.md`, then update the affected component
specs so the corpus is internally consistent.

When two components describe the same parameter, resolve the conflict by
moving the parameter to the configuration SSOT in
`docs/spec-config-ssot.md` or by marking it as a semantic invariant in
the owning component spec. Repeating a parameter as a local default in
multiple components is not an acceptable target state.

## Current Implementation Notes

This section is non-normative. It records current repository facts for
orientation only and is not part of target acceptance. Migration routes
belong in local ignored execution guides.

The current repository already contains substantial dataflow, fabric,
mapping, simulator, report, and audit scaffolding, but implementation
coverage is uneven. `fabric.module` and several SpatialCore-level Fabric
operations are implemented. The current `fabric.system` and `SystemBuilder`
path still uses generic node kinds, open parameters, and link records. That
path is runnable baseline plumbing only and is not the typed system target.

## Hardware Model

The target machine model is:

```text
System = HostCore + heterogeneous AccCores + memory/services + transport
AccCore = InstructionCore + SpatialCore
SpatialCore = CGRA-like fabric described by fabric.module
```

`fabric.system` owns typed AccCore, InstructionCore, SpatialCore attachment,
memory/service, Transport Architecture, Interconnect Implementation, external
boundary, and domain facts. Exact typed operation and attribute schemas that
remain open are not replaced by generic node kinds or parameter dictionaries.
The current generic `Fabric_NodeOp` path is non-normative and remains only
until a complete typed replacement can migrate all producers and consumers in
one change.

Loom does not model virtual addresses in Fabric ADG. System memory
semantics start from physical address spaces, memory regions,
DMA/scratchpad resources, coherent domains, and shared-memory
coherence. The final target includes cache hierarchy and coherence
protocol evidence for multi-core heterogeneous SoCs.

Arbitrary topology is the default. Meshes, arrays, x/y coordinates, and
Manhattan-distance routing are optional conveniences supplied by an
architecture builder or by user metadata. They are never the baseline
semantic assumption. `fabric.module` connectivity is represented by
Graph-region SSA values plus tile connectivity tables or matrices where
those tile specs define them. `fabric.system` connectivity is represented by
fully elaborated typed endpoints, resources, and explicit directed
connections. Protocol-specific ports and links belong to Interconnect
Implementation refinement rather than replacing architecture-level transport
capability.

## Software Representation

The software and Mapping flow has four explicit owners:

* AccCore outlining selects structured regions that become
  `dataflow.thread` definitions;
* SpatialCore outlining selects compiler-internal `loom.spatial_region`
  boundaries and lowers them to canonical `dataflow.graph` definitions and
  launches;
* TechMapping selects Compute and Memory Realizations and exact Fabric
  semantic encodings; and
* Spatial PnR consumes immutable TechMapping and selects concrete SpatialCore
  realization in a complete SpatialMapping.

These are not instances of one generic placement framework. The Structured
Program Candidate owns software ownership decisions, Mapping verification owns
legality, Evaluation owns observations, and central DSE owns candidate
acceptance.

`dataflow.thread` is a software execution-domain carrier. A dynamic
thread instance is a logical execution cell until later binding maps it
onto physical AccCore resources. The accepted direction is a two-level
model: front-end IR preserves logical parallel structure, while SystemMapping
binds selected logical instance domains to AccCore resource-time relations. A
thread body may contain InstructionCore code and
`dataflow.graph.launch` ops, but `dataflow.thread.launch` appears only
in caller-side host/runtime orchestration outside every thread or graph
definition. An InstructionCore-only thread body is legal. This legality is not
an implicit fallback: it must be selected by the Structured Program Candidate
and covered by SystemMapping. Failed graph extraction must not create a new
AccCore disposition or silently redirect a selected SpatialCore graph.

Thread completion and dataflow control are separate token domains.
`!dataflow.thread_token` represents inter-thread asynchronous
completion from `dataflow.thread.launch`. `none`-typed control values
represent graph launch control, graph completion, streaming control, and
memory-order tokens inside dataflow. There is no implicit cast or bridge
between these domains. `dataflow.thread.wait` consumes one or more
thread completion tokens for caller-side causal synchronization and
produces no graph-control value or memory barrier.

Thread grid mapping uses domain-neutral logical-axis attributes. The
target spelling is `#loom.thread_axis<parallel, axis>` or
`#loom.thread_axis<multiplexed, axis>`, with an optional logical-domain
symbol as the third parameter. `parallel` means distinct dynamic values
along that logical axis may be bound to distinct AccCore slots and run
concurrently when resources and policy allow it. `multiplexed` means
distinct dynamic values along that logical axis may reuse an AccCore
slot through time multiplexing. Neither kind is a hardware coordinate,
x/y axis, mesh coordinate, PE coordinate, route, or topology statement.
Physical core selection, reuse order, routing, and resource arbitration
belong to binding/PnR artifacts.

`dataflow.graph` represents SpatialCore software dataflow. The target
form is a symbol-bearing, module-scope callable definition. It executes
only through `dataflow.graph.launch` inside a `dataflow.thread` body.
There is no regional graph form or separate graph callable surface.

TechMapping owns actor grouping, selected FU encodings, actor/op
correspondence, and boundary-port correspondence without persisting a second
software graph. Resource sharing and tags remain Mapping, PnR, or Fabric facts.

## Hardware Representation

The fabric dialect is the hardware-side representation for both CGRA
SpatialCore templates and system-level architecture graphs.
`fabric.module` remains the SpatialCore or CGRA fabric template.
The core SpatialCore tile matrix is `fabric.{pe,switch,mem}` crossed
with `[spatial|temporal]` schedules. A `fabric.module` body is a graph
region whose connectivity is represented by SSA values. It does not use
`fabric.link` for internal connectivity. `fabric.fu` is a functional
unit container inside `fabric.pe`; it is not a module-level tile
parallel to PE, switch, or memory. `fabric.fifo`, `fabric.boundary`, and
`fabric.instantiate` are required SpatialCore support constructs for
buffering, spatial/temporal boundary conversion, and template reuse.

`fabric.system` owns the typed system-level hardware description for
HostCore and AccCore occurrences, their node-local InstructionCore facts,
SpatialCore template occurrences and attachments, memory and service
capabilities, Transport Architecture, Interconnect Implementation, external
boundaries, and hardware domains.

System topology is fully elaborated and explicit through typed resources,
endpoints, and directed connectivity. Selection, replication, arbitration,
transport, and implementation refinement are represented by their owning
typed Fabric concepts rather than generic node kinds or a prescribed link
operation. Exact system operations, attributes, endpoint records, and
assembly syntax remain open until the typed `fabric.system` schema is closed.
The detailed ownership contract is in `docs/spec-fabric-system-adg.md`.

An ergonomic C++ ADG Builder is required. It should let users construct
heterogeneous systems and arbitrary-topology fabrics quickly, then emit
MLIR hardware descriptions suitable for mapping, simulation, and RTL or
estimation flows. Its target API contract is in
`docs/spec-adg-builder.md`; it is a construction frontend and must emit
explicit Fabric ADG rather than preserve separate builder-only hardware
semantics.

## Mapping and Simulation

Loom needs two simulation levels:

* DFG-sim simulates pure dataflow software semantics without hardware
  resource limits. Its results are expected to be optimistic. Its
  target contract is specified in `docs/spec-sim-dfg.md`.
* CGRA-sim simulates one mapped SpatialCore execution on its concrete
  `fabric.module` resources with routing, memory, buffering, and
  resource-time limits. InstructionCore, cache, coherence, and system
  interconnect execution belong to the external system simulator. The
  SpatialCore simulation contract is specified in `docs/spec-sim-cgra.md`.

TechMapping connects canonical software semantics to Fabric capability by
recording complete Compute Realizations in the Mapping Artifact specified in
`docs/spec-mapping-artifact.md`. PnR consumes that predecessor as `T` in the
exact `D/T/F/C/K` authority boundary. Its tool contract is specified in
`docs/spec-pnr.md`. A successful run produces a
complete `SpatialMapping` that references the exact TechMapping predecessor
without restating or changing its Compute or Memory Realizations. Exact
persistent SpatialMapping bindings, Route Trees, resource use, Physical Tags,
buffers, memory/service choices, and mapping-visible configuration remain
open. Mapping diagnostics and metrics are ordinary reports or Evaluation
Evidence, not Mapping records. Detailed
mapping identity, placement, routing, schedule/buffer, memory, verification,
visualization, and search contracts are specified by
`docs/spec-mapping-identity.md`,
`docs/spec-mapping-placement.md`, `docs/spec-mapping-routing.md`,
`docs/spec-mapping-schedule-buffer.md`, `docs/spec-mapping-memory.md`,
`docs/spec-mapping-verification.md`,
`docs/spec-mapping-visualization.md`, and
`docs/spec-mapping-search.md`.

For each accepted run, PnR emits one complete SpatialMapping bound to its exact
immutable TechMapping predecessor. CGRA-sim consumes that complete
SpatialMapping, the exact predecessor coupling, and runtime inputs to report
hardware-aware behavior. CGRA-sim may reject inconsistent inputs, but it must
not choose or repair placements, Route Trees, resource use, Physical Tags,
buffers, memory/service choices, or configuration.

Mapping artifacts and Fabric ADG may carry optional visualization
metadata. Visualization metadata helps GUI tools draw regular
topologies such as two-dimensional meshes or three-dimensional grids
and overlay software-to-hardware mappings. It must not affect software
semantics, hardware semantics, PnR legality, simulation, RTL lowering,
or estimation.

DFG-sim and CGRA-sim results must be compared for the same workload and
input data. Differences are acceptable only when they are explained by
hardware constraints that DFG-sim intentionally ignores. The comparison
contract is specified in `docs/spec-sim-comparison.md`.

The cross-tool fidelity ladder is specified in
`docs/spec-fidelity-ladder.md`. Full-stack reporting, including
cycle/frequency/power/area derived metrics and compact simulator cycle
summary exports, is specified in
`docs/spec-full-stack-reporting.md`. End-to-end demonstrator
requirements are specified in
`docs/spec-end-to-end-demonstrators.md`.
Current mapping and simulation artifact formats are described in
`docs/spec-intermediate-artifacts.md`.

## Runtime ABI

The runtime ABI connects compiled host code, accelerator work packages,
mapping artifacts, memory descriptors, simulator hooks, and hardware
targets. It does not redefine dataflow or fabric semantics. Runtime
launch handles are host-visible dynamic execution handles; they are not
`!dataflow.thread_token` values and are not `none`-typed dataflow
control tokens.

The target runtime contract is specified in
`docs/spec-runtime-abi.md`.

## RTL and Estimation

Fabric hardware descriptions must eventually lower to synthesizable and
simulatable SystemVerilog. The RTL path must support fast sanity checks
and a higher-fidelity evaluation path. Frequency, power, and area
estimation should be combined with CGRA-sim cycle counts to form a
cycle-frequency-power-area feedback loop for software and hardware
design-space exploration.

The target RTL lowering contract is specified in
`docs/spec-rtl-lowering.md`. The target frequency, power, and area
contract is specified in `docs/spec-fpa-estimation.md`. Portable EDA
tool discovery and backend profile rules are specified in
`docs/spec-eda-tooling.md`.

Full-stack artifact traceability from source input through compiler,
dataflow, hardware, mapping, simulation, runtime, RTL, EDA, FPA,
reporting, and DSE is specified in
`docs/spec-full-stack-traceability.md`. DSE feedback records and
candidate-generation boundaries are specified in
`docs/spec-dse-feedback.md`.

Public Loom specs describe tool classes, profile contracts, artifacts,
and reports. Site-specific activation commands, install paths, license
details, and private library paths belong in local ignored
configuration or temporary execution guides, not in public project
specs.
