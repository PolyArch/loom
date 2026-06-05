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
`dataflow.demux`. These are the semantic base for DFG simulation and for
later hardware mapping.

## Core Dialect Boundary

The target ownership boundary between dataflow software semantics,
fabric hardware structure, mapping artifacts, runtime execution, and
simulation evidence is specified in
`docs/spec-core-dialect-boundary.md`. That boundary is authoritative
when a fact could otherwise be assigned to more than one subsystem.

## Hardware Model

The target machine model is:

```text
System = HostCore + AccCore x M + memory hierarchy + interconnect
AccCore = ScalarCore + SpatialCore
SpatialCore = CGRA-like fabric described through fabric IR
```

AccCores may be heterogeneous. The system must not assume that all
AccCores share the same ScalarCore, SpatialCore, memory attachment, or
interconnect cost.

Arbitrary topology is the default. Meshes, arrays, x/y coordinates, and
Manhattan-distance routing are optional conveniences supplied by an
architecture builder or by user metadata. They are never the baseline
semantic assumption. Hardware connectivity is represented as graph
connectivity through explicit links, directed channel endpoints, and
connectivity tables or matrices.

## Software Representation

The software side lowers selected LLVM/SCF regions into dataflow IR. The
compiler owns three placement problems:

| Name | Boundary | Meaning |
|------|----------|---------|
| L1 accelerator placement | HostCore vs AccCore | Select which program regions execute on the accelerator fabric. |
| L2 graph placement | ScalarCore vs SpatialCore | Select which code inside an accelerator kernel becomes a SpatialCore dataflow graph. |
| L3 FU placement | Spatial graph vs fabric FU template | Partition a software graph into subgraphs that can map to function-unit templates. |

All three are optimization problems. A deterministic baseline policy is
allowed and useful for tests, but fixed syntactic lowering is not the
final design.

`dataflow.thread` is a software execution-domain carrier. A dynamic
thread instance is a logical execution cell until later binding maps it
onto physical AccCore resources. The accepted direction is a two-level
model: front-end IR preserves logical parallel structure, while PnR or a
binding artifact assigns selected innermost executable instances to
AccCore execution slots.

An innermost executable thread is a `dataflow.thread` whose body, at the
thread-body placement level, does not launch another `dataflow.thread`.
It may contain ScalarCore residual code and `dataflow.graph.launch` ops.
Only dynamic instances of such threads are eligible to become one
AccCore execution slot after binding. Non-innermost threads remain
logical hierarchy and scheduling structure. Before binding, hierarchy
transforms may reorder independent thread levels, collapse adjacent
independent levels, or tile and split levels only when they preserve the
logical instance set, per-instance scalar values, memory-order
constraints, async launch/fence ordering, and strict thread/graph
layering. The deterministic baseline policy stops at annotation and
canonicalization. Nontrivial hierarchy transforms are explicit
optimization policies, not verifier side effects, and must be enabled
through documented placement policies.

Thread nesting is strictly layered. A non-innermost thread may contain
ScalarCore orchestration code and child `dataflow.thread.launch` ops,
but it must not directly contain `dataflow.graph.launch` ops. An
innermost executable thread may contain ScalarCore residual code and
`dataflow.graph.launch` ops, but must not directly contain child
`dataflow.thread.launch` ops. A single thread-body placement level must
never directly mix thread launches and graph launches. A ScalarCore-only
thread body with neither launch shape is legal and is an innermost
scalar-only AccCore binding candidate. This legality is not an implicit
offload decision: a scalar-only thread is retained as AccCore work only
when L1 placement, source intent, or an explicit DSE policy selected that
region for accelerator execution. Failed L2 graph extraction must not
create a new accelerator offload by itself.

Thread completion and dataflow control are separate token domains.
`!dataflow.thread_token` represents inter-thread asynchronous
completion from `dataflow.thread.launch`. `none`-typed control values
represent graph launch control, graph completion, streaming control, and
memory-order tokens inside dataflow. There is no implicit cast between
these domains. `dataflow.thread.fence` is the explicit bridge from
thread completion and graph-control dependencies to a `none` control
result; `dataflow.thread.wait` consumes thread completion tokens for
host or parent-context synchronization and produces no graph-control
value.

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
only through `dataflow.graph.launch` inside an innermost executable
thread body. The target dataflow dialect has no separate
`dataflow.graph.func` surface.

`dataflow.subgraph` represents an L3 software partition inside a
`dataflow.graph` definition. It is a candidate unit for matching or
generalizing against `fabric.fu` templates. It is not a hardware
hierarchy node, PE, route, schedule, tag, or time-sharing statement.
The dataflow dialect does not attach spatial / temporal schedule
semantics to subgraphs; resource sharing and tags are introduced by
binding, PnR, or fabric-side IR.

## Hardware Representation

The fabric dialect is the hardware-side representation for both CGRA
SpatialCore templates and system-level architecture graphs.
`fabric.module` remains the SpatialCore or CGRA fabric template.
Existing concepts such as `fabric.pe`, `fabric.fu`, `fabric.switch`,
`fabric.mem`, `fabric.fifo`, `fabric.boundary`, and
`fabric.instantiate` form the starting point for SpatialCore
descriptions.

`fabric.system` is the system-level architecture description graph for
`HostCore + AccCore x M` systems, memory hierarchy, external memory,
and interconnect. It contains physical nodes, protocol ports, directed
channels, explicit one-to-one links, optional domain metadata, and
coherence or consistency declarations. An `acc_core` system node
references a `fabric.module` symbol as its SpatialCore template while
remaining an independent physical instance.

System topology is explicit graph connectivity. The system ADG does not
introduce hardware primitives named `mux` or `demux`; system-level
selection and routing use precise primitive node kinds such as
`route_decoder`, `arbiter`, and `broadcast`. The detailed target
contract is in `docs/spec-fabric-system-adg.md`.

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
* CGRA-sim simulates mapped software on a concrete hardware graph with
  resource, routing, memory, buffering, and temporal-sharing limits.
  Despite the name, CGRA-sim is hardware-aware simulation for mapped
  Loom workloads, not only simulation of a `fabric.module` or
  SpatialCore. Its target contract is specified in
  `docs/spec-sim-cgra.md`.

PnR connects the two sides. It takes software dataflow IR plus hardware
fabric/ADG IR and emits the independent mapping artifact specified in
`docs/spec-mapping-artifact.md`; the PnR tool contract is specified in
`docs/spec-pnr.md`. The artifact records placed software nodes, routed
edges, memory bindings, resource sharing, buffers, schedule slots,
temporal tags, diagnostics, and metrics required by the selected
hardware mapping. Detailed mapping identity, placement, routing,
schedule/buffer, memory, verification, visualization, and search
contracts are specified by `docs/spec-mapping-identity.md`,
`docs/spec-mapping-placement.md`, `docs/spec-mapping-routing.md`,
`docs/spec-mapping-schedule-buffer.md`, `docs/spec-mapping-memory.md`,
`docs/spec-mapping-verification.md`,
`docs/spec-mapping-visualization.md`, and
`docs/spec-mapping-search.md`.

PnR chooses and records a mapping. CGRA-sim consumes that mapping plus
runtime inputs and reports hardware-aware behavior. CGRA-sim may reject
an inconsistent mapping artifact, but it must not choose placements,
routes, schedules, or bindings.

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
Intermediate artifact gate schemas for mid-run evidence, summary
exports, and content audits are specified in
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
