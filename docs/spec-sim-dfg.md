# DFG-sim

This document owns the execution contract for Loom's hardware-unaware
Canonical Dataflow Program simulator. Persistent Evaluation schemas are owned
by [DSE and Evaluation](spec-dse-feedback.md); this document defines only the
model-specific subject, behavior, and observations. Shared workload, runtime
input, execution, activity, future trace, and terminal schemas are owned by
[Simulation Artifacts](spec-simulation-artifacts.md).

## Purpose And Boundary

DFG-sim answers:

```text
How does this Canonical Dataflow Program execute when operation timing is
modeled but spatial hardware resources are unlimited?
```

The descriptor references the shared DFG-simulation case signature, whose one
required role is `program: CanonicalDataflowProgram`, bound to finalized `D`.
`workload_ref` and `runtime_input_ref` bind the
exact spatial workload and concrete values, streams, dynamic parameters, and
logical-memory state. The resolved simulator model belongs to
`ResolvedModelBinding`. Trace capture and physical execution limits belong to
the invocation and execution attempt rather than Request semantics.

DFG-sim does not consume Fabric or Mapping. It does not choose placements,
routes, tags, buffers, memory services, or hardware configurations. Supplying
Fabric metadata must not change DFG-sim semantics.

A workload-running invocation declares one typed `SimulationExecution` output
slot and the complete mandatory terminal FindingQuery set. It produces:

```text
SimulationExecution + EvaluationEvidence
```

`SimulationExecution` owns contract-aligned functional observations, progress,
and activity. `EvaluationEvidence` owns
normalized outcome, metrics, findings, and binds the execution through that
output slot. `Retired` returns every mandatory terminal finding as `Absent`;
`Halted` returns the corresponding finding as `Present` and all others as
`Absent`. A CLI report is a
removable projection of those artifacts, never another result authority. The
legacy element-formatted `final_memory_state` projection belongs only to the
fixture-oriented developer CLI. A typed workload returns memory state only
through the byte-addressed Spatial functional-observation contract; simulator
execution must not derive or validate a second element-level terminal state.

## Admission

The entire subject must pass Canonical Dataflow Program finalization before
runtime state is created. Residual `scf.*`, `cf.*`, imperative regions, or
other operations outside the canonical surface are invalid. Parse-time dialect
registration does not grant execution semantics.

DFG-sim imports the Dataflow-owned read-only
`CanonicalDataflowProgramView`, resolves the workload's exact
`RootedGraphLaunchRef`, derives its called `GraphRef`, and uses the imported
`ActorRef` and typed endpoint relations directly. It neither requires a
Mapping Artifact nor creates a simulator-local persistent actor catalog.
Missing, stale, foreign-artifact, or wrong-kind launch, graph, or actor
references fail admission before execution state exists.

Each graph memory argument is instantiated from its exact memref formal and
the Dataflow-owned launch memory binding. An imported linear view reuses the
runtime object and byte offset bound to its `LogicalMemoryRootRef`; exact
memref bindings reuse their existing root or view. DFG-sim has no
pointer-to-memref cast event, simulator-private alias bridge, or conversion
semantics. Residual `builtin.unrealized_conversion_cast` is invalid Canonical
Dataflow input. Registered first-class pointer operations execute through the
same OperationSchema transition authority as every other actor.

DFG-sim represents a defined pointer value by the exact runtime object ordinal,
signed byte offset within that object, address space, and exact DataLayout-owned
representation bits. It never serializes or compares a native host address.
The runtime object registry is the only pointer-to-object authority. GEP,
pointer casts, comparisons, pointer payload memory operations, and
PointerAddressed accesses all consume that same representation. An access whose
pointer does not resolve to the selected service or an admitted object range is
an execution failure; the simulator does not repair it by rebasing the pointer
or selecting another object.

An admitted actor without implemented semantics is `unsupported`. It must not
be approximated, skipped, or interpreted through a compatibility path.

### Source-Backed Validation

Source-backed validation starts from the exact `StructuredProgram` roots of
`SimulationWorkload` and `SimulationRuntimeInput`. These are production
persistent inputs shared by source execution and Structured DSE, not a
developer-only fixture schema. After a selected Structured candidate lowers to
Canonical Dataflow, the validator may mechanically derive candidate-specific
Spatial workload/input pairs for each observed graph activation. That
derivation is ephemeral and does not add source ABI, native pointers, call
paths, or candidate fields to either persistent Simulation family.

For whole-callable Spatial ownership, the oracle resolves each exact finite,
acyclic direct-call path from the execution entry to the selected callable.
For operation-owned Spatial ownership, it consumes the exact invocation-local
ownership derivation, reimports that derivation's parent Structured Program,
reapplies the selected typed decision without changing ownership, and observes
the selected operation at its explicit value and memory boundary. The leaf
call supplies runtime values; the complete path traces callable arguments to
finite backing objects and gates observation to that exact dynamic invocation
context. Distinct static paths are replayed independently. Indirect,
recursive, noncontiguous, or unbounded invocation lineage is typed
`Unsupported`. Source locations, symbol position, operation position, and
printer order are never persistent identity or ownership authority.
One finite static path may contribute zero dynamic invocations for the exact
workload. Its source and selected-decision replays must agree on that empty
sequence, and validation continues with the other paths. The complete
source-backed validation still requires at least one observed selected-region
invocation, one externally observable result, and nonempty DFG execution; an
all-empty aggregate cannot report success.

Graph value inputs are classified totally as `Fixed` or `Runtime`. Fixed
inputs preserve defined, poison, or undef state; runtime inputs are captured in
graph ABI order. Memory roots are projected onto finite byte-addressed backing
objects, and roots that share one object ordinal preserve aliasing. Unknown
extent, stream input capture, ambiguous ownership lineage, or an unsafe native
execution target is typed `Unsupported`, not repaired with fabricated input.
When an operation-owned root is a region-carried or dynamically indexed view,
the static derivation must prove one unique finite backing object. The oracle
then derives that root's exact byte offset from the concrete boundary pointer
for every dynamic invocation. A static offset may not stand in for this
invocation-local binding, and the ephemeral pointer does not enter either
persistent Simulation schema.

Source-backed capture uses one invocation-local object registry for globals,
stack allocations, runtime allocations, descriptor-reached objects, aliases,
and byte offsets. Object ownership is a property of the concrete runtime
allocation, never of an enclosing callable. Every observed host pointer is
resolved to one registry object plus byte offset before constructing the
canonical runtime input. Shared resolutions use one object ordinal and thereby
preserve aliasing. No descriptor-specific pointer table, call-local alias map,
or raw host address enters a graph or persistent Simulation Artifact.
When a boundary pointer is a visible direct-call result, capture derives its
origin by projecting every exact callee return operand back through that call's
operands. All reachable return paths must resolve to the same finite runtime
object; indirect calls, unavailable bodies, or divergent object origins are
`Unsupported` rather than guessed from a symbol name.

Pointer-producing descriptor loads and analogous aggregate traversal are
normalized to rooted memory capabilities when that relation is proved.
Otherwise a candidate may retain the pointer as first-class graph data under
the closed pointer contract. If neither a finite rooted view nor an exact
pointer-capable provider exists at one concrete activation, that activation is
typed `Unsupported`. The oracle never infers a static pointer identity from one
reaching store.
The canonical memory-actor relation also determines whether independent
native replays must agree on an imported object's pre-activation bytes. If any
aliasing root loads, performs RMW, or otherwise may read initial state, those
bytes are part of the replay input and must agree. Output-only roots may have
different concrete storage before independent executions; unchanged bytes in
those objects are not compiler-output differences. Each captured execution
still supplies one complete concrete Defined runtime memory object to DFG-sim.
Unknown capability consumers conservatively require initial-state agreement.

An operation-owned oracle executes an ephemeral clone of the exact selected
target Structured region. It does not reselect ownership, rerun DSE, or match
the region against a separately compiled host operation. Before assigning the
host JIT triple and DataLayout to that clone, the oracle rejects inline
assembly and target-specific intrinsics and proves that every root
execution-layout property, used pointer address space, used type layout, and
used struct element offset is equal. The target triple must be present but its
name need not equal the host triple. Equivalent DataLayout spellings are
accepted through their effective projections; a real layout difference fails
closed. Retargeting removes target CPU, feature, and tuning attributes only
from the ephemeral execution clone. The target Structured Program, Canonical
Dataflow Program, typed semantic decisions, Fabric target, and Mapping
identity remain unchanged. Residual host work executes as host work and is
never inserted into the Canonical Dataflow graph. This execution is a
functional oracle, not a target timing or architecture model.

Sys-sim does not consume this retargeted clone. It executes the exact
Deployment-selected target binary through `Gem5SimulationBinding`; a
layout-compatible host oracle therefore cannot stand in for RISC-V execution,
InstructionCore timing, NoC behavior, coherence, or external-memory behavior.

Source-backed functional validation has three comparisons over clones of the
same immutable runtime input:

1. the unmodified source execution produces the workload reference result;
2. the exact selected Structured candidate produces an equivalent whole-program
   result; and
3. every dynamically observed graph activation replays in DFG-sim with the
   same graph-boundary values, memory effects, and completion as the selected
   Structured execution.

The selected candidate and source executions must start from independently
cloned objects derived from the same `SimulationRuntimeInput`; process-global
mutable state cannot leak between them. A selected region that has no dynamic
activation for the workload is inapplicable and cannot report successful
source-backed acceleration. A mismatch is a hard semantic-gate failure and
cannot be repaired by falling back to host execution or selecting another
unreported graph. Exact integer and byte semantics must agree. Any accepted
floating variance must be attributed to an explicitly legal floating
transformation and reported as such; a generic numeric tolerance is not a
correctness authority.

The unmodified source execution may differ from the selected execution only
when the selected lineage contains an explicit typed floating-point decision,
while observable call inputs and every changed byte outside uniformly floating
canonical write relations remain exact. `llvm.intr.fmuladd` permits its fused
or split execution shape independently at each occurrence, so one source
execution is not required to equal an artificial all-fused or all-split replay
of the complete callable. The selected callable itself remains one exact typed
member and is the native oracle for DFG-sim. Allowed differing bytes are
reported as selected floating-decision variance; integer, address, input, and
non-floating memory differences fail closed.

## Execution Semantics

DFG-sim and CGRA-sim share one dependency-driven progress protocol:

```text
arrival or resource change
-> reevaluate affected actors
-> derive canonical semantic transition
-> admit and reserve
-> atomic commit
-> schedule publication, retirement, and release
```

DFG-sim supplies ideal per-actor resources to that protocol. It is
deterministic and event driven:

* every logical edge carries an ordered sequence of typed tokens;
* an actor transition fires only when its semantic inputs and guards permit;
* one firing consumes and publishes exactly the cardinalities defined by the
  Dataflow operation specs;
* operation latency and initiation interval determine cycle-stamped retirement
  and next-fire eligibility;
* hardware resources are unlimited, so unrelated ready transitions do not
  contend for PE, route, port, bank, buffer, or tag capacity;
* equal-cycle events use a canonical structural action key rooted in the
  imported typed `ActorRef`, execution-local occurrence, firing ordinal, and
  typed endpoint ordinals rather than pointer, container, host-thread, symbol,
  or printer order.

`AbstractCycle`, defined by the exact DFG timing model, is the simulator time
unit. It is not a physical time in nanoseconds and does not imply a target
clock frequency. Timed events use the Simulation Artifact
`EventCoordinate = (reference_cycle, delta)`, where an integral abstract cycle
`N` has the sole persistent encoding `N/1`. `delta` orders causally related
zero-registered-delay propagation inside one cycle and never increments a
cycle metric. DFG-sim may therefore estimate logical latency and throughput in
abstract cycles without claiming hardware-aware timing.

The Dataflow-owned registered `OperationSchemaId` projection owns actor
identity, closed semantic attributes, instance validity, and transition
descriptor identity. The DFG provider dispatches its executable transition
implementation by that same ID. It does not maintain an operation-name
whitelist or copy arbitrary attributes. A separately identified model binding
may provide latency and initiation interval. Neither projection is copied into
the request or result as a second authority.

### Exact Semantic Values

The simulator value domain distinguishes defined, poison, and undef. A defined
integer or floating value uses semantics equivalent to arbitrary-precision
`APInt` or `APFloat`; host `int64_t` and `double` are not authorities. Fixed
vectors carry this state independently per lane. Logical memory and any other
admitted non-bit value use an exact typed identity rather than a host pointer.

Operation schemas determine propagation and observation. `select` does not
observe its unselected operand. An inactive masked-memory lane observes neither
address nor data. An active store may store poison, and a subsequent load
restores that state. Undef remains an unconstrained semantic value until an
owning operation observes or freezes it; it is never rewritten to zero.

`freeze` chooses a legal defined value and keeps it stable for the resulting
SSA value. The exact model derives that deterministic choice from the resolved
semantic seed, `ActorRef`, firing ordinal, and lane. This key makes replay
stable without writing the chosen bits into the Canonical Dataflow Program.

Graph outputs may carry poison or undef. A consuming operation schema decides
whether it propagates, masks, freezes, or triggers undefined behavior. DFG-sim
has no global "terminal poison" failure rule.

## Control And Stateful Actors

`dataflow.stream`, `carry`, `invariant`, and `gate` follow the canonical phase
algebra and reset after a complete legal activation. A false close emits no
sentinel induction value. Ordered token flow, explicit close events, and graph
completion determine quiescence.

`dataflow.sync`, `mux`, `demux`, constants, arithmetic, math, vector adapters,
and other admitted actors follow their owning Dataflow or upstream-dialect
semantic specs. Packed and vector payloads use arbitrary-width bit-accurate
storage; host integer width is not a semantic limit.

## Memory

Logical memories are addressed software memory spaces. `dataflow.load` and
`dataflow.store` fire and retire through their explicit `ctrl`/`done` network.
For each alias partition, the canonical memory-order state is the pair
`(write_frontier, read_frontier)` defined by the graph-memory lowering spec.

Actor-transition commit issues one logical memory operation. The DFG memory
provider then linearizes and retires it under the shared lifecycle and
`MemoryAction` projection in
`docs/spec-dataflow-memory-consistency.md`. DFG-sim models neither cache
hierarchy, coherence traffic, NoC transport, bank conflicts, physical memory
ports, nor Fabric memory-service capacity.

Plain conflicting accesses without an explicit causal order must not become
deterministic merely because of simulator traversal order. When such a
conflict depends on runtime addresses, the exact initial DFG model returns
`Unsupported` rather than choosing an arbitrary result or reporting a
deadlock.

Element, contiguous, indexed, and masked accesses execute the exact Dataflow
semantics in `docs/spec-dataflow-vectorization.md`. DFG-sim suppresses inactive
lane addresses, zero-fills inactive load lanes, preserves canonical row-major
result order, completes an all-zero mask without a memory effect, and retires
one vector actor firing as one load `data + done` or store `done` event. It does
not expose Fabric lane or beat transactions. Active accesses preserve the
exact defined, poison, or undef state stored in logical memory.

The software contract for atomic, RMW, compare-exchange, fence, and volatile
actors is defined by `docs/spec-dataflow-memory-consistency.md`. The DFG
provider executes that contract with logical memory and unlimited physical
resources. It owns deterministic legal choices for modification order,
reads-from, synchronizes-with, and sequentially-consistent order. Simultaneous
eligible actions use the stable action-key derivation defined by the exact DFG
model descriptor; MLIR traversal, container order, and host scheduling are
forbidden tie breaks.

The first exact DFG model gives weak compare-exchange no spurious failures.
Any future model that explores permitted spurious failure or other legal
nondeterminism requires a different exact model identity and explicit
configuration or seed. For each resolved software synchronization scope, the
provider derives an execution-local abstract participant domain from the exact
workload; this is not a Fabric or persistent Artifact domain. An unresolved
target synchronization scope is `Unsupported`. A scope whose participants
extend beyond the spatial workload requires an explicit external service model
or sys-sim. Volatile ordinary storage follows the shared at-most-once
observation contract; volatile MMIO requires an exact external device model.
Hardware coherence remains outside DFG-sim, and none of these actors may be
reinterpreted as plain load/store.

## Trace And Termination

Schema 1.0 has no persistent trace-manifest field because the raw
detailed-bundle owner and importer are not yet defined. DFG-sim may retain a
diagnostic trace in attempt or scratch storage, but it cannot place paths,
opaque bytes, or unchecked Artifact references in `SimulationExecution` or
`EvaluationEvidence`. The future trace contract may encode a complete
launch-to-terminal trace or a gap-free launch-rooted prefix; it cannot encode
an interior loss as partial coverage. Frames are strictly ordered by
`EventCoordinate`, cannot cross future chunk boundaries, and use canonical
within-frame event order.

DFG-sim supports the exact `Firing` and `Semantic` levels owned by Simulation
Artifacts. `Firing` contains `ActorCommitted` and `ActorRetired`.
`Semantic` strictly includes those events and adds every `TokenPublished` and
`MemoryLinearized` event in the covered interval. Every semantic publication
stores its exact one-token `CanonicalValueSequence`; payload omission is not a
capture option. DFG-sim has no physical action source, so a
`Microarchitecture` trace request is `Unsupported`.

Firing remains atomic actor-transition commit, not readiness. A transition
ordinal is allocated when the complete semantic transition is formed and is
also its firing ordinal once committed, allowing a retained execution to name
a final pending transition without inventing another identity. Stable actor
identity remains the Dataflow-owned `ActorRef`; invocation, transition, token,
and memory-action occurrences are execution-local coordinates, never Dataflow
entities, Mapping IDs, or physical Tags.

Launch, graph-retirement, and terminal markers are projected from the
execution's `SpatialProgressObservations`, not emitted as duplicate trace
events.

Successful termination requires all of the following:

* the graph completion frontier has retired;
* no actor has a required pending transition;
* no delayed retirement event remains;
* no required output, stream close, or memory effect remains unpublished.

Quiescence without legal completion is deadlock only when a closed wait-set
witness proves that no future arrival, guaranteed release, or escape can
restore progress. A long run or an empty event queue alone is insufficient.
The execution terminal is exactly `Retired`, `Halted {finding,witness}`, or
`StoppedByLimit`, with the Evidence mapping defined by Simulation Artifacts.
DFG-sim fills the exact `SpatialProgressObservations` anchors. The DFG timing
model's `AbstractCycle` is their reference domain. `Retired` includes visible
graph retirement; a halt or retained stop may occur before or after it. Cycle
count is derived from accepted Spatial Launch through visible graph retirement
without `delta`, rather than stored in the execution. Wall time, host
parallelism, and license availability may interrupt execution but must not
select a different formal result.

## Observations

DFG-sim fills the positional value, stream, and memory arrays defined by
Simulation Artifacts. It derives their targets and order from the exact
workload rather than emitting graph names or simulator-local keys. At
`Retired`, every selected value is published and every selected stream is
closed. `Halted` and a retained `StoppedByLimit` execution preserve published
values, open stream prefixes, and visible logical-memory state at their exact
terminal coordinate. Lack of model capability is `Unsupported`, not an
unavailable output entry.

When requested and exactly observed, DFG-sim may retain an
`ActorTransitions` activity summary over either progress-defined window. A
complete summary is total over the rooted launch's canonical actor inventory;
a partial summary names only continuously observed actors, and omitted actors
are unknown. DFG-sim does not fabricate Fabric-resource or
implementation-signal activity. Actor activity can be projected to a physical
Evaluation target only when the exact Request closure also supplies the
required Mapping and implementation lineage.

Supported normalized observations may include logical cycle count, actor fire
and retirement counts, operation-class activity, terminal observables, and
deadlock or other proven execution-halting findings declared by the model.
They are derived into Evaluation Evidence rather than copied into the
execution. Capability rejection is `Unsupported` and does not become a
finding. Every metric is identified through the central metric registry and
carries its unit and provenance.

## DFG/CGRA Relation

DFG-sim and CGRA-sim use the same canonical actor and logical-memory
semantics. Given equivalent runtime inputs and a legal complete mapping, exact
terminal-value equality is required only when the requested observable
contract or comparison oracle proves that observation deterministic. Each
exact model still produces one deterministic legal execution. Different legal
atomic orders may therefore produce different per-actor values without
violating the shared semantics. Different cycle behavior is expected because
CGRA-sim adds finite resources and hardware timing.

Comparison is an ordinary Evaluation model specified by
[Simulation Comparison](spec-sim-comparison.md). It consumes role-labeled
executions rather than simulator-specific report files.

## Anchor Verification

Stable anchor tests cover:

* rejection of non-finalized subjects before execution;
* graph and actor import from the exact Dataflow Artifact without Mapping;
* rejection of foreign-artifact and wrong-kind Dataflow references;
* dispatch through the registered `OperationSchemaId` without an independent
  operation-name table;
* lazy poison through selection, deterministic `freeze`, active memory
  poison round trip, per-lane exceptional vector state, and legal terminal
  poison or undef;
* exact token cardinality and state reset for canonical control actors;
* `ctrl`/`done` memory order and terminal memory diffs;
* contiguous, indexed, and masked vector-memory semantics with one actor
  retirement;
* relaxed atomic histogram execution;
* release/acquire synchronization, including its fence form;
* repeated-address `PerLane` atomic execution with one actor retirement;
* at-most-once volatile MMIO observation through an exact external model;
* ordered progress anchors and required retirement presence;
* complete and partial actor-activity inventory semantics;
* version-1 rejection of persistent trace-manifest fields and diagnostic trace
  capture noninterference;
* exact terminal and Evidence outcome mapping, including deadlock witnesses;
* explicit unsupported and deadlock outcomes; and
* deterministic or oracle-governed comparison with one legal CGRA-sim
  execution.

Tests must not preserve text report layouts, container order, a fixture matrix,
or implementation-specific scheduler classes.
