# DFG-sim

This document owns the execution contract for Loom's hardware-unaware
Canonical Dataflow Program simulator. Persistent Evaluation schemas are owned
by [DSE and Evaluation](spec-dse-feedback.md); this document defines only the
model-specific subject, behavior, and observations. Shared workload, runtime
input, execution, trace-manifest, and terminal schemas are owned by
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
SimulationExecution + EvaluationEvidence + optional raw detailed bundle
```

`SimulationExecution` owns contract-aligned functional observations, progress,
activity, and the trace manifest. `EvaluationEvidence` owns
normalized outcome, metrics, findings, and binds the execution through that
output slot. `Retired` returns every mandatory terminal finding as `Absent`;
`Halted` returns the corresponding finding as `Present` and all others as
`Absent`. A CLI report is a
removable projection of those artifacts, never another result authority.

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

An admitted actor without implemented semantics is `unsupported`. It must not
be approximated, skipped, or interpreted through a compatibility path.

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
clock frequency. Timed events use `EventCoordinate = (abstract_cycle, delta)`.
`delta` orders causally related zero-registered-delay propagation inside one
cycle and never increments a cycle metric. DFG-sim may therefore estimate
logical latency and throughput in abstract cycles without claiming
hardware-aware timing.

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

When requested, the `SimulationExecution` trace manifest orders
content-addressed chunk references whose opaque payloads are retained by the
raw detailed bundle. Records are strictly ordered by `EventCoordinate` and
canonical within-frame event order. Firing is the atomic actor-transition
commit, not readiness; publication and retirement may occur later. A firing record
identifies the stable actor, execution-local occurrence, per-actor firing
ordinal, consumed and produced logical endpoints, and relevant state
transition. Raw payload inclusion is controlled by the invocation's capture
request.

The stable actor field is the exact Dataflow-owned `ActorRef`. The occurrence
and firing ordinal are execution-local coordinates and never become Dataflow
entities, Mapping IDs, or physical Tags.

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
Cycle count spans accepted Spatial Launch through visible graph retirement.
Wall time, host parallelism, and license availability may interrupt execution
but must not select a different formal result.

## Observations

DFG-sim fills the positional value, stream, and memory arrays defined by
Simulation Artifacts. It derives their targets and order from the exact
workload rather than emitting graph names or simulator-local keys. At
`Retired`, every selected value is published and every selected stream is
closed. `Halted` and a retained `StoppedByLimit` execution preserve published
values, open stream prefixes, and visible logical-memory state at their exact
terminal coordinate. Lack of model capability is `Unsupported`, not an
unavailable output entry.

Supported normalized observations may include logical cycle count, actor fire
and retirement counts, operation-class activity, terminal observables, and
deadlock or other proven execution-halting findings declared by the model.
Capability rejection is `Unsupported` and does not become a finding. Every
metric is identified through the central metric registry and carries its unit
and provenance.

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
* deterministic `EventCoordinate` and within-frame trace order;
* trace observer noninterference;
* exact terminal and Evidence outcome mapping, including deadlock witnesses;
* explicit unsupported and deadlock outcomes; and
* deterministic or oracle-governed comparison with one legal CGRA-sim
  execution.

Tests must not preserve text report layouts, container order, a fixture matrix,
or implementation-specific scheduler classes.
