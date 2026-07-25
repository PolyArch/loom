# CGRA-sim

This document owns the execution contract for Loom's hardware-aware
SpatialCore simulator. Persistent Evaluation schemas are owned by
[DSE and Evaluation](spec-dse-feedback.md). Shared workload, runtime input,
execution, activity, future trace, and terminal schemas are owned by
[Simulation Artifacts](spec-simulation-artifacts.md).

## Purpose And Subject

CGRA-sim answers:

```text
How does a Canonical Dataflow Program execute on this exact SpatialCore and
complete SpatialMapping under its concrete timing and contention rules?
```

The descriptor references the shared CGRA-simulation case signature, whose
three required roles are:

```text
program: CanonicalDataflowProgram
hardware: FabricHardwareDescription
spatial_mapping: SpatialMapping
```

The Mapping is already bound to its exact TechMapping and Fabric inputs.
`workload_ref` and `runtime_input_ref` bind the exact spatial workload and its
concrete values, complete input-stream definitions, dynamic parameters, and
logical memory. The resolved simulator model belongs to
`ResolvedModelBinding`. Trace capture and physical execution limits belong to
invocation and attempt state.

CGRA-sim declares one typed `SimulationExecution` output slot and the complete
mandatory terminal FindingQuery set. It produces one execution, one
`EvaluationEvidence`. Their ownership is identical to DFG-sim: execution owns
contract-aligned functional observations, progress, and activity; Evidence owns
normalized outcome, metrics, findings, and the typed execution output binding.
Raw traces and tool material remain attempt or scratch state until the raw
detailed-bundle owner and a later Simulation Artifacts schema minor are defined.
`Retired` returns every mandatory terminal finding as `Absent`; `Halted`
returns the corresponding finding as `Present` and all others as `Absent`.

## Scope

CGRA-sim models only a SpatialCore invocation. It may model selected:

* PE, FU, switch, memory, FIFO, boundary, and transport occurrences;
* exact physical endpoints, traversals, Route Trees, and shared trunks;
* Fabric-owned latency, initiation interval, capacity, use patterns, and grant
  policies;
* configured functions and `sw_configs` derived from Mapping;
* spatial and temporal PE execution, operand buffers, register-file forwarding,
  local Physical Tags, and context state;
* memory-service bindings, ports, banks, queues, and Mapping-visible local
  protocol behavior; and
* backpressure, arbitration, buffering, contention, and deadlock progress.

It does not simulate HostCore or InstructionCore execution, system cache and
coherence state, system memory hierarchy, or AccCore-to-AccCore NoC behavior.
Those belong to sys-sim through gem5. A Fabric-declared consistency domain
whose complete closure is local to the SpatialCore remains inside CGRA-sim.

## Admission And Mapping Boundary

Before simulation starts, CGRA-sim validates exact Artifact coupling and the
complete final SpatialMapping. It must reject stale identities, unresolved
references, incomplete realization coverage, illegal routes, conflicting
resource use, invalid tags, unsupported configurations, and incomplete memory
service bindings.

The simulator never places, reroutes, recolors, allocates a missing buffer,
selects a different memory service, repairs a Mapping, or invents a Fabric
configuration. Dynamic execution may arbitrate among legal requests according
to Fabric policy, but it cannot change the Mapping decision.

Boundary execution uses the exact atomic join/fork equations declared by
Fabric: no input or output leg may transfer independently, and the base
boundary has no simulator-owned holding state. Temporal-PE operand queues use
the exact required `operand_buffer_size`, mode-derived allocation units,
one-enqueue/one-dequeue service contract, and canonical grant state. The
simulator must not supply the former implicit depth 2 or any other capacity,
port, or priority default.

## Event Model

CGRA-sim and DFG-sim share the dependency-driven progress protocol defined by
DFG-sim. Canonical actor transitions remain the dynamic software firing unit.
A transition can progress only when both its Dataflow inputs and all selected
physical mechanisms permit it. Commit atomically reserves its current input,
output-holding, and execution obligations; it does not pre-lock an entire
future route.

The simulator derives disposable runtime queues, occupancy tables, calendars,
and conflict caches from Fabric and Mapping. These are not persistent Mapping
records. Mapping does not provide an absolute cycle-slot schedule.

Single-clock SpatialCore sessions advance in nonnegative integer cycles, but
persist every cycle `N` as the canonical `ExactRatio` value `N/1`. A
multi-clock session represents every event in exact reference cycles. The
reference domain is derived mechanically from the exact Fabric, complete
SpatialMapping, and mapped Spatial Launch boundary; it is not selected by the
workload or a simulator-private default. The Evaluation base condition owns
the exact relative period and phase selected for clocks that Fabric does not
hard-fix. All timed events use
`EventCoordinate = (reference_cycle, delta)`; host floating point and rounded
nanoseconds cannot determine event order. `delta` expresses only causal
combinational propagation inside one cycle. Equal coordinates use canonical
structural serialization that never substitutes for Fabric arbitration.
Registered resources retire at their declared future cycles, and a mapped
logical edge cannot bypass selected physical latency or capacity.
Buffered and bypassed `fabric.fifo` occurrences use the exact cycle and
backpressure contract in `docs/spec-fabric-fifo.md`; the simulator does not
invent fall-through or hidden storage.

## Tags, Contexts, And Ordered Dataflow

Physical Tags are local encodings used only inside their declared Fabric tag
domains. They are neither global token identities nor firing numbers.
`InstructionContextRef` identifies a resident configured-graph state namespace;
the canonical actor transition remains the execution atom.

Temporal resources may interleave transitions only when operand association,
result ordering, tag continuity, and Fabric-defined progress remain valid. An
implementation that can produce a later ordered token before an earlier token
must buffer or arbitrate to preserve ordered Dataflow semantics.

## Memory

Software memrefs remain logical address spaces. SpatialMapping binds their
accesses and exported capabilities to Fabric memory services and service
routes. CGRA-sim models the selected physical width, capacity, ports, queues,
banks, internal dependency forwarding, and protocol-visible timing.

For every load or store, the simulator derives the same
`CanonicalMemoryAccessView` used by TechMapping and executes the exact selected
Fabric operation port and use pattern. It distinguishes element, contiguous,
and indexed accesses even at equal payload width; consumes a dynamic mask as
one ordinary token; suppresses inactive-lane requests; zero-fills inactive
load lanes; and completes an all-zero mask without a service transaction.

A declared use pattern may issue several internal lane or beat transactions.
CGRA-sim models their resource conflicts and timing, assembles active load
lanes in canonical row-major order, and exposes only one canonical actor firing
and one load `data + done` or store `done` retirement. A vector token has one
Physical Tag wherever a Tag is required; lane indices are never Tags.

Actor-transition commit issues one logical memory operation through the shared
`MemoryAction` projection. The exact CGRA provider derives admission,
linearization timing, visibility acknowledgement, and retirement from the
selected Fabric operation port, `MemoryConsistencyDomain`, use pattern,
resource state, grant policy, and Mapping. Hardware delay may postpone any
provider event, but it cannot change the logical memory-order contract.

For every non-memory resource use, CGRA-sim acquires the complete Fabric claim
envelope at the declared acquire event, applies the optional owner-defined
resource transition atomically at its commit event, and returns the complete
claim envelope at its release event. It never treats durable queue occupancy as
an outstanding claim or lets one use release another use's claim. Concrete
resource state and event ordering come from the selected Fabric use pattern and
its timing contract, not a simulator-private scheduler.

The provider must implement the exact domain release-visibility point and
`BoundedCompletion` or `FairEventual` progress guarantee. It cannot replace
them with a simulator timeout, zero-latency default, or private completion
policy. A bounded guarantee is checked in rising-edge ticks of its exact Fabric
`progress_clock`; a fair-eventual guarantee is checked against the declared
grant and downstream-progress premises.

The Dataflow software contract for atomic, RMW, compare-exchange, fence, and
volatile actors is defined by `docs/spec-dataflow-memory-consistency.md`.
A `MemoryConsistencyDomain` whose complete participant and service closure is
inside the simulated SpatialCore executes through the shared consistency
semantics and exact Fabric-local provider. The provider models physical
contention while preserving one actor issue and one retirement even when the
selected use pattern contains several lane or beat transactions.

A manager or other external endpoint requires a descriptor-owned exact
external-service model. CGRA-sim must not assume zero latency, implicit
coherence, or an arbitrary response policy. If a reachable execution requires
provider behavior that the exact model does not define, including permitted
weak compare-exchange spurious failure under contention, the result is
`Unsupported`. System cache, coherence, memory hierarchy, and cross-AccCore
ordering remain sys-sim responsibilities. None of these actors may execute as
plain load/store.

Terminal logical-memory state and actor values obey the same software
semantics as DFG-sim. Exact value equality is required only for observations
that the requested observable contract or comparison oracle proves
deterministic; different deterministic legal executions may select different
atomic orders.

CGRA-sim uses the same positional value, stream, and memory records as
DFG-sim. Physical stalls and arbitration may change publication coordinates
and open prefixes at a non-retired terminal, but cannot change target order,
memory-diff normalization, or the meaning of published software values.
Missing runtime-dependent hardware-model capability is `Unsupported`, never
an omitted or placeholder functional observation.

## Deadlock And Termination

Successful completion requires the Dataflow completion frontier and every
selected physical obligation to retire. No required route transfer, buffered
token, memory response, configuration transition, or delayed event may remain.

Quiescence before legal completion is deadlock only when a dynamic closed
wait-set witness proves that no future external arrival, guaranteed release,
or escape can restore progress. Diagnostics identify blocked canonical
transitions and the Fabric-owned resource/policy state that prevents progress.
An invalid Mapping is never classified as dynamic deadlock.

The execution terminal is exactly `Retired`, `Halted {finding,witness}`, or
`StoppedByLimit`, with the Evidence mapping defined by Simulation Artifacts.
CGRA-sim fills the exact `SpatialProgressObservations` anchors in the derived
reference domain. `Retired` includes visible graph retirement; a halt or
retained stop may occur before or after it, including a halt during
post-retirement self-reset. Cycle count is derived from accepted Spatial Launch
through visible graph retirement without `delta`, rather than stored in the
execution.

## Trace And Observations

The trace uses increasing `EventCoordinate` frames and canonical within-frame
order. CGRA-sim supports all three levels owned by Simulation Artifacts.
`Firing` records actor commit and retirement. `Semantic` strictly includes
firing and records every exact token publication and memory linearization.
`Microarchitecture` strictly includes semantic and records the request, grant,
and retirement lifecycle of each selected physical action.

A physical action names either one Fabric use pattern or the exact selected
traversal set with an optional Fabric-owned atomic use pattern. Compute,
memory, temporal-switch, broadcast, buffered-route, and direct point-transfer
behavior therefore use the same closed algebra. The request-to-grant interval
is the stall; equal coordinates mean no stall. Queue changes, occupancy,
resource-state transitions, Tags, configurations, and blocker strings are
derived from the exact Fabric contract, Mapping, and lifecycle and are not
duplicated as trace events.

Implementation lane or beat actions remain deterministic child physical
actions of one canonical actor transition and do not appear as additional
actor firings. In the future persistent trace schema, `SimulationExecution`
owns the typed manifest, level, ordering, and complete or launch-rooted prefix
coverage, while its one exact same-Request raw detailed bundle owns canonical
chunk bytes and their Common `BlobDigest` inventory. That schema is unavailable
in version 1.0; diagnostic traces remain attempt or scratch material and do not
create a `SimulationTrace` artifact.

Launch, graph-retirement, and terminal markers are projected from
`SpatialProgressObservations`; CGRA-sim does not serialize duplicate boundary
events.

Trace capture is observational. Enabling or changing it cannot affect grants,
event scheduling, outputs, terminal form, cycle count, metrics, or findings.

When requested and exactly observed, CGRA-sim may retain
`ActorTransitions` and `FabricResources` activity summaries over either
progress-defined window. Actor tables resolve through the exact Dataflow
program. Fabric use and occupancy tables resolve through the exact Fabric and
complete SpatialMapping and use only Fabric-owned use-pattern and
resource-state references. A descriptor that cannot continuously observe an
entire requested target inventory emits a partial summary with missing-is-
unknown semantics or rejects the request; it cannot infer zero activity.
CGRA-sim does not fabricate implementation-signal activity from Fabric
activity.

Normalized Evidence may expose cycle count, latency, throughput, initiation
behavior, stalls, occupancy, utilization, traffic, contention, and deadlock
findings when the model supports them. These values are derived from exact
execution facts and model semantics rather than copied into activity
summaries. Metric names, units, and provenance come from the central registry.
Evidence never becomes Mapping state.

## Standalone And System Integration

The standalone CGRA tool and the gem5 Bridge reuse one Loom-owned SpatialCore
simulation library. A CLI is a thin request/projection surface, not another
semantic implementation.

In sys-sim, gem5 is the only whole-system time authority. The Bridge advances a
SpatialCore session to its next externally observable event and translates that
cycle-relative event at the exact launch/service boundary. The SpatialCore
library does not run an independent whole-system clock. A consistency domain
fully local to the SpatialCore may continue in the Loom provider. A request
whose domain crosses the system boundary is delegated through the typed
Spatial Service boundary; gem5 alone owns the external modification order,
reads-from, cache, coherence, and system-order state.

## Anchor Verification

Stable anchor tests cover:

* exact `{D,F,SpatialMapping}` admission and rejection of stale or incomplete
  Mapping;
* finite-route, buffer, memory, and temporal-resource contention;
* boundary partial-valid and partial-ready stalls without partial transfer;
* replay-visible Temporal PE behavior for explicit operand-buffer depths 1 and
  2;
* contiguous, indexed, masked, and multi-transaction memory execution with one
  logical retirement and one Tag per vector token;
* local atomic and fence execution through one exact Fabric consistency
  domain;
* repeated-address `PerLane` atomics and at-most-once volatile MMIO service;
* rejection of incomplete external or weak-compare-exchange provider behavior;
* exact single- and multi-clock event order and delta nonconvergence;
* mechanically derived progress reference domain and ordered progress anchors;
* complete and partial actor/Fabric activity inventory semantics and Fabric
  capacity bounds;
* version-1 rejection of persistent trace-manifest fields and diagnostic trace
  capture noninterference;
* ordered-token preservation under temporal interleaving;
* deadlock versus invalid-Mapping classification; and
* deterministic or oracle-governed agreement with DFG-sim.

Tests must not pin text reports, disposable simulator caches, or a broad
microarchitecture fixture matrix.
