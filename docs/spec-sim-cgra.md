# CGRA-sim

This document owns the execution contract for Loom's hardware-aware
SpatialCore simulator. Persistent Evaluation schemas are owned by
[DSE and Evaluation](spec-dse-feedback.md). Shared workload, runtime input,
execution, trace-manifest, and terminal schemas are owned by
[Simulation Artifacts](spec-simulation-artifacts.md).

## Purpose And Subject

CGRA-sim answers:

```text
How does a Canonical Dataflow Program execute on this exact SpatialCore and
complete SpatialMapping under its concrete timing and contention rules?
```

The descriptor has three required role-labeled subject slots:

```text
program: CanonicalDataflowProgram
hardware: FabricHardwareDescription
spatial_mapping: SpatialMapping
```

The Mapping is already bound to its exact TechMapping and Fabric inputs.
`workload_ref` and `runtime_input_ref` bind the exact spatial workload and its
concrete values, streams, dynamic parameters, external arrivals, and logical
memory. The resolved simulator model belongs to `ResolvedModelBinding`. Trace
capture and physical execution limits belong to invocation and attempt state.

CGRA-sim declares one typed `SimulationExecution` output slot and the complete
mandatory terminal FindingQuery set. It produces one execution, one
`EvaluationEvidence`, and an optional raw detailed bundle. Their ownership is identical to DFG-sim:
execution owns terminal observables, activity, and trace; Evidence owns
normalized outcome, metrics, findings, and the typed execution output binding.
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

It does not simulate HostCore or InstructionCore execution, system caches,
coherence, system memory hierarchy, or AccCore-to-AccCore NoC behavior. Those
belong to sys-sim through gem5.

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

Single-clock SpatialCore sessions advance in nonnegative integer cycles. A
multi-clock session chooses an explicit reference clock and represents every
event in exact rational reference cycles. All timed events use
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

Hardware resource delay may postpone issue or retirement, but it cannot change
the logical memory-order contract. Visibility and terminal memory diffs must
match DFG-sim for a legal execution. System cache and coherence behavior is
outside this simulator.

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
Cycle count spans accepted Spatial Launch through visible graph retirement in
the workload's declared completion clock domain.

## Trace And Observations

The trace uses increasing `EventCoordinate` frames and canonical within-frame
order. Firing is actor-transition commit; result publication and retirement
may be later events. It can identify actor firings, selected physical
occurrences and traversals, resource grants, stalls, queue changes, logical
memory requests, implementation lane or beat transactions, and retirements.
Child memory transactions remain correlated with their one parent firing and
do not appear as additional actor firings. `SimulationExecution` owns the typed
manifest and ordering; the raw detailed bundle owns referenced opaque chunk
payloads.
Neither creates a separate `SimulationTrace` artifact.

Trace capture is observational. Enabling or changing it cannot affect grants,
event scheduling, outputs, terminal form, cycle count, metrics, or findings.

Normalized Evidence may expose cycle count, latency, throughput, initiation
behavior, stalls, occupancy, utilization, traffic, contention, and deadlock
findings when the model supports them. Metric names, units, and provenance come
from the central registry. Evidence never becomes Mapping state.

## Standalone And System Integration

The standalone CGRA tool and the gem5 Bridge reuse one Loom-owned SpatialCore
simulation library. A CLI is a thin request/projection surface, not another
semantic implementation.

In sys-sim, gem5 is the only whole-system time authority. The Bridge advances a
SpatialCore session to its next externally observable event and translates that
cycle-relative event at the exact launch/service boundary. The SpatialCore
library does not run an independent whole-system clock.

## Anchor Verification

Stable anchor tests cover:

* exact `{D,F,SpatialMapping}` admission and rejection of stale or incomplete
  Mapping;
* finite-route, buffer, memory, and temporal-resource contention;
* contiguous, indexed, masked, and multi-transaction memory execution with one
  logical retirement and one Tag per vector token;
* exact single- and multi-clock event order and delta nonconvergence;
* trace observer noninterference;
* ordered-token preservation under temporal interleaving;
* deadlock versus invalid-Mapping classification; and
* agreement with DFG-sim on terminal software observables.

Tests must not pin text reports, disposable simulator caches, or a broad
microarchitecture fixture matrix.
