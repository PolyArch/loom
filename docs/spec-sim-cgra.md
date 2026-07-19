# CGRA-sim

## Implementation Status

This document is a target contract. Loom does not currently build a true CGRA
simulator, the shared SpatialCore simulation library, or the gem5 Bridge
described below. The implemented simulator is DFG-sim and does not satisfy
this hardware-aware contract.

## Target Architecture

CGRA-sim is the hardware-aware simulator for one mapped SpatialCore
execution. It executes canonical graph semantics on concrete `fabric.module`
resources under one complete SpatialMapping.

CGRA-sim is not a whole-system simulator. HostCore, InstructionCore, caches,
coherence, system memory hierarchy, NoC, and system time belong to an external
system simulator such as gem5. The target Loom Bridge will invoke the shared
SpatialCore simulation library at the Spatial Launch ABI boundary.

## Exact Inputs

A CGRA-sim request identifies at least:

* one finalized Canonical Dataflow Program and graph subject;
* one finalized Fabric Hardware Description containing the selected
  SpatialCore template and elaborated resources;
* one complete SpatialMapping bound to the exact Dataflow, TechMapping, and
  Fabric inputs;
* concrete graph input values, stream messages, and memory capabilities;
* initial visible memory state where required; and
* resolved simulator configuration and model identity.

The exact persistent request and Evidence schemas remain owned by Evaluation
and are not defined here.

## Boundary With DFG-sim

DFG-sim executes the same Canonical Dataflow graph without concrete Fabric
capacity, routes, buffers, tags, or resource contention. CGRA-sim adds those
SpatialCore constraints.

Functional comparison requires the same graph identity, input identity,
initial visible memory identity, and observable-output contract. Different
performance is expected when concrete hardware constrains execution.

## Boundary With Mapping

CGRA-sim consumes and validates a complete SpatialMapping. It must not choose
placements, reroute a logical net, assign a tag, allocate a buffer, select a
memory occurrence, repair resource use, or complete a missing record.

Unsupported inputs and invalid Mapping are ordinary failures. A simulator
report cannot make an invalid Mapping legal and is not copied into Mapping as
diagnostics or metrics.

## Hardware Scope

The simulator may model SpatialCore resources referenced by the exact
Mapping, including:

* PE, FU, switch, memory, FIFO, boundary, and transport occurrences;
* explicit directed endpoints, point-to-point arcs, and resource traversals;
* configured functions and mapping-visible modes;
* Route Trees, physical buffers, local Physical Tags, and selected memory
  services when their persistent schemas are closed; and
* Fabric-owned latency, initiation, capacity, arbitration, and use patterns.

It does not model InstructionCore execution or system interconnect as CGRA
resources. Those components interact through typed Spatial Launch and service
boundaries driven by the external system simulator.

## Execution Model

Execution is event-driven and deterministic for exact semantic inputs,
resolved configuration, and simulator model identity. Canonical Dataflow
edges carry software values and causal events. Fabric resources constrain when
those events can progress.

Mapping does not provide an absolute schedule-slot table. The simulator
instantiates event-relative `ResourceUse` and Fabric-owned use patterns for a
dynamic graph invocation. It may derive queues, calendars, occupancy, or
conflict caches, but those are disposable simulator state rather than Mapping
records.

Physical Tags are interpreted only in their Fabric-owned domains. They are
not global token IDs, firing numbers, or dynamic invocation identities.

The simulator may apply backpressure or wait for capacity. It cannot alter the
selected SpatialMapping while doing so.

## Target Shared SpatialCore Library

Standalone DFG/CGRA tools and the gem5 Bridge must reuse one Loom-owned
event-driven SpatialCore simulation library. A CLI is a thin request and
reporting surface, not a second semantic implementation.

The gem5 event queue is the only whole-system time authority. A SpatialCore
session advances under Bridge control to its next externally observable event
and returns that event without running an independent system clock.

## Evidence And Metrics

CGRA-sim produces Evaluation Evidence with exact subject, model, configuration,
runtime-input, and Mapping identities. Evidence may contain cycle count,
latency, throughput, stalls, utilization observations, memory traffic, route
activity, and other supported metrics with explicit provenance.

Metrics are observations, not Mapping fields or verifier exceptions. Missing
or unsupported observations remain explicit typed results.

## Determinism

Deterministic ordering derives from canonical identities, typed structural
keys, explicit event order, and resolved simulator rules. Host thread
scheduling, container traversal, source order, stable symbols, and printer
order are not tie breakers.

## Open Boundaries

The following remain open and must not be invented by the simulator spec:

* exact persistent SpatialMapping physical records;
* complete Evaluation request and Evidence schemas;
* the full SpatialCore microarchitecture model inventory;
* gem5 Simulation Binding fields; and
* fidelity-specific metric availability.

## Validation

Anchor tests should cover exact input coupling, rejection of invalid Mapping,
deterministic graph execution, explicit route and capacity effects, memory
visibility, and agreement with DFG-sim on shared functional observables. They
must not pin schedule-slot records, InstructionCore simulation, or a textual
report fixture matrix.
