# CGRA-sim

This document specifies Loom CGRA-sim, the hardware-aware simulator for
mapped Loom workloads. Despite the name, CGRA-sim is not limited to
simulating a `fabric.module` or SpatialCore. It simulates mapped
software against a concrete hardware graph and mapping artifact.

## Purpose

CGRA-sim answers this question:

```text
How does this mapped dataflow workload behave under the selected
hardware resources, routes, buffers, memory system, and schedules?
```

CGRA-sim consumes:

* dataflow IR;
* Fabric ADG;
* a mapping artifact specified in `docs/spec-mapping-artifact.md`;
* runtime input data;
* initial memory image or memory configuration;
* simulator configuration.

CGRA-sim produces:

* functional outputs;
* final memory state or memory diffs;
* hardware-aware cycle count;
* resource activity;
* queue and buffer occupancy;
* route activity;
* memory and coherence activity;
* stalls and bottleneck attribution;
* temporal reuse and reconfiguration activity;
* diagnostics;
* a CGRA-sim report usable by comparison, PnR feedback, DSE, and FPA
  integration.

## Boundary With DFG-sim

DFG-sim is the pure software semantic baseline specified in
`docs/spec-sim-dfg.md`. CGRA-sim must preserve the same software
semantics for legal mappings, then add hardware constraints.

CGRA-sim may reuse DFG-sim semantic components for dataflow operation
behavior, token semantics, and memory-order semantics. It must not
replace those semantics with a different interpretation of dataflow IR.

The comparison protocol is specified in
`docs/spec-sim-comparison.md`.

## Boundary With PnR

PnR is specified in `docs/spec-pnr.md`. PnR chooses and records
placements, routes, schedules, buffer bindings, memory bindings,
resource sharing, and temporal tags in a mapping artifact.

CGRA-sim consumes that artifact. It may reject an inconsistent or stale
artifact, but it must not repair it and must not choose a new mapping.
If simulation feedback should change the mapping, a later PnR or DSE
run produces a new mapping artifact.

## Hardware Scope

CGRA-sim models the selected hardware graph at the abstraction level of
Fabric ADG and the mapping artifact. Target hardware scope includes:

* `acc_core` execution contexts;
* SpatialCore resources inside referenced `fabric.module` templates;
* modeled ScalarCore residual execution when present in the mapping;
* `fabric.pe`, `fabric.fu`, `fabric.mem`, `fabric.switch`,
  `fabric.boundary`, and `fabric.fifo` resources when referenced;
* system-level nodes, ports, channels, links, adapters, routers,
  network endpoints, arbiters, route decoders, broadcasts, memories,
  and caches;
* clock-domain crossings and link latencies when modeled;
* memory hierarchy, coherence-domain effects, and consistency-model
  constraints at the target abstraction level;
* temporal sharing, temporal tags, schedule slots, buffers, and
  backpressure.

CGRA-sim is not required to be cycle-accurate RTL simulation. Its
accuracy level is controlled by simulator configuration and reported in
the output.

## Execution Model

CGRA-sim advances hardware-aware time. Each simulated event must be
consistent with:

* dataflow token availability and operation semantics;
* mapping artifact placement records;
* route and buffer records;
* resource capacity;
* schedule and temporal-tag records;
* memory bindings and memory-order constraints;
* Fabric ADG link, protocol, latency, bandwidth, domain, and resource
  metadata;
* simulator configuration.

If a required mapping fact is missing, CGRA-sim must diagnose the
mapping artifact instead of inventing a default placement, route, or
schedule.

## Hardware-Aware Metrics

Baseline CGRA-sim metrics include:

* total cycles;
* per-node and per-resource active cycles;
* per-resource utilization;
* route use counts and contention;
* queue occupancy over time;
* stall cycles by cause;
* memory request counts, bandwidth, latency, and coherence activity;
* temporal tag use;
* reconfiguration activity;
* scalar residual execution activity when modeled;
* output values and final memory diffs;
* diagnostics.

CGRA-sim reports may be consumed by DSE and later PnR runs as feedback.
They do not modify the original mapping artifact unless a separate tool
explicitly creates a new artifact or mapping-set manifest.

## Determinism

Given the same dataflow IR, Fabric ADG, mapping artifact, runtime input,
initial memory state, and simulator configuration, CGRA-sim must produce
the same report. Any stochastic simulation mode must be explicit and
must record its seed.

## Report Contract

A CGRA-sim report must identify:

* software IR root;
* selected `fabric.system`;
* mapping artifact identity and fingerprint when available;
* simulator schema version;
* runtime input identity or fingerprint when available;
* simulator configuration and fidelity level;
* functional outputs and memory diffs;
* hardware-aware metrics;
* trace location or inline trace summary;
* diagnostics.

## Non-Goals

CGRA-sim is not PnR. It does not choose mappings.

CGRA-sim is not DFG-sim. It does not ignore hardware resource limits.

CGRA-sim is not RTL simulation. It may be checked against RTL
simulation later, but it does not replace RTL validation.

CGRA-sim is not FPA estimation. It may produce activity data consumed
by the FPA flow specified in `docs/spec-fpa-estimation.md`, but it does
not by itself produce final frequency, power, or area estimates.

## Acceptance Criteria

CGRA-sim is complete at the target-spec level when:

* it consumes dataflow IR, Fabric ADG, a mapping artifact, and runtime
  input data;
* it rejects stale or inconsistent mapping artifacts;
* it preserves DFG-sim functional behavior for legal mappings;
* it reports hardware-aware cycles, activity, stalls, route activity,
  queue occupancy, memory activity, and temporal reuse;
* it does not choose placements, routes, schedules, buffers, memory
  bindings, or resource sharing;
* its reports can be compared against DFG-sim reports through
  `docs/spec-sim-comparison.md`;
* its reports can feed later PnR, DSE, or FPA flows as explicit input
  evidence.
