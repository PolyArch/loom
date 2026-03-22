# LOOM Mapper Specification

## Overview

The mapper is the bridge between:

- the software execution graph
- the hardware architecture graph

Its job is not only place-and-route. In LOOM it also depends on a prior
tech-mapping step that locks `function_unit` configurations before final
routing.

## Mapper Responsibilities

The LOOM mapper must:

1. build compatibility candidates between DFG nodes and FU resources
2. select FU internal configurations where configurable FUs exist
3. place software nodes onto legal hardware resources
4. route software edges through legal hardware paths
5. reconstruct PE mux or demux selections and switch route tables
6. emit mapping reports and config fragments
7. diagnose feasibility failures clearly

## Stage Boundaries

The normative mapper pipeline is:

1. tech-mapping
2. coarse placement
3. placement refinement
4. boundary rebinding and route preparation
5. interleaved place-and-route refinement
6. local or exact repair
7. validation
8. output generation

More specifically:

- Layer 2 tech-mapping may instantiate configurable `fabric.function_unit`
  bodies through demand-driven structural states, derive an effective FU graph,
  and contract a matched software subgraph into one placeable unit
- Layer 2 emits explicit support classes, config classes, temporal
  compatibility, conservative fallback information, and DSE-facing metrics; the
  mapper consumes that contract and must not rediscover it ad hoc
- coarse placement may come from greedy placement or from a bounded CP-SAT
  solve on small enough problems
- placement refinement may include route-aware simulated annealing and
  congestion-aware scoring, adaptive cooling, and bounded reheating, but must
  not mutate FU-internal configuration
- routing is not a one-shot terminal stage in the current architecture;
  LOOM may run multiple interleaved place or route rounds, negotiated
  congestion routing passes, and selective rip-up or reroute attempts
- negotiated routing may use both per-iteration historical congestion and
  per-pass present-demand feedback while edges are routed, plus bounded
  decay and early-stop heuristics to avoid late-iteration overfitting
- when explicitly enabled in mapper config, negotiated routing may also use a
  temporary relaxed-overuse mode for non-tagged switch-owned routing outputs;
  this intermediate state must always be followed by legalization before a
  checkpoint may be accepted as the current best mapping
- repair may include targeted local re-placement, exact routing repair, or
  bounded CP-SAT neighborhood repair
- the default implementation may explore multiple parallel mapper lanes with
  different seeds and then select the best deterministic winner by routed-edge
  count, cost, and lane index
- mapper execution is governed by one global wall-clock budget for the full
  mapping run, not separate per-stage budgets
- optional progress snapshots may export the current best expanded checkpoint
  at configured mapper-round or elapsed-time intervals
- final reports are expanded back to original DFG-node and DFG-edge identity

## Read-Only Inputs

The mapper consumes:

- a DFG
- an ADG
- flattening metadata
- a Layer-2 plan containing contracted-unit semantics, selected FU
  configurations, config classes, and temporal-compatibility metadata
- search policy such as seed, time budget, and mapper base configuration

The mapper does not mutate hardware topology. It selects legal use of the
existing topology.

## Core LOOM-Specific Concerns

Compared with the legacy design, LOOM mapping must account for:

- explicit `function_unit` containment
- FU-internal configuration selection
- support-class capacity and config-class compatibility exported by Layer 2
- non-positional PE port routing through muxes and demuxes
- decomposable switch semantics
- shared-memory bridge structure and runtime tag representability
- interleaved place-and-route instead of one fixed place-then-route pass
- negotiated congestion routing and bounded exact repair
- visualization payloads that preserve component-local route meaning

## Related Documents

- [spec-mapper-config.md](./spec-mapper-config.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
