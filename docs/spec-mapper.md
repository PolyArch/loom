# FCC Mapper Specification

## Overview

The mapper is the bridge between:

- the software execution graph
- the hardware architecture graph

Its job is not only place-and-route. In FCC it also depends on a prior
tech-mapping step that locks `function_unit` configurations before final
routing.

## Mapper Responsibilities

The FCC mapper must:

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

- Layer 2 tech-mapping may enumerate `fabric.mux` selections, derive an
  effective FU graph, and contract a matched software subgraph into one
  placeable unit
- coarse placement may come from greedy placement or from a bounded CP-SAT
  solve on small enough problems
- placement refinement may include route-aware simulated annealing and
  congestion-aware scoring, but must not mutate FU-internal configuration
- routing is not a one-shot terminal stage in the current architecture;
  FCC may run multiple interleaved place or route rounds, negotiated
  congestion routing passes, and selective rip-up or reroute attempts
- repair may include targeted local re-placement, exact routing repair, or
  bounded CP-SAT neighborhood repair
- the default implementation may explore multiple parallel mapper lanes with
  different seeds and then select the best deterministic winner by routed-edge
  count, cost, and lane index
- final reports are expanded back to original DFG-node and DFG-edge identity

## Read-Only Inputs

The mapper consumes:

- a DFG
- an ADG
- flattening metadata
- search policy such as seed or time budget

The mapper does not mutate hardware topology. It selects legal use of the
existing topology.

## Core FCC-Specific Concerns

Compared with Loom, FCC mapping must account for:

- explicit `function_unit` containment
- FU-internal configuration selection
- non-positional PE port routing through muxes and demuxes
- decomposable switch semantics
- shared-memory bridge structure and runtime tag representability
- interleaved place-and-route instead of one fixed place-then-route pass
- negotiated congestion routing and bounded exact repair
- visualization payloads that preserve component-local route meaning

## Related Documents

- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
