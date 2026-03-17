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
2. initial placement
3. placement refinement
4. routing
5. discard and disconnect assignment
6. validation
7. output generation

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
- visualization payloads that preserve component-local route meaning

## Related Documents

- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-output.md](./spec-mapper-output.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
