# Loom Mapper Algorithm Specification

## Overview

This document defines algorithm-facing interfaces and required behavior for
mapper search procedures.

It does not prescribe a single heuristic. Instead, it defines the contract that
all mapper algorithms must satisfy.

Data structures and constraints referenced here are defined in
[spec-mapper-model.md](./spec-mapper-model.md).

## Algorithm Contract

Any mapper algorithm implementation must:

1. Start from an empty `MappingState`.
2. Produce either:
   - a valid mapping satisfying all hard constraints, or
   - a failure result with diagnostics.
3. Preserve state consistency after every committed action.

## Required Action Primitives

Algorithms operate through primitive actions. Implementations may expose more
actions, but these primitives are mandatory:

- `MapNode(swNode, hwNode)`
- `UnmapNode(swNode)`
- `MapPort(swPort, hwPort)`
- `UnmapPort(swPort)`
- `MapEdge(swEdge, optionalPathHint)`
- `UnmapEdge(swEdge)`

Batch variants are optional but, if provided, must be transactional.

## Action Semantics

### Deterministic Validation

Each action must run deterministic validation against hard constraints and
return one of:

- `success`
- `failed_hard_constraint`
- `failed_resource_unavailable`
- `failed_internal_error`

Status naming is implementation-defined, but categories must be distinguishable.

### Atomicity

- Single action failure must not partially mutate state.
- Batch action failure must roll back all elements in the batch.

### Side Effects

`MapNode` may trigger default port alignment where policy enables it. If this
happens, side effects must be explicit in returned metadata.

## Baseline Search Workflow

The following workflow is recommended as the default deterministic algorithm:

1. Candidate construction:
   - Build legal hardware candidate sets for each software node.
2. Placement pass:
   - Visit software nodes in dependency order (topological where possible).
   - Choose hardware candidates using local feasibility and cost tie-breaks.
3. Routing pass:
   - Route software edges between placed endpoints using shortest legal paths.
   - Reserve resources as required by sharing policy.
   - Emit per-switch routing selections (`fabric.switch` `route_table`) that
     realize the selected paths.
4. Temporal assignment pass:
   - Assign slot/tag/opcode/register metadata for `fabric.temporal_pe`.
   - Assign slot/tag/route metadata for `fabric.temporal_sw`.
5. Repair loop:
   - If a conflict appears, locally unmap and remap the minimal conflicting
     region before global restart, using bounded retry limits.
6. Final validation and config emission:
   - Run full-state validation and emit configuration fragments.

Alternative strategies (simulated annealing, beam search, RL-guided search) are
allowed if they preserve this contract.

## Tie-Breaking Rules

To keep runs reproducible, tie breaks should be deterministic by default:

- Stable sort by candidate identifier
- Stable sort by path length, then identifier
- Seeded randomness only when explicitly enabled

If randomness is enabled, the seed must be part of mapper input parameters.

## Conflict Resolution Policy

When no legal local action exists, implementations should attempt limited
rollback and repair before declaring global failure.

Recommended escalation order:

1. Reroute conflicting edges.
2. Reassign conflicting node placements.
3. Reassign temporal metadata (`temporal_pe` and `temporal_sw`).
4. Restart from checkpoint or fail with minimal conflict report.

Repair loops must be bounded. Implementations must support configurable limits:

- `max_local_repairs`
- `max_rollback_depth`
- `max_global_restarts`

If any limit is exceeded, the mapper must stop retrying, emit a minimal
conflict diagnostic, and return failure.

## Algorithm Quality Requirements

At minimum, a production mapper should provide:

- One deterministic baseline algorithm
- One cost-aware optimization mode
- Deterministic failure diagnostics

Performance optimizations must not violate hard constraints.

## Diagnostics Requirements

On failure, the algorithm output should identify:

- First unsatisfied hard-constraint class
- Conflicting software/hardware resource identifiers
- Last successful checkpoint (if checkpoints are used)

Diagnostic formatting is implementation-defined, but terms should map to
entities in [spec-mapper-model.md](./spec-mapper-model.md).

## Related Documents

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
