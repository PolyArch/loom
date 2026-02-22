# Loom Mapper Algorithm Specification

## Overview

This document defines algorithm-facing interfaces and required behavior for
mapper search procedures.

It does not prescribe a single heuristic. Instead, it defines the contract that
all mapper algorithms must satisfy, and specifies the recommended PnR pipeline.

Data structures and constraints referenced here are defined in
[spec-mapper-model.md](./spec-mapper-model.md).

## Algorithm Contract

Any mapper algorithm implementation must:

1. Start from an empty `MappingState`.
2. Produce either:
   - a valid mapping satisfying all hard constraints, or
   - a failure result with diagnostics.
3. Preserve state consistency after every committed action.
4. Record every action in an ordered action log for reproducibility and
   potential RL training (see Action Sequencing).

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

### Action Logging

Every successfully committed action must be appended to an ordered action log.
The log records:

- Action type and arguments (node/port/edge IDs, candidates chosen).
- Pre-action cost snapshot (optional, for RL reward computation).
- Post-action cost delta.
- Constraint check result.

This log enables:

- Deterministic replay of a mapping solution.
- Training data for reinforcement learning policies.
- Debugging and diagnostic analysis.

## PnR Pipeline

The mapper follows a multi-stage Place-and-Route pipeline. Each stage can
succeed, partially succeed (requiring repair), or fail.

### Stage 0: Preprocessing

**ADG flattening**: Inline all `fabric.instance` references. Build the
two-level ADG structure described in
[spec-mapper-model.md](./spec-mapper-model.md).

**Connectivity matrix construction**: Scan all routing nodes to build a
global reachability structure mapping output ports to reachable input ports
through routing infrastructure. For each routing node type:

- `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`, `fabric.fifo`:
  single `inputPortId -> outputPortId` pass-through.
- `fabric.switch`, `fabric.temporal_sw`: `inputPortId ->
  [outputPortId, ...]` per `connectivity_table`.

Combined with physical edges (`outputPortId -> inputPortId`), this forms
the complete connectivity matrix for routing queries.

**Minimum routing cost precomputation**: For each pair of hardware nodes,
precompute the minimum-hop routing cost (shortest path in the connectivity
matrix). This table is used during placement to estimate routing quality
without actually routing.

### Stage 1: Tech-Mapping and Candidate Construction

**Tech-mapping**: For each unique `fabric.pe` body pattern in the ADG:

1. Extract the body subgraph (operation names, connectivity).
2. Deduplicate identical patterns across PE instances.
3. For single-operation patterns: scan all DFG nodes for compatible
   operations (respecting exclusivity and runtime-config rules from
   [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md)).
4. For multi-operation patterns: run subgraph isomorphism matching against
   the DFG. Each match identifies a group of DFG operations that can map
   to one PE as a unit.

**Overlapping operation group handling**: When multiple multi-operation PE
body patterns share a DFG node, all valid matches are retained as
candidates. Both resolution strategies are supported; heuristic is the
default:

- **Heuristic (default)**: Greedy filtering during placement. When
  placing a DFG node, prefer the longest matching operation group
  (covers the most DFG nodes per PE). On tie, prefer the group with
  fewer candidates (more constrained). Once a group is selected, all
  overlapping groups sharing any DFG node with the selected group are
  invalidated.
- **CP-SAT**: Mutual exclusivity constraints ensure at most one group
  covering each DFG node is selected. The solver finds the globally
  optimal combination.

The heuristic strategy is used by all profiles except `cpsat_full`.
The CP-SAT strategy is used when `cpsat_full` is selected or when
sub-problem refinement targets an overlapping region.

**Candidate set construction**: For each DFG node (or operation group),
build `CandidateSet(swNode)` containing all compatible hardware nodes.

**Memory candidate construction**: Match `handshake.memory` to
`fabric.memory` and `handshake.extmemory` to `fabric.extmemory` per
capacity and type constraints (C1, C4).

**Reachability check**: Verify that every DFG operation has at least one
candidate. If any operation has an empty candidate set, report
`CPL_MAPPER_NO_COMPATIBLE_HW` and abort.

### Stage 2: Initial Heuristic Placement

Place all DFG operation nodes onto ADG nodes without routing. Sentinel nodes
(`ModuleInputNode`, `ModuleOutputNode`) are excluded from placement; they
are fixed at the module I/O boundary. Instead, sentinel ports are bound to
their ADG counterparts via `MapPort` (type and bit-width compatible). Use
precomputed minimum routing costs to estimate placement quality.

**Node ordering**: Process DFG operation nodes in topological order (data
dependency). Memory operations are placed first (most constrained), then
compute operations.

**Candidate selection heuristic**: For each DFG node, score candidates by:

1. Proximity to already-placed neighbors (using precomputed min-hop cost).
2. Resource utilization balance (avoid overloading one region).
3. Connectivity to already-placed nodes.

Select the best candidate and commit `MapNode`.

**Operation group handling**: When a multi-operation PE body match is
selected, all operations in the group are placed simultaneously to the
same hardware node via batch `MapNode`.

### Stage 3: Initial Routing

Route all DFG edges through the ADG connectivity matrix. This includes
edges between operation nodes and edges incident to sentinel nodes. For
sentinel-incident edges, routing starts or ends at the bound hardware
sentinel port (established via `MapPort` in Stage 2).

**Edge ordering**: Route edges in order of estimated difficulty (longest
minimum-hop distance first, fan-out edges before simple edges).

**Routing algorithm**: BFS for initial routing, upgradable to A* with
cost-aware heuristic:

1. Start from the source hardware output port (either a placed operation's
   output port or a bound sentinel output port).
2. Expand through routing node internal connections and physical edges.
3. Find the shortest legal path to the destination hardware input port
   (either a placed operation's input port or a bound sentinel input port).
4. Commit `MapEdge(swEdge, path)`.

**Fan-out routing**: For DFG output ports with multiple consumers (SSA
fan-out), route each consumer edge independently. The routing
infrastructure (`fabric.switch` broadcast, `fabric.temporal_sw`) handles
physical fan-out. Multiple consumer edges sharing a common prefix through
the same routing infrastructure is both allowed and encouraged for
efficiency.

**Tag allocation during routing**: When multiple software edges must share
a hardware edge (tagged sharing):

1. Assign the smallest available tag value to each edge.
2. If no tag value fits within `TAG_WIDTH`, the route fails.
3. Tag values must remain consistent along the entire path: once a tag is
   assigned (via `fabric.add_tag` or `fabric.map_tag`), subsequent routing
   nodes must preserve or legally transform it.

**Route failure**: If no legal path exists for an edge, flag it for the
repair stage.

### Stage 4: Joint Place-and-Route Refinement

Iteratively improve placement and routing together.

**Repair loop**: For failed routes or high-cost mappings:

1. Identify the conflicting region (edges that failed routing, congested
   nodes).
2. Unmap the conflicting nodes and their connected edges.
3. Re-place with alternative candidates, considering routing feasibility.
4. Re-route affected edges.

**Refinement strategies** (in priority order):

1. **Rip-up and re-route** (attempts 0-4): Remove routes for conflicting
   edges. Re-route with A* using increased congestion penalty. Cheapest
   strategy; resolves most routing conflicts.
2. **Node migration** (attempts 5-7): Unmap a placed DFG node and re-place
   to a less congested candidate. Re-route all its edges. Used when
   rip-up alone cannot resolve congestion.
3. **Local swap** (attempt 8): Exchange placements of two DFG nodes if
   their neighbors suggest a swap would reduce routing cost. Used when
   migration fails because both candidates are equally congested.
4. **Partial restart** (attempt 9): Unmap a cluster of DFG nodes (the
   conflicting region + immediate neighbors) and re-place/re-route them.
   Last resort before global restart.

The attempt ranges above are defaults. Strategy selection must be
deterministic: given the same attempt number, the same strategy is
always chosen.

**Bounded iteration**: Repair loops are bounded by configurable limits:

- `max_local_repairs`: maximum repair attempts per conflict (default: 10).
- `max_global_restarts`: maximum full restart attempts (default: 3).
- `max_refinement_iterations`: maximum refinement passes (default: 50).

### Stage 5: Temporal Assignment

For DFG operations mapped to temporal PE FUs:

- Assign instruction slots sequentially (slot 0, 1, 2, ...). Slot
  assignment is trivial because all slots are equivalent.
- Assign tag values for instruction matching.
- Identify intra-temporal-PE edges and assign registers (sequential
  allocation; all registers are equivalent).

For DFG edges routed through `fabric.temporal_sw`:

- Assign route-table slots, match tags, and route masks.

**Resource exhaustion handling (fail-fast)**: Stage 5 validates temporal
resource limits immediately during assignment, not deferred to Stage 6:

- If total mapped operations across all FUs of a temporal PE exceeds
  `num_instruction`, assignment returns `failed_resource_unavailable`.
- If total register-routed edges exceeds `num_register`, assignment
  returns `failed_resource_unavailable`.

On failure, Stage 5 does not partially assign the temporal PE. Any
slot, tag, or register assignments made to the failing temporal PE
before the limit was detected are rolled back, leaving the temporal PE
in its pre-Stage-5 state (no partial temporal metadata remains).
The failing temporal PE and its mapped operations are flagged for the
repair loop (Stage 4). The repair strategy for temporal exhaustion is
node migration: move one or more operations from the overloaded temporal
PE to an alternative candidate (a different temporal PE or a non-temporal
PE), then re-route affected edges and retry temporal assignment.

### Stage 6: Validation and Configuration Emission

- Run full-state validation against all constraint classes (C1-C6).
- Compute cost metrics per [spec-mapper-cost.md](./spec-mapper-cost.md).
- Generate per-node configuration fragments.
- Assemble into `config_mem` image per
  [spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## CP-SAT Integration

The mapper supports two CP-SAT modes, selectable by `--mapper-profile` or
automatic threshold based on DFG size.

### Full-Problem Mode

For small DFG instances (default threshold: 50 nodes), formulate the entire
placement and routing problem as a CP-SAT model.

**Decision variables**:

- `x[sw][hw]`: boolean, DFG node `sw` placed on ADG node `hw`.
  Only created for `(sw, hw)` pairs where `hw` is in `CandidateSet(sw)`.
- `r[se][he]`: boolean, DFG edge `se` uses ADG edge `he`.
- `t[sw][slot]`: boolean, DFG node `sw` uses temporal slot `slot`
  (only for temporal PE mappings).

**Hard constraints (layered)**:

| Layer | Constraints | CP-SAT Treatment |
|-------|------------|------------------|
| Modeled in CP-SAT | C1 (node compatibility, group atomicity/exclusivity), C3.1-C3.4 (route connectivity), C4.1-C4.5 (capacity: exclusive/tagged edges, memory), C5.1-C5.3 (temporal slots, registers) | Encoded as CP-SAT constraints |
| Deferred to post-solve | C2 (type compatibility, pre-filtered via CandidateSets), C3.5 (memory done-token wiring), C3.6 (fan-out broadcast enforcement), C4.6 (addr_offset_table assignment), C5.4 (tag uniqueness per slot), C6 (configuration encoding) | Validated after CP-SAT produces a solution; solution rejected if violated |

Post-solve validation runs before comparing the CP-SAT solution with the
heuristic baseline. A CP-SAT solution that fails deferred constraints is
discarded in favor of the heuristic result.

**Objective**: minimize `total_cost` from
[spec-mapper-cost.md](./spec-mapper-cost.md).

**Warm start**: initialize from the heuristic solution (Stages 2-5) to
accelerate convergence.

**Time budget**: configurable via `--mapper-budget` (default: 60 seconds).

### Sub-Problem Refinement Mode

For larger DFG instances, use CP-SAT only for bounded local sub-problems:

- **Local re-placement**: extract a congested region (up to 50 DFG nodes
  and their candidates), formulate as CP-SAT, solve, and integrate.
- **Conflict resolution**: when heuristic repair loops fail, formulate the
  conflicting sub-graph as CP-SAT.
- **Temporal optimization**: formulate temporal slot/register assignment
  as CP-SAT when greedy assignment has poor quality.

### Mode Selection

| Profile | Behavior |
|---------|----------|
| `balanced` | auto-select based on DFG size (threshold: 50 nodes) |
| `cpsat_full` | force full-problem mode regardless of size |
| `heuristic_only` | disable CP-SAT entirely |
| `throughput_first` | heuristic + sub-problem CP-SAT, throughput weights |
| `area_power_first` | heuristic + sub-problem CP-SAT, area/power weights |
| `deterministic_debug` | heuristic only, maximum determinism |

## Action Sequencing for Reinforcement Learning

The mapper's action-based interface is designed to support RL training as a
future optimization path.

### State Representation

The RL state at any point during mapping consists of:

- Current `MappingState` (occupancy, utilization).
- DFG topology features (node degrees, critical path position, types).
- ADG resource availability (per-node occupancy, per-edge congestion,
  remaining temporal slots).
- Candidate sets for unmapped nodes.

### Action Space

Each RL action corresponds to one action primitive:

- `MapNode(swNode, hwNode)`: choose which unmapped node to place and where.
- `MapEdge(swEdge, pathIndex)`: choose which unrouted edge to route and
  which path to select from precomputed legal paths.
- `UnmapNode` / `UnmapEdge`: undo actions for repair.

The action space at each step is the set of legal actions given the current
state.

### Reward Signal

- **Intermediate**: negative cost delta after each action.
- **Terminal**: large positive for valid complete mapping; large negative
  for failure.
- **Shaping**: bonus for reducing candidate-set entropy, penalty for
  creating routing congestion.

### Episode Structure

One episode is one complete mapping attempt. The heuristic mapper produces
training episodes. An RL policy can later replace or augment decisions.

### RL Compatibility Requirements

1. Expose `step(action) -> (state, reward, done)` interface.
2. Deterministic given same state and action.
3. Support checkpoint/restore of `MappingState` for tree search.
4. Record complete action sequences with state snapshots.

## Tie-Breaking Rules

Deterministic by default:

- Stable sort by candidate identifier.
- Stable sort by path length, then identifier.
- Seeded randomness only when explicitly enabled via `--mapper-seed`.

## Conflict Resolution Policy

Escalation order:

1. Reroute conflicting edges.
2. Reassign conflicting node placements.
3. Reassign temporal metadata.
4. Invoke CP-SAT sub-problem refinement (if enabled).
5. Restart from checkpoint or fail with minimal conflict report.

All repair loops are bounded by configurable limits.

## Algorithm Quality Requirements

At minimum, a production mapper should provide:

- One deterministic baseline algorithm (the PnR pipeline above).
- One cost-aware optimization mode (CP-SAT integration).
- Deterministic failure diagnostics.

Performance optimizations must not violate hard constraints.

## Diagnostics Requirements

On failure, the algorithm output should identify:

- First unsatisfied hard-constraint class.
- Conflicting software/hardware resource identifiers.
- Candidate set sizes for unmapped nodes.
- Congestion hotspots for failed routing.
- Last successful checkpoint (if checkpoints are used).

## Related Documents

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
