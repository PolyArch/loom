# Route-Aware SA as Main Optimization Loop

## Problem

The current mapper pipeline runs placement-only SA refinement (blind to routing)
as the main optimization pass, then uses route-aware SA only as a fallback when
routing fails in the interleaved retry loop. This means:

- The main SA optimizes a proxy (Manhattan distance) that may not reflect
  actual routability.
- Route-aware SA only triggers on failure paths, not proactively.
- The interleaved retry loop is coarse-grained and deterministic, lacking
  fine-grained stochastic exploration.

## Solution

Promote route-aware SA from a fallback mechanism to the main optimization loop.
After initial placement and routing, enter an SA loop where every iteration
swaps/relocates nodes, rips up affected edges, re-routes them, and
accepts/rejects based on a combined placement + routing cost function with
Boltzmann acceptance criterion.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Lane structure | Merge into single unified lane | Simpler architecture |
| Placement SA | Keep brief warmup (~10% budget) | Gives routing a reasonable starting point |
| Multi-lane | Placement lanes then beam-filter to full-pipeline lanes | Avoids wasting SA budget on bad placements |
| Interleaved loop | Replace with route-aware SA + single local repair pass | SA handles perturbation; local repair is targeted cleanup |
| Re-routing strategy | Tiered: cheap re-route every move, exact repair on small failures, periodic full re-route | Every move is route-aware; periodic global optimization |
| Budget | Configurable fractions, default majority (~60%) to route-aware SA | Flexibility for different workloads |

## Pipeline

```
Phase 1: Placement Lanes (parallel, N lanes)
  +-- Constructive placement (CP-SAT global or Greedy)
  +-- Warmup SA (placement-only, ~10% budget)
  +-- Sentinel binding
  +-- Return LaneAttempt with placement metrics

Beam-filter: top K lanes by placement quality

Phase 2: Full-Pipeline Lanes (parallel, K lanes)
  +-- CP-SAT boundary repair (existing, unchanged)
  +-- Initial routing (negotiated or single-pass)
  +-- Route-aware SA main loop (~60% budget)
  |     for each iteration:
  |       swap/relocate node
  |       -> rip up affected edges
  |       -> cheap re-route via findPath (per-edge)
  |       -> if cheap re-route fails on <= cap edges: exact repair
  |       -> Boltzmann accept/reject on route-aware cost
  |       -> periodic full re-route (every ~20 accepted moves)
  |       -> cool temperature
  +-- If unrouted edges remain: single runLocalRepair() pass
  +-- Return LaneAttempt with final metrics

Best Lane Selection -> Final Polish -> Tech-Map Feedback -> Bufferization
```

Downstream stages (final polish, tech-map feedback, bufferization) remain
unchanged.

## `Mapper::runRouteAwareSA()`

```
bool Mapper::runRouteAwareSA(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const DenseMap<IdIndex, SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds,
    const Options &opts);
```

Returns true if all edges are routed at the end. Takes `edgeKinds` as a
non-nullable reference (route-aware mode is always on).

### Comparison with `runRefinement()` (warmup)

| Aspect | runRefinement (warmup) | runRouteAwareSA (main loop) |
|---|---|---|
| Route awareness | Off (placement-only) | Always on |
| Per-move re-route | N/A | Tiered: cheap findPath first, exact repair if needed (cap=32) |
| Full re-route | N/A | Every ~20 accepted moves |
| Iteration budget | Iteration count based | Time budget based |
| Cost function | Placement distance + timing proxy | Placement + unrouted penalty (1M) + path length + timing |
| Cooling schedule | Geometric + adaptive | Geometric + adaptive (separate config) |
| Termination | Iteration count or time | Time budget or all edges routed |

### Early termination behavior

If all edges become routed during SA, iterations continue (to optimize path
quality and timing) with accelerated cooling: multiply the current cooling
rate by 0.95 (e.g. 0.997 becomes ~0.947) to converge faster once
routability is achieved. The loop still respects the time budget.

### Shared SA helpers

`runRouteAwareSA()` and `runRefinement()` share the following helpers
in the `loom::mapper_detail` namespace:

- `SACostState` / `initializeSACostState()` / savepoint helpers
- `SAAdaptiveState` / `applyAdaptiveCoolingWindow()`
- `collectPlacementDeltaEdges()`
- `applyPlacementDeltaForMovedNodes()`
- `computeRouteAwareCheckpointCost()`
- `trySwapMove()` / `tryRelocateMove()` (node move helpers)

Savepoint ownership stays in the caller (the SA loop). The caller creates
both `MappingState` and `SACostState` savepoints before calling the move
helper, and rolls back or commits both based on the return value. The
helpers perform `unmapNode`/`mapNode`/`bindMappedNodePorts` and return
whether the move succeeded, but do not touch savepoints.

## Tiered Per-Move Re-Routing Strategy

Each SA iteration uses a tiered approach to re-route affected edges:

1. **Cheap re-route (always):** For each affected edge, call `findPath()`
   directly to attempt a single-shot A* re-route. This is O(ports * log
   ports) per edge, typically sub-millisecond. Most moves that improve
   placement will also improve routing, so this succeeds most of the time.

2. **Exact repair (conditional):** If cheap re-route leaves some edges
   unrouted AND the count of failed edges is <=
   `routeAwareSANeighborhoodEdgeCap` (default 32), call
   `runExactRoutingRepair()` with a hard per-move micro-budget of
   `routeAwareSAExactRepairMicroBudgetMs` (default 50ms). This prevents
   exact repair from dominating SA iteration time.

3. **Coarse fallback:** If too many edges fail cheap re-route (exceeding
   cap), skip exact repair and evaluate cost based on placement delta alone.
   These moves accumulate toward the checkpoint batch counter.

4. **Periodic full re-route:** Every `routeAwareSACheckpointMoveBatch`
   (default 20) accepted moves, call `runRouting()` to globally re-optimize
   all paths. If quality degrades, restore checkpoint.

### Initial routing failure handling

If initial routing (before the SA loop) fails to route any edges (e.g.
due to an extremely congested placement), the SA loop starts with all
edges unrouted. The cheap re-route (findPath) on each move will gradually
establish routes as placement improves. The SA cost function heavily
penalizes unrouted edges (1M multiplier) so the SA will aggressively seek
routable placements. The periodic full re-route checkpoint also provides a
recovery mechanism.

## Configuration

### New options in `MapperRefinementOptions`

```
routeAwareSABudgetFraction = 0.60         // Main SA loop budget fraction
warmupBudgetFraction = 0.10               // Placement-only warmup budget
initialRoutingBudgetFraction = 0.10       // Initial routing budget
routeAwareSANeighborhoodEdgeCap = 32      // Max edges for exact repair fallback
routeAwareSAExactRepairMicroBudgetMs = 50 // Per-move exact repair time cap (ms)
routeAwareSACheckpointMoveBatch = 20      // Full re-route frequency
routeAwareSAInitialTemperature = 100.0    // Separate from warmup temperature
routeAwareSACoolingRate = 0.997           // Slower cooling for more exploration
routeAwareSAMinTemperature = 0.005        // Lower floor for longer tail
```

### Feature toggle in `MapperOptions`

```
enableRouteAwareSAMainLoop = true         // false = old interleaved path
```

When `enableRouteAwareSAMainLoop = false`, all new options are ignored and
the mapper uses the existing `runInterleavedPlaceRoute()` path with no
behavioral change.

### CLI flag

```
--mapper-enable-route-aware-sa=true|false
```

### YAML config keys

```yaml
mapper:
  enable_route_aware_sa_main_loop: true
  refinement:
    route_aware_sa_budget_fraction: 0.60
    warmup_budget_fraction: 0.10
    initial_routing_budget_fraction: 0.10
    route_aware_sa_neighborhood_edge_cap: 32
    route_aware_sa_exact_repair_micro_budget_ms: 50
    route_aware_sa_checkpoint_move_batch: 20
    route_aware_sa_initial_temperature: 100.0
    route_aware_sa_cooling_rate: 0.997
    route_aware_sa_min_temperature: 0.005
```

### Budget allocation

For a given `budgetSeconds` (default 60s), when
`enableRouteAwareSAMainLoop = true`:

| Stage                        | Fraction option                  | Default | Example (60s) |
|------------------------------|----------------------------------|---------|---------------|
| Placement warmup SA          | warmupBudgetFraction             | 0.10    | 6s            |
| Initial routing              | initialRoutingBudgetFraction     | 0.10    | 6s            |
| Route-aware SA main loop     | routeAwareSABudgetFraction       | 0.60    | 36s           |
| Local repair + final polish  | (remainder)                      | 0.20    | 12s           |

Validation: the three explicit fractions must sum to <= 0.85 (leaving at
least 15% for local repair, final polish, and overhead).

### Checkpoint batch semantics

The `routeAwareSACheckpointMoveBatch` counter counts all accepted moves
(both those with successful neighborhood repair and coarse moves). This
differs from the existing `routeAwareCheckpointAcceptedMoveBatch` which
only counts coarse moves. The rationale: in the main loop, cheap re-route
is always attempted, so coarse-only counting would rarely trigger.

### Naming distinction

The new `routeAwareSANeighborhoodEdgeCap` (default 32, controls the main
SA loop) is distinct from the existing `routeAwareNeighborhoodEdgeCap`
(default 16, controls the fallback path in `runRefinement`). Both coexist
in `MapperRefinementOptions`.

## Logging and Diagnostics

`runRouteAwareSA()` emits progress logging (gated by `opts.verbose`):

- On entry: "Route-aware SA: starting with N placed nodes, M routed edges"
- Periodic (every `adaptiveWindow` iterations): temperature, acceptance
  rate, unrouted count, best cost
- On all-routed milestone: "Route-aware SA: all edges routed at iter N,
  switching to accelerated cooling"
- On exit: "Route-aware SA: K accepted moves, best cost X, unrouted Y"

Reuse existing `MapperSearchSummary` counters
(`routeAwareRefinementPasses`, `routeAwareNeighborhoodAttempts`,
`routeAwareNeighborhoodAcceptedMoves`, `routeAwareCoarseFallbackMoves`,
`routeAwareCheckpointRescorePasses`, `routeAwareCheckpointRestoreCount`).
No new counters needed; the semantics are compatible.

## Acceptance Criteria

1. All `check-loom-unit` tests pass (125+ tests)
2. `check-loom-e2e`: `vecadd` on `mesh-6x6-extmem-2` passes
3. `check-loom-e2e`: `sum-array` on `mesh-6x6-extmem-1` passes
4. A/B benchmark via `mapper_benchmark.py`: the new path must route at
   least as many test cases as the old path (same or higher success count)
5. Fallback path (`enableRouteAwareSAMainLoop = false`) produces
   bit-identical output to the current code
