# PE Port Binding, Sentinel Routing, and Temporal Register Toggle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the mapper's PE-FU port binding model so that input mux / output demux selection participates in routing, sentinels participate in placement SA, and temporal PE register assignment is a dynamic routing decision.

**Architecture:** Introduce explicit PE nodes into the flattened ADG with PE exterior ports. Extend ConnectivityMatrix with `peInToFuIn` / `fuOutToPeOut`. Make scalar sentinels participate in SA swap/relocate. Add a temporal register toggle move to Route-aware SA.

**Tech Stack:** C++ (MLIR/LLVM codebase), loom Mapper, ADGFlattener, ConfigGen

---

## Background and Root Cause

### Problem 1: PE-FU Port Binding

The ADG flattener currently creates FU nodes with direct FU-to-FU edges (no PE-level ports in the graph). Each PE result is mapped to ALL FU output ports via all-to-all wiring, producing duplicate edges between the same port pairs with different `pe_output_index` attributes. The router's `spatialPEOutputIndexUsedByDifferentFUResult` check uses `findEdgeByPortsLocal` which returns only the first matching edge. When duplicate edges exist, this returns the wrong `pe_output_index`, causing false conflict detection.

This blocks routing on direct PE-to-PE topologies (no switches). In switch-based ADGs, the problem is masked because the first routing hop goes to a switch input, and the switch provides alternate paths.

### Problem 2: Sentinel Routing

Input sentinels have `rebindScalarInputSentinels` for cost-based rebinding, but output sentinels have no corresponding rebind. Neither input nor output sentinels participate in SA placement (swap/relocate). Only `OperationNode` types are collected for SA moves.

### Problem 3: `globalAllRouted` Semantic Bug

In `MapperLocalRepairExact.cpp`, `globalAllRouted = localAllRouted` incorrectly sets the global "all routed" flag when only a repair neighborhood's edges are routed. This causes `updateBest(allRouted=true)` to overwrite a better routing state (e.g. 7/9 routed) with a worse local repair state (e.g. 1/9 routed).

### Problem 4: Temporal Register Assignment

Current `classifyTemporalRegisterEdges` statically marks all eligible edges as `TemporalReg` before routing. There is no mechanism to choose which edges should use registers vs external routing when the number of eligible edges exceeds `numRegister`. The assignment does not participate in Route-aware SA iteration.

---

## Change 1: Flattened ADG Representation

### PE Node

ADGFlattener creates one PE node per `fabric.spatial_pe` / `fabric.temporal_pe` instance, added to `Graph::nodes`. The PE node owns:

- PE exterior input ports: count = `peFnType.getNumInputs()`, types from PE function type
- PE exterior output ports: count = `peFnType.getNumResults()`, types from PE function type
- Node attributes: `op_kind = "pe_container"`, `pe_kind = "spatial_pe" / "temporal_pe"`, `pe_name = instanceName`

PE nodes are not `OperationNode`. They do not participate in DFG op placement. They are hardware topology entities only.

FU nodes remain unchanged: each FU has its own ports from `fuOp.getFunctionType()`, with `pe_name` pointing to the owning PE.

`PEContainment` gains a `peNodeId` field pointing to the new PE node.

### ConnectivityMatrix Extension

```
struct ConnectivityMatrix {
  // Cross-module: PE output -> PE input, sentinel -> PE input, PE output -> sentinel, etc.
  DenseMap<IdIndex, SmallVector<IdIndex, 4>> outToIn;

  // Routing node internal: switch input -> switch output, fifo, add/map/del_tag
  DenseMap<IdIndex, SmallVector<IdIndex, 4>> inToOut;

  // PE internal: PE exterior input -> FU input (input mux, full crossbar)
  DenseMap<IdIndex, SmallVector<IdIndex, 4>> peInToFuIn;

  // PE internal: FU output -> PE exterior output (output demux, full crossbar)
  DenseMap<IdIndex, SmallVector<IdIndex, 4>> fuOutToPeOut;
};
```

**peInToFuIn construction:** For each PE exterior input port, connect to all input ports of all FUs in the PE. This represents the input mux full crossbar.

**fuOutToPeOut construction:** For each FU output port, connect to all PE exterior output ports. This represents the output demux full crossbar.

**Sizing note for mux/demux matrices:**
- Input mux matrix: `numPEInputs x maxFUInputs` (max across all FUs in the PE)
- Output demux matrix: `maxFUOutputs x numPEOutputs` (max across all FUs in the PE)

### outToIn Changes

Currently `outToIn` connects FU output port -> downstream FU input port directly. After this change:
- `outToIn` connects PE exterior output port -> downstream PE exterior input port (or switch input, sentinel input, etc.)
- Sentinel output port -> PE exterior input port is in `outToIn`
- PE exterior output port -> sentinel input port is in `outToIn`
- FU-to-FU direct edges are removed from `outToIn`

### Removal of pe_output_index / pe_input_index Edge Attributes

These edge attributes are no longer needed. In the new representation, PE ports are explicit graph ports. Config gen reads mux/demux selections directly from the routing path positions.

### Removal of spatialPEOutputIndexUsedByDifferentFUResult

This function and its call in `isEdgeLegalCached` are removed. PE output exclusivity is enforced naturally by the routing port occupancy model (for spatial PE) or by tagged path validation (for temporal PE).

---

## Change 2: Router findPath Adaptation

### Path Format

Routing paths change from `[fuA_out, nextFuB_in]` to `[fuA_out, peA_out, ..., peB_in, fuB_in]`.

The path includes PE internal traversal hops. For a direct PE-to-PE connection, the minimal path is: `[fuA_out, peA_out, peB_in, fuB_in]`.

### Search State Machine

`findPath` currently branches on port direction:
- Output port -> traverse `outToIn`
- Input port -> traverse `inToOut`

Extended logic:
1. **FU Output port** (parentNode is FU node) -> traverse `fuOutToPeOut` to candidate PE output ports
2. **PE Output port** (parentNode is PE node) -> traverse `outToIn` to cross-module destinations
3. **Switch/routing Input port** (parentNode is routing node) -> traverse `inToOut` as before
4. **PE Input port** (parentNode is PE node) -> traverse `peInToFuIn` to candidate FU input ports
5. **Non-routing Output port** (sentinel output, etc.) -> traverse `outToIn` as before
6. **Memory node** -> maintain existing pass-through blocking logic

### Legality Checks

For `fuOutToPeOut` traversal:
- Spatial PE: the PE output port must not already be used by a different sw edge (exclusivity)
- Temporal PE with tagged ports: sharing allowed, tag representability checked by existing tagged path validation

For `peInToFuIn` traversal:
- Same exclusivity rules as output side for spatial PE
- Temporal PE allows sharing via tag

### Path Cost

PE internal hops (fuOut->peOut, peIn->fuIn) have zero routing cost. They represent PE-internal mux/demux configuration decisions, not external routing resource consumption. The router's A* heuristic should not penalize these hops.

### isNonRoutingBroadcastTransitionConflict Adaptation

This constraint applies to PE exterior output ports in the new representation:
- Non-tagged PE output (spatial PE): one PE output port can carry only one sw edge. Enforced by port occupancy.
- Tagged PE output (temporal PE): multiple sw edges may share one PE output port if tag values are representable within the port's tag width.

---

## Change 3: Scalar Sentinel SA Participation

### Candidate Building

`buildCandidates` is extended to build candidate lists for scalar sentinels:
- Each DFG scalar input sentinel: candidates = all type-compatible ADG input sentinels
- Each DFG scalar output sentinel: candidates = all type-compatible ADG output sentinels
- Compatibility condition: `bitWidth(swPort.type) <= bitWidth(hwPort.type)`

Memref sentinels do NOT build candidates and do NOT participate in SA. Design rationale: memref sentinel binding is determined automatically by memory interface placement. Once a `handshake.extmemory` / `handshake.memory` is mapped to a `fabric.extmemory` / `fabric.memory`, the memref input/output sentinels bind to the corresponding memory interface's memref ports. This binding follows from the memory interface placement, not from independent sentinel routing. See spec-dataflow-memory and spec-fabric-memory-interface for the memory model.

### SA Move Types for Sentinels

The SA (runPlacement, runRefinement, runRouteAwareSA) collects scalar sentinels alongside operation nodes. When a sentinel is selected for a move:

- **Sentinel swap**: exchange two sentinels' ADG bindings. Only between same-side sentinels (input-input or output-output). Both must remain width-compatible after swap.
- **Sentinel relocate**: move a sentinel to another type-compatible ADG boundary port.
- Sentinels and OperationNodes do not cross-swap (semantically different node types).

### Output Sentinel Rebind

New function `rebindScalarOutputSentinels`, symmetric to `rebindScalarInputSentinels`:
- Collects all DFG scalar output sentinels and ADG output sentinels
- Estimates routing cost for each pair (based on incoming edge placement distance)
- Uses backtracking search for optimal assignment
- Called after `rebindScalarInputSentinels`, before routing

Call sites:
- `runInterleavedPlaceRoute`: after `rebindScalarInputSentinels`
- `runLanePlacementSeed` / `runLaneRouting`: same position

### Existing rebindScalarInputSentinels

Retained as-is. It continues to serve as a final optimization before routing.

---

## Change 4: globalAllRouted Semantic Fix

### Location

`MapperLocalRepairExact.cpp`, function `runExactRoutingRepair`.

### Fix

Remove the assignment `globalAllRouted = localAllRouted` (where `localAllRouted = bestLocalRouted == repairEdges.size()`). This value only means "all edges in the repair neighborhood are routed", not "all DFG edges are routed".

`globalAllRouted` should only be set to `true` at the point where `collectUnroutedEdges` returns an empty set (confirming all DFG edges are routed).

The neighborhood-level success (`localAllRouted`) remains used for local flow control (early return from `attemptExactNeighborhood`), but does not propagate as global routing success.

### Impact

This is an independent fix. After correction, `updateBest(repaired=true)` in `runInterleavedPlaceRoute` will no longer overwrite a better routing state with a worse local repair state.

---

## Change 5: Temporal PE Register Toggle

### Data Structure

New field in `MappingState`:

```
// sw edge -> register index within the temporal PE, or absent if not using register
DenseMap<IdIndex, unsigned> temporalRegisterAssignment;
```

### Assignment Constraints

- Only edges where both endpoints are mapped to FUs within the same temporal PE are eligible
- Unique register indices assigned per temporal PE must not exceed `numRegister`
- Same register index can only be used by edges sharing the same source port (broadcast semantics: one register's dequeue side serves multiple readers)
- All registers within a PE are equivalent (no ordering preference)

### Interaction with Routing

- Assigned to register: routing skips the edge, `swEdgeToHwPaths[edgeId]` is empty
- Not assigned: must route externally via `[fuOut, peOut, ..., peIn, fuIn]` with tag conflict checks
- When an edge is removed from register: external route must be found
- When an edge is assigned to register: its external route resources are released

### Route-aware SA Register Toggle Move

Third move type alongside swap and relocate:

1. Select a temporal PE that has at least one eligible internal dependency edge
2. Select an edge, toggle its state:
   - If currently on register: remove register assignment, mark for external routing
   - If currently on external route and registers not full: assign a register, rip-up external route
3. Re-route affected edges
4. Evaluate cost change, accept/reject per SA criteria

### Replacing TechMappedEdgeKind::TemporalReg

`TechMappedEdgeKind::TemporalReg` is no longer statically assigned by `classifyTemporalRegisterEdges`. Instead, `classifyTemporalRegisterEdges` initializes `temporalRegisterAssignment` with a greedy allocation for eligible edges (respecting `numRegister` limit). Subsequent SA iterations can dynamically adjust assignments via register toggle moves.

The routing code checks `temporalRegisterAssignment` to decide whether to skip an edge, replacing the static `edgeKinds[edgeId] == TemporalReg` checks.

### Config Gen

When `temporalRegisterAssignment[edgeId]` has a value:
- Producer instruction result config: `res_is_reg = 1, res_reg_idx = registerIndex`
- Consumer instruction operand config: `op_is_reg = 1, op_reg_idx = registerIndex`

When the edge routes externally:
- Producer: `res_is_reg = 0`, output demux sel from path
- Consumer: `op_is_reg = 0`, input mux sel from path

---

## Change 6: Config Gen Adaptation

### collectPERouteSummary

No longer reads `pe_output_index` / `pe_input_index` from edge attributes. Instead, extracts PE port information directly from routing paths:

For a path `[fuA_out, peA_out, ..., peB_in, fuB_in]`:
- Source side: find `peA_out` position in PE node's `outputPorts` -> output demux sel value. Find `fuA_out` position in FU node's `outputPorts` -> FU output index.
- Destination side: find `peB_in` position in PE node's `inputPorts` -> input mux sel value. Find `fuB_in` position in FU node's `inputPorts` -> FU input index.

`PERouteSummary.inputPortSelects[fuId][fuInputIdx]` and `outputPortSelects[fuId][fuOutputIdx]` are populated from path positions. The downstream consumers (`buildSpatialPEConfig`, `buildTemporalPEConfig`) are unchanged since they consume the same `PERouteSummary` interface.

### Visualization

map.json / viz.html edge routing information adapts to the new path format. port_table includes PE exterior ports so that visualization can render PE-to-FU mux/demux wiring.

---

## Change 7: Validation

### Adapted Checks

**C3 (Unrouted edges):** Edges in `temporalRegisterAssignment` are treated as routed (skip check). `TechMappedEdgeKind::IntraFU` edges still skipped. All other edges must have non-empty `swEdgeToHwPaths`.

**C5.2 (Temporal register overflow):** Counts unique register indices in `temporalRegisterAssignment` per temporal PE, checks against `numRegister`.

### New Checks

**C-PE-1 (Spatial PE single-FU):** Each spatial PE has at most one active FU in the final mapping. Note: Route-aware SA may temporarily violate this for escaping local optima, but the final legal solution must satisfy it.

**C-PE-2 (Spatial PE output demux exclusivity):** Each spatial PE output port is used by at most one sw edge's routing path.

**C-PE-3 (Temporal PE instruction count):** Total sw ops mapped to FU nodes within a temporal PE does not exceed `numInstruction`.

**C-PE-4 (PE port width compatibility):** For fuOut->peOut traversal: `bitWidth(fuOut.type) <= bitWidth(peOut.type)`. For peIn->fuIn traversal: `bitWidth(fuIn.type) <= bitWidth(peIn.type)`.

---

## Testing

### Direct Fix Targets

- **dataflow-flowops**: 3 spatial PEs directly connected, should achieve 9/9 routing
- **mapper-fifo-recurrence-guard**: same topology + bypassable fifo, should pass
- **temporal-pe-registers**: temporal PE port binding and register routing

### Regression

- **check-loom-unit** (127 tests): no degradation
- **check-loom-e2e** (50 tests): switch-based ADGs behave identically
- **KHG ADG vecadd** (24 variants): KHG-generated ADGs map correctly

### Key Regression Risk Areas

1. **Path format change**: all consumers of `swEdgeToHwPaths` must handle the longer path with PE internal hops (config gen, visualization, map.json export)
2. **Placement cost**: functions using placement distance must handle PE node position consistently with FU node positions
3. **TechMapper contracted DFG**: `expandPlanMapping` must adapt to the new path format when expanding contracted state to original state
