# PE Port Binding, Sentinel Routing, and Temporal Register Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mapper's PE-FU port binding model by introducing explicit PE nodes into the flattened ADG, making mux/demux selection part of routing, sentinels part of SA, and temporal register assignment a dynamic decision.

**Architecture:** The flattened ADG gains PE container nodes with exterior ports. ConnectivityMatrix gains `peInToFuIn`/`fuOutToPeOut` for PE internal crossbar. Router `findPath` traverses PE internals. Scalar sentinels participate in SA. Temporal register assignment becomes a Route-aware SA toggle move. The `globalAllRouted` semantic bug is fixed independently.

**Tech Stack:** C++ (MLIR/LLVM), loom Mapper, ADGFlattener, ConfigGen

**Spec:** `docs/superpowers/specs/2026-04-01-pe-port-binding-and-sentinel-routing-design.md`

---

## File Structure

### Modified Files

| File | Responsibility | Key Changes |
|------|---------------|-------------|
| `include/loom/Mapper/ADGFlattener.h` | ConnectivityMatrix, PEContainment structs | Add `peInToFuIn`/`fuOutToPeOut` to ConnectivityMatrix; add `peNodeId` to PEContainment |
| `lib/loom/Mapper/ADGFlattener.cpp` | Node creation during flattening | Create PE nodes with PE exterior ports |
| `lib/loom/Mapper/ADGFlattenerWiring.cpp` | Edge/connectivity creation | Rebuild `outToIn` to use PE ports; populate `peInToFuIn`/`fuOutToPeOut`; remove `pe_output_index`/`pe_input_index` edge attributes |
| `lib/loom/Mapper/MapperRouting.cpp` | Path finding and legality | Adapt `findPath` for PE internal traversal; remove `spatialPEOutputIndexUsedByDifferentFUResult`; adapt `isEdgeLegalCached`; adapt `isNonRoutingBroadcastTransitionConflict` for tagged/non-tagged PE output |
| `lib/loom/Mapper/MapperRoutingStrategy.cpp` | Routing orchestration | Adapt `routeOnePass` for new path format |
| `lib/loom/Mapper/MapperSentinels.cpp` | Sentinel binding | Add `rebindScalarOutputSentinels` |
| `lib/loom/Mapper/MapperPlacement.cpp` | Placement helpers | Adapt `classifyTemporalRegisterEdges` to initialize `temporalRegisterAssignment`; adapt `computeRoutingEdgeStats`/`collectUnroutedEdges` |
| `lib/loom/Mapper/MapperPlacementMethods.cpp` | Candidate building | Extend `buildCandidates` for scalar sentinels |
| `lib/loom/Mapper/MapperRefinement.cpp` | SA refinement | Include scalar sentinels in `placedNodes`; add sentinel swap/relocate move logic |
| `lib/loom/Mapper/MapperRouteAwareSA.cpp` | Route-aware SA | Include scalar sentinels; add register toggle move type |
| `lib/loom/Mapper/MapperInterleaved.cpp` | Interleaved place-route | Add `rebindScalarOutputSentinels` call |
| `lib/loom/Mapper/MapperLocalRepairExact.cpp` | Exact routing repair | Fix `globalAllRouted` semantic bug |
| `lib/loom/Mapper/Mapper.cpp` | Main orchestration, validation | Add C-PE-1/2/3/4 validation checks; adapt C3/C5.2; adapt lane routing calls |
| `lib/loom/Mapper/MappingState.cpp` | State management | Add `temporalRegisterAssignment` to save/restore/init; adapt `mapNode` if needed |
| `include/loom/Mapper/Mapper.h` | Declarations | Add `rebindScalarOutputSentinels` declaration |
| `lib/loom/Mapper/MapperInternal.h` | Internal helpers | Add helper declarations for PE node identification |
| `lib/loom/Mapper/ConfigGenConfig.cpp` | Config generation | Rewrite `collectPERouteSummary` to read from path positions; adapt temporal register config |
| `lib/loom/Mapper/TechMapperGraphs.cpp` | Contracted DFG | Adapt `expandPlanMapping` for new path format |
| `lib/loom/Mapper/MapperLocalRepair.cpp` | Local repair driver | Adapt `updateBest` for `temporalRegisterAssignment` |

---

## Task 1: Fix globalAllRouted Semantic Bug

**Files:**
- Modify: `lib/loom/Mapper/MapperLocalRepairExact.cpp` -- function `runExactRoutingRepair`, near the `attemptExactNeighborhood` lambda

This is an independent fix with immediate value. It prevents local repair from overwriting a better routing state.

- [ ] **Step 1: Identify the bug location**

In `runExactRoutingRepair`, inside the `attemptExactNeighborhood` lambda, find:
```
globalAllRouted = localAllRouted;
```
where `localAllRouted = bestLocalRouted == repairEdges.size()`. This sets the global flag based on a neighborhood-only result.

- [ ] **Step 2: Fix the assignment**

Remove the `globalAllRouted = localAllRouted` assignment. `globalAllRouted` should only be set to `true` at the point where `currentFailed.empty()` confirms all DFG edges are routed (already exists in the outer loop).

The `localAllRouted` variable and its use for early return from `attemptExactNeighborhood` remain unchanged.

- [ ] **Step 3: Build and run dataflow-flowops test**

Run: `ninja -C build loom-unit-dataflow-flowops`

Expected: The test still fails (because the PE port binding issue remains), but the failure should now show 7/9 routed instead of 1/9 routed. The `bestCheckpoint` is no longer overwritten by local repair.

- [ ] **Step 4: Run full unit test suite**

Run: `ninja -C build check-loom-unit`

Expected: No regressions. The 3 previously-failing tests (dataflow-flowops, mapper-fifo-recurrence-guard, temporal-pe-registers) still fail but with better diagnostic output.

- [ ] **Step 5: Commit**

```
git add lib/loom/Mapper/MapperLocalRepairExact.cpp
git commit -m "Fix globalAllRouted semantic bug in exact routing repair"
```

---

## Task 2: Extend ConnectivityMatrix and PEContainment

**Files:**
- Modify: `include/loom/Mapper/ADGFlattener.h` -- `ConnectivityMatrix` struct, `PEContainment` struct

- [ ] **Step 1: Add new fields to ConnectivityMatrix**

Add two new `DenseMap` fields to `ConnectivityMatrix`:

```cpp
// PE internal: PE exterior input port -> FU input ports (input mux crossbar)
llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> peInToFuIn;

// PE internal: FU output port -> PE exterior output ports (output demux crossbar)
llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> fuOutToPeOut;
```

- [ ] **Step 2: Add peNodeId to PEContainment**

Add to `PEContainment`:

```cpp
IdIndex peNodeId = INVALID_ID;  // Graph node ID of the PE container node
```

- [ ] **Step 3: Add PE node identification helpers to MapperInternal.h**

Add helper functions:

```cpp
bool isPEContainerNode(const Node *node);
// Returns true if node has op_kind == "pe_container"

bool isSpatialPEContainer(const Node *node);
// Returns true if isPEContainerNode && pe_kind == "spatial_pe"

bool isTemporalPEContainer(const Node *node);
// Returns true if isPEContainerNode && pe_kind == "temporal_pe"

const PEContainment *findPEContainmentByNodeId(const ADGFlattener &flattener, IdIndex peNodeId);
// Look up PEContainment by the PE container node ID
```

- [ ] **Step 4: Implement helpers in MapperInternal.h or a suitable .cpp**

Implement the helper functions using node attribute checks (`getNodeAttrStr`).

- [ ] **Step 5: Build**

Run: `ninja -C build loom`

Expected: Clean build. No functional changes yet.

- [ ] **Step 6: Commit**

```
git add include/loom/Mapper/ADGFlattener.h lib/loom/Mapper/MapperInternal.h
git commit -m "Extend ConnectivityMatrix with peInToFuIn/fuOutToPeOut and add PE node helpers"
```

---

## Task 3: Create PE Nodes in ADGFlattener

**Files:**
- Modify: `lib/loom/Mapper/ADGFlattener.cpp` -- `flattenCreateNodes` (or equivalent PE creation code)

- [ ] **Step 1: Create PE container node for each spatial_pe / temporal_pe instance**

In the flattener's node creation pass, after creating FU nodes for a PE, create one PE container node:

Pseudocode:
```
for each PE instance:
    // (existing) create FU nodes with FU ports
    ...
    // (new) create PE container node
    peNode = new Node()
    peNode.kind = OperationNode   // or a new kind if preferred
    set peNode attributes: op_kind="pe_container", pe_kind=peKind, pe_name=instanceName

    // Create PE exterior input ports from PE function type
    for i in 0..peFnType.numInputs:
        peInputPort = new Port(direction=Input, type=peFnType.getInput(i))
        peInputPort.parentNode = peNodeId
        peNode.inputPorts.push_back(peInputPort)

    // Create PE exterior output ports from PE function type
    for i in 0..peFnType.numResults:
        peOutputPort = new Port(direction=Output, type=peFnType.getResult(i))
        peOutputPort.parentNode = peNodeId
        peNode.outputPorts.push_back(peOutputPort)

    peNodeId = adg.addNode(peNode)
    pe.peNodeId = peNodeId
    nodeGridPos[peNodeId] = gridPos  // same grid position as FU nodes
```

- [ ] **Step 2: Update the flattener summary output**

Update the diagnostic output (the `ADGFlattener: N nodes, M ports, E edges` line) to reflect the new PE nodes and ports.

- [ ] **Step 3: Build and verify**

Run: `ninja -C build loom-unit-dataflow-flowops`

Expected: Build succeeds. Test still fails (wiring not yet updated), but the flattener summary should show more nodes and ports than before.

- [ ] **Step 4: Commit**

```
git add lib/loom/Mapper/ADGFlattener.cpp
git commit -m "Create PE container nodes with exterior ports in ADGFlattener"
```

---

## Task 4: Rebuild ADGFlattener Wiring for PE Ports

**Files:**
- Modify: `lib/loom/Mapper/ADGFlattenerWiring.cpp` -- `flattenWireEdges`

This is the core wiring change. The `outToIn` map must use PE exterior ports instead of direct FU-to-FU connections. New `peInToFuIn` / `fuOutToPeOut` maps are populated.

- [ ] **Step 1: Populate peInToFuIn and fuOutToPeOut**

After PE and FU nodes are created, build the PE internal crossbar connectivity:

Pseudocode:
```
for each PEContainment pe:
    peNode = adg.getNode(pe.peNodeId)

    // peInToFuIn: each PE input -> all FU inputs in this PE (full crossbar)
    for peInPort in peNode.inputPorts:
        for fuId in pe.fuNodeIds:
            fuNode = adg.getNode(fuId)
            for fuInPort in fuNode.inputPorts:
                connectivity.peInToFuIn[peInPort].push_back(fuInPort)

    // fuOutToPeOut: each FU output -> all PE outputs (full crossbar)
    for fuId in pe.fuNodeIds:
        fuNode = adg.getNode(fuId)
        for fuOutPort in fuNode.outputPorts:
            for peOutPort in peNode.outputPorts:
                connectivity.fuOutToPeOut[fuOutPort].push_back(peOutPort)
```

- [ ] **Step 2: Rewrite outToIn to use PE exterior ports**

Currently, `outToIn` connects FU output ports directly to downstream FU input ports. Change the wiring logic:

Pseudocode:
```
// Current: valueSrcPorts maps PE result Value -> [(fuOutputPort, peOutputIndex)]
// New: valueSrcPorts maps PE result Value -> [(peExteriorOutputPort)]

// For PE results:
for each PE instance pi:
    for r in 0..pi.op.numResults:
        val = pi.op.getResult(r)
        peNode = adg.getNode(pi.peNodeId)
        peOutPort = peNode.outputPorts[r]
        valueSrcPorts[val] = [{peOutPort, -1}]  // no pe_output_index needed

// For non-PE ops (switches, sentinels, etc.): unchanged

// When creating edges from valueSrcPorts to destination PE inputs:
for each PE instance pi (destination):
    for j in 0..pi.op.numOperands:
        operand = pi.op.getOperand(j)
        srcPorts = valueSrcPorts[operand]
        peNode = adg.getNode(pi.peNodeId)
        peInPort = peNode.inputPorts[j]
        for srcBinding in srcPorts:
            create edge: srcBinding.portId -> peInPort
            connectivity.outToIn[srcBinding.portId].push_back(peInPort)

// For non-PE destinations (switches, sentinels, etc.):
// wire from PE exterior output ports (or sentinel/switch output ports) as before
```

- [ ] **Step 3: Remove pe_output_index / pe_input_index edge attribute generation**

Remove all `setEdgeAttr(edge, "pe_output_index", ...)` and `setEdgeAttr(edge, "pe_input_index", ...)` calls from the wiring code. These attributes are no longer needed since PE ports are explicit.

- [ ] **Step 4: Handle non-PE-to-PE connections**

Ensure that connections involving non-PE entities (sentinel -> PE, PE -> sentinel, switch -> PE, PE -> switch, etc.) use the PE exterior ports in `outToIn`, not FU ports.

For sentinel -> PE input: `outToIn[sentinelOutPort].push_back(peExteriorInPort)`
For PE output -> sentinel: `outToIn[peExteriorOutPort].push_back(sentinelInPort)`

- [ ] **Step 5: Update the yield wiring (module outputs)**

The `fabric.yield` inside `fabric.module` yields values from PE results to module output sentinels. Update to use PE exterior output ports as the source.

- [ ] **Step 6: Build and verify flattener output**

Run: `ninja -C build loom-unit-dataflow-flowops`

Expected: Build succeeds. The flattener should now report non-zero `in->out` equivalent entries for PE internal connectivity. The test will likely crash or fail differently since the router hasn't been adapted yet.

- [ ] **Step 7: Commit**

```
git add lib/loom/Mapper/ADGFlattenerWiring.cpp
git commit -m "Rebuild flattener wiring to use PE exterior ports in outToIn"
```

---

## Task 5: Adapt Router findPath for PE Internal Traversal

**Files:**
- Modify: `lib/loom/Mapper/MapperRouting.cpp` -- `findPath`, `isEdgeLegalCached`, `isEdgeRoutable`
- Modify: `lib/loom/Mapper/MapperRoutingStrategy.cpp` -- `routeOnePass` if needed

- [ ] **Step 1: Adapt findPath search to handle PE internal hops**

In the `findPath` A* search loop, extend the port-direction branching:

Pseudocode:
```
// Current logic:
// if port is Output -> expand via outToIn
// if port is Input  -> expand via inToOut

// New logic:
if port is Output:
    parentNode = adg.getNode(port.parentNode)
    if parentNode is FU node (op_kind == "function_unit"):
        // FU output -> traverse fuOutToPeOut to PE exterior outputs
        expand via connectivity.fuOutToPeOut[portId]
        // cost = 0 for PE internal hop
    else:
        // PE output, sentinel output, etc. -> traverse outToIn
        expand via connectivity.outToIn[portId]
        // cost = 1 (or existing cost model)

if port is Input:
    parentNode = adg.getNode(port.parentNode)
    if isPEContainerNode(parentNode):
        // PE exterior input -> traverse peInToFuIn to FU inputs
        expand via connectivity.peInToFuIn[portId]
        // cost = 0 for PE internal hop
    else if isRoutingNode(parentNode):
        // Switch input -> traverse inToOut
        expand via connectivity.inToOut[portId]
    // else: destination reached (FU input, sentinel input, memory input)
```

- [ ] **Step 2: Adapt direct connection check**

The current direct connection check tests if `outToIn[src]` contains `dst`. In the new model, a "direct" PE-to-PE connection requires 4 hops: `[fuOut, peOut, peIn, fuIn]`. Update the fast-path check:

Pseudocode:
```
// Check if fuOut can reach fuIn through exactly one PE-to-PE hop:
for peOut in fuOutToPeOut[srcHwPort]:
    for peIn in outToIn[peOut]:
        if fuIn in peInToFuIn[peIn]:
            // Found a 4-hop direct path
            directPath = [srcHwPort, peOut, peIn, dstHwPort]
            if isEdgeLegal for this path:
                return directPath
```

- [ ] **Step 3: Remove spatialPEOutputIndexUsedByDifferentFUResult**

Delete the function `spatialPEOutputIndexUsedByDifferentFUResult` and remove its call from `isEdgeLegalCached`.

- [ ] **Step 4: Adapt isEdgeLegalCached for PE internal hops**

For PE internal hops (fuOut -> peOut and peIn -> fuIn), add legality checks:

Pseudocode:
```
// For fuOut -> peOut hop (output demux):
// Spatial PE: peOut must not be used by a different sw edge
// Temporal PE (tagged): sharing allowed

// For peIn -> fuIn hop (input mux):
// Same rules as output side
```

Add a new helper function `isInternalPEHopLegal(srcPort, dstPort, swEdgeId, state, adg)` that checks port occupancy based on PE kind (spatial vs temporal/tagged).

- [ ] **Step 5: Adapt isNonRoutingBroadcastTransitionConflict**

This constraint now applies at PE exterior output ports:
- Non-tagged PE output: one sw edge per output port (enforced by PE internal hop legality, so broadcast check is redundant for PE outputs -- but keep it for non-PE non-routing nodes like sentinels)
- Tagged PE output: exempt from broadcast constraint

Add a check: if srcPort belongs to a PE container node with tagged output, skip the broadcast conflict check.

- [ ] **Step 6: Adapt isEdgeRoutable**

Currently checks sentinel node mapping. Extend to also verify that port mapping includes PE ports. Since paths now start at FU ports, `swPortToHwPort` still maps sw ports to FU hw ports. The router uses `fuOutToPeOut` / `peInToFuIn` to traverse PE internals. No change needed to `isEdgeRoutable` if `swPortToHwPort` still points to FU ports.

- [ ] **Step 7: Build and test**

Run: `ninja -C build loom-unit-dataflow-flowops`

Expected: The dataflow-flowops test should now route all 9/9 edges (the PE port binding bug is fixed). The test may still fail in config gen or validation (next tasks).

- [ ] **Step 8: Commit**

```
git add lib/loom/Mapper/MapperRouting.cpp lib/loom/Mapper/MapperRoutingStrategy.cpp
git commit -m "Adapt router findPath for PE internal traversal via fuOutToPeOut/peInToFuIn"
```

---

## Task 6: Adapt Config Gen for New Path Format

**Files:**
- Modify: `lib/loom/Mapper/ConfigGenConfig.cpp` -- `collectPERouteSummary`, `buildTemporalConfigPlan`

- [ ] **Step 1: Rewrite collectPERouteSummary**

Currently reads `pe_output_index` / `pe_input_index` from edge attributes. Rewrite to extract from path positions:

Pseudocode:
```
for each sw edge with non-empty path:
    path = state.swEdgeToHwPaths[edgeId]
    // Path format: [fuOut, peOut, ..., peIn, fuIn]

    // Source side: first two elements are fuOut, peOut
    fuOutPort = path[0]
    peOutPort = path[1]
    fuOutNode = adg.getPort(fuOutPort).parentNode  // FU node
    peOutNode = adg.getPort(peOutPort).parentNode  // PE node

    // Find FU output index
    fuOutputIdx = indexOf(fuOutPort, adg.getNode(fuOutNode).outputPorts)
    // Find PE output index (= output demux sel value)
    peOutputIdx = indexOf(peOutPort, adg.getNode(peOutNode).outputPorts)

    summary.outputPortSelects[fuOutNode][fuOutputIdx] = peOutputIdx

    // Destination side: last two elements are peIn, fuIn
    peInPort = path[path.size() - 2]
    fuInPort = path[path.size() - 1]
    fuInNode = adg.getPort(fuInPort).parentNode
    peInNode = adg.getPort(peInPort).parentNode

    fuInputIdx = indexOf(fuInPort, adg.getNode(fuInNode).inputPorts)
    peInputIdx = indexOf(peInPort, adg.getNode(peInNode).inputPorts)

    summary.inputPortSelects[fuInNode][fuInputIdx] = peInputIdx
```

- [ ] **Step 2: Adapt buildTemporalConfigPlan for temporalRegisterAssignment**

When a sw edge is in `temporalRegisterAssignment`, the config should use register-based operand/result config instead of mux-based:

Pseudocode:
```
for each temporal register binding (edgeId, registerIndex):
    // Producer side: set res_is_reg=1, res_reg_idx=registerIndex
    // Consumer side: set op_is_reg=1, op_reg_idx=registerIndex
    // Skip the mux/demux path extraction for this edge
```

- [ ] **Step 3: Remove edge attribute reads**

Remove all `getUIntEdgeAttr(edge, "pe_output_index")` and `getUIntEdgeAttr(edge, "pe_input_index")` from config gen code.

- [ ] **Step 4: Build and test**

Run: `ninja -C build loom-unit-dataflow-flowops`

Expected: The dataflow-flowops test should now pass (routing + config gen both work).

- [ ] **Step 5: Run full test suite**

Run: `ninja -C build check-loom-unit && ninja -C build check-loom-e2e`

Expected: All previously-passing tests still pass. dataflow-flowops and mapper-fifo-recurrence-guard should now pass.

- [ ] **Step 6: Commit**

```
git add lib/loom/Mapper/ConfigGenConfig.cpp
git commit -m "Adapt config gen to read mux/demux selections from routing path positions"
```

---

## Task 7: Adapt TechMapper expandPlanMapping

**Files:**
- Modify: `lib/loom/Mapper/TechMapperGraphs.cpp` -- `expandPlanMapping`

- [ ] **Step 1: Adapt edge path expansion**

`expandPlanMapping` copies `contractedState.swEdgeToHwPaths[contractedEdgeId]` to `expandedState.swEdgeToHwPaths[swEdgeId]`. The paths now include PE ports. Since PE ports are shared between contracted and original DFG (they're ADG ports, not DFG ports), the path copying should work without changes.

However, verify that the hw edge tracking (`hwEdgeToSwEdges`) correctly handles the new path format. The path iteration at `i += 2` assumes alternating out-in pairs. The new path `[fuOut, peOut, nextIn, ..., peIn, fuIn]` still alternates out-in, so the iteration should be correct.

Verify by reading the code and confirming no changes are needed, or make minimal adaptations.

- [ ] **Step 2: Adapt contracted DFG construction**

In `buildContractedDFG` (same file), the contracted node ports are created from `hwNode->outputPorts` (FU node ports). This remains correct -- the contracted DFG uses FU ports, and `mapNode` maps contracted ports to FU hw ports. The PE internal traversal is handled by the router, not by the contracted DFG.

Verify by reading the code and confirming no changes are needed.

- [ ] **Step 3: Build and run e2e tests**

Run: `ninja -C build check-loom-e2e`

Expected: All 50 e2e tests pass.

- [ ] **Step 4: Commit (if changes were needed)**

```
git add lib/loom/Mapper/TechMapperGraphs.cpp
git commit -m "Adapt TechMapper expansion for new PE port path format"
```

---

## Task 8: Adapt Validation

**Files:**
- Modify: `lib/loom/Mapper/Mapper.cpp` -- `runValidation`

- [ ] **Step 1: Adapt C3 (unrouted edges)**

Replace `TechMappedEdgeKind::TemporalReg` check with `temporalRegisterAssignment` lookup:

Pseudocode:
```
// Old:
if edgeKinds[i] == IntraFU || edgeKinds[i] == TemporalReg: continue

// New:
if edgeKinds[i] == IntraFU: continue
if state.temporalRegisterAssignment.contains(i): continue
```

- [ ] **Step 2: Adapt C5.2 (temporal register overflow)**

Change from counting unique source ports to counting unique register indices per temporal PE from `temporalRegisterAssignment`:

Pseudocode:
```
DenseMap<StringRef, DenseSet<unsigned>> regIndicesByPE;
for (edgeId, regIdx) in state.temporalRegisterAssignment:
    peName = findPENameForEdge(edgeId, state, dfg, adg)
    regIndicesByPE[peName].insert(regIdx)

for (peName, indices) in regIndicesByPE:
    pe = findPEContainmentByName(flattener, peName)
    if indices.size() > pe.numRegister:
        // C5.2 violation
```

- [ ] **Step 3: Add C-PE-1 (spatial PE single-FU)**

Pseudocode:
```
for each PEContainment pe where pe.peKind == "spatial_pe":
    activeFUs = 0
    for fuId in pe.fuNodeIds:
        if hwNodeToSwNodes[fuId] is non-empty:
            activeFUs++
    if activeFUs > 1:
        // C-PE-1 violation
```

- [ ] **Step 4: Add C-PE-2 (spatial PE output demux exclusivity)**

Pseudocode:
```
for each PEContainment pe where pe.peKind == "spatial_pe":
    peNode = adg.getNode(pe.peNodeId)
    for peOutPort in peNode.outputPorts:
        edgeCount = countSwEdgesUsingPort(peOutPort, state)
        if edgeCount > 1:
            // C-PE-2 violation
```

- [ ] **Step 5: Add C-PE-3 (temporal PE instruction count)**

Pseudocode:
```
for each PEContainment pe where pe.peKind == "temporal_pe":
    totalMappedOps = 0
    for fuId in pe.fuNodeIds:
        totalMappedOps += hwNodeToSwNodes[fuId].size()
    if totalMappedOps > pe.numInstruction:
        // C-PE-3 violation
```

- [ ] **Step 6: Add C-PE-4 (PE port width compatibility)**

Pseudocode:
```
for each routed path in state.swEdgeToHwPaths:
    // Check first hop: fuOut -> peOut
    fuOutPort = path[0], peOutPort = path[1]
    if fuOutPort belongs to FU and peOutPort belongs to PE:
        assert bitWidth(fuOutPort.type) <= bitWidth(peOutPort.type)

    // Check last hop: peIn -> fuIn
    peInPort = path[path.size()-2], fuInPort = path[path.size()-1]
    if peInPort belongs to PE and fuInPort belongs to FU:
        assert bitWidth(fuInPort.type) <= bitWidth(peInPort.type)
```

- [ ] **Step 7: Build and run full test suite**

Run: `ninja -C build check-loom-unit && ninja -C build check-loom-e2e`

Expected: All tests pass. New validation checks should not trigger on any existing test.

- [ ] **Step 8: Commit**

```
git add lib/loom/Mapper/Mapper.cpp
git commit -m "Adapt validation for PE nodes, temporal register assignment, and new PE constraints"
```

---

## Task 9: Add rebindScalarOutputSentinels

**Files:**
- Modify: `lib/loom/Mapper/MapperSentinels.cpp` -- add new function
- Modify: `include/loom/Mapper/Mapper.h` -- add declaration
- Modify: `lib/loom/Mapper/MapperInterleaved.cpp` -- add call site
- Modify: `lib/loom/Mapper/Mapper.cpp` -- add call site in lane routing

- [ ] **Step 1: Declare rebindScalarOutputSentinels**

Add to `Mapper` class in `Mapper.h`:

```cpp
bool rebindScalarOutputSentinels(MappingState &state, const Graph &dfg,
                                 const Graph &adg,
                                 const ADGFlattener &flattener);
```

- [ ] **Step 2: Implement rebindScalarOutputSentinels**

Symmetric to `rebindScalarInputSentinels`. The logic:

Pseudocode:
```
function rebindScalarOutputSentinels(state, dfg, adg, flattener):
    // Collect DFG scalar output sentinels
    dfgScalarOutputSentinels = [node for node in dfg if node.kind == ModuleOutputNode
                                 and node.inputPorts[0].type is not memref]

    // Collect ADG output sentinels
    adgOutputSentinels = [node for node in adg if node.kind == ModuleOutputNode]

    // For each (DFG sentinel, ADG sentinel) pair, estimate routing cost
    // based on INCOMING edges (the edge that feeds into this output sentinel)
    estimateAssignmentCost = function(swSentinel, adgSentinel):
        cost = 0
        for each incoming edge to swSentinel:
            srcHwNode = state.swNodeToHwNode[edge.srcNode]
            cost += placementDistance(srcHwNode, adgSentinel)
        return cost

    // Build candidate lists, sort by cost
    // Backtracking search for optimal assignment
    // Unmap all output sentinels, remap with optimal assignment
    // (same algorithm as rebindScalarInputSentinels, but for output side)
```

- [ ] **Step 3: Add call sites**

In `MapperInterleaved.cpp`, after `rebindScalarInputSentinels`:
```cpp
rebindScalarOutputSentinels(state, dfg, adg, flattener);
```

In `Mapper.cpp` lane routing, same pattern -- wherever `rebindScalarInputSentinels` is called, add the output counterpart immediately after.

- [ ] **Step 4: Build and test**

Run: `ninja -C build check-loom-unit`

Expected: All tests pass. Output sentinel rebinding may improve routing quality for some tests.

- [ ] **Step 5: Commit**

```
git add lib/loom/Mapper/MapperSentinels.cpp include/loom/Mapper/Mapper.h lib/loom/Mapper/MapperInterleaved.cpp lib/loom/Mapper/Mapper.cpp
git commit -m "Add rebindScalarOutputSentinels symmetric to input sentinel rebinding"
```

---

## Task 10: Scalar Sentinel SA Participation

**Files:**
- Modify: `lib/loom/Mapper/MapperPlacementMethods.cpp` -- `buildCandidates`
- Modify: `lib/loom/Mapper/MapperRefinement.cpp` -- `runRefinement`
- Modify: `lib/loom/Mapper/MapperRouteAwareSA.cpp` -- `runRouteAwareSA`

- [ ] **Step 1: Extend buildCandidates for scalar sentinels**

In `buildCandidates`, after the existing OperationNode loop, add sentinel candidate building:

Pseudocode:
```
// Existing: skip non-OperationNode
// New: also build candidates for scalar sentinels

for each DFG node where kind == ModuleInputNode:
    swPort = node.outputPorts[0]
    if isMemrefType(swPort.type): continue  // skip memref sentinels
    // Design note: memref sentinel binding is determined by memory interface
    // placement, not by independent sentinel routing. See spec-dataflow-memory
    // and spec-fabric-memory-interface.
    candidates[nodeId] = []
    for each ADG node where kind == ModuleInputNode:
        hwPort = adgNode.outputPorts[0]
        if canMapSoftwareTypeToHardware(swPort.type, hwPort.type):
            candidates[nodeId].push_back(adgNodeId)

// Same for ModuleOutputNode (using inputPorts[0])
```

- [ ] **Step 2: Extend runRefinement to include sentinels**

In `runRefinement`, extend the `placedNodes` collection:

Pseudocode:
```
// Existing:
for node in dfg where kind == OperationNode and mapped:
    placedNodes.push_back(nodeId)

// New: also include scalar sentinels
for node in dfg where kind == ModuleInputNode or ModuleOutputNode:
    if node is mapped and not memref sentinel:
        placedNodes.push_back(nodeId)
```

In the move selection, when a sentinel is chosen:
- For swap: only swap with another sentinel of the same side (input-input or output-output). Check width compatibility after swap.
- For relocate: move to another candidate from the sentinel's candidate list.
- Never cross-swap sentinel with OperationNode.

- [ ] **Step 3: Extend runRouteAwareSA to include sentinels**

Same pattern as runRefinement: extend `placedNodes`, add sentinel-specific move logic.

- [ ] **Step 4: Build and test**

Run: `ninja -C build check-loom-unit && ninja -C build check-loom-e2e`

Expected: All tests pass. Sentinel SA may improve placement quality.

- [ ] **Step 5: Commit**

```
git add lib/loom/Mapper/MapperPlacementMethods.cpp lib/loom/Mapper/MapperRefinement.cpp lib/loom/Mapper/MapperRouteAwareSA.cpp
git commit -m "Include scalar sentinels in SA swap/relocate moves"
```

---

## Task 11: Temporal Register Toggle Data Structure

**Files:**
- Modify: `include/loom/Mapper/Mapper.h` or `lib/loom/Mapper/MappingState.cpp` -- add `temporalRegisterAssignment`
- Modify: `lib/loom/Mapper/MapperPlacement.cpp` -- adapt `classifyTemporalRegisterEdges`
- Modify: `lib/loom/Mapper/MappingState.cpp` -- add to save/restore/init

- [ ] **Step 1: Add temporalRegisterAssignment to MappingState**

Add field to MappingState:
```cpp
llvm::DenseMap<IdIndex, unsigned> temporalRegisterAssignment;
// sw edge ID -> register index within the temporal PE
```

- [ ] **Step 2: Add to save/restore/init**

In `MappingState::init()`: clear the map.
In `MappingState::save()`: copy to checkpoint.
In `MappingState::restore()`: restore from checkpoint.
Add to the `Checkpoint` struct.

- [ ] **Step 3: Adapt classifyTemporalRegisterEdges**

Change from statically setting `edgeKinds[edgeId] = TemporalReg` to populating `temporalRegisterAssignment` with greedy allocation:

Pseudocode:
```
function classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds):
    state.temporalRegisterAssignment.clear()

    // Group eligible edges by PE
    DenseMap<StringRef, SmallVector<IdIndex>> eligibleEdgesByPE

    for each edge in dfg:
        if edgeKinds[edgeId] == IntraFU: continue
        srcHw = state.swNodeToHwNode[srcNode]
        dstHw = state.swNodeToHwNode[dstNode]
        if srcHw == INVALID or dstHw == INVALID: continue
        if not isTemporalPENode(srcHw) or not isTemporalPENode(dstHw): continue
        srcPE = getNodeAttrStr(srcHw, "pe_name")
        dstPE = getNodeAttrStr(dstHw, "pe_name")
        if srcPE != dstPE: continue
        pe = findPEContainmentByName(flattener, srcPE)
        if not pe or pe.numRegister == 0: continue
        eligibleEdgesByPE[srcPE].push_back(edgeId)

    // Greedy allocation per PE
    for (peName, edges) in eligibleEdgesByPE:
        pe = findPEContainmentByName(flattener, peName)
        DenseMap<IdIndex, unsigned> regBySrcPort  // srcPort -> regIndex
        for edgeId in edges:
            srcPort = dfg.getEdge(edgeId).srcPort
            if srcPort in regBySrcPort:
                regIdx = regBySrcPort[srcPort]
            else:
                regIdx = regBySrcPort.size()
            if regIdx >= pe.numRegister:
                continue  // no more registers, this edge stays external
            regBySrcPort[srcPort] = regIdx
            state.temporalRegisterAssignment[edgeId] = regIdx

    // Still call initializeRouteStats but exclude register-assigned edges
    // (replace the old TemporalReg check with temporalRegisterAssignment check)
```

- [ ] **Step 4: Update routing code to check temporalRegisterAssignment**

Replace all `edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg` checks with `state.temporalRegisterAssignment.contains(edgeId)` in:
- `routeOnePass` (MapperRoutingStrategy.cpp)
- `collectUnroutedEdges` (MapperPlacement.cpp)
- `countRoutedEdges` (MapperPlacement.cpp)
- Routing repair functions
- Any other location that checks TemporalReg

- [ ] **Step 5: Build and test**

Run: `ninja -C build check-loom-unit`

Expected: All tests pass. temporal-pe-registers test should still work correctly.

- [ ] **Step 6: Commit**

```
git add include/loom/Mapper/Mapper.h lib/loom/Mapper/MappingState.cpp lib/loom/Mapper/MapperPlacement.cpp lib/loom/Mapper/MapperRoutingStrategy.cpp
git commit -m "Replace static TemporalReg classification with dynamic temporalRegisterAssignment"
```

---

## Task 12: Route-aware SA Register Toggle Move

**Files:**
- Modify: `lib/loom/Mapper/MapperRouteAwareSA.cpp` -- `runRouteAwareSA`

- [ ] **Step 1: Add register toggle move type**

Extend the SA move selection to include a third move type. Current: 50% swap, 50% relocate. New: for example 40% swap, 40% relocate, 20% register-toggle (tunable).

Pseudocode:
```
// In the move selection:
double moveRoll = rng.nextDouble()
if moveRoll < 0.4:
    // swap move (existing)
elif moveRoll < 0.8:
    // relocate move (existing)
else:
    // register toggle move (new)
    tryRegisterToggle(state, dfg, adg, flattener, edgeKinds)
```

- [ ] **Step 2: Implement tryRegisterToggle**

Pseudocode:
```
function tryRegisterToggle(state, dfg, adg, flattener, edgeKinds):
    // Find all temporal PEs with eligible internal edges
    eligiblePEs = []
    for each PEContainment pe where pe.peKind == "temporal_pe" and pe.numRegister > 0:
        internalEdges = findInternalDependencyEdges(pe, state, dfg, adg)
        if not internalEdges.empty():
            eligiblePEs.push_back(pe)

    if eligiblePEs.empty(): return false

    // Pick a random PE and edge
    pe = randomChoice(eligiblePEs)
    internalEdges = findInternalDependencyEdges(pe, state, dfg, adg)
    edgeId = randomChoice(internalEdges)

    // Toggle
    if state.temporalRegisterAssignment.contains(edgeId):
        // Currently on register -> move to external routing
        regIdx = state.temporalRegisterAssignment[edgeId]
        state.temporalRegisterAssignment.erase(edgeId)
        // The edge now needs external routing (will be picked up by reroute)
    else:
        // Currently external -> try to assign a register
        usedRegs = countUsedRegisters(pe, state)
        if usedRegs >= pe.numRegister: return false  // no room
        // Find available register index
        regIdx = findAvailableRegister(pe, edgeId, state, dfg)
        if regIdx == INVALID: return false
        // Remove external route if it exists
        state.unmapEdge(edgeId, dfg, adg)
        state.temporalRegisterAssignment[edgeId] = regIdx

    return true  // move applied, SA will evaluate and accept/reject
```

- [ ] **Step 3: Build and test**

Run: `ninja -C build check-loom-unit`

Expected: All tests pass, including temporal-pe-registers.

- [ ] **Step 4: Commit**

```
git add lib/loom/Mapper/MapperRouteAwareSA.cpp
git commit -m "Add register toggle move to Route-aware SA for temporal PE optimization"
```

---

## Task 13: Integration Testing and Cleanup

**Files:**
- Potentially modify: test check.py files, expect-fail.txt files

- [ ] **Step 1: Run full unit test suite**

Run: `ninja -C build check-loom-unit`

Expected: 127/127 pass (including the 3 previously failing: dataflow-flowops, mapper-fifo-recurrence-guard, temporal-pe-registers).

- [ ] **Step 2: Run full e2e test suite**

Run: `ninja -C build check-loom-e2e`

Expected: 50/50 pass.

- [ ] **Step 3: Run KHG ADG vecadd tests**

Run: `ninja -C build check-loom-khg-vecadd` (or equivalent target)

Expected: 24/24 pass.

- [ ] **Step 4: Remove any remaining dead code**

Search for and remove:
- Any remaining references to `pe_output_index` / `pe_input_index` edge attributes
- Any remaining `TechMappedEdgeKind::TemporalReg` static checks that should now use `temporalRegisterAssignment`
- The `spatialPEOutputIndexUsedByDifferentFUResult` function if not already removed

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "Complete PE port binding, sentinel routing, and temporal register toggle integration"
```

---

## Dependency Graph

```
Task 1 (globalAllRouted fix)           -- independent
Task 2 (ConnectivityMatrix extension)  -- independent
Task 3 (PE nodes in flattener)         -- depends on Task 2
Task 4 (Flattener wiring)              -- depends on Task 3
Task 5 (Router findPath)               -- depends on Task 4
Task 6 (Config gen)                    -- depends on Task 5
Task 7 (TechMapper expansion)          -- depends on Task 5
Task 8 (Validation)                    -- depends on Task 5, Task 11
Task 9 (Output sentinel rebind)        -- depends on Task 5
Task 10 (Sentinel SA)                  -- depends on Task 9
Task 11 (Temporal register data)       -- depends on Task 5
Task 12 (Register toggle SA)           -- depends on Task 11
Task 13 (Integration testing)          -- depends on all above
```

Tasks 1 and 2 can be done in parallel. Tasks 6, 7, 8, 9, 11 can be partially parallelized after Task 5.
