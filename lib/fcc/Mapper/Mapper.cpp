#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace fcc {

using namespace mapper_detail;

// ---------------------------------------------------------------------------
// Mapper::bindSentinels
// ---------------------------------------------------------------------------

bool Mapper::bindSentinels(MappingState &state, const Graph &dfg,
                           const Graph &adg) {
  // Collect DFG and ADG sentinels.
  std::vector<IdIndex> dfgInputSentinels;
  std::vector<IdIndex> dfgOutputSentinels;
  std::vector<IdIndex> adgInputSentinels;
  std::vector<IdIndex> adgOutputSentinels;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      dfgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      dfgOutputSentinels.push_back(i);
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      adgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      adgOutputSentinels.push_back(i);
  }

  llvm::outs() << "  DFG sentinels: " << dfgInputSentinels.size()
               << " inputs, " << dfgOutputSentinels.size() << " outputs\n";
  llvm::outs() << "  ADG sentinels: " << adgInputSentinels.size()
               << " inputs, " << adgOutputSentinels.size() << " outputs\n";

  // Separate DFG input sentinels into memref and non-memref.
  std::vector<IdIndex> dfgMemrefSentinels;
  std::vector<IdIndex> dfgScalarSentinels;

  for (IdIndex sid : dfgInputSentinels) {
    const Node *node = dfg.getNode(sid);
    if (!node || node->outputPorts.empty())
      continue;
    mlir::Type portType = dfg.getPort(node->outputPorts[0])->type;
    if (isMemrefType(portType))
      dfgMemrefSentinels.push_back(sid);
    else
      dfgScalarSentinels.push_back(sid);
  }

  llvm::outs() << "    DFG memref inputs: " << dfgMemrefSentinels.size()
               << ", scalar inputs: " << dfgScalarSentinels.size() << "\n";

  // For memref sentinels: these are NOT mapped to ADG sentinels (the ADG
  // doesn't have memref sentinels since memrefs bind directly to extmemory).
  // Instead, map memref sentinel -> first available memory node in ADG.
  // The mapper already handles extmemory binding through buildCandidates.
  // We leave memref sentinels unmapped here; they are handled by the
  // extmemory matching in buildCandidates.

  // For scalar sentinels: bind DFG scalar inputs to ADG input sentinels.
  llvm::DenseSet<size_t> usedAdgIn;
  for (size_t di = 0; di < dfgScalarSentinels.size(); ++di) {
    IdIndex dfgSid = dfgScalarSentinels[di];

    // Find a matching ADG input sentinel (by index order).
    bool bound = false;
    for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
      if (usedAdgIn.count(ai))
        continue;

      IdIndex adgSid = adgInputSentinels[ai];
      const Node *dfgNode = dfg.getNode(dfgSid);
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->outputPorts.empty() ||
          adgNode->outputPorts.empty())
        continue;
      const Port *swPort = dfg.getPort(dfgNode->outputPorts[0]);
      const Port *hwPort = adg.getPort(adgNode->outputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgIn.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG input sentinel " << dfgSid
                      << " -> ADG input sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG input sentinel "
                    << dfgSid << "\n";
    }
  }

  // For output sentinels: bind DFG output sentinels to ADG output sentinels.
  llvm::DenseSet<size_t> usedAdgOut;
  for (size_t di = 0; di < dfgOutputSentinels.size(); ++di) {
    IdIndex dfgSid = dfgOutputSentinels[di];
    const Node *dfgNode = dfg.getNode(dfgSid);
    if (!dfgNode || dfgNode->inputPorts.empty())
      continue;
    const Port *swPort = dfg.getPort(dfgNode->inputPorts[0]);

    bool bound = false;
    for (size_t ai = 0; ai < adgOutputSentinels.size(); ++ai) {
      if (usedAdgOut.count(ai))
        continue;

      IdIndex adgSid = adgOutputSentinels[ai];
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->inputPorts.empty() ||
          adgNode->inputPorts.empty())
        continue;
      const Port *hwPort = adg.getPort(adgNode->inputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgOut.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG output sentinel " << dfgSid
                      << " -> ADG output sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG output sentinel "
                    << dfgSid << "\n";
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Mapper::bindMemrefSentinels
// ---------------------------------------------------------------------------

bool Mapper::bindMemrefSentinels(MappingState &state, const Graph &dfg,
                                  const Graph &adg) {
  // For each DFG memref input sentinel, find the edge to its downstream
  // extmemory node, and pre-route it as a direct binding. The memref sentinel
  // itself stays unmapped (it has no ADG counterpart), but its edge to the
  // extmemory node is marked as routed with a synthetic path.

  for (IdIndex sid = 0; sid < static_cast<IdIndex>(dfg.nodes.size()); ++sid) {
    const Node *sNode = dfg.getNode(sid);
    if (!sNode || sNode->kind != Node::ModuleInputNode)
      continue;
    if (sNode->outputPorts.empty())
      continue;

    // Check if this is a memref sentinel.
    const Port *outPort = dfg.getPort(sNode->outputPorts[0]);
    if (!outPort || !mlir::isa<mlir::MemRefType>(outPort->type))
      continue;

    // Find the edge(s) from this memref sentinel to extmemory nodes.
    for (IdIndex opId : sNode->outputPorts) {
      const Port *op = dfg.getPort(opId);
      if (!op)
        continue;
      for (IdIndex eid : op->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge || edge->srcPort != opId)
          continue;

        // Get the destination node.
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID)
          continue;

        IdIndex dstNodeId = dstPort->parentNode;
        const Node *dstNode = dfg.getNode(dstNodeId);
        if (!dstNode)
          continue;

        // Verify destination is an extmemory node.
        llvm::StringRef dstOpName = getNodeAttrStr(dstNode, "op_name");
        if (!dstOpName.contains("extmemory"))
          continue;

        // The extmemory DFG node should already be placed on an ADG node.
        if (dstNodeId >= state.swNodeToHwNode.size() ||
            state.swNodeToHwNode[dstNodeId] == INVALID_ID)
          continue;

        IdIndex hwExtMemNodeId = state.swNodeToHwNode[dstNodeId];
        const Node *hwExtMemNode = adg.getNode(hwExtMemNodeId);
        if (!hwExtMemNode)
          continue;

        // Get the memref input port on the ADG extmemory node (port index 0,
        // the memref port from the function type).
        if (hwExtMemNode->inputPorts.empty())
          continue;

        IdIndex hwMemrefInPort = hwExtMemNode->inputPorts[0];

        // Create a synthetic output port mapping for the memref sentinel.
        // We use the memref input port of the ADG extmemory as both
        // source and destination since this is a direct binding.
        // The path just contains [hwMemrefInPort, hwMemrefInPort] as a
        // sentinel marker for "direct memref binding".
        llvm::SmallVector<IdIndex, 8> syntheticPath;
        syntheticPath.push_back(hwMemrefInPort);
        syntheticPath.push_back(hwMemrefInPort);

        auto result = state.mapEdge(eid, syntheticPath, dfg, adg);
        if (result == ActionResult::Success) {
          llvm::outs() << "    Pre-routed memref edge " << eid
                        << " (sentinel " << sid << " -> extmem " << dstNodeId
                        << ") as direct binding\n";
        }
      }
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Mapper::run
// ---------------------------------------------------------------------------

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const ADGFlattener &flattener,
                           mlir::ModuleOp adgModule, const Options &opts) {
  Result result;
  TechMapper techMapper;
  TechMapper::Plan techPlan;
  if (!techMapper.buildPlan(dfg, adgModule, adg, techPlan)) {
    result.diagnostics = "Tech-mapping failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  MappingState contractedState;
  contractedState.init(techPlan.contractedDFG, adg);
  result.edgeKinds = techPlan.originalEdgeKinds;
  std::vector<TechMappedEdgeKind> contractedEdgeKinds(
      techPlan.contractedDFG.edges.size(), TechMappedEdgeKind::Routed);

  // Copy connectivity from flattener.
  connectivity = flattener.getConnectivity();

  // Bind sentinels (DFG boundary nodes -> ADG boundary nodes).
  llvm::outs() << "Mapper: binding sentinels...\n";
  bindSentinels(contractedState, techPlan.contractedDFG, adg);

  llvm::outs() << "Mapper: building candidates...\n";
  auto candidates = buildCandidates(techPlan.contractedDFG, adg);
  for (const auto &entry : techPlan.contractedCandidates)
    candidates[entry.first] = entry.second;

  if (detectForcedTemporalConfigConflict(techPlan, adg, result.diagnostics)) {
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  // Check that all operation nodes have candidates.
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(techPlan.contractedDFG.nodes.size()); ++i) {
    const Node *node = techPlan.contractedDFG.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (candidates.find(i) == candidates.end() || candidates[i].empty()) {
      result.diagnostics = "No hardware candidates for DFG node " +
                           std::to_string(i) + " (" +
                           getNodeAttrStr(node, "op_name").str() + ")";
      llvm::errs() << "Mapper: " << result.diagnostics << "\n";
      return result;
    }
    if (opts.verbose) {
      llvm::outs() << "  Node " << i << " ("
                    << getNodeAttrStr(node, "op_name")
                    << "): " << candidates[i].size() << " candidates\n";
    }
  }

  llvm::outs() << "Mapper: placing...\n";
  if (!runPlacement(contractedState, techPlan.contractedDFG, adg, flattener,
                    candidates, opts)) {
    result.diagnostics = "Placement failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  llvm::outs() << "Mapper: refining placement...\n";
  runRefinement(contractedState, techPlan.contractedDFG, adg, flattener,
                candidates, opts);

  classifyTemporalRegisterEdges(contractedState, techPlan.contractedDFG, adg,
                                flattener, contractedEdgeKinds);

  llvm::outs() << "Mapper: binding memref sentinels...\n";
  bindMemrefSentinels(contractedState, techPlan.contractedDFG, adg);
  auto preRoutingCheckpoint = contractedState.save();

  llvm::outs() << "Mapper: routing...\n";
  bool routingSucceeded = runRouting(contractedState, techPlan.contractedDFG,
                                     adg, contractedEdgeKinds, opts.seed);
  if (!routingSucceeded) {
    auto failedEdges =
        collectUnroutedEdges(contractedState, techPlan.contractedDFG, contractedEdgeKinds);
    if (!failedEdges.empty()) {
      llvm::outs() << "Mapper: local repair...\n";
      contractedState.restore(preRoutingCheckpoint);
      routingSucceeded = runLocalRepair(contractedState, preRoutingCheckpoint,
                                        failedEdges, techPlan.contractedDFG,
                                        adg, flattener, candidates,
                                        contractedEdgeKinds, opts);
    }
    if (!routingSucceeded) {
      result.diagnostics = "Routing failed";
      llvm::errs() << "Mapper: " << result.diagnostics << "\n";
      // Continue anyway to produce partial output.
    }
  }

  if (!techMapper.expandPlanMapping(dfg, adg, techPlan, contractedState,
                                    result.state, result.fuConfigs)) {
    result.diagnostics = "Tech-mapping expansion failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= result.edgeKinds.size() ||
        result.edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    if (edgeId >= techPlan.originalEdgeToContractedEdge.size())
      continue;
    IdIndex contractedEdgeId = techPlan.originalEdgeToContractedEdge[edgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedEdgeKinds.size())
      continue;
    if (contractedEdgeKinds[contractedEdgeId] == TechMappedEdgeKind::TemporalReg)
      result.edgeKinds[edgeId] = TechMappedEdgeKind::TemporalReg;
  }

  llvm::outs() << "Mapper: validating...\n";
  bool validationSucceeded = runValidation(result.state, dfg, adg, flattener,
                                          result.edgeKinds, result.diagnostics);
  if (!validationSucceeded) {
    llvm::errs() << "Mapper: validation issues: " << result.diagnostics
                 << "\n";
    // Proceed with partial result.
  }

  result.success = routingSucceeded && validationSucceeded;
  llvm::outs() << "Mapper: done.\n";
  return result;
}

// ---------------------------------------------------------------------------
// Mapper::runValidation
// ---------------------------------------------------------------------------

bool Mapper::runValidation(const MappingState &state, const Graph &dfg,
                           const Graph &adg, const ADGFlattener &flattener,
                           llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                           std::string &diagnostics) {
  bool valid = true;

  // C1: All operation nodes are placed. Memref sentinels are exempt
  // (they bind directly to extmemory, not through the ADG).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    // Skip memref sentinels (they're expected to be unmapped).
    if (node->kind == Node::ModuleInputNode) {
      if (!node->outputPorts.empty()) {
        const Port *p = dfg.getPort(node->outputPorts[0]);
        if (p && mlir::isa<mlir::MemRefType>(p->type))
          continue;
      }
    }
    if (node->kind == Node::ModuleOutputNode) {
      if (!node->inputPorts.empty()) {
        const Port *p = dfg.getPort(node->inputPorts[0]);
      }
    }
    if (node->kind != Node::OperationNode &&
        node->kind != Node::ModuleInputNode &&
        node->kind != Node::ModuleOutputNode)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      diagnostics += "C1: unmapped node " + std::to_string(i) + "\n";
      valid = false;
    }
  }

  // C3: All edges are routed. Only warn if both endpoints are placed
  // in the ADG (memref sentinel edges are exempt since memref sentinels
  // bind directly to extmemory without routing).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i < edgeKinds.size() &&
        (edgeKinds[i] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[i] == TechMappedEdgeKind::TemporalReg))
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      const Port *sp = dfg.getPort(edge->srcPort);
      const Port *dp = dfg.getPort(edge->dstPort);
      if (sp && dp && sp->parentNode != INVALID_ID &&
          dp->parentNode != INVALID_ID) {
        IdIndex srcNodeId = sp->parentNode;
        IdIndex dstNodeId = dp->parentNode;
        // Check both endpoints are actually mapped in the ADG.
        bool srcMapped = srcNodeId < state.swNodeToHwNode.size() &&
                         state.swNodeToHwNode[srcNodeId] != INVALID_ID;
        bool dstMapped = dstNodeId < state.swNodeToHwNode.size() &&
                         state.swNodeToHwNode[dstNodeId] != INVALID_ID;
        if (srcMapped && dstMapped) {
          diagnostics +=
              "C3: unrouted edge " + std::to_string(i) + "\n";
          valid = false;
        }
      }
    }
  }

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.size() < 3)
      continue;

    for (size_t pathIdx = 1; pathIdx + 1 < path.size(); ++pathIdx) {
      const Port *port = adg.getPort(path[pathIdx]);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      const Node *owner = adg.getNode(port->parentNode);
      if (!owner)
        continue;
      if (getNodeAttrStr(owner, "resource_class") != "functional")
        continue;
      diagnostics += "C9: routed edge " + std::to_string(edgeId) +
                     " illegally traverses functional node " +
                     getNodeAttrStr(owner, "op_name").str() + "\n";
      valid = false;
      break;
    }
  }

  llvm::DenseMap<IdIndex, IdIndex> firstHopByNonRoutingSource;
  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.size() < 2)
      continue;

    IdIndex srcPortId = path.front();
    IdIndex firstHopInputId = path[1];
    const Port *srcPort = adg.getPort(srcPortId);
    const Port *firstHopInput = adg.getPort(firstHopInputId);
    if (!srcPort || !firstHopInput || srcPort->direction != Port::Output ||
        firstHopInput->direction != Port::Input ||
        srcPort->parentNode == INVALID_ID)
      continue;

    const Node *owner = adg.getNode(srcPort->parentNode);
    if (isRoutingResourceNode(owner))
      continue;

    auto it = firstHopByNonRoutingSource.find(srcPortId);
    if (it == firstHopByNonRoutingSource.end()) {
      firstHopByNonRoutingSource[srcPortId] = firstHopInputId;
      continue;
    }
    if (it->second == firstHopInputId)
      continue;

    diagnostics += "C10: non-routing source port " + std::to_string(srcPortId) +
                   " fans out to multiple next hops (" +
                   std::to_string(it->second) + " and " +
                   std::to_string(firstHopInputId) + ")\n";
    valid = false;
  }

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
    if (hwId < state.hwNodeToSwNodes.size() &&
        static_cast<int64_t>(state.hwNodeToSwNodes[hwId].size()) > numRegion) {
      diagnostics += "C4: memory region overflow on hw_node " +
                     std::to_string(hwId) + "\n";
      valid = false;
    }

    BridgeInfo bridge = BridgeInfo::extract(hwNode);
    if (!bridge.hasBridge || hwId >= state.hwNodeToSwNodes.size())
      continue;

    bool isExtMem = (getNodeAttrStr(hwNode, "op_kind") == "extmemory");
    llvm::SmallVector<BridgeLaneRange, 4> usedLaneRanges;
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
      auto laneRange = inferBridgeLaneRange(bridge, memInfo, swNode, state);
      if (!laneRange) {
        diagnostics += "C4: missing bridge lane range for sw_node " +
                       std::to_string(swId) + " on hw_node " +
                       std::to_string(hwId) + "\n";
        valid = false;
        continue;
      }
      for (const auto &usedRange : usedLaneRanges) {
        if (laneRange->start < usedRange.end &&
            usedRange.start < laneRange->end) {
          diagnostics += "C4: overlapping bridge lane range [" +
                         std::to_string(laneRange->start) + ", " +
                         std::to_string(laneRange->end) + ") on hw_node " +
                         std::to_string(hwId) + "\n";
          valid = false;
          break;
        }
      }
      usedLaneRanges.push_back(*laneRange);
    }
  }

  llvm::StringMap<llvm::DenseSet<IdIndex>> activeSpatialFUsByPE;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    if (hwId >= state.hwNodeToSwNodes.size() || state.hwNodeToSwNodes[hwId].empty())
      continue;
    const Node *hwNode = adg.getNode(hwId);
    llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
    if (peName.empty() || !isSpatialPEName(flattener, peName))
      continue;
    activeSpatialFUsByPE[peName].insert(hwId);
  }
  for (const auto &entry : activeSpatialFUsByPE) {
    if (entry.getValue().size() > 1) {
      diagnostics += "C8: multiple active function_unit instances in spatial_pe " +
                     entry.getKey().str() + "\n";
      valid = false;
    }
  }

  llvm::StringMap<llvm::DenseSet<IdIndex>> temporalRegsByPE;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= edgeKinds.size() ||
        edgeKinds[edgeId] != TechMappedEdgeKind::TemporalReg)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    IdIndex srcNodeId = srcPort->parentNode;
    IdIndex dstNodeId = dstPort->parentNode;
    if (srcNodeId >= state.swNodeToHwNode.size() ||
        dstNodeId >= state.swNodeToHwNode.size())
      continue;
    IdIndex srcHwId = state.swNodeToHwNode[srcNodeId];
    IdIndex dstHwId = state.swNodeToHwNode[dstNodeId];
    const Node *srcHwNode = adg.getNode(srcHwId);
    const Node *dstHwNode = adg.getNode(dstHwId);
    if (!isTemporalPENode(srcHwNode) || !isTemporalPENode(dstHwNode))
      continue;
    llvm::StringRef peName = getNodeAttrStr(srcHwNode, "pe_name");
    if (peName.empty() || peName != getNodeAttrStr(dstHwNode, "pe_name"))
      continue;
    temporalRegsByPE[peName].insert(edge->srcPort);
  }

  for (const auto &entry : temporalRegsByPE) {
    const PEContainment *pe =
        findPEContainmentByName(flattener, entry.getKey());
    if (!pe)
      continue;
    if (entry.getValue().size() > pe->numRegister) {
      diagnostics += "C5.2: temporal register overflow on " +
                     entry.getKey().str() + "\n";
      valid = false;
    }
  }

  if (!validateTaggedPathConflicts(state, dfg, adg, edgeKinds, diagnostics))
    valid = false;

  return valid;
}

} // namespace fcc
