//===-- TechMapper.cpp - Technology mapping for DFG to ADG ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/TypeCompat.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"

#include <functional>

namespace loom {

namespace {

/// Get the "op_name" attribute from a node, or empty string.
llvm::StringRef getOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Get the "resource_class" attribute from a node, or empty string.
llvm::StringRef getResourceClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Get an array attribute from a node by name.
mlir::ArrayAttr getArrayAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto arrAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue()))
        return arrAttr;
    }
  }
  return {};
}

/// Check if two types are compatible for mapping purposes.
/// For routing nodes (pass-through), only bit-width matters.
/// For functional/memory nodes, strict type checking applies.
/// Get bit width from a type for tech-mapping purposes.
unsigned getTechMapBitWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto bitsType = mlir::dyn_cast<loom::dataflow::BitsType>(type))
    return bitsType.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth();
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return floatTy.getWidth();
  if (type.isIndex())
    return loom::ADDR_BIT_WIDTH;
  return 0;
}

bool typesCompatible(mlir::Type swType, mlir::Type hwType,
                     bool /*isRoutingNode*/) {
  if (!swType || !hwType)
    return true; // Untyped ports are compatible.

  return isTypeWidthCompatible(swType, hwType);
}

/// Simple hash combining function for pattern hashing.
uint64_t hashCombine(uint64_t seed, uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  return seed;
}

} // namespace

PEBodyPattern TechMapper::extractPEPattern(const Graph &adg, IdIndex nodeId) {
  PEBodyPattern pattern;
  pattern.hwNodeId = nodeId;

  const Node *node = adg.getNode(nodeId);
  if (!node)
    return pattern;

  // Try to get body_ops attribute (array of string attrs from ADG flattener).
  mlir::ArrayAttr bodyOps = getArrayAttr(node, "body_ops");
  if (bodyOps) {
    for (auto attr : bodyOps) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
        pattern.opNames.push_back(strAttr.getValue().str());
    }
  }

  // Fallback: if no body_ops attribute, use op_name as single-op pattern.
  if (pattern.opNames.empty()) {
    llvm::StringRef opName = getOpName(node);
    if (!opName.empty() && opName != "fabric.pe")
      pattern.opNames.push_back(opName.str());
  }

  // Read body_edges attribute for internal connectivity (array of int pairs).
  mlir::ArrayAttr bodyEdges = getArrayAttr(node, "body_edges");
  if (bodyEdges) {
    for (size_t i = 0; i + 1 < bodyEdges.size(); i += 2) {
      auto srcAttr = mlir::dyn_cast<mlir::IntegerAttr>(bodyEdges[i]);
      auto dstAttr = mlir::dyn_cast<mlir::IntegerAttr>(bodyEdges[i + 1]);
      if (srcAttr && dstAttr) {
        pattern.internalEdges.emplace_back(
            static_cast<unsigned>(srcAttr.getInt()),
            static_cast<unsigned>(dstAttr.getInt()));
      }
    }
  }

  // Compute pattern hash for deduplication.
  uint64_t h = 0;
  for (const auto &opName : pattern.opNames) {
    h = hashCombine(h, std::hash<std::string>{}(opName));
  }
  for (const auto &edge : pattern.internalEdges) {
    h = hashCombine(h, static_cast<uint64_t>(edge.first));
    h = hashCombine(h, static_cast<uint64_t>(edge.second));
  }
  pattern.patternHash = h;

  return pattern;
}

bool TechMapper::isOpNameCompatible(llvm::StringRef swOp,
                                    llvm::StringRef hwOp) {
  if (swOp == hwOp)
    return true;

  // arith.cmpi/arith.cmpf: match regardless of predicate variant.
  if (swOp.starts_with("arith.cmpi") && hwOp.starts_with("arith.cmpi"))
    return true;
  if (swOp.starts_with("arith.cmpf") && hwOp.starts_with("arith.cmpf"))
    return true;

  // dataflow.stream: match if both are stream ops (cont_cond is configurable).
  if (swOp.starts_with("dataflow.stream") &&
      hwOp.starts_with("dataflow.stream"))
    return true;

  // ub.poison produces an undefined constant value; map to constant PE.
  if (swOp == "ub.poison" && hwOp == "handshake.constant")
    return true;

  return false;
}

bool TechMapper::isTypeCompatible(const Graph &dfg, IdIndex swPort,
                                  const Graph &adg, IdIndex hwPort) {
  const Port *sp = dfg.getPort(swPort);
  const Port *hp = adg.getPort(hwPort);
  if (!sp || !hp)
    return false;

  // Determine if the hardware port belongs to a routing node.
  const Node *hwNode = adg.getNode(hp->parentNode);
  bool isRouting = hwNode && getResourceClass(hwNode) == "routing";

  return typesCompatible(sp->type, hp->type, isRouting);
}

bool TechMapper::isSingleOpCompatible(const Graph &dfg, IdIndex swNode,
                                      const Graph &adg, IdIndex hwNode) {
  const Node *sw = dfg.getNode(swNode);
  const Node *hw = adg.getNode(hwNode);
  if (!sw || !hw)
    return false;

  // Sentinel nodes are not mapped via tech-mapping.
  if (sw->kind != Node::OperationNode || hw->kind != Node::OperationNode)
    return false;

  // Only functional and memory resources are placement targets.
  llvm::StringRef hwClass = getResourceClass(hw);
  if (hwClass != "functional" && hwClass != "memory")
    return false;

  llvm::StringRef swOp = getOpName(sw);
  llvm::StringRef hwOp = getOpName(hw);

  if (swOp.empty() || hwOp.empty())
    return false;

  // Memory operation matching.
  // Handshake and fabric memory use per-category ordering; for multi-port
  // memory (ldCount>1 or stCount>1), the ADG uses single tagged ports while
  // the DFG uses per-lane native ports. Bridge boundary metadata (added by
  // ADGFlattener Phase F) provides per-lane native boundary ports.
  if (hwOp == "fabric.memory" || hwOp == "fabric.extmemory") {
    if (!(swOp.contains("load") || swOp.contains("store") ||
          swOp.contains("memory")))
      return false;

    // Check for bridge-boundary metadata (multi-port memory).
    mlir::DenseI32ArrayAttr bridgeInPorts;
    mlir::DenseI32ArrayAttr bridgeOutPorts;
    for (auto &attr : hw->attributes) {
      if (attr.getName() == "bridge_input_ports")
        bridgeInPorts =
            mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
      else if (attr.getName() == "bridge_output_ports")
        bridgeOutPorts =
            mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    }

    if (bridgeInPorts || bridgeOutPorts) {
      // Multi-port memory with bridge: match against bridge boundary ports.
      // Bridge boundary ports are native (untagged) and in DFG-compatible
      // order, so we can use positional type matching.

      // For extmemory, DFG port 0 is memref. It maps directly to ADG port 0
      // (not through bridge). Remaining DFG ports map to bridge boundary.
      unsigned swInSkip = 0;
      if (hwOp == "fabric.extmemory") {
        // DFG extmemory has memref at input[0]; check memref compatibility.
        if (sw->inputPorts.empty())
          return false;
        const Port *sp0 = dfg.getPort(sw->inputPorts[0]);
        if (!sp0 || !mlir::isa<mlir::MemRefType>(sp0->type))
          return false;
        // Compare memref element type against ADG extmemory memref port.
        if (!hw->inputPorts.empty()) {
          const Port *hp0 = adg.getPort(hw->inputPorts[0]);
          if (hp0 && mlir::isa<mlir::MemRefType>(hp0->type)) {
            auto swMemRef = mlir::cast<mlir::MemRefType>(sp0->type);
            auto hwMemRef = mlir::cast<mlir::MemRefType>(hp0->type);
            if (swMemRef.getElementType() != hwMemRef.getElementType())
              return false;
          }
        }
        swInSkip = 1;
      }

      // Input port count: DFG non-memref inputs must fit bridge boundary.
      unsigned swInCount = sw->inputPorts.size() - swInSkip;
      unsigned hwBridgeInCount = bridgeInPorts ? bridgeInPorts.size() : 0;
      if (swInCount > hwBridgeInCount)
        return false;

      // Output port count: DFG outputs must fit bridge boundary.
      unsigned hwBridgeOutCount = bridgeOutPorts ? bridgeOutPorts.size() : 0;
      if (sw->outputPorts.size() > hwBridgeOutCount)
        return false;

      // Input type matching: multiset (order may differ across categories).
      auto typeSortKey =
          [](mlir::Type t) -> std::tuple<int, unsigned, unsigned> {
        if (mlir::isa<mlir::MemRefType>(t))
          return {0, 0, 0};
        if (mlir::isa<mlir::NoneType>(t))
          return {2, 0, 0};
        return {1, getTechMapBitWidth(t), 0};
      };

      llvm::SmallVector<mlir::Type, 4> swInTypes, hwInTypes;
      for (unsigned i = swInSkip; i < sw->inputPorts.size(); ++i) {
        const Port *p = dfg.getPort(sw->inputPorts[i]);
        if (p && p->type)
          swInTypes.push_back(p->type);
      }
      for (int32_t pid : bridgeInPorts.asArrayRef()) {
        const Port *p = adg.getPort(static_cast<IdIndex>(pid));
        if (p && p->type)
          hwInTypes.push_back(p->type);
      }
      llvm::sort(swInTypes, [&typeSortKey](mlir::Type a, mlir::Type b) {
        return typeSortKey(a) < typeSortKey(b);
      });
      llvm::sort(hwInTypes, [&typeSortKey](mlir::Type a, mlir::Type b) {
        return typeSortKey(a) < typeSortKey(b);
      });
      if (swInTypes.size() != hwInTypes.size())
        return false;
      for (size_t i = 0; i < swInTypes.size(); ++i) {
        if (!typesCompatible(swInTypes[i], hwInTypes[i], false))
          return false;
      }

      // Output type matching: positional (bridge outputs are in DFG order).
      for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
        const Port *sp = dfg.getPort(sw->outputPorts[i]);
        if (!sp || i >= static_cast<size_t>(bridgeOutPorts.size()))
          return false;
        const Port *hp =
            adg.getPort(static_cast<IdIndex>(bridgeOutPorts[i]));
        if (sp && hp && !typesCompatible(sp->type, hp->type, false))
          return false;
      }

      return true;
    }

    // Single-port memory (no bridge): original matching logic.
    // Port count check.
    if (sw->inputPorts.size() > hw->inputPorts.size())
      return false;
    if (sw->outputPorts.size() > hw->outputPorts.size())
      return false;

    // Multiset type matching for inputs.
    auto typeSortKey = [](mlir::Type t) -> std::tuple<int, unsigned, unsigned> {
      if (mlir::isa<mlir::MemRefType>(t))
        return {0, 0, 0};
      if (mlir::isa<mlir::NoneType>(t))
        return {2, 0, 0};
      if (auto tagged = mlir::dyn_cast<loom::dataflow::TaggedType>(t)) {
        unsigned valW = getTechMapBitWidth(tagged.getValueType());
        unsigned tagW = 0;
        if (auto tagInt = mlir::dyn_cast<mlir::IntegerType>(tagged.getTagType()))
          tagW = tagInt.getWidth();
        return {1, valW, tagW};
      }
      return {1, getTechMapBitWidth(t), 0};
    };
    auto collectTypes = [&typeSortKey](const Graph &g,
                           llvm::ArrayRef<IdIndex> portIds) {
      llvm::SmallVector<mlir::Type, 4> types;
      for (IdIndex pid : portIds) {
        const Port *p = g.getPort(pid);
        if (p && p->type)
          types.push_back(p->type);
      }
      llvm::sort(types, [&typeSortKey](mlir::Type a, mlir::Type b) {
        return typeSortKey(a) < typeSortKey(b);
      });
      return types;
    };

    auto swInTypes = collectTypes(dfg, sw->inputPorts);
    auto hwInTypes = collectTypes(adg, hw->inputPorts);
    if (swInTypes.size() != hwInTypes.size())
      return false;
    for (size_t i = 0; i < swInTypes.size(); ++i) {
      if (!typesCompatible(swInTypes[i], hwInTypes[i], false))
        return false;
    }

    // Output types: positional check.
    for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
      const Port *sp = dfg.getPort(sw->outputPorts[i]);
      const Port *hp = adg.getPort(hw->outputPorts[i]);
      if (sp && hp && !typesCompatible(sp->type, hp->type, false))
        return false;
    }

    return true;
  }

  // For PE nodes, check the body pattern.
  if (hwOp == "fabric.pe") {
    PEBodyPattern bodyPattern = extractPEPattern(adg, hwNode);

    if (bodyPattern.opNames.empty()) {
      // No body_ops attribute: PE without body info, accept generically.
      goto check_ports;
    }

    // Multi-op PE bodies are handled by findGroupCandidates, not single-op.
    if (bodyPattern.opNames.size() > 1)
      return false;

    // Single-op PE body: check if the DFG op matches the body op.
    if (!isOpNameCompatible(swOp, bodyPattern.opNames[0]))
      return false;

    goto check_ports;
  }

  // Direct name match (including arith.cmp* variants).
  if (!isOpNameCompatible(swOp, hwOp))
    return false;

check_ports:
  // Check port count compatibility.
  if (sw->inputPorts.size() > hw->inputPorts.size())
    return false;
  if (sw->outputPorts.size() > hw->outputPorts.size())
    return false;

  // Check type compatibility for each port pair.
  // Temporal PE FU nodes have tagged ports but operate on native values
  // internally -- tags are stripped at the FU boundary.
  // Non-temporal tagged PEs (with output_tag attribute) also accept native DFG
  // ops -- the tag wrapper is transparent at the PE boundary.
  bool isRouting = (hwClass == "routing");
  bool temporalFU = isTemporalPEFU(hw);
  bool taggedPE = false;
  if (!temporalFU) {
    for (auto &attr : hw->attributes) {
      if (attr.getName().getValue() == "output_tag") {
        taggedPE = true;
        break;
      }
    }
  }
  for (size_t i = 0; i < sw->inputPorts.size(); ++i) {
    const Port *sp = dfg.getPort(sw->inputPorts[i]);
    const Port *hp = adg.getPort(hw->inputPorts[i]);
    if (!sp || !hp)
      continue;
    if (temporalFU) {
      if (!isTypeWidthCompatibleForTemporalFU(sp->type, hp->type))
        return false;
    } else if (taggedPE) {
      if (!isTypeWidthCompatibleForTaggedPE(sp->type, hp->type))
        return false;
    } else {
      if (!typesCompatible(sp->type, hp->type, isRouting))
        return false;
    }
  }
  for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
    const Port *sp = dfg.getPort(sw->outputPorts[i]);
    const Port *hp = adg.getPort(hw->outputPorts[i]);
    if (!sp || !hp)
      continue;
    if (temporalFU) {
      if (!isTypeWidthCompatibleForTemporalFU(sp->type, hp->type))
        return false;
    } else if (taggedPE) {
      if (!isTypeWidthCompatibleForTaggedPE(sp->type, hp->type))
        return false;
    } else {
      if (!typesCompatible(sp->type, hp->type, isRouting))
        return false;
    }
  }

  return true;
}

/// Collect DFG neighbor node IDs reachable from a set of matched nodes
/// (both downstream via output edges and upstream via input edges).
static void collectNeighbors(const Graph &dfg,
                             llvm::ArrayRef<IdIndex> matched,
                             const llvm::DenseSet<IdIndex> &usedNodes,
                             llvm::SmallVectorImpl<IdIndex> &neighbors) {
  llvm::DenseSet<IdIndex> seen;
  for (IdIndex prevSwId : matched) {
    const Node *prevSw = dfg.getNode(prevSwId);
    if (!prevSw)
      continue;

    // Downstream neighbors (output edges).
    for (IdIndex outPortId : prevSw->outputPorts) {
      const Port *outPort = dfg.getPort(outPortId);
      if (!outPort)
        continue;
      for (IdIndex edgeId : outPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort)
          continue;
        IdIndex candId = dstPort->parentNode;
        if (candId != INVALID_ID && !usedNodes.count(candId) &&
            seen.insert(candId).second) {
          const Node *candNode = dfg.getNode(candId);
          if (candNode && candNode->kind == Node::OperationNode)
            neighbors.push_back(candId);
        }
      }
    }

    // Upstream neighbors (input edges).
    for (IdIndex inPortId : prevSw->inputPorts) {
      const Port *inPort = dfg.getPort(inPortId);
      if (!inPort)
        continue;
      for (IdIndex edgeId : inPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (!srcPort)
          continue;
        IdIndex candId = srcPort->parentNode;
        if (candId != INVALID_ID && !usedNodes.count(candId) &&
            seen.insert(candId).second) {
          const Node *candNode = dfg.getNode(candId);
          if (candNode && candNode->kind == Node::OperationNode)
            neighbors.push_back(candId);
        }
      }
    }
  }
}

/// Check if there is a DFG edge from srcSwId to dstSwId.
static bool hasDFGEdge(const Graph &dfg, IdIndex srcSwId, IdIndex dstSwId) {
  const Node *srcSw = dfg.getNode(srcSwId);
  if (!srcSw)
    return false;

  for (IdIndex outPortId : srcSw->outputPorts) {
    const Port *outPort = dfg.getPort(outPortId);
    if (!outPort)
      continue;
    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *dp = dfg.getPort(edge->dstPort);
      if (dp && dp->parentNode == dstSwId)
        return true;
    }
  }
  return false;
}

/// Recursive backtracking matcher for multi-op PE body patterns.
/// Tries to assign each pattern op to a DFG node with correct op-name
/// compatibility and internal edge correspondence.
static bool matchPatternRecursive(
    const Graph &dfg, const PEBodyPattern &pattern, size_t patIdx,
    llvm::SmallVectorImpl<IdIndex> &matched,
    llvm::DenseSet<IdIndex> &usedNodes, TechMapper &mapper) {

  if (patIdx >= pattern.opNames.size())
    return true; // All ops matched.

  // Collect candidate DFG neighbors from already-matched nodes.
  llvm::SmallVector<IdIndex, 8> neighbors;
  collectNeighbors(dfg, matched, usedNodes, neighbors);

  for (IdIndex candId : neighbors) {
    llvm::StringRef candOp = getOpName(dfg.getNode(candId));
    if (!mapper.isOpNameCompatible(candOp, pattern.opNames[patIdx]))
      continue;

    // Check operand/result correspondence: verify that required internal
    // edges to/from already-matched nodes exist.
    bool edgesOk = true;
    for (const auto &[srcIdx, dstIdx] : pattern.internalEdges) {
      if (srcIdx == patIdx && dstIdx < matched.size()) {
        if (!hasDFGEdge(dfg, candId, matched[dstIdx])) {
          edgesOk = false;
          break;
        }
      }
      if (dstIdx == patIdx && srcIdx < matched.size()) {
        if (!hasDFGEdge(dfg, matched[srcIdx], candId)) {
          edgesOk = false;
          break;
        }
      }
    }
    if (!edgesOk)
      continue;

    // Try this assignment.
    matched.push_back(candId);
    usedNodes.insert(candId);

    if (matchPatternRecursive(dfg, pattern, patIdx + 1, matched,
                              usedNodes, mapper))
      return true;

    // Backtrack.
    matched.pop_back();
    usedNodes.erase(candId);
  }

  return false;
}

void TechMapper::findGroupCandidates(
    const Graph &dfg, const Graph &adg,
    const std::vector<PEBodyPattern> &patterns, CandidateSet &candidates) {
  // For each multi-op PE body pattern, find matching DFG subgraphs.
  for (const auto &pattern : patterns) {
    if (pattern.opNames.size() <= 1)
      continue; // Skip single-op patterns.

    const Node *hwNode = adg.getNode(pattern.hwNodeId);
    if (!hwNode)
      continue;

    // For each DFG node, check if it could be the root of a subgraph
    // matching this PE body pattern.
    for (IdIndex startSwId = 0;
         startSwId < static_cast<IdIndex>(dfg.nodes.size()); ++startSwId) {
      const Node *startSw = dfg.getNode(startSwId);
      if (!startSw || startSw->kind != Node::OperationNode)
        continue;

      llvm::StringRef startOp = getOpName(startSw);
      if (!isOpNameCompatible(startOp, pattern.opNames[0]))
        continue;

      // Use recursive backtracking to match remaining ops.
      llvm::SmallVector<IdIndex, 4> matched;
      matched.push_back(startSwId);

      llvm::DenseSet<IdIndex> usedNodes;
      usedNodes.insert(startSwId);

      if (!matchPatternRecursive(dfg, pattern, 1, matched, usedNodes,
                                 *this))
        continue;

      // Verify ALL internal edge connectivity matches the pattern.
      bool edgesOk = true;
      for (const auto &[srcIdx, dstIdx] : pattern.internalEdges) {
        if (srcIdx >= matched.size() || dstIdx >= matched.size()) {
          edgesOk = false;
          break;
        }
        if (!hasDFGEdge(dfg, matched[srcIdx], matched[dstIdx])) {
          edgesOk = false;
          break;
        }
      }
      if (!edgesOk)
        continue;

      // Check port compatibility: HW PE must have enough ports.
      if (hwNode->inputPorts.size() == 0 && hwNode->outputPorts.size() == 0)
        continue;

      // Valid group match! Create a group candidate for each matched node.
      Candidate groupCand;
      groupCand.hwNodeId = pattern.hwNodeId;
      groupCand.isGroup = true;
      for (IdIndex swId : matched)
        groupCand.swNodeIds.push_back(swId);

      // Add this group candidate to each node in the group.
      for (IdIndex swId : matched) {
        candidates[swId].push_back(groupCand);
      }
    }
  }
}

void TechMapper::mergeCandidates(CandidateSet &candidates) {
  // For each DFG node, if it participates in any group candidate,
  // remove single-op candidates that are superseded by group matches.
  // Group candidates have priority when they cover more DFG nodes.

  // Collect nodes that are part of group candidates.
  llvm::DenseSet<IdIndex> groupCoveredNodes;
  for (auto &[swId, cands] : candidates) {
    for (const auto &cand : cands) {
      if (cand.isGroup && cand.swNodeIds.size() > 1) {
        for (IdIndex nodeId : cand.swNodeIds)
          groupCoveredNodes.insert(nodeId);
      }
    }
  }

  // For nodes covered by groups, partition candidates: keep group candidates
  // and remove single-op candidates for the same HW node if a group
  // candidate exists. Keep single-op candidates as fallback.
  for (IdIndex coveredNode : groupCoveredNodes) {
    auto it = candidates.find(coveredNode);
    if (it == candidates.end())
      continue;

    auto &cands = it->second;

    // Check if there are both group and single-op candidates.
    bool hasGroup = false;
    for (const auto &c : cands) {
      if (c.isGroup) {
        hasGroup = true;
        break;
      }
    }

    if (!hasGroup)
      continue;

    // Sort: group candidates first (they have priority).
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate &a, const Candidate &b) {
                       if (a.isGroup != b.isGroup)
                         return a.isGroup > b.isGroup;
                       return a.swNodeIds.size() > b.swNodeIds.size();
                     });
  }
}

CandidateSet TechMapper::map(const Graph &dfg, const Graph &adg) {
  CandidateSet candidates;

  // Extract PE body patterns from all ADG PE nodes.
  std::vector<PEBodyPattern> pePatterns;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;

    llvm::StringRef hwOp = getOpName(hwNode);
    llvm::StringRef hwClass = getResourceClass(hwNode);
    if (hwOp == "fabric.pe" || hwClass == "functional") {
      PEBodyPattern pattern = extractPEPattern(adg, hwId);
      if (!pattern.opNames.empty())
        pePatterns.push_back(std::move(pattern));
    }
  }

  // Single-op matching: for each DFG operation, find compatible ADG nodes.
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;

    std::vector<Candidate> &nodeCandidates = candidates[swId];

    for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
         ++hwId) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode || hwNode->kind != Node::OperationNode)
        continue;

      if (isSingleOpCompatible(dfg, swId, adg, hwId)) {
        Candidate c;
        c.hwNodeId = hwId;
        c.swNodeIds.push_back(swId);
        c.isGroup = false;
        nodeCandidates.push_back(c);
      }
    }

  }

  // Multi-op group matching for PE bodies with multiple operations.
  findGroupCandidates(dfg, adg, pePatterns, candidates);

  // Merge and prioritize candidates.
  mergeCandidates(candidates);

  // Sort each candidate list so exact-port-count matches appear first.
  // This prevents a DFG node with fewer ports (e.g., 1-input join) from
  // stealing a larger PE (e.g., 3-input join) when an exact-match PE exists.
  for (auto &[swId, cands] : candidates) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    size_t swIn = swNode->inputPorts.size();
    size_t swOut = swNode->outputPorts.size();
    std::stable_sort(cands.begin(), cands.end(),
                     [&](const Candidate &a, const Candidate &b) {
                       // Group candidates always come first.
                       if (a.isGroup != b.isGroup)
                         return a.isGroup > b.isGroup;
                       const Node *ha = adg.getNode(a.hwNodeId);
                       const Node *hb = adg.getNode(b.hwNodeId);
                       if (!ha || !hb)
                         return false;
                       bool aExact = (ha->inputPorts.size() == swIn &&
                                      ha->outputPorts.size() == swOut);
                       bool bExact = (hb->inputPorts.size() == swIn &&
                                      hb->outputPorts.size() == swOut);
                       return aExact > bExact;
                     });
  }

  return candidates;
}

} // namespace loom
