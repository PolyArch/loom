//===-- TechMapper.cpp - Technology mapping for DFG to ADG ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/TechMapper.h"

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

/// Check if two MLIR types are width-compatible.
/// Allows matching native types (i32, f32) with bits<N> types of the same
/// width. IndexType maps to ADDR_BIT_WIDTH (57) for width comparison.
static bool isTypeWidthCompatible(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;

  auto getWidth = [](mlir::Type t) -> std::optional<unsigned> {
    if (auto bits = mlir::dyn_cast<loom::dataflow::BitsType>(t))
      return bits.getWidth();
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(t))
      return intTy.getWidth();
    if (mlir::isa<mlir::Float32Type>(t))
      return 32u;
    if (mlir::isa<mlir::Float64Type>(t))
      return 64u;
    if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(t))
      return 16u;
    if (t.isIndex())
      return (unsigned)loom::ADDR_BIT_WIDTH;
    if (mlir::isa<mlir::NoneType>(t))
      return 0u;
    return std::nullopt;
  };

  // For tagged types, check tag types match AND value widths match.
  auto tagA = mlir::dyn_cast<loom::dataflow::TaggedType>(a);
  auto tagB = mlir::dyn_cast<loom::dataflow::TaggedType>(b);
  if (tagA && tagB) {
    if (tagA.getTagType() != tagB.getTagType())
      return false;
    auto wA = getWidth(tagA.getValueType());
    auto wB = getWidth(tagB.getValueType());
    return wA && wB && *wA == *wB;
  }
  if (tagA || tagB)
    return false; // one tagged, one not

  auto wA = getWidth(a);
  auto wB = getWidth(b);
  return wA && wB && *wA == *wB;
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
  // Note: handshake.memory and fabric.memory use different operand orderings
  // (handshake: [st_data, st_addr, ld_addr], fabric: [ld_addr, st_addr, st_data])
  // so we use multiset type matching instead of positional checking.
  if (hwOp == "fabric.memory" || hwOp == "fabric.extmemory") {
    if (!(swOp.contains("load") || swOp.contains("store") ||
          swOp.contains("memory")))
      return false;

    // Port count check.
    if (sw->inputPorts.size() > hw->inputPorts.size())
      return false;
    if (sw->outputPorts.size() > hw->outputPorts.size())
      return false;

    // Multiset type matching for inputs: collect types, sort, compare.
    auto collectTypes = [](const Graph &g,
                           llvm::ArrayRef<IdIndex> portIds) {
      llvm::SmallVector<mlir::Type, 4> types;
      for (IdIndex pid : portIds) {
        const Port *p = g.getPort(pid);
        if (p && p->type)
          types.push_back(p->type);
      }
      llvm::sort(types, [](mlir::Type a, mlir::Type b) {
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
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

    // Output types: positional check (output ordering is compatible).
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
  bool isRouting = (hwClass == "routing");
  for (size_t i = 0; i < sw->inputPorts.size(); ++i) {
    const Port *sp = dfg.getPort(sw->inputPorts[i]);
    const Port *hp = adg.getPort(hw->inputPorts[i]);
    if (sp && hp && !typesCompatible(sp->type, hp->type, isRouting))
      return false;
  }
  for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
    const Port *sp = dfg.getPort(sw->outputPorts[i]);
    const Port *hp = adg.getPort(hw->outputPorts[i]);
    if (sp && hp && !typesCompatible(sp->type, hp->type, isRouting))
      return false;
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

  return candidates;
}

} // namespace loom
