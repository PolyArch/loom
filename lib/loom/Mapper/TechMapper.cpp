//===-- TechMapper.cpp - Technology mapping for DFG to ADG ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/TechMapper.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

/// Check if two types are compatible for mapping purposes.
/// For routing nodes (pass-through), only bit-width matters.
/// For functional/memory nodes, strict type checking applies.
bool typesCompatible(mlir::Type swType, mlir::Type hwType,
                     bool isRoutingNode) {
  if (!swType || !hwType)
    return true; // Untyped ports are compatible.

  if (swType == hwType)
    return true;

  if (isRoutingNode) {
    // For routing nodes, check bit-width compatibility.
    if (auto swInt = mlir::dyn_cast<mlir::IntegerType>(swType)) {
      if (auto hwInt = mlir::dyn_cast<mlir::IntegerType>(hwType))
        return swInt.getWidth() <= hwInt.getWidth();
    }
    if (auto swFloat = mlir::dyn_cast<mlir::FloatType>(swType)) {
      if (auto hwFloat = mlir::dyn_cast<mlir::FloatType>(hwType))
        return swFloat.getWidth() <= hwFloat.getWidth();
    }
    // Integer -> Float width comparison for pass-through.
    if (auto swInt = mlir::dyn_cast<mlir::IntegerType>(swType)) {
      if (auto hwFloat = mlir::dyn_cast<mlir::FloatType>(hwType))
        return swInt.getWidth() <= hwFloat.getWidth();
    }
    if (auto swFloat = mlir::dyn_cast<mlir::FloatType>(swType)) {
      if (auto hwInt = mlir::dyn_cast<mlir::IntegerType>(hwType))
        return swFloat.getWidth() <= static_cast<unsigned>(hwInt.getWidth());
    }
  }

  return false;
}

} // namespace

std::vector<std::string> TechMapper::extractPEPattern(const Graph &adg,
                                                       IdIndex nodeId) {
  std::vector<std::string> pattern;
  const Node *node = adg.getNode(nodeId);
  if (!node)
    return pattern;

  llvm::StringRef opName = getOpName(node);
  if (!opName.empty())
    pattern.push_back(opName.str());

  return pattern;
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

  // Special case: arith.cmpi/arith.cmpf match regardless of predicate.
  if (swOp.starts_with("arith.cmp") && hwOp.starts_with("arith.cmp")) {
    if (swOp.starts_with("arith.cmpi") && hwOp.starts_with("arith.cmpi"))
      goto check_ports;
    if (swOp.starts_with("arith.cmpf") && hwOp.starts_with("arith.cmpf"))
      goto check_ports;
  }

  // For PE nodes, check if the PE body contains the required operation.
  // The ADG PE node's op_name may represent the PE definition name, not
  // individual operations. We check if they are from the same operation class.
  if (hwOp == "fabric.pe") {
    // PE nodes can execute any operation in their body.
    // For now, we accept if the PE exists (body-level matching
    // would require walking the PE body in the original MLIR).
    goto check_ports;
  }

  // Memory operation matching.
  if (hwOp == "fabric.memory" || hwOp == "fabric.extmemory") {
    if (swOp.contains("load") || swOp.contains("store") ||
        swOp.contains("memory"))
      goto check_ports;
    return false;
  }

  // Direct name match.
  if (swOp != hwOp)
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

CandidateSet TechMapper::map(const Graph &dfg, const Graph &adg) {
  CandidateSet candidates;

  // For each DFG operation node, find compatible ADG nodes.
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
        nodeCandidates.push_back(c);
      }
    }
  }

  return candidates;
}

} // namespace loom
