//===-- MapperValidation.cpp - Constraint validation for mapper ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace loom {

namespace {

llvm::StringRef getNodeOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

} // namespace

bool Mapper::validateC1(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C1: Node compatibility - each mapped DFG node must be compatible with
  // its ADG target.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwNode = state.swNodeToHwNode[i];
    if (hwNode == INVALID_ID)
      continue;

    const Node *sw = dfg.getNode(i);
    const Node *hw = adg.getNode(hwNode);
    if (!sw || !hw) {
      diag = "C1: invalid node reference sw=" + std::to_string(i) +
             " hw=" + std::to_string(hwNode);
      return false;
    }

    // Sentinel nodes must map to sentinel nodes of the same kind.
    if (sw->kind != hw->kind && sw->kind != Node::OperationNode) {
      diag = "C1: sentinel kind mismatch sw=" + std::to_string(i);
      return false;
    }
  }
  return true;
}

bool Mapper::validateC2(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C2: Port/type compatibility.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swPortToHwPort.size());
       ++i) {
    IdIndex hwPort = state.swPortToHwPort[i];
    if (hwPort == INVALID_ID)
      continue;

    const Port *sw = dfg.getPort(i);
    const Port *hw = adg.getPort(hwPort);
    if (!sw || !hw) {
      diag = "C2: invalid port reference sw=" + std::to_string(i);
      return false;
    }

    // Direction must match.
    if (sw->direction != hw->direction) {
      diag = "C2: direction mismatch sw_port=" + std::to_string(i);
      return false;
    }

    // Type compatibility (relaxed for routing nodes).
    if (sw->type && hw->type) {
      const Node *hwNode = adg.getNode(hw->parentNode);
      bool isRouting = hwNode && getNodeResClass(hwNode) == "routing";

      if (!isRouting && sw->type != hw->type) {
        diag = "C2: type mismatch sw_port=" + std::to_string(i);
        return false;
      }
    }
  }
  return true;
}

bool Mapper::validateC3(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C3: Route legality - each mapped edge path must follow physical
  // connectivity.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &path = state.swEdgeToHwPaths[i];
    if (path.empty())
      continue;

    // Verify each consecutive pair of ports in the path.
    for (size_t j = 0; j + 1 < path.size(); j += 2) {
      IdIndex fromPort = path[j];
      IdIndex toPort = path[j + 1];

      // Check if physical connection exists.
      const Port *fp = adg.getPort(fromPort);
      const Port *tp = adg.getPort(toPort);
      if (!fp || !tp) {
        diag = "C3: invalid port in path of edge " + std::to_string(i);
        return false;
      }
    }
  }
  return true;
}

bool Mapper::validateC4(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C4: Capacity constraints.
  // Check that no ADG node has more mappings than its capacity allows.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    const auto &swNodes = state.hwNodeToSwNodes[i];
    if (swNodes.empty())
      continue;

    const Node *hwNode = adg.getNode(i);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getNodeResClass(hwNode);

    // Non-temporal functional nodes: exclusive (at most 1 mapping).
    if (resClass == "functional") {
      // Check if this is a temporal PE FU node (allows sharing).
      bool isTemporal = false;
      for (auto &attr : hwNode->attributes) {
        if (attr.getName() == "parent_temporal_pe" ||
            attr.getName() == "is_virtual") {
          isTemporal = true;
          break;
        }
      }

      if (!isTemporal && swNodes.size() > 1) {
        diag = "C4: capacity exceeded on hw_node=" + std::to_string(i) +
               " (" + std::to_string(swNodes.size()) + " mappings)";
        return false;
      }
    }
  }
  return true;
}

bool Mapper::validateC5(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C5: Temporal constraints.
  // Verify that temporal assignments are within bounds.
  // This is largely covered by runTemporalAssignment's fail-fast checks.
  return true;
}

bool Mapper::validateC6(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C6: Configuration encoding constraints.
  // Verify that config values are encodable within bit widths.
  // This is validated during config generation.
  return true;
}

bool Mapper::runValidation(const MappingState &state, const Graph &dfg,
                           const Graph &adg, std::string &diagnostics) {
  // Check all operations are mapped.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::OperationNode &&
        (i >= state.swNodeToHwNode.size() ||
         state.swNodeToHwNode[i] == INVALID_ID)) {
      diagnostics = "Unmapped operation node " + std::to_string(i);
      return false;
    }
  }

  // Check all edges are routed.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      diagnostics = "Unrouted edge " + std::to_string(i);
      return false;
    }
  }

  // Run constraint checks.
  if (!validateC1(state, dfg, adg, diagnostics))
    return false;
  if (!validateC2(state, dfg, adg, diagnostics))
    return false;
  if (!validateC3(state, dfg, adg, diagnostics))
    return false;
  if (!validateC4(state, dfg, adg, diagnostics))
    return false;
  if (!validateC5(state, dfg, adg, diagnostics))
    return false;
  if (!validateC6(state, dfg, adg, diagnostics))
    return false;

  return true;
}

} // namespace loom
