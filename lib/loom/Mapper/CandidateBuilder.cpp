//===-- CandidateBuilder.cpp - Candidate set assembly --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/CandidateBuilder.h"
#include "mlir/IR/BuiltinAttributes.h"

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
} // namespace

CandidateBuilder::Result CandidateBuilder::build(const Graph &dfg,
                                                  const Graph &adg) {
  Result result;

  TechMapper mapper;
  result.candidates = mapper.map(dfg, adg);

  // Check that every operation node has at least one candidate.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;

    auto it = result.candidates.find(i);
    if (it == result.candidates.end() || it->second.empty()) {
      result.success = false;
      result.failedNode = i;
      llvm::StringRef opName = getNodeOpName(node);
      std::string portInfo;
      portInfo += " in=[";
      for (size_t p = 0; p < node->inputPorts.size(); ++p) {
        if (p > 0) portInfo += ",";
        const Port *port = dfg.getPort(node->inputPorts[p]);
        if (port && port->type) {
          std::string ts;
          llvm::raw_string_ostream os(ts);
          port->type.print(os);
          portInfo += ts;
        } else {
          portInfo += "?";
        }
      }
      portInfo += "] out=[";
      for (size_t p = 0; p < node->outputPorts.size(); ++p) {
        if (p > 0) portInfo += ",";
        const Port *port = dfg.getPort(node->outputPorts[p]);
        if (port && port->type) {
          std::string ts;
          llvm::raw_string_ostream os(ts);
          port->type.print(os);
          portInfo += ts;
        } else {
          portInfo += "?";
        }
      }
      portInfo += "]";
      result.diagnostics = "CPL_MAPPER_NO_COMPATIBLE_HW: DFG node " +
                            std::to_string(i) + " has no compatible hardware"
                            " (op=" + opName.str() + portInfo + ")";
      return result;
    }
  }

  result.success = true;
  return result;
}

} // namespace loom
