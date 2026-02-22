//===-- CandidateBuilder.cpp - Candidate set assembly --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/CandidateBuilder.h"

namespace loom {

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
      result.diagnostics = "CPL_MAPPER_NO_COMPATIBLE_HW: DFG node " +
                            std::to_string(i) + " has no compatible hardware";
      return result;
    }
  }

  result.success = true;
  return result;
}

} // namespace loom
