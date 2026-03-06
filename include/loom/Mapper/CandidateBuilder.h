//===-- CandidateBuilder.h - Candidate set assembly ---------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Assembles candidate sets from TechMapper results and performs early failure
// checking (CPL_MAPPER_NO_COMPATIBLE_HW).
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_CANDIDATEBUILDER_H
#define LOOM_MAPPER_CANDIDATEBUILDER_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/Types.h"

#include <string>

namespace loom {

class CandidateBuilder {
public:
  struct Result {
    bool success = false;
    CandidateSet candidates;
    /// If !success, which DFG node had no compatible hardware.
    IdIndex failedNode = INVALID_ID;
    std::string diagnostics;
  };

  /// Build candidate sets for all DFG nodes. Fails early if any operation
  /// node has an empty candidate set.
  Result build(const Graph &dfg, const Graph &adg);
};

} // namespace loom

#endif // LOOM_MAPPER_CANDIDATEBUILDER_H
