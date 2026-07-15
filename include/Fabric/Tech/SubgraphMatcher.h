#ifndef FABRIC_TECH_SUBGRAPHMATCHER_H
#define FABRIC_TECH_SUBGRAPHMATCHER_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <utility>

namespace fabric {

struct FuMatchResult {
  // Whether the pattern subgraph is implementable by `fu`.
  bool matched = false;
  // Reference to the matching FU.
  FuOp fu;
  // Index into the FU's Fabric-owned valid_encodings array.
  std::size_t encodingIndex = 0;
  // Pattern actor order to physical fabric.op resource index.
  ::llvm::SmallVector<unsigned, 8> actorToFabricOp;
  // Software boundary port to FU boundary port correspondence.
  ::llvm::SmallVector<std::pair<unsigned, unsigned>, 4> inputPorts;
  ::llvm::SmallVector<std::pair<unsigned, unsigned>, 4> outputPorts;
};

// Match a legacy dataflow.subgraph input against the FU's explicit semantic
// encodings. The result selects one encoding and carries the complete mapping
// witness; it never produces or persists workload-selected sw_configs.
FuMatchResult mapPatternToFu(::dataflow::SubgraphOp pattern, FuOp fu);

} // namespace fabric

#endif // FABRIC_TECH_SUBGRAPHMATCHER_H
