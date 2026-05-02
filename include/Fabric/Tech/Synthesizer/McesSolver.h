#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCESSOLVER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCESSOLVER_H

#include "Fabric/Tech/Synthesizer/McsGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <chrono>
#include <cstddef>
#include <string>

namespace loom::fabric::tech {

struct McesSharedNode {
  unsigned id = 0;
  ::llvm::SmallVector<unsigned, 4> nodeIndexByGraph;
};

struct McesCandidate {
  ::llvm::SmallVector<McesSharedNode, 4> sharedNodes;
  std::string debugLabel;
};

struct McesSearchOptions {
  std::size_t candidateCap = 0;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
};

struct McesSearchResult {
  ::llvm::SmallVector<McesCandidate, 4> candidates;
  std::size_t generatedCandidates = 0;
  bool hitCap = false;
  bool hitTimeout = false;
};

class McesSolver {
public:
  McesSearchResult enumerate(::llvm::ArrayRef<McsGraph> graphs,
                             const McesSearchOptions &options) const;

  ::llvm::SmallVector<McesCandidate, 4>
  enumerate(::llvm::ArrayRef<McsGraph> graphs, std::size_t cap) const;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_MCESSOLVER_H
