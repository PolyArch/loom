#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_EXACTMCESSOLVER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_EXACTMCESSOLVER_H

#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/McesSolver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <chrono>
#include <cstddef>

namespace loom::fabric::tech {

struct ExactMcesSearchOptions {
  std::size_t candidateCap = 0;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  unsigned workers = 1;
  AreaWeights costWeights;
};

struct ExactMcesSearchResult {
  ::llvm::SmallVector<McesCandidate, 4> candidates;
  std::size_t generatedCandidates = 0;
  bool hitCap = false;
  bool hitTimeout = false;
  bool provedOptimal = false;
};

class ExactMcesSolver {
public:
  ExactMcesSearchResult enumerate(::llvm::ArrayRef<McsGraph> graphs,
                                  const ExactMcesSearchOptions &options) const;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_EXACTMCESSOLVER_H
