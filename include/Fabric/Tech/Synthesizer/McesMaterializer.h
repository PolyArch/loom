#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCESMATERIALIZER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCESMATERIALIZER_H

#include "Fabric/Tech/Synthesizer/McesSolver.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <chrono>
#include <cstddef>
#include <string>

namespace loom::fabric::tech {

struct McesMaterializedCandidate {
  std::size_t candidateIndex = 0;
  double cost = 0.0;
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  CoverageReport coverage;
  ::llvm::SmallVector<std::string, 4> notes;
};

class McesMaterializer {
public:
  ::llvm::SmallVector<McesMaterializedCandidate, 4>
  materializeExactCoverCandidates(
      const SynthInputs &inputs,
      ::llvm::ArrayRef<McesCandidate> candidates) const;

  ::llvm::SmallVector<McesMaterializedCandidate, 4>
  materializeExactCoverCandidates(
      const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
      ::llvm::ArrayRef<McesCandidate> candidates) const;

  ::llvm::SmallVector<McesMaterializedCandidate, 4>
  materializeExactCoverCandidates(
      const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
      ::llvm::ArrayRef<McesCandidate> candidates,
      std::chrono::steady_clock::time_point deadline) const;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_MCESMATERIALIZER_H
