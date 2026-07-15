#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H

// CoverageVerifier projects the Fabric-defined valid semantic encodings and
// matches them against canonical ConfiguredFunctions.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "CoverageVerifier".
//
// When parallel matching is enabled, each input is matched independently
// against the shared read-only projected candidate vector.

#include "Common/SynthConfig.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h" // for CoverageReport

#include "llvm/ADT/ArrayRef.h"

namespace loom::fabric::tech {

class CoverageVerifier {
public:
  // Reads `coverage_verifier.parallel_match` and
  // `parallelism.workers` from `cfg`.
  explicit CoverageVerifier(const ::loom::SynthConfig &cfg);

  // Each successful slot includes the encoding index, actor-to-node mapping,
  // and software/FU boundary-port correspondence.
  CoverageReport verify(::fabric::FuOp fu,
                        ::llvm::ArrayRef<::fabric::ConfiguredFunction> inputs);

private:
  bool parallelMatch;
  unsigned parallelismWorkers;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H
