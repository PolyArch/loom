#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H

// CoverageVerifier wraps `fabric::enumerateFuSubgraphs` and
// `fabric::subgraphsIsomorphic` so the synthesis pipeline can ask
// "is every input dataflow.subgraph isomorphic to at least one
// materialization of this fabric.fu?". It is the synthesizer's
// correctness oracle.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "CoverageVerifier".
//
// Threading: `verify` constructs a fresh scratch ModuleOp per call
// and discards it deterministically before returning, so the user's
// own ModuleOp never receives the enumerator's appended candidate
// `func.func`s. When `coverage_verifier.parallel_match` is true and
// `parallelism.workers > 1`, the per-input matching loop runs on a
// `WorkerPool`; the candidate vector itself is read-only during the
// loop so concurrent reads are safe.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h" // for CoverageReport

#include "llvm/ADT/ArrayRef.h"

namespace loom::fabric::tech {

class CoverageVerifier {
public:
  // Reads `coverage_verifier.parallel_match` and
  // `parallelism.workers` from `cfg`.
  explicit CoverageVerifier(const ::loom::SynthConfig &cfg);

  // Materializes `fu` by:
  //   1. Constructing a scratch ModuleOp owned by this call.
  //   2. Cloning the wrapper func.func + fabric.fu into the scratch
  //      module (so SubgraphEnumerator's append behavior does not
  //      pollute the user's module).
  //   3. Invoking `fabric::enumerateFuSubgraphs` on the scratch
  //      module.
  //   4. Matching each input subgraph against the appended
  //      candidates with `fabric::subgraphsIsomorphic` (parallel
  //      per `parallel_match`).
  //   5. Discarding the scratch module deterministically before
  //      return (RAII).
  CoverageReport verify(::fabric::FuOp fu,
                        ::llvm::ArrayRef<::dataflow::SubgraphOp> inputs);

private:
  bool parallelMatch;
  unsigned parallelismWorkers;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_COVERAGEVERIFIER_H
