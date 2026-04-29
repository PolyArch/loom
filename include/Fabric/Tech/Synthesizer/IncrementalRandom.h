#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_RANDOM_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_RANDOM_H

// IncrementalRandom strategy: parallel multi-restart wrapper around the
// Incremental synthesizer. Generates `restarts` random permutations of
// the input subgraph list (seeded by `cfg.incrementalRandomSeed` for
// determinism), runs each via an internal `WorkerPool::parallelMap`, and
// returns the lowest-cost successful FU. Ties are broken by the lowest
// permutation index so the chosen wrapper is reproducible across runs
// with the same seed and input set.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: incremental_random" and
// "Acceptance criteria (incremental_random)".
//
// Threading: each restart runs in a fresh sub-`MLIRContext` so the
// concurrent strategy invocations do not race on the outer scratch
// context's `StorageUniquer`. The winning sub-context wrapper is
// serialized to text and re-parsed into `SynthInputs.context` before
// being returned, mirroring the cross-context handoff used by the pass
// itself for outer-scratch -> user-context.

#include "Common/SynthConfig.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

namespace loom::fabric::tech {

class IncrementalRandomSynthesizer final : public Synthesizer {
public:
  explicit IncrementalRandomSynthesizer(const ::loom::SynthConfig &cfg);
  SynthResult run(const SynthInputs &) override;

private:
  const ::loom::SynthConfig &cfg;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_RANDOM_H
