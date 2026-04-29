#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_H

// Incremental strategy: left-fold over input subgraphs. Start from a
// trivial FU built from input_0 (one fabric.op per body op, mirroring
// the subgraph's topology); for each subsequent input either confirm
// it is already covered by the current FU or extend the FU with the
// lowest-cost legal candidate produced by op-list widening or
// mux/demux insertion. Tier C structural extension (back-edge SCCs)
// is delegated to a follow-up task; this implementation handles tier
// A and tier B.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: incremental" and "Acceptance criteria (incremental)".
//
// Threading: the strategy must build its candidate wrapper inside the
// worker-local `MLIRContext` provided via `SynthInputs.context`. The
// pass's main thread re-homes the returned wrapper into the user's
// module context (see `GeneralizeSubgraphsToFuPass`'s splice loop).

#include "Common/SynthConfig.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

namespace loom::fabric::tech {

class IncrementalSynthesizer final : public Synthesizer {
public:
  explicit IncrementalSynthesizer(const ::loom::SynthConfig &cfg);
  SynthResult run(const SynthInputs &) override;

private:
  const ::loom::SynthConfig &cfg;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTAL_H
