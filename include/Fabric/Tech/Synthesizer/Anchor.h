#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H

// Anchor strategy: lock-step BFS from yield anchors across all input
// `dataflow.subgraph`s in a synth group. Designed to handle tier A
// inputs (topology-isomorphic; only the op identity at each node
// position varies) plus the restricted tier B case where local
// `fabric.mux` legalizes a cross-share-group node position.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "Strategy: anchor (tier A by default)".
//
// Threading: the strategy must build its candidate wrapper inside the
// worker-local `MLIRContext` provided via `SynthInputs.context`. The
// pass's main thread re-homes the returned wrapper into the user's
// module context (see `GeneralizeSubgraphsToFuPass`'s splice loop).

#include "Common/SynthConfig.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

namespace loom::fabric::tech {

// Anchor strategy: lock-step BFS from yield anchors. Handles tier-A
// topology-isomorphic input groups; cross-share-group decisions go
// through a CostModel-ranked `decide_op_node` that may insert a local
// `fabric.mux` when `SynthConfig.anchorAllowIntraPositionMux` is true.
class AnchorSynthesizer final : public Synthesizer {
public:
  explicit AnchorSynthesizer(const ::loom::SynthConfig &cfg);
  SynthResult run(const SynthInputs &) override;

private:
  const ::loom::SynthConfig &cfg;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
