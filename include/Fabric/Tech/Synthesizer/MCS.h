#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H

// MCS strategy: cost-prioritized maximum-common-edge-skeleton synthesis
// across the input group. The implementation generates lock-step and
// positional pure-DAG shared-prefix candidates, verifies coverage, and ranks
// legal candidates by `CostModel::evaluate`.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: mcs" and "Acceptance criteria (mcs)".
//
// Stateful graph-region inputs keep using the Tier-C-aware compatibility
// path in the implementation so existing feedback-alignment behavior
// remains covered while pure-DAG MCES handles divergent same-length DAGs.
//
// Threading: candidate construction uses the worker-local
// `SynthInputs.context`. Compatibility branches use sub-`MLIRContext`
// handoff and reparse the winning wrapper into `SynthInputs.context`.

#include "Common/SynthConfig.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

namespace loom::fabric::tech {

class MCSSynthesizer final : public Synthesizer {
public:
  explicit MCSSynthesizer(const ::loom::SynthConfig &cfg);
  SynthResult run(const SynthInputs &) override;

private:
  const ::loom::SynthConfig &cfg;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H
