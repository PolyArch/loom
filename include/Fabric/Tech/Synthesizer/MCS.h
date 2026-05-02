#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H

// MCS strategy: cost-prioritized configurable FU synthesis across the input
// group. The implementation generates lock-step candidates, exact graph-region
// MCES candidates, and bounded graph-region candidates. Legal candidates are
// accepted by the enumerator/matcher coverage roundtrip and ranked by
// `CostModel::evaluate`.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: mcs" and "Acceptance criteria (mcs)".
//
// Threading: candidate construction uses the caller-provided
// `SynthInputs.context`; exact graph-region search can shard independent
// choices through the synthesizer worker setting.

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
