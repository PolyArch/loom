#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCS_H

// MCS strategy: cost-prioritized branch-and-bound enumeration of
// candidate FUs across the input group. Each branch produces a candidate
// FU by delegating to `IncrementalSynthesizer` over a distinct seed
// ordering of the input subgraphs; the best-cost legal candidate wins.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: mcs" and "Acceptance criteria (mcs)".
//
// Pragmatic implementation note: the spec's reference algorithm is a
// MCES enumeration (NP-hard) anchored at yield positions and grown by
// share-group/width-compatible alignment. The current implementation
// uses the spec-sanctioned fallback ("delegate to incremental for K
// canonical orderings") which gives correct, cost-ranked candidates and
// honours the same termination knobs (`mcs.timeout_sec`,
// `mcs.candidate_cap`, `mcs.branch_workers`). Each branch is one
// candidate; branches run in parallel across `branch_workers`. Candidate
// generation deterministically enumerates anchor-rooted orderings plus
// the seeded random restarts that `IncrementalRandom` already uses, so
// the MCS branch space is a strict superset of the random-restart
// space.
//
// Threading: identical handoff pattern to `IncrementalRandom`. Each
// branch runs in its own sub-`MLIRContext`; the winning wrapper is
// serialized to text and reparsed into `SynthInputs.context` before
// being returned.

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
