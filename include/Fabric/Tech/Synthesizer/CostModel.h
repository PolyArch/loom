#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_COSTMODEL_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_COSTMODEL_H

// Analytic hardware-area cost model for `fabric.fu` synthesis candidates.
//
// `CostModel::evaluate(fu)` is a pure function over a built `fabric.fu`: no
// MLIR mutation, no I/O, no PRNG, no logging. It is the only ranking metric
// shared across all synthesis strategies, MCS branch candidates, and the
// regression metric in perf tests; thread-safe and called inline because
// each evaluation is cheap.
//
// Formula (verbatim from `docs/spec-generalize-subgraphs-to-fu.md`):
//
//   cost(fu) = sum baseArea(shareGroup, bw)               (per non-carry op)
//            + sum carry_penalty * bw                     (per carry op)
//            + sum mux_penalty   * portCount * bw         (per fabric.mux)
//            + sum demux_penalty * portCount * bw         (per fabric.demux)
//
// `baseArea(group, bw) = baseUnit[group] * (bw / 32.0)`. The `baseUnit`
// table lives as a function-local static in `CostModel.cpp`; ops outside
// the table fall back to the singleton baseUnit `1.0`.

#include "Common/SynthConfig.h"
#include "Fabric/IR/FabricOps.h"

#include <cstddef>
#include <optional>

namespace loom::fabric::tech {

// Plain-old-data weight bundle: copied out of `SynthConfig` at construction
// so `evaluate` is independent of any heap memory the config owns.
struct AreaWeights {
  double muxPenalty   = 1.5;
  double demuxPenalty = 1.5;
  double carryPenalty = 2.0;
};

class CostModel {
public:
  // Constructed from a SynthConfig: pulls cost.mux_penalty etc.
  explicit CostModel(const ::loom::SynthConfig &config);

  // Evaluate the analytic hardware area of a fully built fabric.fu.
  // Pure function: no side effects; thread-safe; same inputs always
  // return identical doubles.
  double evaluate(::fabric::FuOp fu) const;

private:
  AreaWeights weights;
};

// Free helper exposed for tests (and reusable by Hwsg.h later).
// baseUnit table is keyed on the share-group index returned by
// `loom::common::findShareGroup`, with a singleton fallback of 1.0.
double baseUnitFor(::std::optional<::std::size_t> shareGroupIndex);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_COSTMODEL_H
