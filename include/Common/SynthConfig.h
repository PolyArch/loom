#ifndef LOOM_COMMON_SYNTH_CONFIG_H
#define LOOM_COMMON_SYNTH_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

// Configuration for the loom-generalize-subgraphs-to-fu pass.
//
// Loaded from a YAML or TOML file via `--config <path>`; missing keys fall
// back to the defaults below. The schema mirrors the YAML reference in
// `docs/spec-generalize-subgraphs-to-fu.md` (section "SynthConfig schema").
//
// All strategies and downstream code read the populated struct once per pass
// invocation; the file is never re-inspected after parsing.
struct SynthConfig {
  // Which strategy to invoke for each share group.
  // Valid values: "anchor", "mcs", "incremental", "incremental_random".
  std::string strategy = "incremental_random";

  // Parallelism knobs.
  // `parallelism.cross_group` parallelizes across loom.synth_group values.
  // `parallelism.workers` is the worker thread count; 0 means
  // std::thread::hardware_concurrency() (also written `auto` in YAML).
  bool parallelismCrossGroup = true;
  unsigned parallelismWorkers = 0;

  // Coverage-verifier knobs.
  // `coverage_verifier.enabled` toggles back-coverage verification of every
  // synthesized FU against its input subgraphs. `parallel_match` parallelizes
  // the per-input matching loop inside the verifier.
  bool coverageVerifierEnabled = true;
  bool coverageVerifierParallelMatch = true;

  // Optional ordered list of strategies to retry on failure. Empty means no
  // fallback (the primary strategy's failure is reported as-is).
  std::vector<std::string> fallbackChain;

  // Cost-model weights. See `lib/Fabric/Tech/CostModel.cpp` for usage.
  double costMuxPenalty = 1.5;
  double costDemuxPenalty = 1.5;
  double costCarryPenalty = 2.0;

  // Anchor (tier A) options.
  bool anchorAllowIntraPositionMux = false;

  // Incremental (deterministic, tier B) options.
  std::string incrementalInputOrderHeuristic = "largest_first";
  bool incrementalCoverageVerifyEachAttempt = true;

  // Incremental-random (tier B) options.
  unsigned incrementalRandomRestarts = 16;
  uint64_t incrementalRandomSeed = 42;
  std::string incrementalRandomInputOrderHeuristic = "random_seeded";

  // MCS (maximum-common-subgraph) options.
  unsigned mcsTimeoutSec = 60;
  unsigned mcsBranchWorkers = 8;
  unsigned mcsCandidateCap = 1000000;

  // Tier-C / tier-B feature flags.
  bool sccFullUnroll = false;
  bool subgraphShareRecurse = false;

  bool operator==(const SynthConfig &o) const {
    return strategy == o.strategy &&
           parallelismCrossGroup == o.parallelismCrossGroup &&
           parallelismWorkers == o.parallelismWorkers &&
           coverageVerifierEnabled == o.coverageVerifierEnabled &&
           coverageVerifierParallelMatch == o.coverageVerifierParallelMatch &&
           fallbackChain == o.fallbackChain &&
           costMuxPenalty == o.costMuxPenalty &&
           costDemuxPenalty == o.costDemuxPenalty &&
           costCarryPenalty == o.costCarryPenalty &&
           anchorAllowIntraPositionMux == o.anchorAllowIntraPositionMux &&
           incrementalInputOrderHeuristic ==
               o.incrementalInputOrderHeuristic &&
           incrementalCoverageVerifyEachAttempt ==
               o.incrementalCoverageVerifyEachAttempt &&
           incrementalRandomRestarts == o.incrementalRandomRestarts &&
           incrementalRandomSeed == o.incrementalRandomSeed &&
           incrementalRandomInputOrderHeuristic ==
               o.incrementalRandomInputOrderHeuristic &&
           mcsTimeoutSec == o.mcsTimeoutSec &&
           mcsBranchWorkers == o.mcsBranchWorkers &&
           mcsCandidateCap == o.mcsCandidateCap &&
           sccFullUnroll == o.sccFullUnroll &&
           subgraphShareRecurse == o.subgraphShareRecurse;
  }
  bool operator!=(const SynthConfig &o) const { return !(*this == o); }
};

// File format auto-detected from extension (.yaml/.yml -> YAML; .toml -> TOML).
// Returns the populated config or an error explaining why parsing failed.
::llvm::Expected<SynthConfig> loadSynthConfig(::llvm::StringRef path);

// Parse YAML/TOML body directly. Used by tests and by the file loader.
::llvm::Expected<SynthConfig> parseSynthConfigYAML(::llvm::StringRef body);
::llvm::Expected<SynthConfig> parseSynthConfigTOML(::llvm::StringRef body);

} // namespace loom

#endif // LOOM_COMMON_SYNTH_CONFIG_H
