#ifndef LOOM_COMMON_SYNTH_CONFIG_H
#define LOOM_COMMON_SYNTH_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom {

// Configuration for the loom-generalize-subgraphs-to-fu pass.
//
// Loaded from a YAML or TOML file via `--config <path>`; missing keys fall
// back to the defaults below. The schema mirrors the YAML reference in
// `docs/spec-generalize-subgraphs-to-fu.md` (section "SynthConfig schema").
//
// The synthesizer and downstream code read the populated struct once per pass
// invocation; the file is never re-inspected after parsing.
struct SynthConfig {
  // Anchor is the only externally selectable canonical strategy.
  std::string strategy = "anchor";

  // Parallelism knobs.
  // `parallelism.cross_group` parallelizes across loom.synth_group values.
  // `parallelism.workers` is the worker thread count; 0 means
  // std::thread::hardware_concurrency() (also written `auto` in YAML).
  bool parallelismCrossGroup = true;
  unsigned parallelismWorkers = 0;

  // Coverage-verifier knobs.
  // `parallel_match` parallelizes the per-input matching loop inside the
  // verifier.
  bool coverageVerifierParallelMatch = true;

  // Cost-model weights. See `lib/Fabric/Tech/CostModel.cpp` for usage.
  double costMuxPenalty = 1.5;
  double costDemuxPenalty = 1.5;
  double costCarryPenalty = 2.0;

  // Anchor options.
  bool anchorAllowIntraPositionMux = false;

  bool operator==(const SynthConfig &o) const {
    return strategy == o.strategy &&
           parallelismCrossGroup == o.parallelismCrossGroup &&
           parallelismWorkers == o.parallelismWorkers &&
           coverageVerifierParallelMatch == o.coverageVerifierParallelMatch &&
           costMuxPenalty == o.costMuxPenalty &&
           costDemuxPenalty == o.costDemuxPenalty &&
           costCarryPenalty == o.costCarryPenalty &&
           anchorAllowIntraPositionMux == o.anchorAllowIntraPositionMux;
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
