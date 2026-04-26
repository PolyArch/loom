#ifndef LOOM_COMMON_CONFIG_H
#define LOOM_COMMON_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom {

// Tech-mapping partitioner configuration.
//
// Loaded from a YAML or TOML file via `--config <path>`; missing keys fall
// back to the defaults below. The chosen algorithm and cost weights are read
// once per pass invocation; downstream code never inspects the file again.
struct TechMapConfig {
  // Cost: alpha * |blocks| + beta * cross_edges - gamma * avg_density.
  double alpha = 1.0;
  double beta = 1.0;
  double gamma = 0.5;

  // Algorithm name. Valid values: "greedy", "list", "beam", "sa", "ilp".
  std::string algorithm = "greedy";

  // Beam-search width when algorithm == "beam".
  unsigned beamWidth = 4;

  // Simulated-annealing parameters when algorithm == "sa".
  unsigned saSteps = 1000;
  uint64_t saSeed = 0xC0DEull;

  // Worker thread count for the candidate cache. 0 means
  // std::thread::hardware_concurrency().
  unsigned threads = 0;

  bool operator==(const TechMapConfig &o) const {
    return alpha == o.alpha && beta == o.beta && gamma == o.gamma &&
           algorithm == o.algorithm && beamWidth == o.beamWidth &&
           saSteps == o.saSteps && saSeed == o.saSeed && threads == o.threads;
  }
  bool operator!=(const TechMapConfig &o) const { return !(*this == o); }
};

// File format auto-detected from extension (.yaml/.yml -> YAML; .toml -> TOML).
// Returns the populated config or an error explaining why parsing failed.
::llvm::Expected<TechMapConfig> loadTechMapConfig(::llvm::StringRef path);

// Parse YAML/TOML body directly. Used by tests and by the file loader.
::llvm::Expected<TechMapConfig> parseTechMapConfigYAML(::llvm::StringRef body);
::llvm::Expected<TechMapConfig> parseTechMapConfigTOML(::llvm::StringRef body);

} // namespace loom

#endif // LOOM_COMMON_CONFIG_H
