#ifndef LOOM_COMMON_RESOLVEDCONFIG_H
#define LOOM_COMMON_RESOLVEDCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

struct ResolvedDseObjective {
  std::string objectiveId;
  double weight = 0.0;
};

enum class FabricTechMapAlgorithm : std::uint8_t {
  Greedy,
  List,
  Beam,
  SimulatedAnnealing,
  ILP,
};

struct ResolvedFabricTechMapConfig {
  double alpha = 1.0;
  double beta = 1.0;
  double gamma = 0.5;
  FabricTechMapAlgorithm algorithm = FabricTechMapAlgorithm::Greedy;
  unsigned beamWidth = 4;
  unsigned saSteps = 1000;
  std::uint64_t saSeed = 0xC0DEull;
  unsigned threads = 0;
};

struct ResolvedGlobalConfig {
  unsigned addrBits = 48;
  unsigned indexWidth = 32;
  unsigned memBusWidth = 32768;
};

struct ResolvedDseConfig {
  std::string rankingPolicy = "weighted_sum";
  std::vector<ResolvedDseObjective> objectives;
};

struct ResolvedConfig {
  std::string configId = "loom.default";
  ResolvedGlobalConfig global;
  ResolvedFabricTechMapConfig fabricTechMap;
  ResolvedDseConfig dse;
};

ResolvedConfig defaultResolvedConfig();

llvm::Expected<ResolvedConfig> loadResolvedConfig(llvm::StringRef path);
llvm::Expected<ResolvedConfig> parseResolvedConfig(llvm::StringRef body,
                                                   llvm::StringRef sourceName);

std::string canonicalResolvedConfigJson(const ResolvedConfig &config);
std::string resolvedConfigFingerprint(const ResolvedConfig &config);

std::string canonicalComponentConfigViewJson(const ResolvedConfig &config,
                                             llvm::StringRef viewId);
std::string componentConfigFingerprint(const ResolvedConfig &config,
                                       llvm::StringRef viewId);

bool isResolvedConfigFingerprint(llvm::StringRef fingerprint);

} // namespace loom

#endif // LOOM_COMMON_RESOLVEDCONFIG_H
