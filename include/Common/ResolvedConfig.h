#ifndef LOOM_COMMON_RESOLVEDCONFIG_H
#define LOOM_COMMON_RESOLVEDCONFIG_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace loom {

struct ResolvedDseObjective {
  std::string objectiveId;
  double weight = 0.0;
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
  static constexpr ArtifactSchemaDescriptor artifactSchema{
      "loom.config.resolved", SchemaVersion{1, 0}};

  std::string configId = "loom.default";
  ResolvedGlobalConfig global;
  ResolvedDseConfig dse;
};

ResolvedConfig defaultResolvedConfig();

llvm::Expected<ResolvedConfig> loadResolvedConfig(llvm::StringRef path);
llvm::Expected<ResolvedConfig> parseResolvedConfig(llvm::StringRef body,
                                                   llvm::StringRef sourceName);

std::string canonicalResolvedConfigJson(const ResolvedConfig &config);
CanonicalSemanticBytes
canonicalResolvedConfigBytes(const ResolvedConfig &config);
ArtifactIdentity resolvedConfigIdentity(const ResolvedConfig &config);

} // namespace loom

#endif // LOOM_COMMON_RESOLVEDCONFIG_H
