#ifndef LOOM_COMMON_RESOLVEDCONFIG_H
#define LOOM_COMMON_RESOLVEDCONFIG_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "Common/ResolvedPnrPolicy.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

struct ResolvedHardwareTargetConfig final {
  std::string templateIdentity;
  SchemaVersion schemaVersion;
  adg::BuiltinTargetScale parameters;
};

struct ResolvedStructuredOwnershipConfig {
  std::uint32_t scopeExpansionLimit = 64;
};

struct ResolvedTechMappingConfig {
  std::uint64_t matchRowAttemptLimit = 65536;
  std::uint64_t partialCoverExpansionLimit = 262144;
  std::uint64_t candidatePublicationLimit = 16;
};

struct ResolvedDseConfig {
  ResolvedStructuredOwnershipConfig structuredOwnership;
  ResolvedTechMappingConfig techMapping;
  ResolvedObjectiveCatalogs objectiveCatalogs;
  ResolvedPnrPolicyConfig spatialPnr;
  ResolvedPnrPolicyConfig systemPnr;
};

struct ResolvedConfig {
  static constexpr ArtifactSchemaDescriptor artifactSchema{
      "loom.config.resolved", SchemaVersion{3, 0}};

  ResolvedHardwareTargetConfig hardwareTarget;
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
