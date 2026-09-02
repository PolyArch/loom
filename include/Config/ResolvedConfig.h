#ifndef LOOM_CONFIG_RESOLVEDCONFIG_H
#define LOOM_CONFIG_RESOLVEDCONFIG_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "Common/ResolvedPnrPolicy.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/Models/OpenRoadStaticFpaConfig.h"
#include "Evaluation/Models/PhysicalRailAnalysisConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
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

struct ResolvedStructuredScheduleConfig {
  std::uint32_t scopeExpansionLimit = 64;
};

struct ResolvedMemoryCommunicationConfig {
  std::uint32_t scopeExpansionLimit = 64;
};

struct ResolvedDataflowRewriteConfig {
  std::uint32_t scopeExpansionLimit = 256;
};

struct ResolvedTechMappingConfig {
  std::uint64_t matchRowAttemptLimit = 65536;
  std::uint64_t partialCoverExpansionLimit = 262144;
  std::uint64_t candidateEvaluationLimit = 128;
  std::uint64_t candidatePublicationLimit = 8;
};

struct ResolvedDseConfig {
  ResolvedStructuredOwnershipConfig structuredOwnership;
  ResolvedStructuredScheduleConfig schedule;
  ResolvedMemoryCommunicationConfig memoryCommunication;
  ResolvedDataflowRewriteConfig dataflowRewrite;
  ResolvedTechMappingConfig techMapping;
  std::vector<dse::ModelAuthorization> modelAuthorizations;
  std::vector<dse::EvidenceObligationTemplate> evidenceObligationTemplates;
  ResolvedObjectiveCatalogs objectiveCatalogs;
  std::vector<dse::QualityGatePolicy> qualityGatePolicies;
  std::vector<dse::DsePlanNodeDefinition> planNodes;
  ResolvedPnrPolicyConfig spatialPnr;
  ResolvedPnrPolicyConfig systemPnr;
};

struct ResolvedEvaluationConfig final {
  std::optional<evaluation::models::CadenceVoltusStaticRailProviderBinding>
      cadenceVoltusStaticRail;
  std::optional<evaluation::models::MappedRtlSimulatorBinding>
      mappedRtlSimulator;
  std::optional<evaluation::models::OpenRoadStaticFpaProviderBinding>
      openRoadStaticFpa;
};

struct ResolvedConfig {
  static constexpr ArtifactSchemaDescriptor artifactSchema{
      "loom.config.resolved", SchemaVersion{11, 3}};

  ResolvedHardwareTargetConfig hardwareTarget;
  ResolvedDseConfig dse;
  ResolvedEvaluationConfig evaluation;
};

/// The resolved value together with the source-level ownership of the PnR
/// policies.  A configuration file may be a partial hardware overlay; in
/// that case omitted PnR policies remain owned by the product profile rather
/// than becoming an accidental exhaustive-search request through defaults.
struct ResolvedConfigProfile final {
  ResolvedConfig config;
  bool spatialPnrAuthored = false;
  bool systemPnrAuthored = false;
};

ResolvedConfig defaultResolvedConfig();
llvm::Expected<ResolvedConfig>
resolveConfigProfile(llvm::StringRef builtinPresetOrConfigPath);
llvm::Expected<ResolvedConfigProfile>
resolveConfigProfileWithProvenance(llvm::StringRef builtinPresetOrConfigPath);
/// True when the spelling names a builtin profile preset (or is empty, which
/// selects the default preset) rather than an explicit configuration file. An
/// explicit file remains the single policy owner for everything it states.
bool isBuiltinConfigProfile(llvm::StringRef builtinPresetOrConfigPath);

llvm::Expected<ResolvedConfig> loadResolvedConfig(llvm::StringRef path);
llvm::Expected<ResolvedConfig> parseResolvedConfig(llvm::StringRef body,
                                                   llvm::StringRef sourceName);

std::string canonicalResolvedConfigJson(const ResolvedConfig &config);
CanonicalSemanticBytes
canonicalResolvedConfigBytes(const ResolvedConfig &config);
ArtifactIdentity resolvedConfigIdentity(const ResolvedConfig &config);

} // namespace loom

#endif // LOOM_CONFIG_RESOLVEDCONFIG_H
