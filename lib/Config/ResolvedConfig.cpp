#include "Config/ResolvedConfig.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <string>

using llvm::StringRef;

static std::optional<loom::ResolvedProfilePreset>
profilePresetForName(llvm::StringRef spelling);
static loom::ResolvedConfig
builtinResolvedConfig(loom::ResolvedProfilePreset preset);

namespace {

llvm::Error makeErr(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 msg.str().c_str());
}

llvm::Error diagnostic(llvm::StringRef code, const llvm::Twine &key,
                       const llvm::Twine &detail = {}) {
  const std::string keyString = key.str();
  const std::string detailString = detail.str();
  std::string message;
  llvm::raw_string_ostream os(message);
  os << code;
  if (!keyString.empty())
    os << ": " << keyString;
  if (!detailString.empty())
    os << ": " << detailString;
  return makeErr(os.str());
}

StringRef stripQuotes(StringRef value) {
  value = value.trim();
  if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
      value.front() == value.back())
    return value.drop_front().drop_back();
  return value;
}

bool yamlScalarWasQuoted(llvm::yaml::Node *node) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  const char *begin = scalar->getSourceRange().Start.getPointer();
  const char *end = scalar->getSourceRange().End.getPointer();
  if (!begin || !end)
    return false;
  while (begin < end && std::isspace(static_cast<unsigned char>(*begin)))
    ++begin;
  return begin < end && (*begin == '"' || *begin == '\'');
}

template <unsigned N>
StringRef scalarValue(llvm::yaml::Node *node, llvm::SmallString<N> &storage) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return {};
  return stripQuotes(scalar->getValue(storage));
}

struct ConfigSyntax final {
  enum class Kind { Null, Scalar, Mapping, Sequence };

  Kind kind = Kind::Null;
  std::string scalar;
  bool quoted = false;
  std::vector<std::pair<std::string, ConfigSyntax>> mapping;
  std::vector<ConfigSyntax> sequence;
};

llvm::Expected<ConfigSyntax> materializeSyntax(llvm::yaml::Node *node,
                                               const llvm::Twine &path) {
  if (llvm::isa<llvm::yaml::NullNode>(node))
    return ConfigSyntax{};
  if (auto *scalar = llvm::dyn_cast<llvm::yaml::ScalarNode>(node)) {
    llvm::SmallString<128> storage;
    const bool quoted = yamlScalarWasQuoted(node);
    const StringRef value = stripQuotes(scalar->getValue(storage));
    if (!quoted &&
        (value == "null" || value == "Null" || value == "NULL" || value == "~"))
      return ConfigSyntax{};
    ConfigSyntax result;
    result.kind = ConfigSyntax::Kind::Scalar;
    result.scalar = value.str();
    result.quoted = quoted;
    return result;
  }
  if (auto *sequence = llvm::dyn_cast<llvm::yaml::SequenceNode>(node)) {
    ConfigSyntax result;
    result.kind = ConfigSyntax::Kind::Sequence;
    std::uint64_t ordinal = 0;
    for (llvm::yaml::Node &entry : *sequence) {
      auto valueOrErr =
          materializeSyntax(&entry, path + "[" + llvm::Twine(ordinal) + "]");
      if (!valueOrErr)
        return valueOrErr.takeError();
      result.sequence.push_back(std::move(*valueOrErr));
      ++ordinal;
    }
    return result;
  }
  auto *mapping = llvm::dyn_cast<llvm::yaml::MappingNode>(node);
  if (!mapping)
    return diagnostic("config_type_mismatch", path, "unsupported YAML node");

  ConfigSyntax result;
  result.kind = ConfigSyntax::Kind::Mapping;
  llvm::StringSet<> seen;
  const std::string pathString = path.str();
  for (auto &kv : *mapping) {
    llvm::SmallString<64> keyStorage;
    StringRef key = scalarValue(kv.getKey(), keyStorage);
    if (key.empty())
      return diagnostic("config_type_mismatch", pathString,
                        "mapping key must be a scalar");
    const std::string fieldPath =
        pathString.empty() ? key.str()
                           : (llvm::Twine(pathString) + "." + key).str();
    if (!seen.insert(key).second)
      return diagnostic("config_duplicate_key", fieldPath);
    auto valueOrErr = materializeSyntax(kv.getValue(), fieldPath);
    if (!valueOrErr)
      return valueOrErr.takeError();
    result.mapping.emplace_back(key.str(), std::move(*valueOrErr));
  }
  return result;
}

llvm::Expected<std::string> requireScalarString(const ConfigSyntax *node,
                                                const llvm::Twine &key) {
  if (!node || node->kind != ConfigSyntax::Kind::Scalar || node->scalar.empty())
    return diagnostic("config_type_mismatch", key, "expected scalar string");
  return node->scalar;
}

llvm::Expected<unsigned> requireUnsigned(const ConfigSyntax *node,
                                         const llvm::Twine &key) {
  if (node && node->quoted)
    return diagnostic("config_type_mismatch", key,
                      "expected unsigned integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::uint64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value) || value == 0)
    return diagnostic("config_type_mismatch", key,
                      "expected positive unsigned integer");
  if (value > std::numeric_limits<unsigned>::max())
    return diagnostic("config_range_violation", key,
                      "unsigned integer exceeds supported range");
  return static_cast<unsigned>(value);
}

llvm::Expected<std::uint64_t> requireU64(const ConfigSyntax *node,
                                         const llvm::Twine &key) {
  if (node && node->quoted)
    return diagnostic("config_type_mismatch", key,
                      "expected unsigned integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::uint64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value))
    return diagnostic("config_type_mismatch", key, "expected unsigned integer");
  return value;
}

llvm::Expected<std::int64_t> requireI64(const ConfigSyntax *node,
                                        const llvm::Twine &key) {
  if (node && node->quoted)
    return diagnostic("config_type_mismatch", key,
                      "expected integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::int64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value))
    return diagnostic("config_type_mismatch", key, "expected integer");
  return value;
}

llvm::Expected<std::uint32_t> requireU32(const ConfigSyntax *node,
                                         const llvm::Twine &key) {
  auto valueOrErr = requireU64(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr > std::numeric_limits<std::uint32_t>::max())
    return diagnostic("config_range_violation", key,
                      "integer exceeds uint32 range");
  return static_cast<std::uint32_t>(*valueOrErr);
}

llvm::Expected<std::uint64_t> requirePositiveU64(const ConfigSyntax *node,
                                                 const llvm::Twine &key) {
  if (node && node->quoted)
    return diagnostic("config_type_mismatch", key,
                      "expected unsigned integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::uint64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value) || value == 0)
    return diagnostic("config_type_mismatch", key,
                      "expected positive unsigned integer");
  return value;
}

struct ConfigPatch {
  std::optional<loom::ResolvedProfilePreset> inheritedPreset;
  std::optional<loom::ResolvedHardwareTargetConfig> hardwareTarget;
  std::optional<
      loom::evaluation::models::CadenceVoltusStaticRailProviderBinding>
      cadenceVoltusStaticRail;
  std::optional<loom::evaluation::models::MappedRtlSimulatorBinding>
      mappedRtlSimulator;
  std::optional<loom::evaluation::models::OpenRoadStaticFpaProviderBinding>
      openRoadStaticFpa;
  std::optional<std::uint32_t> ownershipScopeExpansionLimit;
  std::optional<std::uint32_t> scheduleScopeExpansionLimit;
  std::optional<std::uint32_t> memoryCommunicationScopeExpansionLimit;
  std::optional<std::uint32_t> dataflowRewriteScopeExpansionLimit;
  std::optional<std::uint64_t> techMappingMatchRowAttemptLimit;
  std::optional<std::uint64_t> techMappingPartialCoverExpansionLimit;
  std::optional<std::uint64_t> techMappingCandidateEvaluationLimit;
  std::optional<std::uint64_t> techMappingCandidatePublicationLimit;
  std::optional<std::vector<loom::dse::ModelAuthorization>> modelAuthorizations;
  std::optional<std::vector<loom::dse::EvidenceObligationTemplate>>
      evidenceObligationTemplates;
  std::optional<loom::ResolvedObjectiveCatalogs> objectiveCatalogs;
  std::optional<std::vector<loom::dse::QualityGatePolicy>> qualityGatePolicies;
  std::optional<std::vector<loom::dse::DsePlanNodeDefinition>> planNodes;
  std::optional<loom::ResolvedPnrPolicyConfig> spatialPnr;
  std::optional<loom::ResolvedPnrPolicyConfig> systemPnr;
  std::set<std::string> touchedKeys;
};

class ClosedMapping final {
public:
  const ConfigSyntax *at(llvm::StringRef key) const {
    auto found = fields_.find(key);
    return found == fields_.end() ? nullptr : found->second;
  }

  static llvm::Expected<ClosedMapping>
  parse(const ConfigSyntax *node, const llvm::Twine &prefix,
        std::initializer_list<llvm::StringRef> required,
        std::initializer_list<llvm::StringRef> optional = {}) {
    const std::string prefixString = prefix.str();
    if (!node || node->kind != ConfigSyntax::Kind::Mapping)
      return diagnostic("config_type_mismatch", prefixString,
                        "expected mapping");

    ClosedMapping parsed;
    llvm::StringSet<> allowed;
    for (llvm::StringRef field : required)
      allowed.insert(field);
    for (llvm::StringRef field : optional)
      allowed.insert(field);

    for (const auto &[keyStorage, value] : node->mapping) {
      StringRef key(keyStorage);
      const std::string fieldKey =
          (llvm::Twine(prefixString) + "." + key).str();
      if (!allowed.count(key))
        return diagnostic("config_unknown_key", fieldKey);
      if (!parsed.fields_.try_emplace(key, &value).second)
        return diagnostic("config_duplicate_key", fieldKey);
    }
    for (llvm::StringRef field : required)
      if (!parsed.fields_.count(field))
        return diagnostic("config_missing_required_profile", prefixString,
                          field);
    return parsed;
  }

private:
  llvm::StringMap<const ConfigSyntax *> fields_;
};

llvm::Expected<const std::vector<ConfigSyntax> *>
requireSequence(const ConfigSyntax *node, const llvm::Twine &key) {
  if (!node || node->kind != ConfigSyntax::Kind::Sequence)
    return diagnostic("config_type_mismatch", key, "expected sequence");
  return &node->sequence;
}

llvm::Error touch(ConfigPatch &patch, llvm::StringRef canonicalKey) {
  patch.touchedKeys.insert(canonicalKey.str());
  return llvm::Error::success();
}

void applyPatch(loom::ResolvedConfig &config, const ConfigPatch &patch) {
  if (patch.hardwareTarget)
    config.hardwareTarget = *patch.hardwareTarget;
  if (patch.cadenceVoltusStaticRail)
    config.evaluation.cadenceVoltusStaticRail = *patch.cadenceVoltusStaticRail;
  if (patch.mappedRtlSimulator)
    config.evaluation.mappedRtlSimulator = *patch.mappedRtlSimulator;
  if (patch.openRoadStaticFpa)
    config.evaluation.openRoadStaticFpa = *patch.openRoadStaticFpa;
  if (patch.ownershipScopeExpansionLimit)
    config.dse.structuredOwnership.scopeExpansionLimit =
        *patch.ownershipScopeExpansionLimit;
  if (patch.scheduleScopeExpansionLimit)
    config.dse.schedule.scopeExpansionLimit =
        *patch.scheduleScopeExpansionLimit;
  if (patch.memoryCommunicationScopeExpansionLimit)
    config.dse.memoryCommunication.scopeExpansionLimit =
        *patch.memoryCommunicationScopeExpansionLimit;
  if (patch.dataflowRewriteScopeExpansionLimit)
    config.dse.dataflowRewrite.scopeExpansionLimit =
        *patch.dataflowRewriteScopeExpansionLimit;
  if (patch.techMappingMatchRowAttemptLimit)
    config.dse.techMapping.matchRowAttemptLimit =
        *patch.techMappingMatchRowAttemptLimit;
  if (patch.techMappingPartialCoverExpansionLimit)
    config.dse.techMapping.partialCoverExpansionLimit =
        *patch.techMappingPartialCoverExpansionLimit;
  if (patch.techMappingCandidateEvaluationLimit)
    config.dse.techMapping.candidateEvaluationLimit =
        *patch.techMappingCandidateEvaluationLimit;
  if (patch.techMappingCandidatePublicationLimit)
    config.dse.techMapping.candidatePublicationLimit =
        *patch.techMappingCandidatePublicationLimit;
  if (patch.modelAuthorizations)
    config.dse.modelAuthorizations = *patch.modelAuthorizations;
  if (patch.evidenceObligationTemplates)
    config.dse.evidenceObligationTemplates = *patch.evidenceObligationTemplates;
  if (patch.objectiveCatalogs)
    config.dse.objectiveCatalogs = *patch.objectiveCatalogs;
  if (patch.qualityGatePolicies)
    config.dse.qualityGatePolicies = *patch.qualityGatePolicies;
  if (patch.planNodes)
    config.dse.planNodes = *patch.planNodes;
  if (patch.spatialPnr)
    config.dse.spatialPnr = *patch.spatialPnr;
  if (patch.systemPnr)
    config.dse.systemPnr = *patch.systemPnr;
}

llvm::Expected<loom::ResolvedExactRatio>
parseExactRatio(const ConfigSyntax *node, const llvm::Twine &key) {
  auto fieldsOrErr =
      ClosedMapping::parse(node, key, {"numerator", "denominator"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto numeratorOrErr =
      requireU64(fieldsOrErr->at("numerator"), key + ".numerator");
  if (!numeratorOrErr)
    return numeratorOrErr.takeError();
  auto denominatorOrErr =
      requireU64(fieldsOrErr->at("denominator"), key + ".denominator");
  if (!denominatorOrErr)
    return denominatorOrErr.takeError();
  return loom::ResolvedExactRatio{*numeratorOrErr, *denominatorOrErr};
}

llvm::Expected<loom::adg::BuiltinFuOccurrenceCounts>
parseBuiltinFuOccurrences(const ConfigSyntax *node, llvm::StringRef key) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, key,
      {"dedicated_scalar_add", "mac", "vector_compute", "loop_control",
       "token_control", "vector_adapter", "vector_structural", "special_math"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  const auto read = [&](llvm::StringRef field) {
    return requireU32(fieldsOrErr->at(field), llvm::Twine(key) + "." + field);
  };
  auto dedicatedScalarAdd = read("dedicated_scalar_add");
  auto mac = read("mac");
  auto vectorCompute = read("vector_compute");
  auto loopControl = read("loop_control");
  auto tokenControl = read("token_control");
  auto vectorAdapter = read("vector_adapter");
  auto vectorStructural = read("vector_structural");
  auto specialMath = read("special_math");
  if (!dedicatedScalarAdd)
    return dedicatedScalarAdd.takeError();
  if (!mac)
    return mac.takeError();
  if (!vectorCompute)
    return vectorCompute.takeError();
  if (!loopControl)
    return loopControl.takeError();
  if (!tokenControl)
    return tokenControl.takeError();
  if (!vectorAdapter)
    return vectorAdapter.takeError();
  if (!vectorStructural)
    return vectorStructural.takeError();
  if (!specialMath)
    return specialMath.takeError();
  return loom::adg::BuiltinFuOccurrenceCounts{
      *dedicatedScalarAdd, *mac,           *vectorCompute,    *loopControl,
      *tokenControl,       *vectorAdapter, *vectorStructural, *specialMath};
}

llvm::Expected<loom::ResolvedHardwareTargetConfig>
parseHardwareTarget(const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, "hardware_target",
      {"template_identity", "schema_major", "schema_minor", "parameters"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto templateOrErr = requireScalarString(fieldsOrErr->at("template_identity"),
                                           "hardware_target.template_identity");
  if (!templateOrErr)
    return templateOrErr.takeError();
  auto majorOrErr = requireU32(fieldsOrErr->at("schema_major"),
                               "hardware_target.schema_major");
  if (!majorOrErr)
    return majorOrErr.takeError();
  auto minorOrErr = requireU32(fieldsOrErr->at("schema_minor"),
                               "hardware_target.schema_minor");
  if (!minorOrErr)
    return minorOrErr.takeError();

  auto parametersOrErr = ClosedMapping::parse(
      fieldsOrErr->at("parameters"), "hardware_target.parameters",
      {"acc_core_count", "mesh_dimension", "spatial_mesh_lanes_per_direction",
       "temporal_mesh_lanes_per_direction", "spatial_pe_count",
       "temporal_pe_count", "spatial_fu_occurrences", "temporal_fu_occurrences",
       "spatial_memory_count", "temporal_memory_count",
       "temporal_resident_contexts", "local_memory_port_variant",
       "cross_schedule_boundary_lanes_per_temporal_pe", "gateway_count",
       "memory_capacity_bytes"});
  if (!parametersOrErr)
    return parametersOrErr.takeError();

  auto positiveU32 =
      [&](llvm::StringRef field) -> llvm::Expected<std::uint32_t> {
    auto valueOrErr = requireUnsigned(parametersOrErr->at(field),
                                      "hardware_target.parameters." + field);
    if (!valueOrErr)
      return valueOrErr.takeError();
    return static_cast<std::uint32_t>(*valueOrErr);
  };
  auto accCores = positiveU32("acc_core_count");
  auto meshDimension = positiveU32("mesh_dimension");
  auto spatialMeshLanes = positiveU32("spatial_mesh_lanes_per_direction");
  auto temporalMeshLanes = positiveU32("temporal_mesh_lanes_per_direction");
  auto spatialPes = positiveU32("spatial_pe_count");
  auto temporalPes = positiveU32("temporal_pe_count");
  auto spatialFuOccurrences = parseBuiltinFuOccurrences(
      parametersOrErr->at("spatial_fu_occurrences"),
      "hardware_target.parameters.spatial_fu_occurrences");
  auto temporalFuOccurrences = parseBuiltinFuOccurrences(
      parametersOrErr->at("temporal_fu_occurrences"),
      "hardware_target.parameters.temporal_fu_occurrences");
  auto spatialMemories = positiveU32("spatial_memory_count");
  auto temporalMemories = positiveU32("temporal_memory_count");
  auto residentContexts = positiveU32("temporal_resident_contexts");
  auto memoryPortVariantSpelling = requireScalarString(
      parametersOrErr->at("local_memory_port_variant"),
      "hardware_target.parameters.local_memory_port_variant");
  if (!memoryPortVariantSpelling)
    return memoryPortVariantSpelling.takeError();
  auto memoryPortVariant =
      loom::adg::parseLocalMemoryPortVariant(*memoryPortVariantSpelling);
  if (!memoryPortVariant)
    return diagnostic("config_unknown_enum",
                      "hardware_target.parameters.local_memory_port_variant",
                      *memoryPortVariantSpelling);
  auto crossScheduleBoundaryLanes =
      positiveU32("cross_schedule_boundary_lanes_per_temporal_pe");
  auto gateways = positiveU32("gateway_count");
  auto memoryCapacity =
      requirePositiveU64(parametersOrErr->at("memory_capacity_bytes"),
                         "hardware_target.parameters.memory_capacity_bytes");
  if (!accCores)
    return accCores.takeError();
  if (!meshDimension)
    return meshDimension.takeError();
  if (!spatialMeshLanes)
    return spatialMeshLanes.takeError();
  if (!temporalMeshLanes)
    return temporalMeshLanes.takeError();
  if (!spatialPes)
    return spatialPes.takeError();
  if (!temporalPes)
    return temporalPes.takeError();
  if (!spatialFuOccurrences)
    return spatialFuOccurrences.takeError();
  if (!temporalFuOccurrences)
    return temporalFuOccurrences.takeError();
  if (!spatialMemories)
    return spatialMemories.takeError();
  if (!temporalMemories)
    return temporalMemories.takeError();
  if (!residentContexts)
    return residentContexts.takeError();
  if (!crossScheduleBoundaryLanes)
    return crossScheduleBoundaryLanes.takeError();
  if (!gateways)
    return gateways.takeError();
  if (!memoryCapacity)
    return memoryCapacity.takeError();

  return loom::ResolvedHardwareTargetConfig{
      *templateOrErr,
      {*majorOrErr, *minorOrErr},
      {*accCores, *meshDimension, *spatialMeshLanes, *temporalMeshLanes,
       *spatialPes, *temporalPes, *spatialFuOccurrences, *temporalFuOccurrences,
       *spatialMemories, *temporalMemories, *residentContexts,
       *memoryPortVariant, *crossScheduleBoundaryLanes, *gateways,
       *memoryCapacity}};
}

enum class ParsedObjectiveSourceKind {
  MappingViolation,
  MappingMeasure,
  EvaluationMetric,
};

llvm::Expected<ParsedObjectiveSourceKind>
parseObjectiveSourceKind(const ConfigSyntax *node, const llvm::Twine &key) {
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr == "mapping_violation")
    return ParsedObjectiveSourceKind::MappingViolation;
  if (*valueOrErr == "mapping_measure")
    return ParsedObjectiveSourceKind::MappingMeasure;
  if (*valueOrErr == "evaluation_metric")
    return ParsedObjectiveSourceKind::EvaluationMetric;
  return diagnostic("config_unknown_enum", key, *valueOrErr);
}

llvm::Expected<loom::ResolvedObjectiveScalar>
parseObjectiveScalar(const ConfigSyntax *node, const llvm::Twine &key) {
  if (!node)
    return diagnostic("config_missing_required_profile", key);
  if (node->kind == ConfigSyntax::Kind::Mapping) {
    auto fields =
        ClosedMapping::parse(node, key, {"coefficient", "base10_exponent"});
    if (!fields)
      return fields.takeError();
    auto coefficient =
        requireI64(fields->at("coefficient"), key + ".coefficient");
    auto exponent =
        requireI64(fields->at("base10_exponent"), key + ".base10_exponent");
    if (!coefficient)
      return coefficient.takeError();
    if (!exponent)
      return exponent.takeError();
    return loom::resolvedObjectiveDecimal(*coefficient, *exponent);
  }
  if (node->quoted || node->kind != ConfigSyntax::Kind::Scalar)
    return diagnostic("config_type_mismatch", key,
                      "expected integer or DecimalValue");
  const StringRef spelling(node->scalar);
  if (spelling.starts_with("-")) {
    auto value = requireI64(node, key);
    if (!value)
      return value.takeError();
    const std::uint64_t magnitude =
        *value < 0 ? static_cast<std::uint64_t>(-(*value + 1)) + 1
                   : static_cast<std::uint64_t>(*value);
    return loom::resolvedObjectiveInteger(magnitude, *value < 0);
  }
  auto value = requireU64(node, key);
  if (!value)
    return value.takeError();
  return loom::resolvedObjectiveInteger(*value);
}

llvm::Expected<loom::ResolvedObjectiveDirection>
parseObjectiveDirection(const ConfigSyntax *node, const llvm::Twine &key) {
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr == "minimize")
    return loom::ResolvedObjectiveDirection::Minimize;
  if (*valueOrErr == "maximize")
    return loom::ResolvedObjectiveDirection::Maximize;
  return diagnostic("config_unknown_enum", key, *valueOrErr);
}

struct ParsedDsePolicyCatalogs final {
  std::vector<loom::dse::ModelAuthorization> modelAuthorizations;
  std::vector<loom::dse::EvidenceObligationTemplate>
      evidenceObligationTemplates;
  loom::ResolvedObjectiveCatalogs objectives;
  std::vector<loom::dse::QualityGatePolicy> qualityGatePolicies;
  std::vector<loom::dse::DsePlanNodeDefinition> planNodes;
};

template <typename T, typename Adopter>
llvm::Expected<std::vector<T>>
parseCanonicalRecordSequence(const ConfigSyntax *node, const llvm::Twine &key,
                             Adopter adopter) {
  auto entriesOrErr = requireSequence(node, key);
  if (!entriesOrErr)
    return entriesOrErr.takeError();
  std::vector<T> records;
  records.reserve((*entriesOrErr)->size());
  std::uint64_t ordinal = 0;
  for (const ConfigSyntax &entry : **entriesOrErr) {
    const std::string entryKey = (key + "[" + llvm::Twine(ordinal) + "]").str();
    auto spelling = requireScalarString(&entry, entryKey);
    if (!spelling)
      return spelling.takeError();
    auto bytes = loom::parseArtifactLocalPayloadHex(*spelling);
    if (!bytes)
      return diagnostic("config_type_mismatch", entryKey,
                        llvm::toString(bytes.takeError()));
    auto record = adopter(*bytes);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
    ++ordinal;
  }
  return records;
}

llvm::Expected<ParsedDsePolicyCatalogs>
parseObjectiveCatalogs(const ConfigSyntax *node) {
  constexpr llvm::StringLiteral prefix =
      "dse.evaluation_and_objective_catalogs";
  auto fieldsOrErr = ClosedMapping::parse(
      node, prefix,
      {"model_authorizations", "evidence_obligation_templates",
       "objective_dimensions", "weighted_levels", "total_orderings",
       "quality_gate_policies", "resolved_plan_nodes"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();

  ParsedDsePolicyCatalogs parsed;
  auto authorizations = requireSequence(fieldsOrErr->at("model_authorizations"),
                                        prefix + ".model_authorizations");
  if (!authorizations)
    return authorizations.takeError();
  std::uint64_t authorizationOrdinal = 0;
  for (const ConfigSyntax &entry : **authorizations) {
    const std::string entryKey = (prefix + ".model_authorizations[" +
                                  llvm::Twine(authorizationOrdinal) + "]")
                                     .str();
    auto authorization = ClosedMapping::parse(
        &entry, entryKey, {"schema_major", "schema_minor", "model_kind"});
    if (!authorization)
      return authorization.takeError();
    auto major = requireU32(authorization->at("schema_major"),
                            entryKey + ".schema_major");
    auto minor = requireU32(authorization->at("schema_minor"),
                            entryKey + ".schema_minor");
    auto kind =
        requireU32(authorization->at("model_kind"), entryKey + ".model_kind");
    if (!major)
      return major.takeError();
    if (!minor)
      return minor.takeError();
    if (!kind)
      return kind.takeError();
    auto descriptor = loom::evaluation::EvaluationModelDescriptorRef::get(
        {*major, *minor}, loom::evaluation::EvaluationModelKind(*kind));
    if (!descriptor)
      return descriptor.takeError();
    parsed.modelAuthorizations.push_back({*descriptor});
    ++authorizationOrdinal;
  }

  auto templates =
      parseCanonicalRecordSequence<loom::dse::EvidenceObligationTemplate>(
          fieldsOrErr->at("evidence_obligation_templates"),
          prefix + ".evidence_obligation_templates",
          loom::dse::adoptEvidenceObligationTemplate);
  if (!templates)
    return templates.takeError();
  parsed.evidenceObligationTemplates = std::move(*templates);

  auto gates = parseCanonicalRecordSequence<loom::dse::QualityGatePolicy>(
      fieldsOrErr->at("quality_gate_policies"),
      prefix + ".quality_gate_policies", loom::dse::adoptQualityGatePolicy);
  if (!gates)
    return gates.takeError();
  parsed.qualityGatePolicies = std::move(*gates);

  auto planNodes =
      parseCanonicalRecordSequence<loom::dse::DsePlanNodeDefinition>(
          fieldsOrErr->at("resolved_plan_nodes"),
          prefix + ".resolved_plan_nodes", loom::dse::adoptDsePlanNode);
  if (!planNodes)
    return planNodes.takeError();
  parsed.planNodes = std::move(*planNodes);

  loom::ResolvedObjectiveCatalogs catalogs;
  auto dimensionsOrErr =
      requireSequence(fieldsOrErr->at("objective_dimensions"),
                      prefix + ".objective_dimensions");
  if (!dimensionsOrErr)
    return dimensionsOrErr.takeError();
  std::uint64_t ordinal = 0;
  for (const ConfigSyntax &entry : **dimensionsOrErr) {
    const std::string entryKey =
        (prefix + ".objective_dimensions[" + llvm::Twine(ordinal) + "]").str();
    auto dimensionOrErr =
        ClosedMapping::parse(&entry, entryKey,
                             {"source_kind", "direction", "origin", "quantum",
                              "lower_index", "upper_index"},
                             {"source_ordinal", "evidence_obligation_template",
                              "metric_request_ordinal"});
    if (!dimensionOrErr)
      return dimensionOrErr.takeError();
    auto sourceKind = parseObjectiveSourceKind(
        dimensionOrErr->at("source_kind"), entryKey + ".source_kind");
    auto direction = parseObjectiveDirection(dimensionOrErr->at("direction"),
                                             entryKey + ".direction");
    auto origin = parseObjectiveScalar(dimensionOrErr->at("origin"),
                                       entryKey + ".origin");
    auto quantum = parseObjectiveScalar(dimensionOrErr->at("quantum"),
                                        entryKey + ".quantum");
    auto lower = requireU64(dimensionOrErr->at("lower_index"),
                            entryKey + ".lower_index");
    auto upper = requireU64(dimensionOrErr->at("upper_index"),
                            entryKey + ".upper_index");
    if (!sourceKind)
      return sourceKind.takeError();
    if (!direction)
      return direction.takeError();
    if (!origin)
      return origin.takeError();
    if (!quantum)
      return quantum.takeError();
    if (!lower)
      return lower.takeError();
    if (!upper)
      return upper.takeError();
    loom::ResolvedObjectiveScalarSource source =
        loom::ResolvedMappingMeasureObjectiveSource{0};
    if (*sourceKind == ParsedObjectiveSourceKind::EvaluationMetric) {
      if (dimensionOrErr->at("source_ordinal"))
        return diagnostic("config_unknown_key", entryKey + ".source_ordinal");
      auto obligation =
          requireU32(dimensionOrErr->at("evidence_obligation_template"),
                     entryKey + ".evidence_obligation_template");
      auto metricRequest =
          requireU64(dimensionOrErr->at("metric_request_ordinal"),
                     entryKey + ".metric_request_ordinal");
      if (!obligation)
        return obligation.takeError();
      if (!metricRequest)
        return metricRequest.takeError();
      source = loom::ResolvedEvaluationMetricObjectiveSource{*obligation,
                                                             *metricRequest};
    } else {
      if (dimensionOrErr->at("evidence_obligation_template") ||
          dimensionOrErr->at("metric_request_ordinal"))
        return diagnostic("config_unknown_key", entryKey,
                          "Evaluation metric source fields");
      auto sourceOrdinal = requireU32(dimensionOrErr->at("source_ordinal"),
                                      entryKey + ".source_ordinal");
      if (!sourceOrdinal)
        return sourceOrdinal.takeError();
      if (*sourceKind == ParsedObjectiveSourceKind::MappingViolation)
        source = loom::ResolvedMappingViolationObjectiveSource{
            static_cast<loom::ResolvedPnrViolationKind>(*sourceOrdinal)};
      else
        source = loom::ResolvedMappingMeasureObjectiveSource{*sourceOrdinal};
    }
    catalogs.dimensions.push_back({std::move(source), *direction,
                                   std::move(*origin), std::move(*quantum),
                                   *lower, *upper});
    ++ordinal;
  }

  auto levelsOrErr = requireSequence(fieldsOrErr->at("weighted_levels"),
                                     prefix + ".weighted_levels");
  if (!levelsOrErr)
    return levelsOrErr.takeError();
  ordinal = 0;
  for (const ConfigSyntax &entry : **levelsOrErr) {
    const std::string levelKey =
        (prefix + ".weighted_levels[" + llvm::Twine(ordinal) + "]").str();
    auto levelOrErr = ClosedMapping::parse(&entry, levelKey, {"terms"});
    if (!levelOrErr)
      return levelOrErr.takeError();
    auto termsOrErr =
        requireSequence(levelOrErr->at("terms"), levelKey + ".terms");
    if (!termsOrErr)
      return termsOrErr.takeError();
    loom::ResolvedWeightedObjectiveLevel level;
    std::uint64_t termOrdinal = 0;
    for (const ConfigSyntax &termNode : **termsOrErr) {
      const std::string termKey =
          (levelKey + ".terms[" + llvm::Twine(termOrdinal) + "]").str();
      auto termOrErr =
          ClosedMapping::parse(&termNode, termKey, {"dimension", "weight"});
      if (!termOrErr)
        return termOrErr.takeError();
      auto dimension =
          requireU32(termOrErr->at("dimension"), termKey + ".dimension");
      auto weight = requireU64(termOrErr->at("weight"), termKey + ".weight");
      if (!dimension)
        return dimension.takeError();
      if (!weight)
        return weight.takeError();
      level.terms.push_back({*dimension, *weight});
      ++termOrdinal;
    }
    catalogs.weightedLevels.push_back(std::move(level));
    ++ordinal;
  }

  auto orderingsOrErr = requireSequence(fieldsOrErr->at("total_orderings"),
                                        prefix + ".total_orderings");
  if (!orderingsOrErr)
    return orderingsOrErr.takeError();
  ordinal = 0;
  for (const ConfigSyntax &entry : **orderingsOrErr) {
    const std::string orderingKey =
        (prefix + ".total_orderings[" + llvm::Twine(ordinal) + "]").str();
    auto orderingOrErr =
        ClosedMapping::parse(&entry, orderingKey, {"weighted_levels"});
    if (!orderingOrErr)
      return orderingOrErr.takeError();
    auto referencesOrErr = requireSequence(orderingOrErr->at("weighted_levels"),
                                           orderingKey + ".weighted_levels");
    if (!referencesOrErr)
      return referencesOrErr.takeError();
    loom::ResolvedTotalOrdering ordering;
    for (const ConfigSyntax &reference : **referencesOrErr) {
      auto valueOrErr =
          requireU32(&reference, orderingKey + ".weighted_levels");
      if (!valueOrErr)
        return valueOrErr.takeError();
      ordering.weightedLevels.push_back(*valueOrErr);
    }
    catalogs.totalOrderings.push_back(std::move(ordering));
    ++ordinal;
  }

  if (llvm::Error error = loom::validateResolvedObjectiveCatalogs(catalogs))
    return std::move(error);
  parsed.objectives = std::move(catalogs);
  return parsed;
}

llvm::Expected<loom::ResolvedPathFinderPriceKernel>
parsePathFinderKernel(const ConfigSyntax *node, const llvm::Twine &key) {
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr == "multiplicative")
    return loom::ResolvedPathFinderPriceKernel::Multiplicative;
  if (*valueOrErr == "additive")
    return loom::ResolvedPathFinderPriceKernel::Additive;
  return diagnostic("config_unknown_enum", key, *valueOrErr);
}

llvm::Expected<loom::ResolvedRoutingNegotiationPolicy>
parseRoutingNegotiationPolicy(const ConfigSyntax *node,
                              const llvm::Twine &key) {
  auto kindProbeOrErr = ClosedMapping::parse(
      node, key, {"kind"},
      {"price_kernel", "present_pressure_initial", "present_pressure_growth",
       "history_pressure_increment", "direction_kernel", "step_schedule"});
  if (!kindProbeOrErr)
    return kindProbeOrErr.takeError();
  auto kindOrErr =
      requireScalarString(kindProbeOrErr->at("kind"), key + ".kind");
  if (!kindOrErr)
    return kindOrErr.takeError();

  if (*kindOrErr == "pathfinder") {
    auto fieldsOrErr = ClosedMapping::parse(
        node, key,
        {"kind", "price_kernel", "present_pressure_initial",
         "present_pressure_growth", "history_pressure_increment"});
    if (!fieldsOrErr)
      return fieldsOrErr.takeError();
    auto kernel = parsePathFinderKernel(fieldsOrErr->at("price_kernel"),
                                        key + ".price_kernel");
    auto initial = requireU64(fieldsOrErr->at("present_pressure_initial"),
                              key + ".present_pressure_initial");
    auto growth = parseExactRatio(fieldsOrErr->at("present_pressure_growth"),
                                  key + ".present_pressure_growth");
    auto history = requireU64(fieldsOrErr->at("history_pressure_increment"),
                              key + ".history_pressure_increment");
    if (!kernel)
      return kernel.takeError();
    if (!initial)
      return initial.takeError();
    if (!growth)
      return growth.takeError();
    if (!history)
      return history.takeError();
    return loom::ResolvedRoutingNegotiationPolicy{
        loom::ResolvedPathFinderPolicy{*kernel, *initial, *growth, *history}};
  }
  if (*kindOrErr != "dual_subgradient")
    return diagnostic("config_unknown_union", key + ".kind", *kindOrErr);

  auto fieldsOrErr = ClosedMapping::parse(
      node, key, {"kind", "direction_kernel", "step_schedule"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto directionProbeOrErr =
      ClosedMapping::parse(fieldsOrErr->at("direction_kernel"),
                           key + ".direction_kernel", {"kind"}, {"beta"});
  if (!directionProbeOrErr)
    return directionProbeOrErr.takeError();
  auto directionName = requireScalarString(directionProbeOrErr->at("kind"),
                                           key + ".direction_kernel.kind");
  if (!directionName)
    return directionName.takeError();

  loom::ResolvedDualDirectionKernel direction;
  std::optional<loom::ResolvedExactRatio> momentum;
  if (*directionName == "projected_signed") {
    direction = loom::ResolvedDualDirectionKernel::ProjectedSigned;
  } else if (*directionName == "positive_violation_only") {
    direction = loom::ResolvedDualDirectionKernel::PositiveViolationOnly;
  } else if (*directionName == "momentum_deflected") {
    direction = loom::ResolvedDualDirectionKernel::MomentumDeflected;
    if (!directionProbeOrErr->at("beta"))
      return diagnostic("config_missing_required_profile",
                        key + ".direction_kernel", "beta");
    auto momentumOrErr = parseExactRatio(directionProbeOrErr->at("beta"),
                                         key + ".direction_kernel.beta");
    if (!momentumOrErr)
      return momentumOrErr.takeError();
    momentum = *momentumOrErr;
  } else {
    return diagnostic("config_unknown_enum", key + ".direction_kernel.kind",
                      *directionName);
  }
  if (direction != loom::ResolvedDualDirectionKernel::MomentumDeflected &&
      directionProbeOrErr->at("beta"))
    return diagnostic("config_inactive_union_field",
                      key + ".direction_kernel.beta");

  auto scheduleProbeOrErr = ClosedMapping::parse(
      fieldsOrErr->at("step_schedule"), key + ".step_schedule", {"kind"},
      {"step", "initial_step", "minimum_step", "decay", "numerator", "offset"});
  if (!scheduleProbeOrErr)
    return scheduleProbeOrErr.takeError();
  auto scheduleName = requireScalarString(scheduleProbeOrErr->at("kind"),
                                          key + ".step_schedule.kind");
  if (!scheduleName)
    return scheduleName.takeError();
  loom::ResolvedDualStepSchedule schedule{};
  if (*scheduleName == "constant") {
    auto exactFields =
        ClosedMapping::parse(fieldsOrErr->at("step_schedule"),
                             key + ".step_schedule", {"kind", "step"});
    if (!exactFields)
      return exactFields.takeError();
    auto step =
        requireU64(exactFields->at("step"), key + ".step_schedule.step");
    if (!step)
      return step.takeError();
    schedule = {loom::ResolvedDualStepScheduleKind::Constant, *step, 0, 0, 0};
  } else if (*scheduleName == "geometric_decay") {
    auto exactFields = ClosedMapping::parse(
        fieldsOrErr->at("step_schedule"), key + ".step_schedule",
        {"kind", "initial_step", "minimum_step", "decay"});
    if (!exactFields)
      return exactFields.takeError();
    auto initial = requireU64(exactFields->at("initial_step"),
                              key + ".step_schedule.initial_step");
    auto minimum = requireU64(exactFields->at("minimum_step"),
                              key + ".step_schedule.minimum_step");
    auto decay =
        parseExactRatio(exactFields->at("decay"), key + ".step_schedule.decay");
    if (!initial)
      return initial.takeError();
    if (!minimum)
      return minimum.takeError();
    if (!decay)
      return decay.takeError();
    schedule = {loom::ResolvedDualStepScheduleKind::GeometricDecay, *initial,
                *minimum, decay->numerator, decay->denominator};
  } else if (*scheduleName == "harmonic_decay") {
    auto exactFields = ClosedMapping::parse(
        fieldsOrErr->at("step_schedule"), key + ".step_schedule",
        {"kind", "numerator", "offset", "minimum_step"});
    if (!exactFields)
      return exactFields.takeError();
    auto numerator = requireU64(exactFields->at("numerator"),
                                key + ".step_schedule.numerator");
    auto offset =
        requireU64(exactFields->at("offset"), key + ".step_schedule.offset");
    auto minimum = requireU64(exactFields->at("minimum_step"),
                              key + ".step_schedule.minimum_step");
    if (!numerator)
      return numerator.takeError();
    if (!offset)
      return offset.takeError();
    if (!minimum)
      return minimum.takeError();
    schedule = {loom::ResolvedDualStepScheduleKind::HarmonicDecay, *numerator,
                *offset, *minimum, 0};
  } else {
    return diagnostic("config_unknown_union", key + ".step_schedule.kind",
                      *scheduleName);
  }

  return loom::ResolvedRoutingNegotiationPolicy{
      loom::ResolvedDualSubgradientPolicy{direction, momentum, schedule}};
}

llvm::Expected<loom::ResolvedPnrViolationKind>
parseViolationKind(const ConfigSyntax *node, const llvm::Twine &key) {
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  using Kind = loom::ResolvedPnrViolationKind;
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  if (*valueOrErr == ConfigSpelling)                                           \
    return Kind::Name;
#include "Common/MappingObjectiveKinds.def"
  return diagnostic("config_unknown_enum", key, *valueOrErr);
}

llvm::Expected<loom::ResolvedPnrPolicyConfig>
parsePnrPolicy(const ConfigSyntax *node, const llvm::Twine &key) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, key,
      {"search_policy", "determinism_policy", "temporary_violation_policy",
       "selected_total_ordering", "selected_search_energy"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto searchOrErr = ClosedMapping::parse(
      fieldsOrErr->at("search_policy"), key + ".search_policy",
      {"completion_goal", "initializer", "action_proposal", "routing",
       "annealing", "exact_repair"});
  if (!searchOrErr)
    return searchOrErr.takeError();

  auto completionName =
      requireScalarString(searchOrErr->at("completion_goal"),
                          key + ".search_policy.completion_goal");
  if (!completionName)
    return completionName.takeError();
  const auto completionGoal =
      loom::parseResolvedPnrCompletionGoal(*completionName);
  if (!completionGoal)
    return diagnostic("config_unknown_enum",
                      key + ".search_policy.completion_goal", *completionName);

  auto initializerOrErr = ClosedMapping::parse(
      searchOrErr->at("initializer"), key + ".search_policy.initializer",
      {"seed_attempt_count", "assignment_attempt_limit_per_seed"});
  if (!initializerOrErr)
    return initializerOrErr.takeError();
  auto seeds =
      requireU32(initializerOrErr->at("seed_attempt_count"),
                 key + ".search_policy.initializer.seed_attempt_count");
  auto assignments = requireU64(
      initializerOrErr->at("assignment_attempt_limit_per_seed"),
      key + ".search_policy.initializer.assignment_attempt_limit_per_seed");
  if (!seeds)
    return seeds.takeError();
  if (!assignments)
    return assignments.takeError();

  auto actionOrErr = ClosedMapping::parse(
      searchOrErr->at("action_proposal"),
      key + ".search_policy.action_proposal",
      {"realization_binding_weight", "transport_routing_weight",
       "resource_allocation_weight"});
  if (!actionOrErr)
    return actionOrErr.takeError();
  auto realization = requireU64(
      actionOrErr->at("realization_binding_weight"),
      key + ".search_policy.action_proposal.realization_binding_weight");
  auto transport = requireU64(
      actionOrErr->at("transport_routing_weight"),
      key + ".search_policy.action_proposal.transport_routing_weight");
  auto resource = requireU64(
      actionOrErr->at("resource_allocation_weight"),
      key + ".search_policy.action_proposal.resource_allocation_weight");
  if (!realization)
    return realization.takeError();
  if (!transport)
    return transport.takeError();
  if (!resource)
    return resource.takeError();

  auto routingOrErr = ClosedMapping::parse(
      searchOrErr->at("routing"), key + ".search_policy.routing",
      {"endpoint_expansion_limit", "negotiation_iteration_limit",
       "no_progress_iteration_limit", "no_progress_trend_window",
       "negotiation_policy"});
  if (!routingOrErr)
    return routingOrErr.takeError();
  auto endpointLimit =
      requireU64(routingOrErr->at("endpoint_expansion_limit"),
                 key + ".search_policy.routing.endpoint_expansion_limit");
  auto negotiationLimit =
      requireU64(routingOrErr->at("negotiation_iteration_limit"),
                 key + ".search_policy.routing.negotiation_iteration_limit");
  auto noProgressLimit =
      requireU64(routingOrErr->at("no_progress_iteration_limit"),
                 key + ".search_policy.routing.no_progress_iteration_limit");
  auto noProgressTrendWindow =
      requireU64(routingOrErr->at("no_progress_trend_window"),
                 key + ".search_policy.routing.no_progress_trend_window");
  auto negotiation = parseRoutingNegotiationPolicy(
      routingOrErr->at("negotiation_policy"),
      key + ".search_policy.routing.negotiation_policy");
  if (!endpointLimit)
    return endpointLimit.takeError();
  if (!negotiationLimit)
    return negotiationLimit.takeError();
  if (!noProgressLimit)
    return noProgressLimit.takeError();
  if (!noProgressTrendWindow)
    return noProgressTrendWindow.takeError();
  if (!negotiation)
    return negotiation.takeError();
  auto annealingOrErr = ClosedMapping::parse(
      searchOrErr->at("annealing"), key + ".search_policy.annealing",
      {"calibration_proposal_count", "positive_delta_quantile",
       "target_initial_acceptance", "fallback_temperature",
       "minimum_temperature", "cooling_ratio", "temperature_level_limit",
       "proposals_per_level_base", "proposals_per_movable_decision"});
  if (!annealingOrErr)
    return annealingOrErr.takeError();
  auto calibration =
      requireU64(annealingOrErr->at("calibration_proposal_count"),
                 key + ".search_policy.annealing.calibration_proposal_count");
  auto quantile =
      parseExactRatio(annealingOrErr->at("positive_delta_quantile"),
                      key + ".search_policy.annealing.positive_delta_quantile");
  auto acceptance = parseExactRatio(
      annealingOrErr->at("target_initial_acceptance"),
      key + ".search_policy.annealing.target_initial_acceptance");
  auto fallback =
      requireU64(annealingOrErr->at("fallback_temperature"),
                 key + ".search_policy.annealing.fallback_temperature");
  auto minimum =
      requireU64(annealingOrErr->at("minimum_temperature"),
                 key + ".search_policy.annealing.minimum_temperature");
  auto cooling =
      parseExactRatio(annealingOrErr->at("cooling_ratio"),
                      key + ".search_policy.annealing.cooling_ratio");
  auto temperatureLevelLimit =
      requireU64(annealingOrErr->at("temperature_level_limit"),
                 key + ".search_policy.annealing.temperature_level_limit");
  auto levelBase =
      requireU64(annealingOrErr->at("proposals_per_level_base"),
                 key + ".search_policy.annealing.proposals_per_level_base");
  auto perMovable = requireU64(
      annealingOrErr->at("proposals_per_movable_decision"),
      key + ".search_policy.annealing.proposals_per_movable_decision");
  if (!calibration)
    return calibration.takeError();
  if (!quantile)
    return quantile.takeError();
  if (!acceptance)
    return acceptance.takeError();
  if (!fallback)
    return fallback.takeError();
  if (!minimum)
    return minimum.takeError();
  if (!cooling)
    return cooling.takeError();
  if (!temperatureLevelLimit)
    return temperatureLevelLimit.takeError();
  if (!levelBase)
    return levelBase.takeError();
  if (!perMovable)
    return perMovable.takeError();

  auto repairProbeOrErr = ClosedMapping::parse(
      searchOrErr->at("exact_repair"), key + ".search_policy.exact_repair",
      {"kind"}, {"max_region_decisions", "max_solver_calls"});
  if (!repairProbeOrErr)
    return repairProbeOrErr.takeError();
  auto repairName = requireScalarString(
      repairProbeOrErr->at("kind"), key + ".search_policy.exact_repair.kind");
  if (!repairName)
    return repairName.takeError();
  loom::ResolvedPnrExactRepairPolicy repair{};
  if (*repairName == "disabled") {
    auto exactFields =
        ClosedMapping::parse(searchOrErr->at("exact_repair"),
                             key + ".search_policy.exact_repair", {"kind"});
    if (!exactFields)
      return exactFields.takeError();
    repair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  } else if (*repairName == "cp_sat") {
    auto exactFields = ClosedMapping::parse(
        searchOrErr->at("exact_repair"), key + ".search_policy.exact_repair",
        {"kind", "max_region_decisions", "max_solver_calls"});
    if (!exactFields)
      return exactFields.takeError();
    auto decisions =
        requireU64(exactFields->at("max_region_decisions"),
                   key + ".search_policy.exact_repair.max_region_decisions");
    auto calls =
        requireU64(exactFields->at("max_solver_calls"),
                   key + ".search_policy.exact_repair.max_solver_calls");
    if (!decisions)
      return decisions.takeError();
    if (!calls)
      return calls.takeError();
    repair = {loom::ResolvedPnrExactRepairKind::CpSat, *decisions, *calls};
  } else {
    return diagnostic("config_unknown_union",
                      key + ".search_policy.exact_repair.kind", *repairName);
  }

  auto determinismOrErr = ClosedMapping::parse(
      fieldsOrErr->at("determinism_policy"), key + ".determinism_policy",
      {"master_seed", "prng_protocol", "acceptance_protocol"});
  if (!determinismOrErr)
    return determinismOrErr.takeError();
  auto masterSeed = requireU64(determinismOrErr->at("master_seed"),
                               key + ".determinism_policy.master_seed");
  auto prng = requireScalarString(determinismOrErr->at("prng_protocol"),
                                  key + ".determinism_policy.prng_protocol");
  auto acceptanceProtocol =
      requireScalarString(determinismOrErr->at("acceptance_protocol"),
                          key + ".determinism_policy.acceptance_protocol");
  if (!masterSeed)
    return masterSeed.takeError();
  if (!prng)
    return prng.takeError();
  if (!acceptanceProtocol)
    return acceptanceProtocol.takeError();
  if (*prng != "sha256_seeded_xoshiro256starstar_1_0")
    return diagnostic("config_unknown_protocol",
                      key + ".determinism_policy.prng_protocol", *prng);
  if (*acceptanceProtocol != "exp_negative_q64_table_1_0")
    return diagnostic("config_unknown_protocol",
                      key + ".determinism_policy.acceptance_protocol",
                      *acceptanceProtocol);

  auto violationsOrErr =
      requireSequence(fieldsOrErr->at("temporary_violation_policy"),
                      key + ".temporary_violation_policy");
  if (!violationsOrErr)
    return violationsOrErr.takeError();
  loom::ResolvedPnrTemporaryViolationPolicy violations;
  for (const ConfigSyntax &violationNode : **violationsOrErr) {
    auto violation =
        parseViolationKind(&violationNode, key + ".temporary_violation_policy");
    if (!violation)
      return violation.takeError();
    violations.admitted.push_back(*violation);
  }

  auto selectedOrdering = requireU32(fieldsOrErr->at("selected_total_ordering"),
                                     key + ".selected_total_ordering");
  auto selectedEnergy = requireU32(fieldsOrErr->at("selected_search_energy"),
                                   key + ".selected_search_energy");
  if (!selectedOrdering)
    return selectedOrdering.takeError();
  if (!selectedEnergy)
    return selectedEnergy.takeError();
  return loom::ResolvedPnrPolicyConfig{
      {loom::ResolvedPnrInitializerPolicy{*seeds, *assignments},
       loom::ResolvedPnrActionProposalPolicy{*realization, *transport,
                                             *resource},
       loom::ResolvedPnrRoutingPolicy{*endpointLimit, *negotiationLimit,
                                      *noProgressLimit, *noProgressTrendWindow,
                                      std::move(*negotiation)},
       loom::ResolvedPnrAnnealingPolicy{
           *calibration, *quantile, *acceptance, *fallback, *minimum, *cooling,
           *temperatureLevelLimit, *levelBase, *perMovable},
       repair, *completionGoal},
      loom::ResolvedPnrDeterminismPolicy{
          *masterSeed,
          loom::ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0,
          loom::ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0},
      std::move(violations),
      loom::ResolvedPnrObjectiveSelection{*selectedOrdering, *selectedEnergy}};
}

llvm::Error parseStructuredOwnership(ConfigPatch &patch,
                                     const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(node, "dse.structured_ownership", {},
                                          {"scope_expansion_limit"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *value = fieldsOrErr->at("scope_expansion_limit")) {
    constexpr llvm::StringLiteral key =
        "dse.structured_ownership.scope_expansion_limit";
    auto valueOrErr = requireUnsigned(value, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    patch.ownershipScopeExpansionLimit = *valueOrErr;
    return touch(patch, key);
  }
  return llvm::Error::success();
}

llvm::Error parseStructuredSchedule(ConfigPatch &patch,
                                    const ConfigSyntax *node) {
  auto fieldsOrErr =
      ClosedMapping::parse(node, "dse.schedule", {}, {"scope_expansion_limit"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *value = fieldsOrErr->at("scope_expansion_limit")) {
    constexpr llvm::StringLiteral key = "dse.schedule.scope_expansion_limit";
    auto valueOrErr = requireUnsigned(value, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    patch.scheduleScopeExpansionLimit = *valueOrErr;
    return touch(patch, key);
  }
  return llvm::Error::success();
}

llvm::Error parseMemoryCommunication(ConfigPatch &patch,
                                     const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(node, "dse.memory_communication", {},
                                          {"scope_expansion_limit"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *value = fieldsOrErr->at("scope_expansion_limit")) {
    constexpr llvm::StringLiteral key =
        "dse.memory_communication.scope_expansion_limit";
    auto valueOrErr = requireUnsigned(value, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    patch.memoryCommunicationScopeExpansionLimit = *valueOrErr;
    return touch(patch, key);
  }
  return llvm::Error::success();
}

llvm::Error parseDataflowRewrite(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(node, "dse.dataflow_rewrite", {},
                                          {"scope_expansion_limit"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *value = fieldsOrErr->at("scope_expansion_limit")) {
    constexpr llvm::StringLiteral key =
        "dse.dataflow_rewrite.scope_expansion_limit";
    auto valueOrErr = requireUnsigned(value, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    patch.dataflowRewriteScopeExpansionLimit = *valueOrErr;
    return touch(patch, key);
  }
  return llvm::Error::success();
}

llvm::Error parseTechMapping(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, "dse.tech_mapping", {},
      {"match_row_attempt_limit", "partial_cover_expansion_limit",
       "candidate_evaluation_limit", "candidate_publication_limit"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  const auto parseLimit =
      [&](llvm::StringRef field,
          std::optional<std::uint64_t> &destination) -> llvm::Error {
    const ConfigSyntax *value = fieldsOrErr->at(field);
    if (!value)
      return llvm::Error::success();
    const std::string key = ("dse.tech_mapping." + field).str();
    auto valueOrErr = requirePositiveU64(value, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    destination = *valueOrErr;
    return touch(patch, key);
  };
  if (llvm::Error error = parseLimit("match_row_attempt_limit",
                                     patch.techMappingMatchRowAttemptLimit))
    return error;
  if (llvm::Error error =
          parseLimit("partial_cover_expansion_limit",
                     patch.techMappingPartialCoverExpansionLimit))
    return error;
  if (llvm::Error error = parseLimit("candidate_evaluation_limit",
                                     patch.techMappingCandidateEvaluationLimit))
    return error;
  if (llvm::Error error =
          parseLimit("candidate_publication_limit",
                     patch.techMappingCandidatePublicationLimit))
    return error;
  return llvm::Error::success();
}

llvm::Error parseDse(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, "dse", {},
      {"structured_ownership", "schedule", "memory_communication",
       "dataflow_rewrite", "tech_mapping", "evaluation_and_objective_catalogs",
       "spatial_pnr", "system_pnr"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *structured = fieldsOrErr->at("structured_ownership"))
    if (llvm::Error error = parseStructuredOwnership(patch, structured))
      return error;
  if (const ConfigSyntax *schedule = fieldsOrErr->at("schedule"))
    if (llvm::Error error = parseStructuredSchedule(patch, schedule))
      return error;
  if (const ConfigSyntax *memory = fieldsOrErr->at("memory_communication"))
    if (llvm::Error error = parseMemoryCommunication(patch, memory))
      return error;
  if (const ConfigSyntax *rewrite = fieldsOrErr->at("dataflow_rewrite"))
    if (llvm::Error error = parseDataflowRewrite(patch, rewrite))
      return error;
  if (const ConfigSyntax *tech = fieldsOrErr->at("tech_mapping"))
    if (llvm::Error error = parseTechMapping(patch, tech))
      return error;
  if (const ConfigSyntax *catalogs =
          fieldsOrErr->at("evaluation_and_objective_catalogs")) {
    auto catalogsOrErr = parseObjectiveCatalogs(catalogs);
    if (!catalogsOrErr)
      return catalogsOrErr.takeError();
    patch.modelAuthorizations = std::move(catalogsOrErr->modelAuthorizations);
    patch.evidenceObligationTemplates =
        std::move(catalogsOrErr->evidenceObligationTemplates);
    patch.objectiveCatalogs = std::move(catalogsOrErr->objectives);
    patch.qualityGatePolicies = std::move(catalogsOrErr->qualityGatePolicies);
    patch.planNodes = std::move(catalogsOrErr->planNodes);
    if (llvm::Error error =
            touch(patch, "dse.evaluation_and_objective_catalogs"))
      return error;
  }
  if (const ConfigSyntax *spatial = fieldsOrErr->at("spatial_pnr")) {
    auto policyOrErr = parsePnrPolicy(spatial, "dse.spatial_pnr");
    if (!policyOrErr)
      return policyOrErr.takeError();
    patch.spatialPnr = std::move(*policyOrErr);
    if (llvm::Error error = touch(patch, "dse.spatial_pnr"))
      return error;
  }
  if (const ConfigSyntax *system = fieldsOrErr->at("system_pnr")) {
    auto policyOrErr = parsePnrPolicy(system, "dse.system_pnr");
    if (!policyOrErr)
      return policyOrErr.takeError();
    patch.systemPnr = std::move(*policyOrErr);
    if (llvm::Error error = touch(patch, "dse.system_pnr"))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<loom::evaluation::models::CadenceVoltusStaticRailProviderBinding>
parseCadenceVoltusStaticRailBinding(const ConfigSyntax *node) {
  constexpr llvm::StringLiteral prefix =
      "evaluation.cadence_voltus_static_rail";
  auto fieldsOrErr = ClosedMapping::parse(node, prefix,
                                          {"stable_provider_build_identity",
                                           "power_grid_library_members",
                                           "power_grid_library_entrypoints"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();

  auto buildOrErr =
      requireScalarString(fieldsOrErr->at("stable_provider_build_identity"),
                          prefix + ".stable_provider_build_identity");
  if (!buildOrErr)
    return buildOrErr.takeError();
  auto membersOrErr =
      requireSequence(fieldsOrErr->at("power_grid_library_members"),
                      prefix + ".power_grid_library_members");
  if (!membersOrErr)
    return membersOrErr.takeError();
  auto entrypointsOrErr =
      requireSequence(fieldsOrErr->at("power_grid_library_entrypoints"),
                      prefix + ".power_grid_library_entrypoints");
  if (!entrypointsOrErr)
    return entrypointsOrErr.takeError();

  std::vector<loom::external_tool::ExternalFileTreeMember> members;
  members.reserve((*membersOrErr)->size());
  for (std::size_t index = 0; index < (*membersOrErr)->size(); ++index) {
    const std::string memberPrefix =
        (prefix + ".power_grid_library_members[" + llvm::Twine(index) + "]")
            .str();
    auto memberOrErr = ClosedMapping::parse(
        &(*membersOrErr)->at(index), memberPrefix, {"relative_path", "sha256"});
    if (!memberOrErr)
      return memberOrErr.takeError();
    auto pathOrErr = requireScalarString(memberOrErr->at("relative_path"),
                                         memberPrefix + ".relative_path");
    if (!pathOrErr)
      return pathOrErr.takeError();
    auto fingerprintTextOrErr = requireScalarString(memberOrErr->at("sha256"),
                                                    memberPrefix + ".sha256");
    if (!fingerprintTextOrErr)
      return fingerprintTextOrErr.takeError();
    auto fingerprintOrErr =
        loom::parseExternalFileFingerprint(*fingerprintTextOrErr);
    if (!fingerprintOrErr)
      return fingerprintOrErr.takeError();
    members.push_back({std::move(*pathOrErr), std::move(*fingerprintOrErr)});
  }

  std::vector<std::string> entrypoints;
  entrypoints.reserve((*entrypointsOrErr)->size());
  for (std::size_t index = 0; index < (*entrypointsOrErr)->size(); ++index) {
    auto entrypointOrErr = requireScalarString(
        &(*entrypointsOrErr)->at(index),
        prefix + ".power_grid_library_entrypoints[" + llvm::Twine(index) + "]");
    if (!entrypointOrErr)
      return entrypointOrErr.takeError();
    entrypoints.push_back(std::move(*entrypointOrErr));
  }

  loom::evaluation::models::CadenceVoltusStaticRailProviderBinding binding{
      std::move(*buildOrErr), std::move(members), std::move(entrypoints)};
  if (llvm::Error error = loom::evaluation::models::
          validateCadenceVoltusStaticRailProviderBinding(binding))
    return std::move(error);
  return binding;
}

llvm::Expected<loom::evaluation::models::MappedRtlSimulatorBinding>
parseMappedRtlSimulatorBinding(const ConfigSyntax *node) {
  constexpr llvm::StringLiteral prefix = "evaluation.mapped_rtl_simulator";
  auto fieldsOrErr = ClosedMapping::parse(
      node, prefix, {"stable_hdl_simulator_build_identity"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto buildOrErr = requireScalarString(
      fieldsOrErr->at("stable_hdl_simulator_build_identity"),
      prefix + ".stable_hdl_simulator_build_identity");
  if (!buildOrErr)
    return buildOrErr.takeError();
  loom::evaluation::models::MappedRtlSimulatorBinding binding{
      std::move(*buildOrErr)};
  if (llvm::Error error =
          loom::evaluation::models::validateMappedRtlSimulatorBinding(binding))
    return std::move(error);
  return binding;
}

llvm::Expected<loom::evaluation::models::OpenRoadStaticFpaProviderBinding>
parseOpenRoadStaticFpaBinding(const ConfigSyntax *node) {
  constexpr llvm::StringLiteral prefix =
      "evaluation.openroad_routed_static_fpa";
  auto fieldsOrErr =
      ClosedMapping::parse(node, prefix, {"stable_provider_build_identity"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto buildOrErr =
      requireScalarString(fieldsOrErr->at("stable_provider_build_identity"),
                          prefix + ".stable_provider_build_identity");
  if (!buildOrErr)
    return buildOrErr.takeError();
  loom::evaluation::models::OpenRoadStaticFpaProviderBinding binding{
      std::move(*buildOrErr)};
  if (llvm::Error error =
          loom::evaluation::models::validateOpenRoadStaticFpaProviderBinding(
              binding))
    return std::move(error);
  return binding;
}

llvm::Error parseEvaluation(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(node, "evaluation", {},
                                          {"cadence_voltus_static_rail",
                                           "mapped_rtl_simulator",
                                           "openroad_routed_static_fpa"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *rail =
          fieldsOrErr->at("cadence_voltus_static_rail")) {
    auto bindingOrErr = parseCadenceVoltusStaticRailBinding(rail);
    if (!bindingOrErr)
      return bindingOrErr.takeError();
    patch.cadenceVoltusStaticRail = std::move(*bindingOrErr);
    if (llvm::Error error =
            touch(patch, "evaluation.cadence_voltus_static_rail"))
      return error;
  }
  if (const ConfigSyntax *mapped = fieldsOrErr->at("mapped_rtl_simulator")) {
    auto bindingOrErr = parseMappedRtlSimulatorBinding(mapped);
    if (!bindingOrErr)
      return bindingOrErr.takeError();
    patch.mappedRtlSimulator = std::move(*bindingOrErr);
    if (llvm::Error error = touch(patch, "evaluation.mapped_rtl_simulator"))
      return error;
  }
  if (const ConfigSyntax *fpa = fieldsOrErr->at("openroad_routed_static_fpa")) {
    auto bindingOrErr = parseOpenRoadStaticFpaBinding(fpa);
    if (!bindingOrErr)
      return bindingOrErr.takeError();
    patch.openRoadStaticFpa = std::move(*bindingOrErr);
    if (llvm::Error error =
            touch(patch, "evaluation.openroad_routed_static_fpa"))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(const ConfigSyntax &topMap,
                            llvm::StringRef sourceName);

llvm::Expected<ConfigPatch> parseConfigFilePatch(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError())
    return makeErr("config_parse_failed: " + path + ": " + ec.message());
  llvm::SourceMgr sourceMgr;
  llvm::yaml::Stream stream((*bufferOrErr)->getBuffer(), sourceMgr);
  auto it = stream.begin();
  if (it == stream.end())
    return ConfigPatch();
  llvm::yaml::Node *root = it->getRoot();
  if (!root)
    return ConfigPatch();
  auto syntaxOrErr = materializeSyntax(root, path);
  if (!syntaxOrErr)
    return syntaxOrErr.takeError();
  ++it;
  if (it != stream.end())
    return diagnostic("config_parse_failed", path,
                      "multiple YAML documents are not supported");
  if (stream.failed())
    return diagnostic("config_parse_failed", path);
  auto patchOrErr = parseConfigPatchFromMapping(*syntaxOrErr, path);
  if (!patchOrErr)
    return patchOrErr.takeError();
  return *patchOrErr;
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(const ConfigSyntax &topMap,
                            llvm::StringRef sourceName) {
  if (topMap.kind != ConfigSyntax::Kind::Mapping)
    return diagnostic("config_type_mismatch", sourceName, "top-level mapping");
  ConfigPatch local;
  for (const auto &[keyStorage, value] : topMap.mapping) {
    StringRef key(keyStorage);
    if (key == "inherits") {
      auto spelling = requireScalarString(&value, "inherits");
      if (!spelling)
        return spelling.takeError();
      local.inheritedPreset = profilePresetForName(*spelling);
      if (!local.inheritedPreset)
        return diagnostic("config_unknown_enum", "inherits", *spelling);
    } else if (key == "hardware_target") {
      auto targetOrErr = parseHardwareTarget(&value);
      if (!targetOrErr)
        return targetOrErr.takeError();
      local.hardwareTarget = std::move(*targetOrErr);
      if (llvm::Error err = touch(local, "hardware_target"))
        return err;
    } else if (key == "dse") {
      if (llvm::Error err = parseDse(local, &value))
        return err;
    } else if (key == "evaluation") {
      if (llvm::Error err = parseEvaluation(local, &value))
        return err;
    } else {
      return diagnostic("config_unknown_key", key);
    }
  }

  return local;
}

llvm::Error validateResolvedConfig(const loom::ResolvedConfig &config) {
  if (config.hardwareTarget.templateIdentity.empty())
    return diagnostic("config_missing_required_profile",
                      "hardware_target.template_identity");
  const loom::adg::BuiltinTargetScale &scale = config.hardwareTarget.parameters;
  if (!loom::adg::isValidBuiltinTargetScale(scale))
    return diagnostic("config_range_violation", "hardware_target.parameters",
                      "base scale values must be positive, mesh lane counts "
                      "must be within the public MeshSwitch domain, "
                      "mesh_dimension must exceed one, and each FU occurrence "
                      "count must not exceed its schedule-local PE count");
  if (config.dse.structuredOwnership.scopeExpansionLimit == 0 ||
      config.dse.schedule.scopeExpansionLimit == 0 ||
      config.dse.memoryCommunication.scopeExpansionLimit == 0 ||
      config.dse.dataflowRewrite.scopeExpansionLimit == 0 ||
      config.dse.techMapping.matchRowAttemptLimit == 0 ||
      config.dse.techMapping.partialCoverExpansionLimit == 0 ||
      config.dse.techMapping.candidateEvaluationLimit == 0 ||
      config.dse.techMapping.candidatePublicationLimit == 0)
    return diagnostic("config_range_violation", "dse",
                      "semantic work limits must be positive");
  auto dseView = loom::dse::projectResolvedDseConfigView(config);
  if (!dseView)
    return dseView.takeError();
  if (llvm::Error error = loom::validateResolvedPnrPolicyConfig(
          config.dse.spatialPnr, config.dse.objectiveCatalogs))
    return error;
  if (config.evaluation.cadenceVoltusStaticRail)
    if (llvm::Error error = loom::evaluation::models::
            validateCadenceVoltusStaticRailProviderBinding(
                *config.evaluation.cadenceVoltusStaticRail))
      return error;
  if (config.evaluation.mappedRtlSimulator)
    if (llvm::Error error =
            loom::evaluation::models::validateMappedRtlSimulatorBinding(
                *config.evaluation.mappedRtlSimulator))
      return error;
  if (config.evaluation.openRoadStaticFpa)
    if (llvm::Error error =
            loom::evaluation::models::validateOpenRoadStaticFpaProviderBinding(
                *config.evaluation.openRoadStaticFpa))
      return error;
  return loom::validateResolvedPnrPolicyConfig(config.dse.systemPnr,
                                               config.dse.objectiveCatalogs);
}

} // namespace

static std::optional<loom::ResolvedProfilePreset>
profilePresetForName(llvm::StringRef spelling) {
  using Preset = loom::ResolvedProfilePreset;
  if (spelling == "report_only")
    return Preset::ReportOnly;
  if (spelling == "quick_explore")
    return Preset::QuickExplore;
  if (spelling == "balanced_explore")
    return Preset::BalancedExplore;
  if (spelling == "performance_explore")
    return Preset::PerformanceExplore;
  if (spelling == "implementation")
    return Preset::Implementation;
  if (spelling == "strict_implementation")
    return Preset::StrictImplementation;
  return std::nullopt;
}

static loom::ResolvedConfig
builtinResolvedConfig(loom::ResolvedProfilePreset preset) {
  loom::ResolvedConfig config;
  config.hardwareTarget = {
      loom::adg::builtinCoverageTarget.templateIdentity.str(),
      {loom::adg::builtinCoverageTarget.schemaMajor,
       loom::adg::builtinCoverageTarget.schemaMinor},
      loom::adg::builtinCoverageTarget.scale};
  config.dse.objectiveCatalogs = loom::resolvedBuiltinObjectiveCatalogs();
  config.dse.spatialPnr = loom::resolvedBuiltinSpatialPnrPolicy(preset);
  config.dse.systemPnr = loom::resolvedBuiltinSystemPnrPolicy(preset);
  llvm::cantFail(loom::validateResolvedPnrPolicyConfig(
      config.dse.spatialPnr, config.dse.objectiveCatalogs));
  llvm::cantFail(loom::validateResolvedPnrPolicyConfig(
      config.dse.systemPnr, config.dse.objectiveCatalogs));
  return config;
}

loom::ResolvedConfig loom::defaultResolvedConfig() {
  return builtinResolvedConfig(ResolvedProfilePreset::BalancedExplore);
}

llvm::Expected<loom::ResolvedConfig>
loom::resolveConfigProfile(llvm::StringRef builtinPresetOrConfigPath) {
  if (builtinPresetOrConfigPath.empty())
    return defaultResolvedConfig();
  if (std::optional<ResolvedProfilePreset> preset =
          profilePresetForName(builtinPresetOrConfigPath))
    return builtinResolvedConfig(*preset);
  return loadResolvedConfig(builtinPresetOrConfigPath);
}

llvm::Expected<loom::ResolvedConfig>
loom::parseResolvedConfig(llvm::StringRef body, llvm::StringRef sourceName) {
  llvm::SourceMgr sourceMgr;
  llvm::yaml::Stream stream(body, sourceMgr);
  auto it = stream.begin();
  if (it == stream.end())
    return defaultResolvedConfig();
  llvm::yaml::Node *root = it->getRoot();
  if (!root)
    return defaultResolvedConfig();
  auto syntaxOrErr = materializeSyntax(root, sourceName);
  if (!syntaxOrErr)
    return syntaxOrErr.takeError();
  ++it;
  if (it != stream.end())
    return diagnostic("config_parse_failed", sourceName,
                      "multiple YAML documents are not supported");
  if (stream.failed())
    return diagnostic("config_parse_failed", sourceName);

  auto patchOrErr = parseConfigPatchFromMapping(*syntaxOrErr, sourceName);
  if (!patchOrErr)
    return patchOrErr.takeError();

  ResolvedConfig config =
      builtinResolvedConfig(patchOrErr->inheritedPreset.value_or(
          ResolvedProfilePreset::BalancedExplore));
  applyPatch(config, *patchOrErr);
  if (llvm::Error error = validateResolvedConfig(config))
    return std::move(error);
  return config;
}

llvm::Expected<loom::ResolvedConfig>
loom::loadResolvedConfig(llvm::StringRef path) {
  auto patchOrErr = parseConfigFilePatch(path);
  if (!patchOrErr)
    return patchOrErr.takeError();

  ResolvedConfig config =
      builtinResolvedConfig(patchOrErr->inheritedPreset.value_or(
          ResolvedProfilePreset::BalancedExplore));
  applyPatch(config, *patchOrErr);
  if (llvm::Error error = validateResolvedConfig(config))
    return std::move(error);
  return config;
}
