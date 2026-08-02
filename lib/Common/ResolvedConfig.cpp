#include "Common/ResolvedConfig.h"

#include "Common/ArtifactFinalizer.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <string>

using llvm::StringRef;

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
  std::optional<loom::ResolvedHardwareTargetConfig> hardwareTarget;
  std::optional<std::uint32_t> ownershipScopeExpansionLimit;
  std::optional<std::uint64_t> techMappingMatchRowAttemptLimit;
  std::optional<std::uint64_t> techMappingPartialCoverExpansionLimit;
  std::optional<std::uint64_t> techMappingCandidatePublicationLimit;
  std::optional<loom::ResolvedObjectiveCatalogs> objectiveCatalogs;
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

llvm::Error mergeSiblingPatch(ConfigPatch &dst, const ConfigPatch &src) {
  for (const std::string &key : src.touchedKeys)
    if (dst.touchedKeys.count(key) != 0)
      return diagnostic("config_conflicting_sources", key);

  if (src.hardwareTarget)
    dst.hardwareTarget = src.hardwareTarget;
  if (src.ownershipScopeExpansionLimit)
    dst.ownershipScopeExpansionLimit = src.ownershipScopeExpansionLimit;
  if (src.techMappingMatchRowAttemptLimit)
    dst.techMappingMatchRowAttemptLimit = src.techMappingMatchRowAttemptLimit;
  if (src.techMappingPartialCoverExpansionLimit)
    dst.techMappingPartialCoverExpansionLimit =
        src.techMappingPartialCoverExpansionLimit;
  if (src.techMappingCandidatePublicationLimit)
    dst.techMappingCandidatePublicationLimit =
        src.techMappingCandidatePublicationLimit;
  if (src.objectiveCatalogs)
    dst.objectiveCatalogs = src.objectiveCatalogs;
  if (src.spatialPnr)
    dst.spatialPnr = src.spatialPnr;
  if (src.systemPnr)
    dst.systemPnr = src.systemPnr;
  dst.touchedKeys.insert(src.touchedKeys.begin(), src.touchedKeys.end());
  return llvm::Error::success();
}

void applyPatch(loom::ResolvedConfig &config, const ConfigPatch &patch) {
  if (patch.hardwareTarget)
    config.hardwareTarget = *patch.hardwareTarget;
  if (patch.ownershipScopeExpansionLimit)
    config.dse.structuredOwnership.scopeExpansionLimit =
        *patch.ownershipScopeExpansionLimit;
  if (patch.techMappingMatchRowAttemptLimit)
    config.dse.techMapping.matchRowAttemptLimit =
        *patch.techMappingMatchRowAttemptLimit;
  if (patch.techMappingPartialCoverExpansionLimit)
    config.dse.techMapping.partialCoverExpansionLimit =
        *patch.techMappingPartialCoverExpansionLimit;
  if (patch.techMappingCandidatePublicationLimit)
    config.dse.techMapping.candidatePublicationLimit =
        *patch.techMappingCandidatePublicationLimit;
  if (patch.objectiveCatalogs)
    config.dse.objectiveCatalogs = *patch.objectiveCatalogs;
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
      {"acc_core_count", "spatial_pe_count", "temporal_pe_count",
       "spatial_memory_count", "temporal_memory_count",
       "temporal_resident_contexts", "gateway_count", "memory_capacity_bytes"});
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
  auto spatialPes = positiveU32("spatial_pe_count");
  auto temporalPes = positiveU32("temporal_pe_count");
  auto spatialMemories = positiveU32("spatial_memory_count");
  auto temporalMemories = positiveU32("temporal_memory_count");
  auto residentContexts = positiveU32("temporal_resident_contexts");
  auto gateways = positiveU32("gateway_count");
  auto memoryCapacity =
      requirePositiveU64(parametersOrErr->at("memory_capacity_bytes"),
                         "hardware_target.parameters.memory_capacity_bytes");
  if (!accCores)
    return accCores.takeError();
  if (!spatialPes)
    return spatialPes.takeError();
  if (!temporalPes)
    return temporalPes.takeError();
  if (!spatialMemories)
    return spatialMemories.takeError();
  if (!temporalMemories)
    return temporalMemories.takeError();
  if (!residentContexts)
    return residentContexts.takeError();
  if (!gateways)
    return gateways.takeError();
  if (!memoryCapacity)
    return memoryCapacity.takeError();

  return loom::ResolvedHardwareTargetConfig{
      *templateOrErr,
      {*majorOrErr, *minorOrErr},
      {*accCores, *spatialPes, *temporalPes, *spatialMemories,
       *temporalMemories, *residentContexts, *gateways, *memoryCapacity}};
}

llvm::Expected<loom::ResolvedObjectiveSourceKind>
parseObjectiveSourceKind(const ConfigSyntax *node, const llvm::Twine &key) {
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr == "mapping_violation")
    return loom::ResolvedObjectiveSourceKind::MappingViolation;
  if (*valueOrErr == "mapping_measure")
    return loom::ResolvedObjectiveSourceKind::MappingMeasure;
  return diagnostic("config_unknown_enum", key, *valueOrErr);
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

llvm::Expected<loom::ResolvedObjectiveCatalogs>
parseObjectiveCatalogs(const ConfigSyntax *node) {
  constexpr llvm::StringLiteral prefix =
      "dse.evaluation_and_objective_catalogs";
  auto fieldsOrErr = ClosedMapping::parse(
      node, prefix,
      {"evidence_obligation_templates", "objective_dimensions",
       "weighted_levels", "total_orderings"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();

  auto templatesOrErr =
      requireSequence(fieldsOrErr->at("evidence_obligation_templates"),
                      prefix + ".evidence_obligation_templates");
  if (!templatesOrErr)
    return templatesOrErr.takeError();
  if (!(*templatesOrErr)->empty())
    return diagnostic("config_owner_contract_unavailable",
                      prefix + ".evidence_obligation_templates");

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
    auto dimensionOrErr = ClosedMapping::parse(
        &entry, entryKey,
        {"source_kind", "source_ordinal", "direction", "origin", "quantum",
         "lower_index", "upper_index"});
    if (!dimensionOrErr)
      return dimensionOrErr.takeError();
    auto sourceKind = parseObjectiveSourceKind(
        dimensionOrErr->at("source_kind"), entryKey + ".source_kind");
    auto sourceOrdinal = requireU32(dimensionOrErr->at("source_ordinal"),
                                    entryKey + ".source_ordinal");
    auto direction = parseObjectiveDirection(dimensionOrErr->at("direction"),
                                             entryKey + ".direction");
    auto origin =
        requireU64(dimensionOrErr->at("origin"), entryKey + ".origin");
    auto quantum =
        requireU64(dimensionOrErr->at("quantum"), entryKey + ".quantum");
    auto lower = requireU64(dimensionOrErr->at("lower_index"),
                            entryKey + ".lower_index");
    auto upper = requireU64(dimensionOrErr->at("upper_index"),
                            entryKey + ".upper_index");
    if (!sourceKind)
      return sourceKind.takeError();
    if (!sourceOrdinal)
      return sourceOrdinal.takeError();
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
    catalogs.dimensions.push_back({*sourceKind, *sourceOrdinal, *direction,
                                   *origin, *quantum, *lower, *upper});
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
  return catalogs;
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
  if (*valueOrErr == "unrouted_obligation")
    return Kind::UnroutedObligation;
  if (*valueOrErr == "capacity_overuse")
    return Kind::CapacityOveruse;
  if (*valueOrErr == "resource_time_overbooking")
    return Kind::ResourceTimeOverbooking;
  if (*valueOrErr == "buffer_overuse")
    return Kind::BufferOveruse;
  if (*valueOrErr == "tag_unassigned")
    return Kind::TagUnassigned;
  if (*valueOrErr == "tag_conflict")
    return Kind::TagConflict;
  if (*valueOrErr == "hard_progress_violation")
    return Kind::HardProgressViolation;
  if (*valueOrErr == "hard_service_contract_shortfall")
    return Kind::HardServiceContractShortfall;
  return diagnostic("config_unknown_enum", key, *valueOrErr);
}

llvm::Expected<loom::ResolvedPnrPolicyConfig>
parsePnrPolicy(const ConfigSyntax *node, const llvm::Twine &key) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, key,
      {"search_policy", "determinism_policy", "temporary_violation_policy",
       "selected_total_ordering", "selected_search_energy",
       "focused_closure_dimensions", "evaluation_interaction_bindings"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  auto searchOrErr = ClosedMapping::parse(
      fieldsOrErr->at("search_policy"), key + ".search_policy",
      {"initializer", "action_proposal", "routing", "annealing",
       "focused_closure", "exact_repair"});
  if (!searchOrErr)
    return searchOrErr.takeError();

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
       "negotiation_policy", "route_guidance_binding"});
  if (!routingOrErr)
    return routingOrErr.takeError();
  auto endpointLimit =
      requireU64(routingOrErr->at("endpoint_expansion_limit"),
                 key + ".search_policy.routing.endpoint_expansion_limit");
  auto negotiationLimit =
      requireU64(routingOrErr->at("negotiation_iteration_limit"),
                 key + ".search_policy.routing.negotiation_iteration_limit");
  auto negotiation = parseRoutingNegotiationPolicy(
      routingOrErr->at("negotiation_policy"),
      key + ".search_policy.routing.negotiation_policy");
  if (!endpointLimit)
    return endpointLimit.takeError();
  if (!negotiationLimit)
    return negotiationLimit.takeError();
  if (!negotiation)
    return negotiation.takeError();
  std::optional<std::uint32_t> routeGuidance;
  const ConfigSyntax *guidanceNode = routingOrErr->at("route_guidance_binding");
  if (guidanceNode->kind != ConfigSyntax::Kind::Null) {
    auto guidance = requireU32(
        guidanceNode, key + ".search_policy.routing.route_guidance_binding");
    if (!guidance)
      return guidance.takeError();
    routeGuidance = *guidance;
  }

  auto annealingOrErr = ClosedMapping::parse(
      searchOrErr->at("annealing"), key + ".search_policy.annealing",
      {"calibration_proposal_count", "positive_delta_quantile",
       "target_initial_acceptance", "fallback_temperature",
       "minimum_temperature", "cooling_ratio", "proposals_per_level_base",
       "proposals_per_movable_decision"});
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
  if (!levelBase)
    return levelBase.takeError();
  if (!perMovable)
    return perMovable.takeError();

  auto focusedOrErr = ClosedMapping::parse(
      searchOrErr->at("focused_closure"),
      key + ".search_policy.focused_closure", {"proposal_limit"});
  if (!focusedOrErr)
    return focusedOrErr.takeError();
  auto focusedLimit =
      requireU64(focusedOrErr->at("proposal_limit"),
                 key + ".search_policy.focused_closure.proposal_limit");
  if (!focusedLimit)
    return focusedLimit.takeError();

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
  auto focusedDimensionsOrErr =
      requireSequence(fieldsOrErr->at("focused_closure_dimensions"),
                      key + ".focused_closure_dimensions");
  if (!focusedDimensionsOrErr)
    return focusedDimensionsOrErr.takeError();
  std::vector<std::uint32_t> focusedDimensions;
  for (const ConfigSyntax &dimensionNode : **focusedDimensionsOrErr) {
    auto dimension =
        requireU32(&dimensionNode, key + ".focused_closure_dimensions");
    if (!dimension)
      return dimension.takeError();
    focusedDimensions.push_back(*dimension);
  }

  auto bindingsOrErr =
      requireSequence(fieldsOrErr->at("evaluation_interaction_bindings"),
                      key + ".evaluation_interaction_bindings");
  if (!bindingsOrErr)
    return bindingsOrErr.takeError();
  std::vector<loom::ResolvedPnrEvaluationBindingSelection> bindings;
  for (const ConfigSyntax &bindingNode : **bindingsOrErr) {
    auto bindingOrErr = ClosedMapping::parse(
        &bindingNode, key + ".evaluation_interaction_bindings",
        {"obligation_template", "interaction_domain"});
    if (!bindingOrErr)
      return bindingOrErr.takeError();
    auto obligation = requireU32(
        bindingOrErr->at("obligation_template"),
        key + ".evaluation_interaction_bindings.obligation_template");
    auto domain =
        requireU32(bindingOrErr->at("interaction_domain"),
                   key + ".evaluation_interaction_bindings.interaction_domain");
    if (!obligation)
      return obligation.takeError();
    if (!domain)
      return domain.takeError();
    bindings.push_back({*obligation, *domain});
  }

  return loom::ResolvedPnrPolicyConfig{
      {loom::ResolvedPnrInitializerPolicy{*seeds, *assignments},
       loom::ResolvedPnrActionProposalPolicy{*realization, *transport,
                                             *resource},
       loom::ResolvedPnrRoutingPolicy{*endpointLimit, *negotiationLimit,
                                      std::move(*negotiation), routeGuidance},
       loom::ResolvedPnrAnnealingPolicy{*calibration, *quantile, *acceptance,
                                        *fallback, *minimum, *cooling,
                                        *levelBase, *perMovable},
       *focusedLimit, repair},
      loom::ResolvedPnrDeterminismPolicy{
          *masterSeed,
          loom::ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0,
          loom::ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0},
      std::move(violations),
      loom::ResolvedPnrObjectiveSelection{*selectedOrdering, *selectedEnergy,
                                          std::move(focusedDimensions)},
      std::move(bindings)};
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

llvm::Error parseTechMapping(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(node, "dse.tech_mapping", {},
                                          {"match_row_attempt_limit",
                                           "partial_cover_expansion_limit",
                                           "candidate_publication_limit"});
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
  if (llvm::Error error =
          parseLimit("candidate_publication_limit",
                     patch.techMappingCandidatePublicationLimit))
    return error;
  return llvm::Error::success();
}

llvm::Error parseDse(ConfigPatch &patch, const ConfigSyntax *node) {
  auto fieldsOrErr = ClosedMapping::parse(
      node, "dse", {},
      {"structured_ownership", "tech_mapping",
       "evaluation_and_objective_catalogs", "spatial_pnr", "system_pnr"});
  if (!fieldsOrErr)
    return fieldsOrErr.takeError();
  if (const ConfigSyntax *structured = fieldsOrErr->at("structured_ownership"))
    if (llvm::Error error = parseStructuredOwnership(patch, structured))
      return error;
  if (const ConfigSyntax *tech = fieldsOrErr->at("tech_mapping"))
    if (llvm::Error error = parseTechMapping(patch, tech))
      return error;
  if (const ConfigSyntax *catalogs =
          fieldsOrErr->at("evaluation_and_objective_catalogs")) {
    auto catalogsOrErr = parseObjectiveCatalogs(catalogs);
    if (!catalogsOrErr)
      return catalogsOrErr.takeError();
    patch.objectiveCatalogs = std::move(*catalogsOrErr);
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

llvm::Expected<std::vector<std::string>> parseIncludes(const ConfigSyntax *node,
                                                       llvm::StringRef key) {
  std::vector<std::string> includes;
  if (auto scalar = requireScalarString(node, key)) {
    includes.push_back(*scalar);
    return includes;
  } else {
    llvm::consumeError(scalar.takeError());
  }
  if (!node || node->kind != ConfigSyntax::Kind::Sequence)
    return diagnostic("config_type_mismatch", key,
                      "include must be a scalar or array of scalars");
  for (const ConfigSyntax &entry : node->sequence) {
    auto valueOrErr = requireScalarString(&entry, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    includes.push_back(*valueOrErr);
  }
  return includes;
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(const ConfigSyntax &topMap,
                            llvm::StringRef sourceName, llvm::StringRef baseDir,
                            std::set<std::string> &activeFiles);

llvm::Expected<ConfigPatch>
parseConfigFilePatch(llvm::StringRef path, std::set<std::string> &activeFiles) {
  if (activeFiles.count(path.str()) != 0)
    return diagnostic("config_parse_failed", path, "cyclic include");
  activeFiles.insert(path.str());

  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError()) {
    activeFiles.erase(path.str());
    return makeErr("config_parse_failed: " + path + ": " + ec.message());
  }
  llvm::SourceMgr sourceMgr;
  llvm::yaml::Stream stream((*bufferOrErr)->getBuffer(), sourceMgr);
  auto it = stream.begin();
  if (it == stream.end()) {
    activeFiles.erase(path.str());
    return ConfigPatch();
  }
  llvm::yaml::Node *root = it->getRoot();
  if (!root) {
    activeFiles.erase(path.str());
    return ConfigPatch();
  }
  auto syntaxOrErr = materializeSyntax(root, path);
  if (!syntaxOrErr) {
    activeFiles.erase(path.str());
    return syntaxOrErr.takeError();
  }
  ++it;
  if (it != stream.end()) {
    activeFiles.erase(path.str());
    return diagnostic("config_parse_failed", path,
                      "multiple YAML documents are not supported");
  }
  if (stream.failed()) {
    activeFiles.erase(path.str());
    return diagnostic("config_parse_failed", path);
  }
  llvm::SmallString<256> base(path);
  llvm::sys::path::remove_filename(base);
  auto patchOrErr =
      parseConfigPatchFromMapping(*syntaxOrErr, path, base.str(), activeFiles);
  activeFiles.erase(path.str());
  if (!patchOrErr)
    return patchOrErr.takeError();
  return *patchOrErr;
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(const ConfigSyntax &topMap,
                            llvm::StringRef sourceName, llvm::StringRef baseDir,
                            std::set<std::string> &activeFiles) {
  if (topMap.kind != ConfigSyntax::Kind::Mapping)
    return diagnostic("config_type_mismatch", sourceName, "top-level mapping");
  ConfigPatch included;
  ConfigPatch local;
  for (const auto &[keyStorage, value] : topMap.mapping) {
    StringRef key(keyStorage);
    if (key == "include") {
      auto includesOrErr = parseIncludes(&value, "include");
      if (!includesOrErr)
        return includesOrErr.takeError();
      for (const std::string &include : *includesOrErr) {
        llvm::SmallString<256> includePath(include);
        if (!llvm::sys::path::is_absolute(includePath)) {
          includePath = baseDir;
          llvm::sys::path::append(includePath, include);
        }
        auto includePatchOrErr =
            parseConfigFilePatch(includePath.str(), activeFiles);
        if (!includePatchOrErr)
          return includePatchOrErr.takeError();
        if (llvm::Error err = mergeSiblingPatch(included, *includePatchOrErr))
          return err;
      }
      continue;
    }
    if (key == "hardware_target") {
      auto targetOrErr = parseHardwareTarget(&value);
      if (!targetOrErr)
        return targetOrErr.takeError();
      local.hardwareTarget = std::move(*targetOrErr);
      if (llvm::Error err = touch(local, "hardware_target"))
        return err;
    } else if (key == "dse") {
      if (llvm::Error err = parseDse(local, &value))
        return err;
    } else {
      return diagnostic("config_unknown_key", key);
    }
  }

  ConfigPatch merged = included;
  for (const std::string &key : local.touchedKeys)
    merged.touchedKeys.insert(key);
  if (local.hardwareTarget)
    merged.hardwareTarget = local.hardwareTarget;
  if (local.ownershipScopeExpansionLimit)
    merged.ownershipScopeExpansionLimit = local.ownershipScopeExpansionLimit;
  if (local.techMappingMatchRowAttemptLimit)
    merged.techMappingMatchRowAttemptLimit =
        local.techMappingMatchRowAttemptLimit;
  if (local.techMappingPartialCoverExpansionLimit)
    merged.techMappingPartialCoverExpansionLimit =
        local.techMappingPartialCoverExpansionLimit;
  if (local.techMappingCandidatePublicationLimit)
    merged.techMappingCandidatePublicationLimit =
        local.techMappingCandidatePublicationLimit;
  if (local.objectiveCatalogs)
    merged.objectiveCatalogs = local.objectiveCatalogs;
  if (local.spatialPnr)
    merged.spatialPnr = local.spatialPnr;
  if (local.systemPnr)
    merged.systemPnr = local.systemPnr;
  (void)sourceName;
  return merged;
}

llvm::StringRef violationName(loom::ResolvedPnrViolationKind violation) {
  using Kind = loom::ResolvedPnrViolationKind;
  switch (violation) {
  case Kind::UnroutedObligation:
    return "unrouted_obligation";
  case Kind::CapacityOveruse:
    return "capacity_overuse";
  case Kind::ResourceTimeOverbooking:
    return "resource_time_overbooking";
  case Kind::BufferOveruse:
    return "buffer_overuse";
  case Kind::TagUnassigned:
    return "tag_unassigned";
  case Kind::TagConflict:
    return "tag_conflict";
  case Kind::HardProgressViolation:
    return "hard_progress_violation";
  case Kind::HardServiceContractShortfall:
    return "hard_service_contract_shortfall";
  }
  llvm_unreachable("all PnR violation kinds are handled");
}

llvm::json::Object ratioJson(const loom::ResolvedExactRatio &ratio) {
  return llvm::json::Object{{"numerator", ratio.numerator},
                            {"denominator", ratio.denominator}};
}

llvm::json::Object
routingNegotiationJson(const loom::ResolvedRoutingNegotiationPolicy &policy) {
  if (const auto *pathFinder =
          std::get_if<loom::ResolvedPathFinderPolicy>(&policy)) {
    return llvm::json::Object{
        {"kind", "pathfinder"},
        {"price_kernel",
         pathFinder->priceKernel ==
                 loom::ResolvedPathFinderPriceKernel::Multiplicative
             ? "multiplicative"
             : "additive"},
        {"present_pressure_initial", pathFinder->presentPressureInitial},
        {"present_pressure_growth",
         ratioJson(pathFinder->presentPressureGrowth)},
        {"history_pressure_increment", pathFinder->historyPressureIncrement}};
  }

  const auto &dual = std::get<loom::ResolvedDualSubgradientPolicy>(policy);
  llvm::json::Object direction;
  switch (dual.directionKernel) {
  case loom::ResolvedDualDirectionKernel::ProjectedSigned:
    direction = llvm::json::Object{{"kind", "projected_signed"}};
    break;
  case loom::ResolvedDualDirectionKernel::PositiveViolationOnly:
    direction = llvm::json::Object{{"kind", "positive_violation_only"}};
    break;
  case loom::ResolvedDualDirectionKernel::MomentumDeflected:
    direction = llvm::json::Object{{"kind", "momentum_deflected"},
                                   {"beta", ratioJson(*dual.momentum)}};
    break;
  }

  const loom::ResolvedDualStepSchedule &schedule = dual.stepSchedule;
  llvm::json::Object scheduleJson;
  switch (schedule.kind) {
  case loom::ResolvedDualStepScheduleKind::Constant:
    scheduleJson =
        llvm::json::Object{{"kind", "constant"}, {"step", schedule.first}};
    break;
  case loom::ResolvedDualStepScheduleKind::GeometricDecay:
    scheduleJson = llvm::json::Object{
        {"kind", "geometric_decay"},
        {"initial_step", schedule.first},
        {"minimum_step", schedule.second},
        {"decay", ratioJson({schedule.third, schedule.fourth})}};
    break;
  case loom::ResolvedDualStepScheduleKind::HarmonicDecay:
    scheduleJson = llvm::json::Object{{"kind", "harmonic_decay"},
                                      {"numerator", schedule.first},
                                      {"offset", schedule.second},
                                      {"minimum_step", schedule.third}};
    break;
  }
  return llvm::json::Object{{"kind", "dual_subgradient"},
                            {"direction_kernel", std::move(direction)},
                            {"step_schedule", std::move(scheduleJson)}};
}

llvm::json::Object pnrPolicyJson(const loom::ResolvedPnrPolicyConfig &policy) {
  const loom::ResolvedPnrSearchPolicy &search = policy.search;
  llvm::json::Array temporaryViolations;
  for (loom::ResolvedPnrViolationKind violation :
       policy.temporaryViolations.admitted)
    temporaryViolations.push_back(violationName(violation));

  llvm::json::Array focusedDimensions;
  for (std::uint32_t dimension :
       policy.objectiveSelection.focusedClosureDimensions)
    focusedDimensions.push_back(dimension);

  llvm::json::Array evaluationBindings;
  for (const loom::ResolvedPnrEvaluationBindingSelection &binding :
       policy.evaluationBindings)
    evaluationBindings.push_back(
        llvm::json::Object{{"obligation_template", binding.obligationTemplate},
                           {"interaction_domain", binding.interactionDomain}});

  llvm::json::Object exactRepair;
  if (search.exactRepair.kind == loom::ResolvedPnrExactRepairKind::Disabled) {
    exactRepair = llvm::json::Object{{"kind", "disabled"}};
  } else {
    exactRepair = llvm::json::Object{
        {"kind", "cp_sat"},
        {"max_region_decisions", search.exactRepair.maxRegionDecisions},
        {"max_solver_calls", search.exactRepair.maxSolverCalls}};
  }

  return llvm::json::Object{
      {"search_policy",
       llvm::json::Object{
           {"initializer",
            llvm::json::Object{
                {"seed_attempt_count", search.initializer.seedAttemptCount},
                {"assignment_attempt_limit_per_seed",
                 search.initializer.assignmentAttemptLimitPerSeed}}},
           {"action_proposal",
            llvm::json::Object{
                {"realization_binding_weight",
                 search.actionProposal.realizationBindingWeight},
                {"transport_routing_weight",
                 search.actionProposal.transportRoutingWeight},
                {"resource_allocation_weight",
                 search.actionProposal.resourceAllocationWeight}}},
           {"routing",
            llvm::json::Object{
                {"endpoint_expansion_limit",
                 search.routing.endpointExpansionLimit},
                {"negotiation_iteration_limit",
                 search.routing.negotiationIterationLimit},
                {"negotiation_policy",
                 routingNegotiationJson(search.routing.negotiation)},
                {"route_guidance_binding",
                 search.routing.routeGuidanceBinding
                     ? llvm::json::Value(*search.routing.routeGuidanceBinding)
                     : llvm::json::Value(nullptr)}}},
           {"annealing",
            llvm::json::Object{
                {"calibration_proposal_count",
                 search.annealing.calibrationProposalCount},
                {"positive_delta_quantile",
                 ratioJson(search.annealing.positiveDeltaQuantile)},
                {"target_initial_acceptance",
                 ratioJson(search.annealing.targetInitialAcceptance)},
                {"fallback_temperature", search.annealing.fallbackTemperature},
                {"minimum_temperature", search.annealing.minimumTemperature},
                {"cooling_ratio", ratioJson(search.annealing.coolingRatio)},
                {"proposals_per_level_base",
                 search.annealing.proposalsPerLevelBase},
                {"proposals_per_movable_decision",
                 search.annealing.proposalsPerMovableDecision}}},
           {"focused_closure",
            llvm::json::Object{
                {"proposal_limit", search.focusedClosureProposalLimit}}},
           {"exact_repair", std::move(exactRepair)}}},
      {"determinism_policy",
       llvm::json::Object{
           {"master_seed", policy.determinism.masterSeed},
           {"prng_protocol", "sha256_seeded_xoshiro256starstar_1_0"},
           {"acceptance_protocol", "exp_negative_q64_table_1_0"}}},
      {"temporary_violation_policy", std::move(temporaryViolations)},
      {"selected_total_ordering",
       policy.objectiveSelection.selectedTotalOrdering},
      {"selected_search_energy",
       policy.objectiveSelection.selectedSearchEnergy},
      {"focused_closure_dimensions", std::move(focusedDimensions)},
      {"evaluation_interaction_bindings", std::move(evaluationBindings)}};
}

llvm::StringRef objectiveSourceName(loom::ResolvedObjectiveSourceKind source) {
  switch (source) {
  case loom::ResolvedObjectiveSourceKind::MappingViolation:
    return "mapping_violation";
  case loom::ResolvedObjectiveSourceKind::MappingMeasure:
    return "mapping_measure";
  }
  llvm_unreachable("all objective source kinds are handled");
}

llvm::StringRef
objectiveDirectionName(loom::ResolvedObjectiveDirection direction) {
  return direction == loom::ResolvedObjectiveDirection::Minimize ? "minimize"
                                                                 : "maximize";
}

llvm::json::Object
objectiveCatalogsJson(const loom::ResolvedObjectiveCatalogs &catalogs) {
  llvm::json::Array dimensions;
  for (const loom::ResolvedObjectiveDimension &dimension : catalogs.dimensions)
    dimensions.push_back(llvm::json::Object{
        {"source_kind", objectiveSourceName(dimension.sourceKind)},
        {"source_ordinal", dimension.sourceOrdinal},
        {"direction", objectiveDirectionName(dimension.direction)},
        {"origin", dimension.origin},
        {"quantum", dimension.quantum},
        {"lower_index", dimension.lowerIndex},
        {"upper_index", dimension.upperIndex}});

  llvm::json::Array levels;
  for (const loom::ResolvedWeightedObjectiveLevel &level :
       catalogs.weightedLevels) {
    llvm::json::Array terms;
    for (const loom::ResolvedWeightedObjectiveTerm &term : level.terms)
      terms.push_back(llvm::json::Object{{"dimension", term.dimension},
                                         {"weight", term.weight}});
    levels.push_back(llvm::json::Object{{"terms", std::move(terms)}});
  }

  llvm::json::Array orderings;
  for (const loom::ResolvedTotalOrdering &ordering : catalogs.totalOrderings) {
    llvm::json::Array levelRefs;
    for (std::uint32_t level : ordering.weightedLevels)
      levelRefs.push_back(level);
    orderings.push_back(
        llvm::json::Object{{"weighted_levels", std::move(levelRefs)}});
  }
  return llvm::json::Object{
      {"evidence_obligation_templates", llvm::json::Array{}},
      {"objective_dimensions", std::move(dimensions)},
      {"weighted_levels", std::move(levels)},
      {"total_orderings", std::move(orderings)}};
}

llvm::json::Object
resolvedConfigJsonObject(const loom::ResolvedConfig &config) {
  const loom::adg::BuiltinTargetScale &scale = config.hardwareTarget.parameters;
  return llvm::json::Object{
      {"hardware_target",
       llvm::json::Object{
           {"template_identity", config.hardwareTarget.templateIdentity},
           {"schema_major", config.hardwareTarget.schemaVersion.major},
           {"schema_minor", config.hardwareTarget.schemaVersion.minor},
           {"parameters",
            llvm::json::Object{
                {"acc_core_count", scale.accCoreCount},
                {"spatial_pe_count", scale.spatialPeCount},
                {"temporal_pe_count", scale.temporalPeCount},
                {"spatial_memory_count", scale.spatialMemoryCount},
                {"temporal_memory_count", scale.temporalMemoryCount},
                {"temporal_resident_contexts", scale.temporalResidentContexts},
                {"gateway_count", scale.gatewayCount},
                {"memory_capacity_bytes", scale.memoryCapacityBytes}}},
       }},
      {"dse",
       llvm::json::Object{
           {"structured_ownership",
            llvm::json::Object{
                {"scope_expansion_limit",
                 static_cast<int64_t>(
                     config.dse.structuredOwnership.scopeExpansionLimit)},
            }},
           {"tech_mapping",
            llvm::json::Object{
                {"match_row_attempt_limit",
                 config.dse.techMapping.matchRowAttemptLimit},
                {"partial_cover_expansion_limit",
                 config.dse.techMapping.partialCoverExpansionLimit},
                {"candidate_publication_limit",
                 config.dse.techMapping.candidatePublicationLimit},
            }},
           {"evaluation_and_objective_catalogs",
            objectiveCatalogsJson(config.dse.objectiveCatalogs)},
           {"spatial_pnr", pnrPolicyJson(config.dse.spatialPnr)},
           {"system_pnr", pnrPolicyJson(config.dse.systemPnr)},
       }},
  };
}

llvm::Error validateResolvedConfig(const loom::ResolvedConfig &config) {
  if (config.hardwareTarget.templateIdentity.empty())
    return diagnostic("config_missing_required_profile",
                      "hardware_target.template_identity");
  const loom::adg::BuiltinTargetScale &scale = config.hardwareTarget.parameters;
  if (scale.accCoreCount == 0 || scale.spatialPeCount == 0 ||
      scale.temporalPeCount == 0 || scale.spatialMemoryCount == 0 ||
      scale.temporalMemoryCount == 0 || scale.temporalResidentContexts == 0 ||
      scale.gatewayCount == 0 || scale.memoryCapacityBytes == 0)
    return diagnostic("config_range_violation", "hardware_target.parameters",
                      "all target scale values must be positive");
  if (config.dse.structuredOwnership.scopeExpansionLimit == 0 ||
      config.dse.techMapping.matchRowAttemptLimit == 0 ||
      config.dse.techMapping.partialCoverExpansionLimit == 0 ||
      config.dse.techMapping.candidatePublicationLimit == 0)
    return diagnostic("config_range_violation", "dse",
                      "semantic work limits must be positive");
  if (llvm::Error error = loom::validateResolvedPnrPolicyConfig(
          config.dse.spatialPnr, config.dse.objectiveCatalogs))
    return error;
  return loom::validateResolvedPnrPolicyConfig(config.dse.systemPnr,
                                               config.dse.objectiveCatalogs);
}

} // namespace

loom::ResolvedConfig loom::defaultResolvedConfig() {
  ResolvedConfig config;
  config.hardwareTarget = {adg::builtinDefaultTarget.templateIdentity.str(),
                           {adg::builtinDefaultTarget.schemaMajor,
                            adg::builtinDefaultTarget.schemaMinor},
                           adg::builtinDefaultTarget.scale};
  config.dse.objectiveCatalogs = resolvedBuiltinObjectiveCatalogs();
  config.dse.spatialPnr =
      resolvedBuiltinPnrPolicy(ResolvedProfilePreset::BalancedExplore);
  config.dse.systemPnr =
      resolvedBuiltinPnrPolicy(ResolvedProfilePreset::BalancedExplore);
  llvm::cantFail(validateResolvedPnrPolicyConfig(config.dse.spatialPnr,
                                                 config.dse.objectiveCatalogs));
  llvm::cantFail(validateResolvedPnrPolicyConfig(config.dse.systemPnr,
                                                 config.dse.objectiveCatalogs));
  return config;
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

  std::set<std::string> activeFiles;
  auto patchOrErr =
      parseConfigPatchFromMapping(*syntaxOrErr, sourceName, "", activeFiles);
  if (!patchOrErr)
    return patchOrErr.takeError();

  ResolvedConfig config = defaultResolvedConfig();
  applyPatch(config, *patchOrErr);
  if (llvm::Error error = validateResolvedConfig(config))
    return std::move(error);
  return config;
}

llvm::Expected<loom::ResolvedConfig>
loom::loadResolvedConfig(llvm::StringRef path) {
  std::set<std::string> activeFiles;
  auto patchOrErr = parseConfigFilePatch(path, activeFiles);
  if (!patchOrErr)
    return patchOrErr.takeError();

  ResolvedConfig config = defaultResolvedConfig();
  applyPatch(config, *patchOrErr);
  if (llvm::Error error = validateResolvedConfig(config))
    return std::move(error);
  return config;
}

std::string
loom::canonicalResolvedConfigJson(const loom::ResolvedConfig &config) {
  return llvm::formatv("{0:2}",
                       llvm::json::Value(resolvedConfigJsonObject(config)))
      .str();
}

loom::CanonicalSemanticBytes
loom::canonicalResolvedConfigBytes(const loom::ResolvedConfig &config) {
  const std::string json = canonicalResolvedConfigJson(config);
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
}

loom::ArtifactIdentity
loom::resolvedConfigIdentity(const loom::ResolvedConfig &config) {
  return finalizeArtifactIdentity(ResolvedConfig::artifactSchema,
                                  canonicalResolvedConfigBytes(config));
}
