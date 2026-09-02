#include "DSE/HardwareMutationRepairRecord.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>
#include <utility>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral kRecordEncoding{
    "loom.dse.hardware_mutation_repair_record.1"};

llvm::Error malformed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "hardware_mutation_repair_record_invalid: " + message);
}

template <typename Owner>
using CounterMember = std::uint64_t Owner::*;

template <typename Owner> struct CounterField final {
  llvm::StringLiteral name;
  CounterMember<Owner> member;
};

constexpr CounterField<JointMappingRebaseAccounting> kAccountingFields[] = {
    {"parent_tech_mappings", &JointMappingRebaseAccounting::parentTechMappings},
    {"parent_spatial_mappings",
     &JointMappingRebaseAccounting::parentSpatialMappings},
    {"preserved_tech_mappings",
     &JointMappingRebaseAccounting::preservedTechMappings},
    {"preserved_spatial_mappings",
     &JointMappingRebaseAccounting::preservedSpatialMappings},
    {"repaired_tech_mappings",
     &JointMappingRebaseAccounting::repairedTechMappings},
    {"repaired_spatial_mappings",
     &JointMappingRebaseAccounting::repairedSpatialMappings},
    {"invalidated_tech_mappings",
     &JointMappingRebaseAccounting::invalidatedTechMappings},
    {"invalidated_spatial_mappings",
     &JointMappingRebaseAccounting::invalidatedSpatialMappings},
    {"parent_tech_decisions",
     &JointMappingRebaseAccounting::parentTechDecisions},
    {"parent_spatial_decisions",
     &JointMappingRebaseAccounting::parentSpatialDecisions},
    {"preserved_tech_decisions",
     &JointMappingRebaseAccounting::preservedTechDecisions},
    {"preserved_spatial_decisions",
     &JointMappingRebaseAccounting::preservedSpatialDecisions},
    {"reopened_tech_decisions",
     &JointMappingRebaseAccounting::reopenedTechDecisions},
    {"reopened_spatial_decisions",
     &JointMappingRebaseAccounting::reopenedSpatialDecisions},
    {"repaired_tech_decisions",
     &JointMappingRebaseAccounting::repairedTechDecisions},
    {"repaired_spatial_decisions",
     &JointMappingRebaseAccounting::repairedSpatialDecisions},
    {"invalidation_root_count",
     &JointMappingRebaseAccounting::invalidationRootCount},
    {"invalidation_cone_decision_count",
     &JointMappingRebaseAccounting::invalidationConeDecisionCount},
    {"parent_route_node_count",
     &JointMappingRebaseAccounting::parentRouteNodeCount},
    {"preserved_route_node_count",
     &JointMappingRebaseAccounting::preservedRouteNodeCount},
    {"reopened_route_node_count",
     &JointMappingRebaseAccounting::reopenedRouteNodeCount},
    {"repaired_route_node_count",
     &JointMappingRebaseAccounting::repairedRouteNodeCount},
    {"parent_service_leg_count",
     &JointMappingRebaseAccounting::parentServiceLegCount},
    {"preserved_service_leg_count",
     &JointMappingRebaseAccounting::preservedServiceLegCount},
    {"reopened_service_leg_count",
     &JointMappingRebaseAccounting::reopenedServiceLegCount},
    {"parent_thread_binding_count",
     &JointMappingRebaseAccounting::parentThreadBindingCount},
    {"preserved_thread_binding_count",
     &JointMappingRebaseAccounting::preservedThreadBindingCount},
    {"reopened_thread_binding_count",
     &JointMappingRebaseAccounting::reopenedThreadBindingCount},
    {"parent_graph_binding_count",
     &JointMappingRebaseAccounting::parentGraphBindingCount},
    {"preserved_graph_binding_count",
     &JointMappingRebaseAccounting::preservedGraphBindingCount},
    {"reopened_graph_binding_count",
     &JointMappingRebaseAccounting::reopenedGraphBindingCount},
    {"parent_resource_use_count",
     &JointMappingRebaseAccounting::parentResourceUseCount},
    {"preserved_resource_use_count",
     &JointMappingRebaseAccounting::preservedResourceUseCount},
    {"reopened_resource_use_count",
     &JointMappingRebaseAccounting::reopenedResourceUseCount},
    {"parent_service_realization_count",
     &JointMappingRebaseAccounting::parentServiceRealizationCount},
    {"preserved_service_realization_count",
     &JointMappingRebaseAccounting::preservedServiceRealizationCount},
    {"reopened_service_realization_count",
     &JointMappingRebaseAccounting::reopenedServiceRealizationCount},
};

constexpr CounterField<HardwareMutationRepairSideRecord> kSideFields[] = {
    {"tech_mapping_invocations",
     &HardwareMutationRepairSideRecord::techMappingInvocations},
    {"spatial_pnr_invocations",
     &HardwareMutationRepairSideRecord::spatialPnrInvocations},
    {"system_pnr_invocations",
     &HardwareMutationRepairSideRecord::systemPnrInvocations},
    {"tech_mapping_dispatches",
     &HardwareMutationRepairSideRecord::techMappingDispatches},
    {"spatial_pnr_dispatches",
     &HardwareMutationRepairSideRecord::spatialPnrDispatches},
    {"system_pnr_dispatches",
     &HardwareMutationRepairSideRecord::systemPnrDispatches},
    {"tech_mapping_journal_replays",
     &HardwareMutationRepairSideRecord::techMappingJournalReplays},
    {"spatial_pnr_journal_replays",
     &HardwareMutationRepairSideRecord::spatialPnrJournalReplays},
    {"system_pnr_journal_replays",
     &HardwareMutationRepairSideRecord::systemPnrJournalReplays},
    {"execution_wall_time_ns",
     &HardwareMutationRepairSideRecord::executionWallTimeNanoseconds},
};

constexpr CounterField<mapping::SystemMappingImportSessionStatistics>
    kVerifierFields[] = {
        {"import_requests",
         &mapping::SystemMappingImportSessionStatistics::importRequests},
        {"cache_hits",
         &mapping::SystemMappingImportSessionStatistics::cacheHits},
        {"cache_misses",
         &mapping::SystemMappingImportSessionStatistics::cacheMisses},
        {"unique_constructions",
         &mapping::SystemMappingImportSessionStatistics::uniqueConstructions},
        {"uncached_constructions",
         &mapping::SystemMappingImportSessionStatistics::uncachedConstructions},
        {"bytes_read",
         &mapping::SystemMappingImportSessionStatistics::bytesRead},
        {"construction_ns",
         &mapping::SystemMappingImportSessionStatistics::
             constructionNanoseconds},
        {"deterministic_work",
         &mapping::SystemMappingImportSessionStatistics::deterministicWork},
        {"retained_bytes",
         &mapping::SystemMappingImportSessionStatistics::retainedBytes},
        {"entry_count",
         &mapping::SystemMappingImportSessionStatistics::entryCount},
};

template <typename Enum, typename Spelling>
llvm::Expected<Enum> parseSpelling(llvm::StringRef spelling,
                                   std::uint8_t count, Spelling spell,
                                   const llvm::Twine &context) {
  for (std::uint8_t ordinal = 0; ordinal != count; ++ordinal) {
    const auto value = static_cast<Enum>(ordinal);
    if (spell(value) == spelling)
      return value;
  }
  return malformed(context + " has an unknown spelling '" + spelling + "'");
}

constexpr std::uint8_t kFamilyCount =
    static_cast<std::uint8_t>(HardwareMutationFamily::SystemMemoryService) + 1;
constexpr std::uint8_t kLocalityCount =
    static_cast<std::uint8_t>(HardwareMutationLocality::GlobalReopen) + 1;
constexpr std::uint8_t kImpactKindCount =
    static_cast<std::uint8_t>(HardwareMappingImpactKind::Reopen) + 1;
constexpr std::uint8_t kMappingDispositionCount =
    static_cast<std::uint8_t>(JointMappingReuseDisposition::ColdFallback) + 1;
constexpr std::uint8_t kSystemDispositionCount =
    static_cast<std::uint8_t>(JointSystemMappingReuseDisposition::ColdFallback) +
    1;
constexpr std::uint8_t kRebaseFailureCount =
    static_cast<std::uint8_t>(
        JointMappingRebaseFailureReason::SpatialRebaseRejected) +
    1;
constexpr std::uint8_t kQualityReasonCount =
    static_cast<std::uint8_t>(
        JointDesignQualityIncompleteReason::CancelledOrTimeout) +
    1;

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

void writeRoot(llvm::json::OStream &json, llvm::StringRef key,
               const ArtifactRootReference &root) {
  json.attributeObject(
      key, [&] { writeArtifactRootReferenceJsonFields(json, root); });
}

void writeRootArray(llvm::json::OStream &json, llvm::StringRef key,
                    llvm::ArrayRef<ArtifactRootReference> roots) {
  json.attributeArray(key, [&] {
    for (const ArtifactRootReference &root : roots)
      json.object([&] { writeArtifactRootReferenceJsonFields(json, root); });
  });
}

template <typename Ref>
void writeFabricRefArray(llvm::json::OStream &json, llvm::StringRef key,
                         llvm::ArrayRef<Ref> refs) {
  json.attributeArray(key, [&] {
    for (const Ref &ref : refs)
      json.value(formatArtifactLocalPayloadHex(fabric::canonicalFabricBytes(ref)));
  });
}

template <typename Owner>
void writeCounters(llvm::json::OStream &json, const Owner &owner,
                   llvm::ArrayRef<CounterField<Owner>> fields) {
  for (const CounterField<Owner> &field : fields)
    json.attribute(field.name, owner.*(field.member));
}

void writeSide(llvm::json::OStream &json,
               const HardwareMutationRepairSideRecord &side) {
  json.object([&] {
    writeRootArray(json, "mappings", side.mappings);
    writeCounters<HardwareMutationRepairSideRecord>(json, side, kSideFields);
    json.attributeObject("verification", [&] {
      writeCounters<mapping::SystemMappingImportSessionStatistics>(
          json, side.verification, kVerifierFields);
    });
  });
}

void writeImpact(llvm::json::OStream &json,
                 const HardwareMutationImpactRecord &impact) {
  json.object([&] {
    json.attributeBegin("child");
    if (impact.child)
      writeArtifactRootReferenceJson(json, *impact.child);
    else
      json.value(nullptr);
    json.attributeEnd();
    json.attribute("family", hardwareMutationFamilySpelling(impact.family));
    json.attribute("locality",
                   hardwareMutationLocalitySpelling(impact.locality));
    json.attributeObject("tech", [&] {
      json.attribute("kind",
                     hardwareMappingImpactKindSpelling(impact.tech.kind));
      writeFabricRefArray<fabric::FabricModulePhysicalOwnerRef>(
          json, "realization_roots", impact.tech.realizationRoots);
    });
    json.attributeObject("spatial", [&] {
      json.attribute("kind",
                     hardwareMappingImpactKindSpelling(impact.spatial.kind));
      writeFabricRefArray<fabric::FabricModulePhysicalOwnerRef>(
          json, "placement_roots", impact.spatial.placementRoots);
      writeFabricRefArray<fabric::FabricTransportEndpointRef>(
          json, "route_roots", impact.spatial.routeRoots);
    });
    json.attributeObject("system", [&] {
      json.attribute("kind",
                     hardwareMappingImpactKindSpelling(impact.system.kind));
      writeFabricRefArray<fabric::AccCoreOccurrenceRef>(
          json, "execution_roots", impact.system.executionRoots);
      writeFabricRefArray<fabric::InstructionCoreContextRef>(
          json, "instruction_context_roots",
          impact.system.instructionContextRoots);
      writeFabricRefArray<fabric::SystemTransportResourceRef>(
          json, "transport_roots", impact.system.transportRoots);
      writeFabricRefArray<fabric::FabricTransportEndpointRef>(
          json, "route_roots", impact.system.routeRoots);
      writeFabricRefArray<fabric::SystemServiceEndpointRef>(
          json, "service_roots", impact.system.serviceRoots);
      writeFabricRefArray<fabric::SystemMemoryServiceRef>(
          json, "memory_service_roots", impact.system.memoryServiceRoots);
      writeFabricRefArray<fabric::FabricMemoryEndpointRef>(
          json, "memory_roots", impact.system.memoryRoots);
    });
  });
}

std::string serialize(const HardwareMutationRepairRecord &record) {
  std::string text;
  llvm::raw_string_ostream output(text);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", kRecordEncoding);
    writeRoot(json, "parent_mapping", record.parentMapping);
    writeRoot(json, "parent_system", record.parentSystem);
    writeRoot(json, "child_system", record.childSystem);
    json.attributeArray("impacts", [&] {
      for (const HardwareMutationImpactRecord &impact : record.impacts)
        writeImpact(json, impact);
    });
    json.attribute(
        "mapping_reuse_disposition",
        jointMappingReuseDispositionSpelling(record.mappingReuseDisposition));
    json.attribute("system_mapping_reuse_disposition",
                   jointSystemMappingReuseDispositionSpelling(
                       record.systemMappingReuseDisposition));
    json.attributeArray("rebase_failures", [&] {
      for (const JointMappingRebaseFailure &failure : record.rebaseFailures)
        json.object([&] {
          json.attribute("reason", jointMappingRebaseFailureReasonSpelling(
                                       failure.reason));
          json.attributeBegin("parent");
          if (failure.parent)
            writeArtifactRootReferenceJson(json, *failure.parent);
          else
            json.value(nullptr);
          json.attributeEnd();
          json.attribute("diagnostic", failure.diagnostic);
        });
    });
    json.attributeObject("accounting", [&] {
      writeCounters<JointMappingRebaseAccounting>(json, record.accounting,
                                                  kAccountingFields);
    });
    json.attributeBegin("cold");
    if (record.cold)
      writeSide(json, *record.cold);
    else
      json.value(nullptr);
    json.attributeEnd();
    json.attributeBegin("incremental");
    writeSide(json, record.incremental);
    json.attributeEnd();
    json.attributeArray("quality_observations", [&] {
      for (const HardwareMutationRepairQualityObservation &observation :
           record.qualityObservations)
        json.object([&] {
          writeRoot(json, "candidate", observation.candidate);
          json.attributeArray("objective_codes", [&] {
            for (std::uint64_t code : observation.objectiveCodes)
              json.value(code);
          });
          json.attributeBegin("incomplete_reason");
          if (observation.incompleteReason)
            json.value(jointDesignQualityIncompleteReasonSpelling(
                *observation.incompleteReason));
          else
            json.value(nullptr);
          json.attributeEnd();
        });
    });
  });
  return text;
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::ArrayRef<llvm::StringRef> allowed,
                                const llvm::Twine &context) {
  for (const auto &field : object) {
    const llvm::StringRef name = field.first;
    if (!llvm::is_contained(allowed, name))
      return malformed(context + " has unknown field '" + name + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  const llvm::json::Object *result = value ? value->getAsObject() : nullptr;
  if (!result)
    return malformed(context + " field '" + key + "' must be an object");
  return result;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  const llvm::json::Array *result = value ? value->getAsArray() : nullptr;
  if (!result)
    return malformed(context + " field '" + key + "' must be an array");
  return result;
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  const auto result = value ? value->getAsString() : std::nullopt;
  if (!result)
    return malformed(context + " field '" + key + "' must be a string");
  return *result;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  const auto result = value ? value->getAsUINT64() : std::nullopt;
  if (!result)
    return malformed(context + " field '" + key +
                     "' must be an unsigned integer");
  return *result;
}

llvm::Expected<ArtifactRootReference>
parseRoot(const llvm::json::Object &object, llvm::StringRef key,
          const llvm::Twine &context) {
  auto root = requireObject(object, key, context);
  if (!root)
    return root.takeError();
  auto parsed = parseArtifactRootReferenceJson(**root);
  if (!parsed)
    return malformed(context + " field '" + key +
                     "' is invalid: " + llvm::toString(parsed.takeError()));
  return parsed;
}

llvm::Expected<std::optional<ArtifactRootReference>>
parseOptionalRoot(const llvm::json::Object &object, llvm::StringRef key,
                  const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  if (value->getAsNull())
    return std::optional<ArtifactRootReference>{};
  auto root = parseRoot(object, key, context);
  if (!root)
    return root.takeError();
  return std::optional<ArtifactRootReference>(std::move(*root));
}

llvm::Expected<std::vector<ArtifactRootReference>>
parseRootArray(const llvm::json::Object &object, llvm::StringRef key,
               const llvm::Twine &context) {
  auto values = requireArray(object, key, context);
  if (!values)
    return values.takeError();
  std::vector<ArtifactRootReference> roots;
  roots.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    const llvm::json::Object *root = value.getAsObject();
    if (!root)
      return malformed(context + " field '" + key +
                       "' must contain only objects");
    auto parsed = parseArtifactRootReferenceJson(*root);
    if (!parsed)
      return malformed(context + " field '" + key +
                       "' is invalid: " + llvm::toString(parsed.takeError()));
    roots.push_back(std::move(*parsed));
  }
  return roots;
}

template <typename Ref>
llvm::Expected<std::vector<Ref>>
parseFabricRefArray(const llvm::json::Object &object, llvm::StringRef key,
                    const llvm::Twine &context) {
  auto values = requireArray(object, key, context);
  if (!values)
    return values.takeError();
  std::vector<Ref> refs;
  std::vector<std::uint8_t> previous;
  refs.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    const auto spelling = value.getAsString();
    if (!spelling)
      return malformed(context + " field '" + key +
                       "' must contain canonical reference payloads");
    auto bytes = parseArtifactLocalPayloadHex(*spelling);
    if (!bytes)
      return malformed(context + " field '" + key + "' is invalid: " +
                       llvm::toString(bytes.takeError()));
    if (!previous.empty() && !(previous < *bytes))
      return malformed(context + " field '" + key +
                       "' is not in canonical order");
    fabric::FabricByteReader reader(*bytes);
    Ref ref{};
    if (llvm::Error error = fabric::decodeFabricRefInto(reader, ref))
      return malformed(context + " field '" + key + "' is invalid: " +
                       llvm::toString(std::move(error)));
    if (!reader.empty() || fabric::canonicalFabricBytes(ref) != *bytes)
      return malformed(context + " field '" + key +
                       "' payload is not canonical");
    previous = std::move(*bytes);
    refs.push_back(std::move(ref));
  }
  return refs;
}

template <typename Owner>
llvm::Error parseCounters(const llvm::json::Object &object, Owner &owner,
                          llvm::ArrayRef<CounterField<Owner>> fields,
                          const llvm::Twine &context) {
  for (const CounterField<Owner> &field : fields) {
    auto value = requireUnsigned(object, field.name, context);
    if (!value)
      return value.takeError();
    owner.*(field.member) = *value;
  }
  return llvm::Error::success();
}

template <typename Owner>
std::vector<llvm::StringRef>
counterNames(llvm::ArrayRef<CounterField<Owner>> fields,
             std::initializer_list<llvm::StringRef> extra) {
  std::vector<llvm::StringRef> names(extra);
  for (const CounterField<Owner> &field : fields)
    names.push_back(field.name);
  return names;
}

llvm::Expected<HardwareMutationRepairSideRecord>
parseSide(const llvm::json::Object &object, const llvm::Twine &context) {
  if (llvm::Error error = rejectUnknownFields(
          object,
          counterNames<HardwareMutationRepairSideRecord>(
              kSideFields, {"mappings", "verification"}),
          context))
    return std::move(error);
  HardwareMutationRepairSideRecord side;
  auto mappings = parseRootArray(object, "mappings", context);
  if (!mappings)
    return mappings.takeError();
  side.mappings = std::move(*mappings);
  if (llvm::Error error = parseCounters<HardwareMutationRepairSideRecord>(
          object, side, kSideFields, context))
    return std::move(error);
  auto verification = requireObject(object, "verification", context);
  if (!verification)
    return verification.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **verification,
          counterNames<mapping::SystemMappingImportSessionStatistics>(
              kVerifierFields, {}),
          context + " verification"))
    return std::move(error);
  if (llvm::Error error =
          parseCounters<mapping::SystemMappingImportSessionStatistics>(
              **verification, side.verification, kVerifierFields,
              context + " verification"))
    return std::move(error);
  return side;
}

llvm::Expected<HardwareMappingImpactKind>
parseImpactKind(const llvm::json::Object &object, const llvm::Twine &context) {
  auto spelling = requireString(object, "kind", context);
  if (!spelling)
    return spelling.takeError();
  return parseSpelling<HardwareMappingImpactKind>(
      *spelling, kImpactKindCount, hardwareMappingImpactKindSpelling,
      context + " kind");
}

llvm::Expected<HardwareMutationImpactRecord>
parseImpact(const llvm::json::Value &value, const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, {"child", "family", "locality", "tech", "spatial", "system"},
          context))
    return std::move(error);
  HardwareMutationImpactRecord impact;
  auto child = parseOptionalRoot(*object, "child", context);
  if (!child)
    return child.takeError();
  impact.child = std::move(*child);
  auto family = requireString(*object, "family", context);
  if (!family)
    return family.takeError();
  auto parsedFamily = parseSpelling<HardwareMutationFamily>(
      *family, kFamilyCount, hardwareMutationFamilySpelling,
      context + " family");
  if (!parsedFamily)
    return parsedFamily.takeError();
  impact.family = *parsedFamily;
  auto locality = requireString(*object, "locality", context);
  if (!locality)
    return locality.takeError();
  auto parsedLocality = parseSpelling<HardwareMutationLocality>(
      *locality, kLocalityCount, hardwareMutationLocalitySpelling,
      context + " locality");
  if (!parsedLocality)
    return parsedLocality.takeError();
  impact.locality = *parsedLocality;

  auto tech = requireObject(*object, "tech", context);
  if (!tech)
    return tech.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **tech, {"kind", "realization_roots"}, context + " tech"))
    return std::move(error);
  auto techKind = parseImpactKind(**tech, context + " tech");
  if (!techKind)
    return techKind.takeError();
  impact.tech.kind = *techKind;
  auto realization = parseFabricRefArray<fabric::FabricModulePhysicalOwnerRef>(
      **tech, "realization_roots", context + " tech");
  if (!realization)
    return realization.takeError();
  impact.tech.realizationRoots = std::move(*realization);

  auto spatial = requireObject(*object, "spatial", context);
  if (!spatial)
    return spatial.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **spatial, {"kind", "placement_roots", "route_roots"},
          context + " spatial"))
    return std::move(error);
  auto spatialKind = parseImpactKind(**spatial, context + " spatial");
  if (!spatialKind)
    return spatialKind.takeError();
  impact.spatial.kind = *spatialKind;
  auto placement = parseFabricRefArray<fabric::FabricModulePhysicalOwnerRef>(
      **spatial, "placement_roots", context + " spatial");
  if (!placement)
    return placement.takeError();
  impact.spatial.placementRoots = std::move(*placement);
  auto spatialRoutes = parseFabricRefArray<fabric::FabricTransportEndpointRef>(
      **spatial, "route_roots", context + " spatial");
  if (!spatialRoutes)
    return spatialRoutes.takeError();
  impact.spatial.routeRoots = std::move(*spatialRoutes);

  auto system = requireObject(*object, "system", context);
  if (!system)
    return system.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **system,
          {"kind", "execution_roots", "instruction_context_roots",
           "transport_roots", "route_roots", "service_roots",
           "memory_service_roots", "memory_roots"},
          context + " system"))
    return std::move(error);
  auto systemKind = parseImpactKind(**system, context + " system");
  if (!systemKind)
    return systemKind.takeError();
  impact.system.kind = *systemKind;
  auto execution = parseFabricRefArray<fabric::AccCoreOccurrenceRef>(
      **system, "execution_roots", context + " system");
  if (!execution)
    return execution.takeError();
  impact.system.executionRoots = std::move(*execution);
  auto instructionContexts =
      parseFabricRefArray<fabric::InstructionCoreContextRef>(
          **system, "instruction_context_roots", context + " system");
  if (!instructionContexts)
    return instructionContexts.takeError();
  impact.system.instructionContextRoots = std::move(*instructionContexts);
  auto transport = parseFabricRefArray<fabric::SystemTransportResourceRef>(
      **system, "transport_roots", context + " system");
  if (!transport)
    return transport.takeError();
  impact.system.transportRoots = std::move(*transport);
  auto systemRoutes = parseFabricRefArray<fabric::FabricTransportEndpointRef>(
      **system, "route_roots", context + " system");
  if (!systemRoutes)
    return systemRoutes.takeError();
  impact.system.routeRoots = std::move(*systemRoutes);
  auto services = parseFabricRefArray<fabric::SystemServiceEndpointRef>(
      **system, "service_roots", context + " system");
  if (!services)
    return services.takeError();
  impact.system.serviceRoots = std::move(*services);
  auto memoryServices = parseFabricRefArray<fabric::SystemMemoryServiceRef>(
      **system, "memory_service_roots", context + " system");
  if (!memoryServices)
    return memoryServices.takeError();
  impact.system.memoryServiceRoots = std::move(*memoryServices);
  auto memories = parseFabricRefArray<fabric::FabricMemoryEndpointRef>(
      **system, "memory_roots", context + " system");
  if (!memories)
    return memories.takeError();
  impact.system.memoryRoots = std::move(*memories);
  return impact;
}

llvm::Expected<HardwareMutationRepairRecord> parse(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return malformed("record is not JSON: " +
                     llvm::toString(value.takeError()));
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return malformed("record root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object,
          {"schema", "parent_mapping", "parent_system", "child_system",
           "impacts", "mapping_reuse_disposition",
           "system_mapping_reuse_disposition", "rebase_failures",
           "accounting", "cold", "incremental", "quality_observations"},
          "record"))
    return std::move(error);
  auto schema = requireString(*object, "schema", "record");
  if (!schema)
    return schema.takeError();
  if (*schema != kRecordEncoding)
    return malformed("record has an unknown encoding");
  auto parentMapping = parseRoot(*object, "parent_mapping", "record");
  if (!parentMapping)
    return parentMapping.takeError();
  auto parentSystem = parseRoot(*object, "parent_system", "record");
  if (!parentSystem)
    return parentSystem.takeError();
  auto childSystem = parseRoot(*object, "child_system", "record");
  if (!childSystem)
    return childSystem.takeError();
  HardwareMutationRepairRecord record{std::move(*parentMapping),
                                      std::move(*parentSystem),
                                      std::move(*childSystem),
                                      {},
                                      JointMappingReuseDisposition::ColdFallback,
                                      JointSystemMappingReuseDisposition::
                                          ColdFallback,
                                      {},
                                      {},
                                      std::nullopt,
                                      {},
                                      {}};
  auto impacts = requireArray(*object, "impacts", "record");
  if (!impacts)
    return impacts.takeError();
  if ((*impacts)->empty())
    return malformed("record has no impact component");
  for (const auto indexed : llvm::enumerate(**impacts)) {
    auto impact = parseImpact(indexed.value(), llvm::Twine("record impact ") +
                                                   llvm::Twine(indexed.index()));
    if (!impact)
      return impact.takeError();
    record.impacts.push_back(std::move(*impact));
  }
  auto mappingDisposition =
      requireString(*object, "mapping_reuse_disposition", "record");
  if (!mappingDisposition)
    return mappingDisposition.takeError();
  auto parsedMappingDisposition = parseSpelling<JointMappingReuseDisposition>(
      *mappingDisposition, kMappingDispositionCount,
      jointMappingReuseDispositionSpelling,
      "record mapping_reuse_disposition");
  if (!parsedMappingDisposition)
    return parsedMappingDisposition.takeError();
  record.mappingReuseDisposition = *parsedMappingDisposition;
  auto systemDisposition =
      requireString(*object, "system_mapping_reuse_disposition", "record");
  if (!systemDisposition)
    return systemDisposition.takeError();
  auto parsedSystemDisposition =
      parseSpelling<JointSystemMappingReuseDisposition>(
          *systemDisposition, kSystemDispositionCount,
          jointSystemMappingReuseDispositionSpelling,
          "record system_mapping_reuse_disposition");
  if (!parsedSystemDisposition)
    return parsedSystemDisposition.takeError();
  record.systemMappingReuseDisposition = *parsedSystemDisposition;

  auto failures = requireArray(*object, "rebase_failures", "record");
  if (!failures)
    return failures.takeError();
  for (const auto indexed : llvm::enumerate(**failures)) {
    const llvm::Twine context =
        llvm::Twine("record rebase failure ") + llvm::Twine(indexed.index());
    const llvm::json::Object *failure = indexed.value().getAsObject();
    if (!failure)
      return malformed(context + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *failure, {"reason", "parent", "diagnostic"}, context))
      return std::move(error);
    auto reason = requireString(*failure, "reason", context);
    if (!reason)
      return reason.takeError();
    auto parsedReason = parseSpelling<JointMappingRebaseFailureReason>(
        *reason, kRebaseFailureCount, jointMappingRebaseFailureReasonSpelling,
        context + " reason");
    if (!parsedReason)
      return parsedReason.takeError();
    auto parent = parseOptionalRoot(*failure, "parent", context);
    if (!parent)
      return parent.takeError();
    auto diagnostic = requireString(*failure, "diagnostic", context);
    if (!diagnostic)
      return diagnostic.takeError();
    record.rebaseFailures.push_back(
        {*parsedReason, std::move(*parent), diagnostic->str()});
  }

  auto accounting = requireObject(*object, "accounting", "record");
  if (!accounting)
    return accounting.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **accounting,
          counterNames<JointMappingRebaseAccounting>(kAccountingFields, {}),
          "record accounting"))
    return std::move(error);
  if (llvm::Error error = parseCounters<JointMappingRebaseAccounting>(
          **accounting, record.accounting, kAccountingFields,
          "record accounting"))
    return std::move(error);

  const llvm::json::Value *cold = object->get("cold");
  if (!cold)
    return malformed("record field 'cold' is required");
  if (!cold->getAsNull()) {
    const llvm::json::Object *coldObject = cold->getAsObject();
    if (!coldObject)
      return malformed("record cold side must be null or an object");
    auto side = parseSide(*coldObject, "record cold");
    if (!side)
      return side.takeError();
    record.cold = std::move(*side);
  }
  auto incremental = requireObject(*object, "incremental", "record");
  if (!incremental)
    return incremental.takeError();
  auto incrementalSide = parseSide(**incremental, "record incremental");
  if (!incrementalSide)
    return incrementalSide.takeError();
  record.incremental = std::move(*incrementalSide);

  auto observations = requireArray(*object, "quality_observations", "record");
  if (!observations)
    return observations.takeError();
  for (const auto indexed : llvm::enumerate(**observations)) {
    const llvm::Twine context = llvm::Twine("record quality observation ") +
                                llvm::Twine(indexed.index());
    const llvm::json::Object *observation = indexed.value().getAsObject();
    if (!observation)
      return malformed(context + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *observation, {"candidate", "objective_codes", "incomplete_reason"},
            context))
      return std::move(error);
    auto candidate = parseRoot(*observation, "candidate", context);
    if (!candidate)
      return candidate.takeError();
    HardwareMutationRepairQualityObservation parsed{std::move(*candidate),
                                                    {},
                                                    std::nullopt};
    auto codes = requireArray(*observation, "objective_codes", context);
    if (!codes)
      return codes.takeError();
    for (const llvm::json::Value &code : **codes) {
      const auto value = code.getAsUINT64();
      if (!value)
        return malformed(context + " objective codes must be unsigned");
      parsed.objectiveCodes.push_back(*value);
    }
    const llvm::json::Value *reason = observation->get("incomplete_reason");
    if (!reason)
      return malformed(context + " field 'incomplete_reason' is required");
    if (!reason->getAsNull()) {
      const auto spelling = reason->getAsString();
      if (!spelling)
        return malformed(context + " incomplete_reason must be null or a "
                                   "string");
      auto parsedReason = parseSpelling<JointDesignQualityIncompleteReason>(
          *spelling, kQualityReasonCount,
          jointDesignQualityIncompleteReasonSpelling,
          context + " incomplete_reason");
      if (!parsedReason)
        return parsedReason.takeError();
      parsed.incompleteReason = *parsedReason;
    }
    record.qualityObservations.push_back(std::move(parsed));
  }
  return record;
}

HardwareMutationRepairSideRecord
projectSide(llvm::ArrayRef<ArtifactRootReference> mappings,
            const JointDesignExecutionSummary &summary,
            const mapping::SystemMappingImportSessionStatistics &verification) {
  HardwareMutationRepairSideRecord side;
  side.mappings.assign(mappings.begin(), mappings.end());
  side.techMappingInvocations = summary.techMappingInvocationCount;
  side.spatialPnrInvocations = summary.spatialPnrInvocationCount;
  side.systemPnrInvocations = summary.systemPnrInvocationCount;
  side.techMappingDispatches = summary.techMappingDispatchCount;
  side.spatialPnrDispatches = summary.spatialPnrDispatchCount;
  side.systemPnrDispatches = summary.systemPnrDispatchCount;
  side.techMappingJournalReplays = summary.techMappingJournalReplayCount;
  side.spatialPnrJournalReplays = summary.spatialPnrJournalReplayCount;
  side.systemPnrJournalReplays = summary.systemPnrJournalReplayCount;
  side.executionWallTimeNanoseconds = summary.executionWallTimeNanoseconds;
  side.verification = verification;
  return side;
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

} // namespace

llvm::Expected<FinalizedHardwareMutationRepairRecord>
publishHardwareMutationRepairRecord(const JointHardwareMutationRepair &repair,
                                    const ArtifactStore &artifacts) {
  if (repair.child.impacts.empty())
    return malformed("repair has no typed impact component");
  HardwareMutationRepairRecord record{repair.parentMapping,
                                      repair.child.impacts.front().parent,
                                      repair.child.system,
                                      {},
                                      repair.rebase.disposition,
                                      repair.systemDisposition,
                                      {},
                                      {},
                                      std::nullopt,
                                      {},
                                      {}};
  for (const HardwareImpactProjection &impact : repair.child.impacts)
    record.impacts.push_back({impact.child, impact.family, impact.locality,
                              impact.tech, impact.spatial, impact.system});
  record.rebaseFailures = repair.rebase.failures;
  record.accounting = repair.rebase.accounting;
  if (repair.coldExecution)
    record.cold = projectSide(repair.coldMappings,
                              repair.coldExecution->summary,
                              repair.coldVerification);
  record.incremental =
      projectSide(repair.incrementalMappings,
                  repair.incrementalExecution.summary,
                  repair.incrementalVerification);
  for (const JointDesignQualityObservation &observation :
       repair.incrementalExecution.summary.qualityObservations)
    record.qualityObservations.push_back({observation.candidate,
                                          observation.objectiveCodes,
                                          observation.incompleteReason});
  const std::string text = serialize(record);
  auto parsed = parse(text);
  if (!parsed)
    return parsed.takeError();
  if (serialize(*parsed) != text)
    return malformed("record failed its canonical roundtrip");
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
  auto identity = artifacts.put(hardwareMutationRepairRecordSchema, bytes);
  if (!identity)
    return identity.takeError();
  if (*identity !=
      finalizeArtifactIdentity(hardwareMutationRepairRecordSchema, bytes))
    return malformed("ArtifactStore returned a foreign record identity");
  return importHardwareMutationRepairRecord(
      {hardwareMutationRepairRecordSchema.identity.str(),
       hardwareMutationRepairRecordSchema.version, *identity},
      artifacts);
}

llvm::Expected<FinalizedHardwareMutationRepairRecord>
importHardwareMutationRepairRecord(const ArtifactRootReference &reference,
                                   const ArtifactStore &artifacts) {
  if (reference.schemaIdentity != hardwareMutationRepairRecordSchema.identity ||
      reference.schemaVersion != hardwareMutationRepairRecordSchema.version)
    return malformed("foreign hardware mutation repair record schema");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  const llvm::StringRef text = asText(bytes->bytes());
  auto record = parse(text);
  if (!record)
    return record.takeError();
  if (serialize(*record) != text)
    return malformed("stored hardware mutation repair record is not "
                     "canonical");
  if (finalizeArtifactIdentity(hardwareMutationRepairRecordSchema, *bytes) !=
      reference.artifact)
    return malformed("hardware mutation repair record identity changed on "
                     "import");
  return FinalizedHardwareMutationRepairRecord(reference, std::move(*record),
                                               std::move(*bytes));
}

} // namespace loom::dse
