#include "DSE/HardwareMutationRepairRecord.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral kRecordEncoding{
    "loom.dse.hardware_mutation_repair_record.2"};

llvm::Error malformed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "hardware_mutation_repair_record_invalid: " + message);
}

template <typename Owner> using CounterMember = std::uint64_t Owner::*;

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
        {"construction_ns", &mapping::SystemMappingImportSessionStatistics::
                                constructionNanoseconds},
        {"deterministic_work",
         &mapping::SystemMappingImportSessionStatistics::deterministicWork},
        {"retained_bytes",
         &mapping::SystemMappingImportSessionStatistics::retainedBytes},
        {"entry_count",
         &mapping::SystemMappingImportSessionStatistics::entryCount},
};

template <typename Enum, typename Spelling>
llvm::Expected<Enum> parseSpelling(llvm::StringRef spelling, std::uint8_t count,
                                   Spelling spell, const llvm::Twine &context) {
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
    static_cast<std::uint8_t>(
        JointSystemMappingReuseDisposition::ColdFallback) +
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
      json.value(
          formatArtifactLocalPayloadHex(fabric::canonicalFabricBytes(ref)));
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

void writeDecisionLineage(llvm::json::OStream &json,
                          const HardwareMutationDecisionLineage &lineage) {
  json.object([&] {
    json.attribute("owner_kind", lineage.owner.ordinal());
    writeRoot(json, "output", lineage.output);
    writeRootArray(json, "parents", lineage.parents);
    json.attribute("owner_payload",
                   formatArtifactLocalPayloadHex(lineage.ownerPayload));
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
    json.attributeArray("decision_lineage", [&] {
      for (const HardwareMutationDecisionLineage &lineage :
           record.decisionLineage)
        writeDecisionLineage(json, lineage);
    });
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

llvm::Expected<HardwareMutationDecisionLineage>
parseDecisionLineage(const llvm::json::Value &value,
                     const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, {"owner_kind", "output", "parents", "owner_payload"},
          context))
    return std::move(error);
  auto owner = requireUnsigned(*object, "owner_kind", context);
  if (!owner)
    return owner.takeError();
  if (*owner > std::numeric_limits<std::uint32_t>::max())
    return malformed(context + " owner kind exceeds u32");
  auto output = parseRoot(*object, "output", context);
  if (!output)
    return output.takeError();
  auto parents = parseRootArray(*object, "parents", context);
  if (!parents)
    return parents.takeError();
  auto payload = requireString(*object, "owner_payload", context);
  if (!payload)
    return payload.takeError();
  auto bytes = parseArtifactLocalPayloadHex(*payload);
  if (!bytes)
    return malformed(context + " owner payload is invalid: " +
                     llvm::toString(bytes.takeError()));
  return HardwareMutationDecisionLineage{
      CandidateGeneratorKind(static_cast<std::uint32_t>(*owner)),
      std::move(*output), std::move(*parents), std::move(*bytes)};
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
      return malformed(context + " field '" + key +
                       "' is invalid: " + llvm::toString(bytes.takeError()));
    if (!previous.empty() && !(previous < *bytes))
      return malformed(context + " field '" + key +
                       "' is not in canonical order");
    fabric::FabricByteReader reader(*bytes);
    Ref ref{};
    if (llvm::Error error = fabric::decodeFabricRefInto(reader, ref))
      return malformed(context + " field '" + key +
                       "' is invalid: " + llvm::toString(std::move(error)));
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
  if (llvm::Error error =
          rejectUnknownFields(object,
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

struct PersistedHardwareMutationImpactRecord final {
  std::optional<ArtifactRootReference> child;
  HardwareMutationFamily family = HardwareMutationFamily::SpatialTopology;
  HardwareMutationLocality locality = HardwareMutationLocality::Unchanged;
  TechMappingImpactProjection tech;
  SpatialMappingImpactProjection spatial;
  SystemMappingImpactProjection system;
};

llvm::Expected<PersistedHardwareMutationImpactRecord>
parseImpact(const llvm::json::Value &value, const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, {"child", "family", "locality", "tech", "spatial", "system"},
          context))
    return std::move(error);
  PersistedHardwareMutationImpactRecord impact;
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

struct ParsedHardwareMutationRepairRecord final {
  HardwareMutationRepairRecord record;
  std::vector<PersistedHardwareMutationImpactRecord> impacts;
};

llvm::Expected<ParsedHardwareMutationRepairRecord> parse(llvm::StringRef text) {
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
           "decision_lineage", "impacts", "mapping_reuse_disposition",
           "system_mapping_reuse_disposition", "rebase_failures", "accounting",
           "cold", "incremental", "quality_observations"},
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
  HardwareMutationRepairRecord record{
      std::move(*parentMapping),
      std::move(*parentSystem),
      std::move(*childSystem),
      {},
      {},
      JointMappingReuseDisposition::ColdFallback,
      JointSystemMappingReuseDisposition::ColdFallback,
      {},
      {},
      std::nullopt,
      {},
      {}};
  auto decisionLineage = requireArray(*object, "decision_lineage", "record");
  if (!decisionLineage)
    return decisionLineage.takeError();
  if ((*decisionLineage)->empty())
    return malformed("record has no canonical hardware decision lineage");
  for (const auto indexed : llvm::enumerate(**decisionLineage)) {
    auto lineage = parseDecisionLineage(
        indexed.value(), llvm::Twine("record decision lineage ") +
                             llvm::Twine(indexed.index()));
    if (!lineage)
      return lineage.takeError();
    record.decisionLineage.push_back(std::move(*lineage));
  }
  auto impacts = requireArray(*object, "impacts", "record");
  if (!impacts)
    return impacts.takeError();
  if ((*impacts)->empty())
    return malformed("record has no impact component");
  std::vector<PersistedHardwareMutationImpactRecord> parsedImpacts;
  parsedImpacts.reserve((*impacts)->size());
  for (const auto indexed : llvm::enumerate(**impacts)) {
    auto impact =
        parseImpact(indexed.value(), llvm::Twine("record impact ") +
                                         llvm::Twine(indexed.index()));
    if (!impact)
      return impact.takeError();
    parsedImpacts.push_back(std::move(*impact));
  }
  auto mappingDisposition =
      requireString(*object, "mapping_reuse_disposition", "record");
  if (!mappingDisposition)
    return mappingDisposition.takeError();
  auto parsedMappingDisposition = parseSpelling<JointMappingReuseDisposition>(
      *mappingDisposition, kMappingDispositionCount,
      jointMappingReuseDispositionSpelling, "record mapping_reuse_disposition");
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
  std::optional<ArtifactRootReference> previousQualityCandidate;
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
    HardwareMutationRepairQualityObservation parsed{
        std::move(*candidate), {}, std::nullopt};
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
    if (parsed.incompleteReason && !parsed.objectiveCodes.empty())
      return malformed(context + " incomplete observation has objective codes");
    if (previousQualityCandidate &&
        !artifactRootReferenceLess(*previousQualityCandidate,
                                   parsed.candidate))
      return malformed(
          "record quality observations are not canonical and unique");
    previousQualityCandidate = parsed.candidate;
    record.qualityObservations.push_back(std::move(parsed));
  }
  return ParsedHardwareMutationRepairRecord{std::move(record),
                                             std::move(parsedImpacts)};
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

bool rootsAreCanonical(llvm::ArrayRef<ArtifactRootReference> roots) {
  return llvm::is_sorted(roots, artifactRootReferenceLess) &&
         std::adjacent_find(roots.begin(), roots.end()) == roots.end();
}

struct ProjectedHardwareDecisionLineage final {
  HardwareImpactProjection impact;
  bool systemDecision = false;
  std::optional<ArtifactRootReference> replacementModule;
  std::optional<fabric::AccCoreOccurrenceRef> replacementTarget;
};

llvm::Expected<ProjectedHardwareDecisionLineage>
validateAndProjectDecisionLineage(
    const HardwareMutationDecisionLineage &lineage,
    const ArtifactStore &artifacts) {
  const CandidateGeneratorDescriptor *descriptor = nullptr;
  if (lineage.owner == spatialTopologyCandidateGeneratorKind)
    descriptor = &spatialTopologyCandidateGeneratorDescriptor();
  else if (lineage.owner == spatialMicroarchitectureCandidateGeneratorKind)
    descriptor = &spatialMicroarchitectureCandidateGeneratorDescriptor();
  else if (lineage.owner == systemCompositionCandidateGeneratorKind)
    descriptor = &systemCompositionCandidateGeneratorDescriptor();
  else
    return malformed("decision lineage has a non-hardware owner");
  if (lineage.parents.size() != 1 ||
      lineage.parents.front() == lineage.output)
    return malformed("decision lineage is not one exact parent-to-child edge");
  auto storedOutput = fabric::importEntireFabricRoot(lineage.output, artifacts);
  if (!storedOutput)
    return malformed("decision lineage output failed strict import: " +
                     llvm::toString(storedOutput.takeError()));
  if (!descriptor->ownerLineagePayload ||
      !descriptor->ownerLineagePayload->validateCanonical)
    return malformed("decision lineage owner has no canonical payload contract");
  if (llvm::Error error =
          descriptor->ownerLineagePayload->validateCanonical(
              lineage.ownerPayload, lineage.output, lineage.parents,
              artifacts))
    return malformed("decision lineage failed its canonical owner: " +
                     llvm::toString(std::move(error)));

  if (lineage.owner == spatialTopologyCandidateGeneratorKind) {
    auto decision = adoptSpatialTopologyDecision(lineage.ownerPayload);
    if (!decision)
      return decision.takeError();
    return ProjectedHardwareDecisionLineage{
        projectHardwareImpact(*decision, lineage.output), false, std::nullopt,
        std::nullopt};
  }
  if (lineage.owner == spatialMicroarchitectureCandidateGeneratorKind) {
    auto decision =
        adoptSpatialMicroarchitectureDecision(lineage.ownerPayload);
    if (!decision)
      return decision.takeError();
    return ProjectedHardwareDecisionLineage{
        projectHardwareImpact(*decision, lineage.output), false, std::nullopt,
        std::nullopt};
  }

  auto decision = adoptSystemCompositionDecision(lineage.ownerPayload);
  if (!decision)
    return decision.takeError();
  std::optional<ArtifactRootReference> replacementModule;
  std::optional<fabric::AccCoreOccurrenceRef> replacementTarget;
  if (const auto *replacement =
          std::get_if<ReplaceSpatialAttachment>(&decision->decision)) {
    replacementModule = replacement->module;
    replacementTarget = replacement->target;
  }
  return ProjectedHardwareDecisionLineage{
      projectHardwareImpact(*decision, lineage.output), true,
      std::move(replacementModule), replacementTarget};
}

llvm::Expected<ArtifactRootReference> systemAttachmentModule(
    const ArtifactRootReference &systemReference,
    fabric::AccCoreOccurrenceRef target, const ArtifactStore &artifacts) {
  auto root = fabric::importEntireFabricRoot(systemReference, artifacts);
  if (!root)
    return root.takeError();
  auto system = fabric::requireSystemRoot(root->view());
  if (!system)
    return system.takeError();
  const auto attachment = system->spatialCoreTarget(target);
  if (!attachment ||
      attachment->dependencyOrdinal >= root->directDependencies().size())
    return malformed("System decision target has no exact Module attachment");
  return root->directDependencies()[attachment->dependencyOrdinal].root;
}

llvm::Expected<bool> systemHasAttachmentToModule(
    const ArtifactRootReference &systemReference,
    const ArtifactRootReference &module, const ArtifactStore &artifacts) {
  auto root = fabric::importEntireFabricRoot(systemReference, artifacts);
  if (!root)
    return root.takeError();
  auto system = fabric::requireSystemRoot(root->view());
  if (!system)
    return system.takeError();
  for (const fabric::AccCoreOccurrenceRef core :
       system->artifact().accCoreOccurrences()) {
    const auto attachment = system->spatialCoreTarget(core);
    if (!attachment ||
        attachment->dependencyOrdinal >= root->directDependencies().size())
      return malformed("System has an AccCore without an exact Module target");
    if (root->directDependencies()[attachment->dependencyOrdinal].root == module)
      return true;
  }
  return false;
}

bool samePersistedImpact(const PersistedHardwareMutationImpactRecord &stored,
                         const HardwareImpactProjection &derived) {
  return stored.child == derived.child && stored.family == derived.family &&
         stored.locality == derived.locality &&
         stored.tech.kind == derived.tech.kind &&
         stored.tech.realizationRoots == derived.tech.realizationRoots &&
         stored.spatial.kind == derived.spatial.kind &&
         stored.spatial.placementRoots == derived.spatial.placementRoots &&
         stored.spatial.routeRoots == derived.spatial.routeRoots &&
         stored.system.kind == derived.system.kind &&
         stored.system.executionRoots == derived.system.executionRoots &&
         stored.system.instructionContextRoots ==
             derived.system.instructionContextRoots &&
         stored.system.transportRoots == derived.system.transportRoots &&
         stored.system.routeRoots == derived.system.routeRoots &&
         stored.system.serviceRoots == derived.system.serviceRoots &&
         stored.system.memoryServiceRoots ==
             derived.system.memoryServiceRoots &&
         stored.system.memoryRoots == derived.system.memoryRoots;
}

llvm::Expected<std::vector<HardwareImpactProjection>>
validateDecisionLineage(HardwareMutationRepairRecord &record,
                        llvm::ArrayRef<PersistedHardwareMutationImpactRecord>
                            persistedImpacts,
                        const ArtifactStore &artifacts) {
  if (record.decisionLineage.empty())
    return malformed("record has no canonical hardware decision lineage");
  std::vector<ProjectedHardwareDecisionLineage> projected;
  projected.reserve(record.decisionLineage.size());
  for (const HardwareMutationDecisionLineage &lineage :
       record.decisionLineage) {
    auto decision = validateAndProjectDecisionLineage(lineage, artifacts);
    if (!decision)
      return decision.takeError();
    projected.push_back(std::move(*decision));
  }

  ArtifactRootReference currentSystem = record.parentSystem;
  std::vector<HardwareImpactProjection> derivedImpacts;
  for (std::size_t ordinal = 0; ordinal != projected.size();) {
    ProjectedHardwareDecisionLineage &decision = projected[ordinal];
    const HardwareMutationDecisionLineage &lineage =
        record.decisionLineage[ordinal];
    if (decision.systemDecision) {
      if (lineage.parents.front() != currentSystem)
        return malformed("System decision lineage is not consecutive");
      currentSystem = lineage.output;
      derivedImpacts.push_back(std::move(decision.impact));
      ++ordinal;
      continue;
    }

    auto currentModules =
        projectJointDesignTargetModules(currentSystem, artifacts);
    if (!currentModules)
      return currentModules.takeError();
    if (!llvm::is_contained(*currentModules, lineage.parents.front()))
      return malformed(
          "Module decision lineage is outside its current System");
    const ArtifactRootReference moduleChild = lineage.output;
    HardwareImpactProjection moduleImpact = std::move(decision.impact);
    ++ordinal;
    bool replacedAttachment = false;
    while (ordinal != projected.size() &&
           projected[ordinal].systemDecision &&
           projected[ordinal].replacementModule == moduleChild) {
      if (!projected[ordinal].replacementTarget)
        return malformed(
            "Module attachment lineage has no exact replacement target");
      const HardwareMutationDecisionLineage &replacement =
          record.decisionLineage[ordinal];
      if (replacement.parents.front() != currentSystem)
        return malformed(
            "Module attachment decision lineage is not consecutive");
      auto attachedModule = systemAttachmentModule(
          currentSystem, *projected[ordinal].replacementTarget, artifacts);
      if (!attachedModule)
        return attachedModule.takeError();
      if (*attachedModule != moduleImpact.parent)
        return malformed(
            "Module attachment decision targets an unrelated parent Module");
      currentSystem = replacement.output;
      replacedAttachment = true;
      ++ordinal;
    }
    if (!replacedAttachment)
      return malformed(
          "Module decision lineage has no owning System replacement");
    auto remainingAttachment = systemHasAttachmentToModule(
        currentSystem, moduleImpact.parent, artifacts);
    if (!remainingAttachment)
      return remainingAttachment.takeError();
    if (*remainingAttachment)
      return malformed(
          "Module decision lineage leaves a parent Module attachment active");
    moduleImpact.child = currentSystem;
    derivedImpacts.push_back(std::move(moduleImpact));
  }
  if (currentSystem != record.childSystem)
    return malformed("decision lineage does not end at the record child");
  if (persistedImpacts.size() != derivedImpacts.size())
    return malformed("record impact inventory differs from decision lineage");
  for (const auto indexed : llvm::enumerate(derivedImpacts)) {
    const PersistedHardwareMutationImpactRecord &stored =
        persistedImpacts[indexed.index()];
    HardwareImpactProjection &derived = indexed.value();
    if (!samePersistedImpact(stored, derived))
      return malformed("record impact differs from its canonical decision");
    record.impacts.push_back(
        {derived.parent, derived.child, derived.moduleEntities, derived.family,
         derived.locality, derived.tech, derived.spatial, derived.system});
  }
  return derivedImpacts;
}

llvm::Error validateProviderWork(
    const HardwareMutationRepairSideRecord &side,
    const llvm::Twine &context) {
  const auto tech = llvm::checkedAddUnsigned(side.techMappingDispatches,
                                             side.techMappingJournalReplays);
  const auto spatial = llvm::checkedAddUnsigned(
      side.spatialPnrDispatches, side.spatialPnrJournalReplays);
  const auto system = llvm::checkedAddUnsigned(
      side.systemPnrDispatches, side.systemPnrJournalReplays);
  if (!tech || *tech != side.techMappingInvocations || !spatial ||
      *spatial != side.spatialPnrInvocations || !system ||
      *system != side.systemPnrInvocations)
    return malformed(context + " provider accounting is not closed");
  return llvm::Error::success();
}

bool sameDeterministicVerification(
    const mapping::SystemMappingImportSessionStatistics &lhs,
    const mapping::SystemMappingImportSessionStatistics &rhs) {
  return lhs.importRequests == rhs.importRequests &&
         lhs.cacheHits == rhs.cacheHits &&
         lhs.cacheMisses == rhs.cacheMisses &&
         lhs.uniqueConstructions == rhs.uniqueConstructions &&
         lhs.uncachedConstructions == rhs.uncachedConstructions &&
         lhs.bytesRead == rhs.bytesRead &&
         lhs.deterministicWork == rhs.deterministicWork &&
         lhs.retainedBytes == rhs.retainedBytes &&
         lhs.entryCount == rhs.entryCount;
}

llvm::Expected<mapping::SystemMappingImportSessionStatistics>
verifyMappingSide(
    const HardwareMutationRepairSideRecord &side,
    const mapping::FinalizedSystemMapping &parentMapping,
    const ArtifactRootReference &childSystem, const ArtifactStore &artifacts,
    const llvm::Twine &context) {
  if (!rootsAreCanonical(side.mappings))
    return malformed(context + " Mapping roots are not canonical and unique");
  if (llvm::Error error = validateProviderWork(side, context))
    return std::move(error);

  mapping::SystemMappingImportSession importSession(
      artifacts, std::max<std::size_t>(1, side.mappings.size()),
      mapping::SystemMappingImportSessionMode::New);
  for (const ArtifactRootReference &mappingReference : side.mappings) {
    auto imported = mapping::importSystemMapping(mappingReference, artifacts);
    if (!imported)
      return imported.takeError();
    if (imported->view().dataflowIdentity() !=
            parentMapping.view().dataflowIdentity() ||
        imported->view().fabricIdentity() != childSystem.artifact)
      return malformed(context + " Mapping has foreign owners");
  }
  return importSession.statistics();
}

llvm::Error validateReuseDispositions(
    const HardwareMutationRepairRecord &record) {
  const JointMappingRebaseAccounting &accounting = record.accounting;
  switch (record.mappingReuseDisposition) {
  case JointMappingReuseDisposition::Preserved:
    if (accounting.invalidatedTechMappings != 0 ||
        accounting.invalidatedSpatialMappings != 0)
      return malformed(
          "preserved Mapping disposition has invalidated lower Mappings");
    break;
  case JointMappingReuseDisposition::LocalRepair:
    if ((accounting.invalidatedTechMappings == 0 &&
         accounting.invalidatedSpatialMappings == 0) ||
        (accounting.preservedTechMappings == 0 &&
         accounting.repairedTechMappings == 0 &&
         accounting.preservedSpatialMappings == 0 &&
         accounting.repairedSpatialMappings == 0))
      return malformed(
          "local Mapping repair has no invalidated and retained lower cone");
    break;
  case JointMappingReuseDisposition::ColdFallback:
    break;
  }

  const bool hasReopenedSystemCone =
      accounting.reopenedThreadBindingCount != 0 ||
      accounting.reopenedGraphBindingCount != 0 ||
      accounting.reopenedResourceUseCount != 0 ||
      accounting.reopenedServiceRealizationCount != 0 ||
      accounting.reopenedServiceLegCount != 0;
  const bool hasPreservedSystemCone =
      accounting.preservedThreadBindingCount != 0 ||
      accounting.preservedGraphBindingCount != 0 ||
      accounting.preservedResourceUseCount != 0 ||
      accounting.preservedServiceRealizationCount != 0 ||
      accounting.preservedServiceLegCount != 0;
  switch (record.systemMappingReuseDisposition) {
  case JointSystemMappingReuseDisposition::Preserved:
    if (hasReopenedSystemCone)
      return malformed(
          "preserved SystemMapping disposition has a reopened System cone");
    break;
  case JointSystemMappingReuseDisposition::Reopened:
    if (!hasReopenedSystemCone)
      return malformed(
          "reopened SystemMapping disposition has no reopened System cone");
    break;
  case JointSystemMappingReuseDisposition::ColdFallback:
    if (hasPreservedSystemCone)
      return malformed(
          "cold SystemMapping fallback claims preserved System decisions");
    break;
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FinalizedHardwareMutationRepairRecord>
publishHardwareMutationRepairRecord(const JointHardwareMutationRepair &repair,
                                    const ArtifactStore &artifacts) {
  if (repair.child.impacts.empty())
    return malformed("repair has no typed impact component");
  auto parentMapping =
      mapping::importSystemMapping(repair.parentMapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  ArtifactRootReference parentSystem{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      parentMapping->view().fabricIdentity()};
  HardwareMutationRepairRecord record{repair.parentMapping,
                                      std::move(parentSystem),
                                      repair.child.system,
                                      repair.child.decisionLineage,
                                      {},
                                      repair.rebase.disposition,
                                      repair.systemDisposition,
                                      {},
                                      {},
                                      std::nullopt,
                                      {},
                                      {}};
  for (const HardwareImpactProjection &impact : repair.child.impacts)
    record.impacts.push_back(
        {impact.parent, impact.child, impact.moduleEntities, impact.family,
         impact.locality, impact.tech, impact.spatial, impact.system});
  record.rebaseFailures = repair.rebase.failures;
  record.accounting = repair.rebase.accounting;
  if (repair.coldExecution)
    record.cold =
        projectSide(repair.coldMappings, repair.coldExecution->summary,
                    repair.coldVerification);
  record.incremental = projectSide(repair.incrementalMappings,
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
  auto parsed = parse(text);
  if (!parsed)
    return parsed.takeError();
  HardwareMutationRepairRecord &record = parsed->record;
  if (finalizeArtifactIdentity(hardwareMutationRepairRecordSchema, *bytes) !=
      reference.artifact)
    return malformed("hardware mutation repair record identity changed on "
                     "import");
  auto parentMapping =
      mapping::importSystemMapping(record.parentMapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  const ArtifactRootReference expectedParentSystem{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      parentMapping->view().fabricIdentity()};
  if (record.parentSystem != expectedParentSystem)
    return malformed("record parent System disagrees with its parent Mapping");
  for (const ArtifactRootReference *system :
       {&record.parentSystem, &record.childSystem}) {
    auto imported = fabric::importEntireFabricRoot(*system, artifacts);
    if (!imported)
      return imported.takeError();
    if (imported->view().rootKind() != fabric::FabricRootKind::System)
      return malformed("record System reference names a non-System root");
  }
  auto decisionImpacts =
      validateDecisionLineage(record, parsed->impacts, artifacts);
  if (!decisionImpacts)
    return decisionImpacts.takeError();
  if (serialize(record) != text)
    return malformed("stored hardware mutation repair record is not "
                     "canonical");
  if (record.accounting.invalidationRootCount !=
      projectJointHardwareInvalidationRootCount(*decisionImpacts))
    return malformed(
        "record invalidation-root count differs from decision lineage");
  if (llvm::Error error = validateJointMappingRebaseAccounting(
          record.accounting))
    return malformed("record rebase accounting is invalid: " +
                     llvm::toString(std::move(error)));
  if (llvm::Error error = validateReuseDispositions(record))
    return std::move(error);

  if (record.cold) {
    auto verification = verifyMappingSide(*record.cold, *parentMapping,
                                          record.childSystem, artifacts,
                                          "record cold side");
    if (!verification)
      return verification.takeError();
    if (!sameDeterministicVerification(*verification,
                                       record.cold->verification))
      return malformed(
          "record cold verifier accounting does not replay exactly");
  }
  auto incrementalVerification = verifyMappingSide(
      record.incremental, *parentMapping, record.childSystem, artifacts,
      "record incremental side");
  if (!incrementalVerification)
    return incrementalVerification.takeError();
  if (!sameDeterministicVerification(*incrementalVerification,
                                     record.incremental.verification))
    return malformed(
        "record incremental verifier accounting does not replay exactly");
  for (const HardwareMutationRepairQualityObservation &observation :
       record.qualityObservations)
    if (!llvm::is_contained(record.incremental.mappings,
                            observation.candidate))
      return malformed(
          "record quality observation is outside the incremental Mapping set");

  auto parentModules =
      projectJointDesignTargetModules(record.parentSystem, artifacts);
  if (!parentModules)
    return parentModules.takeError();
  for (const JointMappingRebaseFailure &failure : record.rebaseFailures)
    if (failure.parent) {
      auto lower = mapping::importLowerMapping(*failure.parent, artifacts);
      if (!lower)
        return lower.takeError();
      llvm::Error ownerError = std::visit(
          [&](const auto &finalized) -> llvm::Error {
            const auto &view = finalized.view();
            if (view.dataflowIdentity() !=
                parentMapping->view().dataflowIdentity())
              return malformed(
                  "rebase failure parent has a foreign Dataflow owner");
            if (!llvm::any_of(*parentModules, [&](const auto &module) {
                  return module.artifact == view.fabricIdentity();
                }))
              return malformed(
                  "rebase failure parent has a foreign Module owner");
            return llvm::Error::success();
          },
          *lower);
      if (ownerError)
        return std::move(ownerError);
    }
  return FinalizedHardwareMutationRepairRecord(reference, std::move(record),
                                               std::move(*bytes));
}

} // namespace loom::dse
