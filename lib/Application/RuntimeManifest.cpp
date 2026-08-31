#include "Application/RuntimeManifest.h"

#include "Application/ActivationDecision.h"
#include "Application/Build.h"
#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Deployment/Package.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

constexpr llvm::StringLiteral applicationPairIdentityDescriptor{
    "loom.application.pair.decision.identity.1"};

llvm::Error reject(ApplicationRuntimeManifestErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ApplicationRuntimeManifestError>(reason,
                                                           message.str());
}

llvm::Error malformed(const llvm::Twine &message) {
  return reject(ApplicationRuntimeManifestErrorReason::MalformedEncoding,
                message);
}

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
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  const llvm::json::Object *result = value->getAsObject();
  if (!result)
    return malformed(context + " field '" + key + "' must be an object");
  return result;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  const llvm::json::Array *result = value->getAsArray();
  if (!result)
    return malformed(context + " field '" + key + "' must be an array");
  return result;
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  auto result = value->getAsString();
  if (!result)
    return malformed(context + " field '" + key + "' must be a string");
  return *result;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  auto result = value->getAsUINT64();
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

void writeReplayCases(
    llvm::json::OStream &json,
    llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference> replayCases) {
  json.attributeArray("source_backed_replay_cases", [&] {
    for (const sim::SourceBackedDfgReplayCaseReference &replay : replayCases)
      json.object([&] {
        writeRoot(json, "workload", replay.workload);
        writeRoot(json, "runtime_input", replay.runtimeInput);
      });
  });
}

llvm::Expected<std::vector<sim::SourceBackedDfgReplayCaseReference>>
parseReplayCases(const llvm::json::Object &object) {
  auto values =
      requireArray(object, "source_backed_replay_cases", "runtime manifest");
  if (!values)
    return values.takeError();
  std::vector<sim::SourceBackedDfgReplayCaseReference> result;
  result.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    const llvm::json::Object *replay = value.getAsObject();
    if (!replay)
      return malformed("runtime manifest source-backed replay case must be an "
                       "object");
    if (llvm::Error error =
            rejectUnknownFields(*replay, {"workload", "runtime_input"},
                                "runtime manifest source-backed replay case"))
      return std::move(error);
    auto workload = parseRoot(*replay, "workload",
                              "runtime manifest source-backed replay case");
    if (!workload)
      return workload.takeError();
    auto runtimeInput = parseRoot(*replay, "runtime_input",
                                  "runtime manifest source-backed replay case");
    if (!runtimeInput)
      return runtimeInput.takeError();
    result.push_back({std::move(*workload), std::move(*runtimeInput)});
  }
  return result;
}

void writeDigestArray(llvm::json::OStream &json, llvm::StringRef key,
                      llvm::ArrayRef<ComponentViewDigest> digests) {
  json.attributeArray(key, [&] {
    for (const ComponentViewDigest &digest : digests)
      json.value(formatComponentViewDigestHex(digest));
  });
}

llvm::Expected<std::vector<ComponentViewDigest>>
parseDigestArray(const llvm::json::Object &object, llvm::StringRef key,
                 const llvm::Twine &context) {
  auto values = requireArray(object, key, context);
  if (!values)
    return values.takeError();
  std::vector<ComponentViewDigest> digests;
  digests.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    auto spelling = value.getAsString();
    if (!spelling)
      return malformed(context + " field '" + key +
                       "' must contain only digests");
    auto digest = parseComponentViewDigestHex(*spelling);
    if (!digest)
      return malformed(context + " field '" + key +
                       "' is invalid: " + llvm::toString(digest.takeError()));
    digests.push_back(*digest);
  }
  return digests;
}

llvm::Expected<std::optional<ArtifactRootReference>>
parseOptionalRoot(const llvm::json::Object &object, llvm::StringRef key,
                  const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  if (value->getAsNull())
    return std::optional<ArtifactRootReference>{};
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return malformed(context + " field '" + key +
                     "' must be null or an object");
  auto parsed = parseArtifactRootReferenceJson(*root);
  if (!parsed)
    return malformed(context + " field '" + key +
                     "' is invalid: " + llvm::toString(parsed.takeError()));
  return std::optional<ArtifactRootReference>{std::move(*parsed)};
}

void writeOptionalRoot(llvm::json::OStream &json, llvm::StringRef key,
                       const std::optional<ArtifactRootReference> &reference) {
  json.attributeBegin(key);
  if (reference)
    writeArtifactRootReferenceJson(json, *reference);
  else
    json.value(nullptr);
  json.attributeEnd();
}

llvm::Expected<std::optional<ComponentViewDigest>>
parseOptionalDigest(const llvm::json::Object &object, llvm::StringRef key,
                    const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  if (value->getAsNull())
    return std::optional<ComponentViewDigest>{};
  auto spelling = value->getAsString();
  if (!spelling)
    return malformed(context + " field '" + key + "' must be null or a digest");
  auto digest = parseComponentViewDigestHex(*spelling);
  if (!digest)
    return malformed(context + " field '" + key +
                     "' is invalid: " + llvm::toString(digest.takeError()));
  return std::optional<ComponentViewDigest>{*digest};
}

void writeOptionalDigest(llvm::json::OStream &json, llvm::StringRef key,
                         const std::optional<ComponentViewDigest> &digest) {
  if (digest)
    json.attribute(key, formatComponentViewDigestHex(*digest));
  else
    json.attribute(key, nullptr);
}

llvm::Expected<std::optional<std::uint64_t>>
parseOptionalUnsigned(const llvm::json::Object &object, llvm::StringRef key,
                      const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return malformed(context + " field '" + key + "' is required");
  if (value->getAsNull())
    return std::optional<std::uint64_t>{};
  auto parsed = value->getAsUINT64();
  if (!parsed)
    return malformed(context + " field '" + key +
                     "' must be null or an unsigned integer");
  return std::optional<std::uint64_t>{*parsed};
}

void writeOptionalUnsigned(llvm::json::OStream &json, llvm::StringRef key,
                           std::optional<std::uint64_t> value) {
  if (value)
    json.attribute(key, *value);
  else
    json.attribute(key, nullptr);
}

template <typename Ref>
void writeDataflowReference(llvm::json::OStream &json,
                            const ArtifactIdentity &artifact,
                            const Ref &reference) {
  auto local = dataflow::encodeDataflowReference(artifact, reference);
  assert(local && "validated Dataflow reference must encode");
  json.object([&] {
    json.attribute("artifact", formatArtifactIdentityHex(artifact));
    json.attribute("local", formatArtifactLocalPayloadHex(*local));
  });
}

template <typename Ref> struct ParsedDataflowReference final {
  ArtifactIdentity artifact;
  Ref reference;
};

template <typename Ref>
llvm::Expected<ParsedDataflowReference<Ref>>
parseDataflowReference(const llvm::json::Value &value,
                       const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, {"artifact", "local"}, context))
    return std::move(error);
  auto artifactSpelling = requireString(*object, "artifact", context);
  if (!artifactSpelling)
    return artifactSpelling.takeError();
  auto artifact = parseArtifactIdentityHex(*artifactSpelling);
  if (!artifact)
    return malformed(context + " has an invalid artifact: " +
                     llvm::toString(artifact.takeError()));
  auto localSpelling = requireString(*object, "local", context);
  if (!localSpelling)
    return localSpelling.takeError();
  auto local = parseArtifactLocalPayloadHex(*localSpelling);
  if (!local)
    return malformed(context + " has an invalid local reference: " +
                     llvm::toString(local.takeError()));
  auto reference = dataflow::decodeDataflowReference<Ref>(*local, *artifact);
  if (!reference)
    return malformed(context + " has an invalid typed reference: " +
                     llvm::toString(reference.takeError()));
  return ParsedDataflowReference<Ref>{std::move(*artifact),
                                      std::move(*reference)};
}

void writeEndpoint(
    llvm::json::OStream &json,
    const pnr::ResourceTimeTransitionEndpointReference &endpoint) {
  json.object([&] {
    writeRoot(json, "mapping", endpoint.mapping);
    writeOptionalRoot(json, "deployment", endpoint.deployment);
  });
}

llvm::Expected<pnr::ResourceTimeTransitionEndpointReference>
parseEndpoint(const llvm::json::Value &value, const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, {"mapping", "deployment"}, context))
    return std::move(error);
  auto mapping = parseRoot(*object, "mapping", context);
  if (!mapping)
    return mapping.takeError();
  auto deployment = parseOptionalRoot(*object, "deployment", context);
  if (!deployment)
    return deployment.takeError();
  return pnr::ResourceTimeTransitionEndpointReference{std::move(*mapping),
                                                      std::move(*deployment)};
}

void writeAllocation(llvm::json::OStream &json,
                     const pnr::ResourceTimeRegionAllocation &allocation) {
  json.object([&] {
    json.attributeBegin("region");
    writeDataflowReference(json, allocation.region.artifact, allocation.region);
    json.attributeEnd();
    json.attributeArray("resources", [&] {
      for (const fabric::FabricPhysicalOccurrenceOwnerRef &resource :
           allocation.resources)
        json.value(fabric::printFabricRef(resource));
    });
  });
}

llvm::Expected<pnr::ResourceTimeRegionAllocation>
parseAllocation(const llvm::json::Value &value, const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, {"region", "resources"}, context))
    return std::move(error);
  const llvm::json::Value *regionValue = object->get("region");
  if (!regionValue)
    return malformed(context + " field 'region' is required");
  auto region = parseDataflowReference<dataflow::RootThreadLaunchRef>(
      *regionValue, context + " region");
  if (!region)
    return region.takeError();
  auto resources = requireArray(*object, "resources", context);
  if (!resources)
    return resources.takeError();
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> parsedResources;
  parsedResources.reserve((*resources)->size());
  for (const llvm::json::Value &resourceValue : **resources) {
    auto spelling = resourceValue.getAsString();
    if (!spelling)
      return malformed(context + " resources must be strings");
    auto resource =
        fabric::parseFabricRef<fabric::FabricPhysicalOccurrenceOwnerRef>(
            *spelling);
    if (!resource)
      return malformed(context + " has an invalid Fabric resource: " +
                       llvm::toString(resource.takeError()));
    parsedResources.push_back(std::move(*resource));
  }
  return pnr::ResourceTimeRegionAllocation{std::move(region->reference),
                                           std::move(parsedResources)};
}

void writeAllocationArray(
    llvm::json::OStream &json, llvm::StringRef key,
    llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> allocations) {
  json.attributeArray(key, [&] {
    for (const pnr::ResourceTimeRegionAllocation &allocation : allocations)
      writeAllocation(json, allocation);
  });
}

llvm::Expected<std::vector<pnr::ResourceTimeRegionAllocation>>
parseAllocationArray(const llvm::json::Object &object, llvm::StringRef key,
                     const llvm::Twine &context) {
  auto values = requireArray(object, key, context);
  if (!values)
    return values.takeError();
  std::vector<pnr::ResourceTimeRegionAllocation> result;
  result.reserve((*values)->size());
  for (const auto indexed : llvm::enumerate(**values)) {
    auto allocation =
        parseAllocation(indexed.value(), context + " " + key + " entry " +
                                             llvm::Twine(indexed.index()));
    if (!allocation)
      return allocation.takeError();
    result.push_back(std::move(*allocation));
  }
  return result;
}

void writeCompletedRoots(llvm::json::OStream &json,
                         llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots) {
  json.attributeArray("completed_before", [&] {
    for (const dataflow::RootThreadLaunchRef &root : roots)
      writeDataflowReference(json, root.artifact, root);
  });
}

llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
parseCompletedRoots(const llvm::json::Object &object,
                    const llvm::Twine &context) {
  auto values = requireArray(object, "completed_before", context);
  if (!values)
    return values.takeError();
  std::vector<dataflow::RootThreadLaunchRef> result;
  result.reserve((*values)->size());
  for (const auto indexed : llvm::enumerate(**values)) {
    auto root = parseDataflowReference<dataflow::RootThreadLaunchRef>(
        indexed.value(),
        context + " completed_before entry " + llvm::Twine(indexed.index()));
    if (!root)
      return root.takeError();
    result.push_back(std::move(root->reference));
  }
  return result;
}

llvm::Expected<pnr::ResourceTimeTransitionStatus>
parseTransitionStatus(llvm::StringRef spelling) {
  for (std::uint8_t ordinal = 0; ordinal != 4; ++ordinal) {
    const auto candidate =
        static_cast<pnr::ResourceTimeTransitionStatus>(ordinal);
    if (spelling == pnr::resourceTimeTransitionStatusSpelling(candidate))
      return candidate;
  }
  return malformed("unknown resource-time transition status '" + spelling +
                   "'");
}

llvm::Expected<pnr::ResourceTimeSafePointKind>
parseSafePointKind(llvm::StringRef spelling) {
  for (std::uint8_t ordinal = 0; ordinal != 2; ++ordinal) {
    const auto candidate = static_cast<pnr::ResourceTimeSafePointKind>(ordinal);
    if (spelling == pnr::resourceTimeSafePointKindSpelling(candidate))
      return candidate;
  }
  return malformed("unknown resource-time safe-point kind '" + spelling + "'");
}

void writeTransition(llvm::json::OStream &json,
                     const pnr::ResourceTimeTransition &transition) {
  assert(transition.safePoint &&
         "verified runtime transition must carry a safe point");
  json.object([&] {
    json.attributeBegin("trigger");
    writeDataflowReference(json, transition.safePoint->artifact.artifact,
                           transition.trigger);
    json.attributeEnd();
    json.attributeBegin("safe_point");
    if (transition.safePoint) {
      json.object([&] {
        writeRoot(json, "artifact", transition.safePoint->artifact);
        json.attribute("kind", pnr::resourceTimeSafePointKindSpelling(
                                   transition.safePoint->kind));
      });
    } else {
      json.value(nullptr);
    }
    json.attributeEnd();
    json.attributeBegin("parent");
    writeEndpoint(json, transition.parent);
    json.attributeEnd();
    json.attributeBegin("child");
    writeEndpoint(json, transition.child);
    json.attributeEnd();
    writeAllocationArray(json, "before_active", transition.beforeActive);
    writeAllocationArray(json, "after_active", transition.afterActive);
    writeCompletedRoots(json, transition.completedBefore);
    writeRootArray(json, "before_live_work", transition.beforeLiveWork);
    writeRootArray(json, "after_live_work", transition.afterLiveWork);
    writeOptionalRoot(json, "token_live_state_correspondence",
                      transition.tokenLiveStateCorrespondence);
    writeOptionalDigest(json, "resource_delta", transition.resourceDeltaDigest);
    writeOptionalDigest(json, "configuration_delta",
                        transition.configurationDeltaDigest);
    writeOptionalDigest(json, "route_delta", transition.routeDeltaDigest);
    writeOptionalUnsigned(json, "reprogramming_time_ps",
                          transition.reprogrammingTimePicoseconds);
    writeOptionalUnsigned(json, "migration_time_ps",
                          transition.migrationTimePicoseconds);
    json.attribute(
        "status", pnr::resourceTimeTransitionStatusSpelling(transition.status));
  });
}

llvm::Expected<pnr::ResourceTimeTransition>
parseTransition(const llvm::json::Value &value, const llvm::Twine &context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return malformed(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object,
          {"trigger", "safe_point", "parent", "child", "before_active",
           "after_active", "completed_before", "before_live_work",
           "after_live_work", "token_live_state_correspondence",
           "resource_delta", "configuration_delta", "route_delta",
           "reprogramming_time_ps", "migration_time_ps", "status"},
          context))
    return std::move(error);
  const llvm::json::Value *triggerValue = object->get("trigger");
  if (!triggerValue)
    return malformed(context + " field 'trigger' is required");
  auto trigger = parseDataflowReference<dataflow::EventFamilyKey>(
      *triggerValue, context + " trigger");
  if (!trigger)
    return trigger.takeError();

  const llvm::json::Value *safePointValue = object->get("safe_point");
  if (!safePointValue)
    return malformed(context + " field 'safe_point' is required");
  std::optional<pnr::ResourceTimeSafePointReference> safePoint;
  if (!safePointValue->getAsNull()) {
    const llvm::json::Object *safePointObject = safePointValue->getAsObject();
    if (!safePointObject)
      return malformed(context + " safe_point must be null or an object");
    if (llvm::Error error = rejectUnknownFields(
            *safePointObject, {"artifact", "kind"}, context + " safe_point"))
      return std::move(error);
    auto artifact =
        parseRoot(*safePointObject, "artifact", context + " safe_point");
    if (!artifact)
      return artifact.takeError();
    auto kindSpelling =
        requireString(*safePointObject, "kind", context + " safe_point");
    if (!kindSpelling)
      return kindSpelling.takeError();
    auto kind = parseSafePointKind(*kindSpelling);
    if (!kind)
      return kind.takeError();
    safePoint =
        pnr::ResourceTimeSafePointReference{std::move(*artifact), *kind};
  }

  const llvm::json::Value *parentValue = object->get("parent");
  const llvm::json::Value *childValue = object->get("child");
  if (!parentValue || !childValue)
    return malformed(context + " requires parent and child endpoints");
  auto parent = parseEndpoint(*parentValue, context + " parent");
  if (!parent)
    return parent.takeError();
  auto child = parseEndpoint(*childValue, context + " child");
  if (!child)
    return child.takeError();
  auto beforeActive = parseAllocationArray(*object, "before_active", context);
  if (!beforeActive)
    return beforeActive.takeError();
  auto afterActive = parseAllocationArray(*object, "after_active", context);
  if (!afterActive)
    return afterActive.takeError();
  auto completedBefore = parseCompletedRoots(*object, context);
  if (!completedBefore)
    return completedBefore.takeError();
  auto beforeLive = parseRootArray(*object, "before_live_work", context);
  if (!beforeLive)
    return beforeLive.takeError();
  auto afterLive = parseRootArray(*object, "after_live_work", context);
  if (!afterLive)
    return afterLive.takeError();
  auto correspondence =
      parseOptionalRoot(*object, "token_live_state_correspondence", context);
  if (!correspondence)
    return correspondence.takeError();
  auto resourceDelta = parseOptionalDigest(*object, "resource_delta", context);
  if (!resourceDelta)
    return resourceDelta.takeError();
  auto configurationDelta =
      parseOptionalDigest(*object, "configuration_delta", context);
  if (!configurationDelta)
    return configurationDelta.takeError();
  auto routeDelta = parseOptionalDigest(*object, "route_delta", context);
  if (!routeDelta)
    return routeDelta.takeError();
  auto reprogramming =
      parseOptionalUnsigned(*object, "reprogramming_time_ps", context);
  if (!reprogramming)
    return reprogramming.takeError();
  auto migration = parseOptionalUnsigned(*object, "migration_time_ps", context);
  if (!migration)
    return migration.takeError();
  auto statusSpelling = requireString(*object, "status", context);
  if (!statusSpelling)
    return statusSpelling.takeError();
  auto status = parseTransitionStatus(*statusSpelling);
  if (!status)
    return status.takeError();

  return pnr::ResourceTimeTransition{std::move(trigger->reference),
                                     std::move(safePoint),
                                     std::move(*parent),
                                     std::move(*child),
                                     std::move(*beforeActive),
                                     std::move(*afterActive),
                                     std::move(*completedBefore),
                                     std::move(*beforeLive),
                                     std::move(*afterLive),
                                     std::move(*correspondence),
                                     std::move(*resourceDelta),
                                     std::move(*configurationDelta),
                                     std::move(*routeDelta),
                                     std::move(*reprogramming),
                                     std::move(*migration),
                                     *status};
}

void writeGraph(llvm::json::OStream &json,
                const pnr::ResourceTimeTransitionGraph &graph) {
  json.object([&] {
    json.attributeBegin("entry");
    writeEndpoint(json, graph.entry);
    json.attributeEnd();
    json.attributeArray("endpoints", [&] {
      for (const auto &endpoint : graph.endpoints)
        writeEndpoint(json, endpoint);
    });
    json.attributeArray("transitions", [&] {
      for (const pnr::ResourceTimeTransition &transition : graph.transitions)
        writeTransition(json, transition);
    });
  });
}

template <typename WriteValue>
std::string canonicalJsonKey(WriteValue &&writeValue) {
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  writeValue(json);
  return output.str().str();
}

void canonicalizeGraphOrder(pnr::ResourceTimeTransitionGraph &graph) {
  for (pnr::ResourceTimeTransition &transition : graph.transitions) {
    for (auto *allocations :
         {&transition.beforeActive, &transition.afterActive}) {
      for (pnr::ResourceTimeRegionAllocation &allocation : *allocations)
        llvm::sort(allocation.resources, [](const auto &lhs, const auto &rhs) {
          return fabric::printFabricRef(lhs) < fabric::printFabricRef(rhs);
        });
      llvm::sort(*allocations, [](const auto &lhs, const auto &rhs) {
        const std::string lhsKey = canonicalJsonKey(
            [&](llvm::json::OStream &json) { writeAllocation(json, lhs); });
        const std::string rhsKey = canonicalJsonKey(
            [&](llvm::json::OStream &json) { writeAllocation(json, rhs); });
        return lhsKey < rhsKey;
      });
    }
    llvm::sort(transition.completedBefore,
               [](const auto &lhs, const auto &rhs) {
                 const std::string lhsKey =
                     canonicalJsonKey([&](llvm::json::OStream &json) {
                       writeDataflowReference(json, lhs.artifact, lhs);
                     });
                 const std::string rhsKey =
                     canonicalJsonKey([&](llvm::json::OStream &json) {
                       writeDataflowReference(json, rhs.artifact, rhs);
                     });
                 return lhsKey < rhsKey;
               });
    llvm::sort(transition.beforeLiveWork, artifactRootReferenceLess);
    llvm::sort(transition.afterLiveWork, artifactRootReferenceLess);
  }
  llvm::sort(graph.endpoints, [](const auto &lhs, const auto &rhs) {
    const std::string lhsKey = canonicalJsonKey(
        [&](llvm::json::OStream &json) { writeEndpoint(json, lhs); });
    const std::string rhsKey = canonicalJsonKey(
        [&](llvm::json::OStream &json) { writeEndpoint(json, rhs); });
    return lhsKey < rhsKey;
  });
  llvm::sort(graph.transitions, [](const auto &lhs, const auto &rhs) {
    const std::string lhsKey = canonicalJsonKey(
        [&](llvm::json::OStream &json) { writeTransition(json, lhs); });
    const std::string rhsKey = canonicalJsonKey(
        [&](llvm::json::OStream &json) { writeTransition(json, rhs); });
    return lhsKey < rhsKey;
  });
}

llvm::Expected<pnr::ResourceTimeTransitionGraph>
parseGraph(const llvm::json::Object &object) {
  if (llvm::Error error =
          rejectUnknownFields(object, {"entry", "endpoints", "transitions"},
                              "resource-time transition graph"))
    return std::move(error);
  const llvm::json::Value *entryValue = object.get("entry");
  if (!entryValue)
    return malformed("resource-time transition graph entry is required");
  auto entry = parseEndpoint(*entryValue, "resource-time graph entry");
  if (!entry)
    return entry.takeError();
  auto endpointValues =
      requireArray(object, "endpoints", "resource-time transition graph");
  if (!endpointValues)
    return endpointValues.takeError();
  std::vector<pnr::ResourceTimeTransitionEndpointReference> endpoints;
  endpoints.reserve((*endpointValues)->size());
  for (const auto indexed : llvm::enumerate(**endpointValues)) {
    auto endpoint =
        parseEndpoint(indexed.value(), "resource-time graph endpoint " +
                                           llvm::Twine(indexed.index()));
    if (!endpoint)
      return endpoint.takeError();
    endpoints.push_back(std::move(*endpoint));
  }
  auto transitionValues =
      requireArray(object, "transitions", "resource-time transition graph");
  if (!transitionValues)
    return transitionValues.takeError();
  std::vector<pnr::ResourceTimeTransition> transitions;
  transitions.reserve((*transitionValues)->size());
  for (const auto indexed : llvm::enumerate(**transitionValues)) {
    auto transition =
        parseTransition(indexed.value(), "resource-time graph transition " +
                                             llvm::Twine(indexed.index()));
    if (!transition)
      return transition.takeError();
    transitions.push_back(std::move(*transition));
  }
  return pnr::ResourceTimeTransitionGraph{
      std::move(*entry), std::move(endpoints), std::move(transitions)};
}

llvm::Expected<ApplicationPairDecisionDisposition>
parsePairDisposition(llvm::StringRef spelling) {
  for (std::uint8_t ordinal = 0; ordinal != 10; ++ordinal) {
    const auto candidate =
        static_cast<ApplicationPairDecisionDisposition>(ordinal);
    if (spelling == toString(candidate))
      return candidate;
  }
  return malformed("unknown application pair disposition '" + spelling + "'");
}

std::string serializeDraft(const ApplicationRuntimeManifestDraft &draft) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", applicationRuntimeManifestSchema.identity);
    json.attribute(
        "schema_version",
        formatSchemaVersion(applicationRuntimeManifestSchema.version));
    writeRoot(json, "source_program", draft.sourceProgram);
    writeRoot(json, "fabric", draft.fabric);
    writeRoot(json, "workload", draft.workload);
    writeRoot(json, "runtime_input", draft.runtimeInput);
    writeReplayCases(json, draft.sourceBackedReplayCases);
    writeRoot(json, "activation_decision", draft.activationDecision);
    json.attribute("pair_identity",
                   formatComponentViewDigestHex(draft.pairIdentity));
    json.attribute("invocation_run_key",
                   formatArtifactLocalPayloadHex(draft.invocationRunKey));
    json.attribute("pair_disposition", toString(draft.pairDisposition));
    json.attribute("selected_candidate", formatComponentViewDigestHex(
                                             draft.selectedCandidateIdentity));
    json.attribute("selected_plan", draft.selectedPlanOrdinal);
    writeDigestArray(json, "selected_schedule_hints",
                     draft.selectedScheduleHintDigests);
    writeRoot(json, "selected_system", draft.selectedSystem);
    writeRoot(json, "selected_mapping", draft.selectedMapping);
    writeRoot(json, "deployment", draft.deployment);
    writeRoot(json, "activation_workload", draft.activationWorkload);
    writeRoot(json, "activation_runtime_input", draft.activationRuntimeInput);
    writeRootArray(json, "runtime_request_dependencies",
                   draft.runtimeRequestDependencies);
    writeRootArray(json, "runtime_evidence", draft.runtimeEvidence);
    writeRootArray(json, "oracle_evidence", draft.oracleEvidence);
    json.attributeBegin("transition_graph");
    if (draft.transitionGraph)
      writeGraph(json, *draft.transitionGraph);
    else
      json.value(nullptr);
    json.attributeEnd();
  });
  return output.str().str();
}

llvm::Expected<ApplicationRuntimeManifestDraft>
parseDraft(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return malformed("runtime manifest JSON cannot be parsed: " +
                     llvm::toString(value.takeError()));
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return malformed("runtime manifest root must be an object");
  if (llvm::Error error = rejectUnknownFields(*root,
                                              {"schema",
                                               "schema_version",
                                               "source_program",
                                               "fabric",
                                               "workload",
                                               "runtime_input",
                                               "source_backed_replay_cases",
                                               "activation_decision",
                                               "pair_identity",
                                               "invocation_run_key",
                                               "pair_disposition",
                                               "selected_candidate",
                                               "selected_plan",
                                               "selected_schedule_hints",
                                               "selected_system",
                                               "selected_mapping",
                                               "deployment",
                                               "activation_workload",
                                               "activation_runtime_input",
                                               "runtime_request_dependencies",
                                               "runtime_evidence",
                                               "oracle_evidence",
                                               "transition_graph"},
                                              "runtime manifest"))
    return std::move(error);
  auto schema = requireString(*root, "schema", "runtime manifest");
  if (!schema)
    return schema.takeError();
  auto versionSpelling =
      requireString(*root, "schema_version", "runtime manifest");
  if (!versionSpelling)
    return versionSpelling.takeError();
  auto version = parseSchemaVersion(*versionSpelling);
  if (!version)
    return malformed("runtime manifest schema version is invalid: " +
                     llvm::toString(version.takeError()));
  if (*schema != applicationRuntimeManifestSchema.identity ||
      *version != applicationRuntimeManifestSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::ForeignSchema,
                  "unsupported Application runtime manifest schema");
  auto source = parseRoot(*root, "source_program", "runtime manifest");
  if (!source)
    return source.takeError();
  auto fabric = parseRoot(*root, "fabric", "runtime manifest");
  if (!fabric)
    return fabric.takeError();
  auto workload = parseRoot(*root, "workload", "runtime manifest");
  if (!workload)
    return workload.takeError();
  auto runtimeInput = parseRoot(*root, "runtime_input", "runtime manifest");
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto replayCases = parseReplayCases(*root);
  if (!replayCases)
    return replayCases.takeError();
  auto activationDecision =
      parseRoot(*root, "activation_decision", "runtime manifest");
  if (!activationDecision)
    return activationDecision.takeError();
  auto pairSpelling = requireString(*root, "pair_identity", "runtime manifest");
  if (!pairSpelling)
    return pairSpelling.takeError();
  auto pairIdentity = parseComponentViewDigestHex(*pairSpelling);
  if (!pairIdentity)
    return malformed("runtime manifest pair identity is invalid: " +
                     llvm::toString(pairIdentity.takeError()));
  auto runKeySpelling =
      requireString(*root, "invocation_run_key", "runtime manifest");
  if (!runKeySpelling)
    return runKeySpelling.takeError();
  auto runKeyBytes = parseArtifactLocalPayloadHex(*runKeySpelling);
  if (!runKeyBytes)
    return malformed("runtime manifest run key is invalid: " +
                     llvm::toString(runKeyBytes.takeError()));
  if (runKeyBytes->size() != 32)
    return malformed("runtime manifest run key must be 32 bytes");
  std::array<std::uint8_t, 32> runKey;
  std::copy(runKeyBytes->begin(), runKeyBytes->end(), runKey.begin());
  auto dispositionSpelling =
      requireString(*root, "pair_disposition", "runtime manifest");
  if (!dispositionSpelling)
    return dispositionSpelling.takeError();
  auto disposition = parsePairDisposition(*dispositionSpelling);
  if (!disposition)
    return disposition.takeError();
  auto candidateSpelling =
      requireString(*root, "selected_candidate", "runtime manifest");
  if (!candidateSpelling)
    return candidateSpelling.takeError();
  auto candidate = parseComponentViewDigestHex(*candidateSpelling);
  if (!candidate)
    return malformed("runtime manifest candidate identity is invalid: " +
                     llvm::toString(candidate.takeError()));
  auto plan = requireUnsigned(*root, "selected_plan", "runtime manifest");
  if (!plan)
    return plan.takeError();
  auto scheduleHints =
      parseDigestArray(*root, "selected_schedule_hints", "runtime manifest");
  if (!scheduleHints)
    return scheduleHints.takeError();
  auto selectedSystem = parseRoot(*root, "selected_system", "runtime manifest");
  if (!selectedSystem)
    return selectedSystem.takeError();
  auto mapping = parseRoot(*root, "selected_mapping", "runtime manifest");
  if (!mapping)
    return mapping.takeError();
  auto deployment = parseRoot(*root, "deployment", "runtime manifest");
  if (!deployment)
    return deployment.takeError();
  auto activationWorkload =
      parseRoot(*root, "activation_workload", "runtime manifest");
  if (!activationWorkload)
    return activationWorkload.takeError();
  auto activationRuntimeInput =
      parseRoot(*root, "activation_runtime_input", "runtime manifest");
  if (!activationRuntimeInput)
    return activationRuntimeInput.takeError();
  auto runtimeRequestDependencies =
      parseRootArray(*root, "runtime_request_dependencies", "runtime manifest");
  if (!runtimeRequestDependencies)
    return runtimeRequestDependencies.takeError();
  auto runtimeEvidence =
      parseRootArray(*root, "runtime_evidence", "runtime manifest");
  if (!runtimeEvidence)
    return runtimeEvidence.takeError();
  auto oracleEvidence =
      parseRootArray(*root, "oracle_evidence", "runtime manifest");
  if (!oracleEvidence)
    return oracleEvidence.takeError();
  const llvm::json::Value *graphValue = root->get("transition_graph");
  if (!graphValue)
    return malformed("runtime manifest transition_graph is required");
  std::optional<pnr::ResourceTimeTransitionGraph> graph;
  if (!graphValue->getAsNull()) {
    const llvm::json::Object *graphObject = graphValue->getAsObject();
    if (!graphObject)
      return malformed("runtime manifest transition_graph must be null or an "
                       "object");
    auto parsed = parseGraph(*graphObject);
    if (!parsed)
      return parsed.takeError();
    graph = std::move(*parsed);
  }
  return ApplicationRuntimeManifestDraft{std::move(*source),
                                         std::move(*fabric),
                                         std::move(*workload),
                                         std::move(*runtimeInput),
                                         std::move(*replayCases),
                                         std::move(*activationDecision),
                                         *pairIdentity,
                                         runKey,
                                         *disposition,
                                         *candidate,
                                         *plan,
                                         std::move(*scheduleHints),
                                         std::move(*selectedSystem),
                                         std::move(*mapping),
                                         std::move(*deployment),
                                         std::move(*activationWorkload),
                                         std::move(*activationRuntimeInput),
                                         std::move(*runtimeRequestDependencies),
                                         std::move(*runtimeEvidence),
                                         std::move(*oracleEvidence),
                                         std::move(graph)};
}

llvm::Error
canonicalizeReferenceSet(std::vector<ArtifactRootReference> &references,
                         llvm::StringRef name) {
  llvm::sort(references, artifactRootReferenceLess);
  for (std::size_t index = 1; index != references.size(); ++index)
    if (references[index - 1] == references[index])
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          name + " repeats one Artifact root");
  return llvm::Error::success();
}

llvm::Error canonicalizeDigestSet(std::vector<ComponentViewDigest> &digests,
                                  llvm::StringRef name) {
  llvm::sort(digests, [](const auto &lhs, const auto &rhs) {
    return lhs.bytes() < rhs.bytes();
  });
  for (std::size_t index = 1; index != digests.size(); ++index)
    if (digests[index - 1] == digests[index])
      return reject(
          ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
          name + " repeats one digest");
  return llvm::Error::success();
}

llvm::Error canonicalizeReplayCases(
    std::vector<sim::SourceBackedDfgReplayCaseReference> &replayCases) {
  llvm::sort(replayCases, [](const auto &lhs, const auto &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  });
  for (std::size_t index = 1; index != replayCases.size(); ++index)
    if (replayCases[index - 1] == replayCases[index])
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "source-backed replay cases repeat one activation input");
  return llvm::Error::success();
}

bool contains(llvm::ArrayRef<ArtifactRootReference> references,
              const ArtifactRootReference &reference) {
  return llvm::is_contained(references, reference);
}

llvm::Error verifyManifestDraft(ApplicationRuntimeManifestDraft &draft,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  auto activation = importApplicationActivationDecision(
      draft.activationDecision, artifacts, blobs);
  if (!activation)
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision failed strict import: " +
            llvm::toString(activation.takeError()));
  const ApplicationActivationDecision &decision = activation->decision();
  if (decision.sourceProgram() != draft.sourceProgram ||
      decision.fabric() != draft.fabric ||
      decision.workload() != draft.workload ||
      decision.runtimeInput() != draft.runtimeInput ||
      decision.sourceBackedReplayCases() !=
          llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference>(
              draft.sourceBackedReplayCases))
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the runtime source lineage");
  if (decision.dseInvocation().occurrence().runKey.bytes() !=
          draft.invocationRunKey ||
      decision.disposition() != draft.pairDisposition ||
      decision.selectedCandidateIdentity() != draft.selectedCandidateIdentity ||
      decision.selectedPlanOrdinal() != draft.selectedPlanOrdinal ||
      decision.selectedSystem() != draft.selectedSystem ||
      decision.selectedMapping() != draft.selectedMapping ||
      decision.runtimeEvidence() !=
          llvm::ArrayRef<ArtifactRootReference>(draft.runtimeEvidence) ||
      decision.oracleEvidence() !=
          llvm::ArrayRef<ArtifactRootReference>(draft.oracleEvidence))
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the selected runtime execution");
  std::vector<ComponentViewDigest> decisionScheduleHints;
  decisionScheduleHints.reserve(decision.selectedScheduleHints().size());
  for (const dse::ResourceTimeScheduleHint &hint :
       decision.selectedScheduleHints()) {
    auto digest = dse::deriveResourceTimeScheduleHintDigest(hint);
    if (!digest)
      return reject(
          ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
          "activation decision schedule hint cannot be derived: " +
              llvm::toString(digest.takeError()));
    decisionScheduleHints.push_back(*digest);
  }
  if (llvm::Error error = canonicalizeDigestSet(
          decisionScheduleHints, "activation decision schedule hints"))
    return error;
  if (decisionScheduleHints != draft.selectedScheduleHintDigests)
    return reject(
        ApplicationRuntimeManifestErrorReason::ActivationDecisionMismatch,
        "activation decision differs from the selected schedule hints");

  auto expectedPair = deriveApplicationPairIdentity(
      draft.sourceProgram, draft.fabric, draft.workload, draft.runtimeInput);
  if (!expectedPair)
    return expectedPair.takeError();
  if (*expectedPair != draft.pairIdentity)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest pair identity does not match its exact "
                  "source, Fabric, workload, and runtime input roots");
  if (draft.pairDisposition !=
          ApplicationPairDecisionDisposition::VerifiedAcceleration &&
      draft.pairDisposition != ApplicationPairDecisionDisposition::
                                   VerifiedFeasibleButNotBeneficial &&
      draft.pairDisposition !=
          ApplicationPairDecisionDisposition::HardwareDseAlternative)
    return reject(ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
                  "runtime manifest does not carry a completed pair decision");
  const bool selectedDifferentSystem = draft.selectedSystem != draft.fabric;
  if (selectedDifferentSystem !=
      (draft.pairDisposition ==
       ApplicationPairDecisionDisposition::HardwareDseAlternative))
    return reject(
        ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
        "hardware-alternative disposition differs from the selected System");
  if (draft.selectedScheduleHintDigests.empty())
    return reject(ApplicationRuntimeManifestErrorReason::PairDecisionIncomplete,
                  "runtime manifest has no selected schedule hint");

  for (const ArtifactRootReference *root :
       {&draft.sourceProgram, &draft.fabric, &draft.workload,
        &draft.runtimeInput, &draft.selectedSystem}) {
    auto stored = artifacts.get(*root);
    if (!stored)
      return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                    "runtime manifest pair root is unavailable: " +
                        llvm::toString(stored.takeError()));
  }
  if (draft.sourceProgram.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      draft.sourceProgram.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      draft.fabric.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      draft.fabric.schemaVersion != fabric::fabricArtifactSchema.version ||
      draft.selectedSystem.schemaIdentity !=
          fabric::fabricArtifactSchema.identity ||
      draft.selectedSystem.schemaVersion !=
          fabric::fabricArtifactSchema.version ||
      draft.workload.schemaIdentity != sim::simulationWorkloadSchema.identity ||
      draft.workload.schemaVersion != sim::simulationWorkloadSchema.version ||
      draft.runtimeInput.schemaIdentity !=
          sim::simulationRuntimeInputSchema.identity ||
      draft.runtimeInput.schemaVersion !=
          sim::simulationRuntimeInputSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest pair roots use foreign schemas");
  if (draft.activationWorkload.schemaIdentity !=
          sim::simulationWorkloadSchema.identity ||
      draft.activationWorkload.schemaVersion !=
          sim::simulationWorkloadSchema.version ||
      draft.activationRuntimeInput.schemaIdentity !=
          sim::simulationRuntimeInputSchema.identity ||
      draft.activationRuntimeInput.schemaVersion !=
          sim::simulationRuntimeInputSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "runtime manifest activation roots use foreign schemas");

  auto sourceInputs = sim::importStructuredProgramSimulationInputs(
      draft.workload, draft.runtimeInput, artifacts);
  if (!sourceInputs)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest source activation failed strict import: " +
                      llvm::toString(sourceInputs.takeError()));
  if (sourceInputs->structuredProgram.identity() !=
      draft.sourceProgram.artifact)
    return reject(ApplicationRuntimeManifestErrorReason::PairIdentityMismatch,
                  "runtime manifest source activation names a foreign "
                  "StructuredProgram");

  auto importedMapping =
      mapping::importSystemMapping(draft.selectedMapping, artifacts);
  if (!importedMapping)
    return reject(ApplicationRuntimeManifestErrorReason::MappingMismatch,
                  "cannot import selected SystemMapping: " +
                      llvm::toString(importedMapping.takeError()));
  if (importedMapping->view().fabricIdentity() != draft.selectedSystem.artifact)
    return reject(ApplicationRuntimeManifestErrorReason::MappingMismatch,
                  "selected SystemMapping names a foreign selected System");
  const ArtifactRootReference dataflow{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      importedMapping->view().dataflowIdentity()};

  auto importedDeployment =
      deployment::importDeployment(draft.deployment, artifacts, blobs);
  if (!importedDeployment)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "cannot import selected Deployment: " +
                      llvm::toString(importedDeployment.takeError()));
  if (importedDeployment->deployment().systemMapping() != draft.selectedMapping)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "Deployment does not bind the selected SystemMapping");
  auto activationInputs = sim::importSystemSimulationInputs(
      draft.activationWorkload, draft.activationRuntimeInput, artifacts, blobs);
  if (!activationInputs)
    return reject(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "runtime manifest Deployment activation failed strict import: " +
            llvm::toString(activationInputs.takeError()));
  if (activationInputs->deployment.reference() != draft.deployment)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "runtime manifest activation inputs name a foreign "
                  "Deployment");
  auto deploymentClosure = deployment::deriveDeploymentPackageClosure(
      *importedDeployment, artifacts, blobs);
  if (!deploymentClosure)
    return reject(ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
                  "cannot derive selected Deployment closure: " +
                      llvm::toString(deploymentClosure.takeError()));

  if (draft.runtimeEvidence.empty() || draft.oracleEvidence.empty())
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "completed Application runtime requires runtime and oracle "
        "Evidence");
  if (draft.sourceBackedReplayCases.empty())
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "completed Application runtime has no source-backed replay case");
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       draft.sourceBackedReplayCases) {
    auto imported = sim::importSpatialSimulationInputs(
        replay.workload, replay.runtimeInput, artifacts);
    if (!imported)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "source-backed replay case failed strict import: " +
              llvm::toString(imported.takeError()));
    if (imported->dataflow.identity() != dataflow.artifact)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "source-backed replay case names a foreign canonical Dataflow");
  }
  for (const ArtifactRootReference &oracle : draft.oracleEvidence)
    if (!contains(draft.runtimeEvidence, oracle))
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "oracle Evidence is outside the selected runtime Evidence set");

  struct EvidenceFacts final {
    ArtifactRootReference evidence;
    evaluation::EvaluationEvidenceDependencyProjection projection;
    std::vector<ArtifactRootReference> requestReferences;
  };
  std::vector<EvidenceFacts> evidenceFacts;
  std::vector<ArtifactRootReference> executionOutputs;
  for (const ArtifactRootReference &evidenceReference : draft.runtimeEvidence) {
    auto projection = evaluation::importEvaluationEvidenceDependencyProjection(
        evidenceReference, artifacts);
    if (!projection)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "cannot project runtime Evidence dependencies: " +
              llvm::toString(projection.takeError()));
    if (projection->outcomeKind != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "selected runtime Evidence is not completed");
    auto requestReferences =
        evaluation::importEvaluationRequestArtifactReferences(
            projection->request, artifacts);
    if (!requestReferences)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "cannot project runtime Evidence Request dependencies: " +
              llvm::toString(requestReferences.takeError()));
    for (const evaluation::ModelOutputBinding &binding :
         projection->outputBindings)
      for (const ArtifactRootReference &output : binding.artifacts) {
        auto stored = artifacts.get(output);
        if (!stored)
          return reject(
              ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
              "runtime Evidence output is unavailable: " +
                  llvm::toString(stored.takeError()));
        if (output.schemaIdentity == sim::simulationExecutionSchema.identity &&
            output.schemaVersion == sim::simulationExecutionSchema.version) {
          auto request =
              sim::simulationExecutionRequestReference(output, artifacts);
          if (!request)
            return reject(
                ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
                "cannot project SimulationExecution Request: " +
                    llvm::toString(request.takeError()));
          if (*request != projection->request)
            return reject(
                ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
                "SimulationExecution and Evidence name different Requests");
          executionOutputs.push_back(output);
        }
      }
    evidenceFacts.push_back({evidenceReference, std::move(*projection),
                             std::move(*requestReferences)});
  }
  llvm::sort(executionOutputs, artifactRootReferenceLess);
  executionOutputs.erase(
      std::unique(executionOutputs.begin(), executionOutputs.end()),
      executionOutputs.end());
  if (executionOutputs.size() < 2)
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "runtime Evidence does not bind both reference and candidate "
        "SimulationExecution roots");

  std::vector<ArtifactRootReference> deploymentAndExecutionClosure =
      deploymentClosure->artifacts().vec();
  deploymentAndExecutionClosure.insert(deploymentAndExecutionClosure.end(),
                                       executionOutputs.begin(),
                                       executionOutputs.end());
  llvm::sort(deploymentAndExecutionClosure, artifactRootReferenceLess);
  deploymentAndExecutionClosure.erase(
      std::unique(deploymentAndExecutionClosure.begin(),
                  deploymentAndExecutionClosure.end()),
      deploymentAndExecutionClosure.end());

  draft.runtimeRequestDependencies.clear();
  for (const EvidenceFacts &facts : evidenceFacts)
    for (const ArtifactRootReference &reference : facts.requestReferences)
      if (!contains(deploymentAndExecutionClosure, reference))
        draft.runtimeRequestDependencies.push_back(reference);
  llvm::sort(draft.runtimeRequestDependencies, artifactRootReferenceLess);
  draft.runtimeRequestDependencies.erase(
      std::unique(draft.runtimeRequestDependencies.begin(),
                  draft.runtimeRequestDependencies.end()),
      draft.runtimeRequestDependencies.end());
  for (const ArtifactRootReference &reference :
       draft.runtimeRequestDependencies) {
    auto stored = artifacts.get(reference);
    if (!stored)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "runtime Evidence Request dependency is unavailable: " +
              formatArtifactRootReferenceJson(reference) + ": " +
              llvm::toString(stored.takeError()));
  }

  const auto spatialMappings =
      importedMapping->view().executionBindings().spatialMappingImports();
  enum class RuntimeEvidenceKind : std::uint8_t { Dfg, Cgra };
  struct ExecutionEvidenceRecord final {
    ArtifactRootReference evidence;
    ArtifactRootReference execution;
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
    evaluation::CaseArtifactResolution resolution;
    RuntimeEvidenceKind kind;
  };
  struct RuntimeInputPair final {
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;

    bool operator==(const RuntimeInputPair &other) const {
      return workload == other.workload && runtimeInput == other.runtimeInput;
    }
  };
  const auto strictExecutionOutput =
      [&](const evaluation::EvaluationEvidence &evidence)
      -> llvm::Expected<ArtifactRootReference> {
    std::optional<ArtifactRootReference> execution;
    for (const evaluation::ModelOutputBinding &binding :
         evidence.outputBindings())
      for (const ArtifactRootReference &output : binding.artifacts) {
        if (output.schemaIdentity != sim::simulationExecutionSchema.identity ||
            output.schemaVersion != sim::simulationExecutionSchema.version)
          continue;
        if (execution)
          return reject(
              ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
              "runtime Evidence repeats its SimulationExecution output");
        execution = output;
      }
    if (!execution)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "runtime Evidence has no SimulationExecution output");
    return *execution;
  };
  std::vector<ExecutionEvidenceRecord> executionEvidence;
  std::vector<const EvidenceFacts *> comparisonEvidence;
  std::vector<RuntimeInputPair> sourceRuntimeInputs;
  std::vector<RuntimeInputPair> mappedRuntimeInputs;
  for (const EvidenceFacts &facts : evidenceFacts) {
    std::optional<ArtifactRootReference> workload;
    std::optional<ArtifactRootReference> runtimeInput;
    for (const ArtifactRootReference &reference : facts.requestReferences) {
      if (reference.schemaIdentity == sim::simulationWorkloadSchema.identity &&
          reference.schemaVersion == sim::simulationWorkloadSchema.version) {
        if (workload)
          return reject(
              ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
              "runtime Evidence Request repeats its SimulationWorkload");
        workload = reference;
      }
      if (reference.schemaIdentity ==
              sim::simulationRuntimeInputSchema.identity &&
          reference.schemaVersion ==
              sim::simulationRuntimeInputSchema.version) {
        if (runtimeInput)
          return reject(
              ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
              "runtime Evidence Request repeats its SimulationRuntimeInput");
        runtimeInput = reference;
      }
    }
    const bool hasDataflow = contains(facts.requestReferences, dataflow);
    std::vector<ArtifactRootReference> selectedSpatialMappings;
    for (const ArtifactRootReference &root : spatialMappings)
      if (contains(facts.requestReferences, root))
        selectedSpatialMappings.push_back(root);
    const bool hasSpatialMapping = !selectedSpatialMappings.empty();
    if (!hasDataflow && !hasSpatialMapping) {
      comparisonEvidence.push_back(&facts);
      continue;
    }
    if ((hasDataflow || hasSpatialMapping) && (!workload || !runtimeInput))
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "runtime Evidence Request has no exact workload and runtime input");
    std::optional<evaluation::CaseArtifactResolution> resolution;
    RuntimeEvidenceKind kind = RuntimeEvidenceKind::Dfg;
    if (hasSpatialMapping) {
      if (selectedSpatialMappings.size() != 1)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "runtime Evidence Request repeats a selected SpatialMapping");
      auto resolved = evaluation::models::resolveCgraSimulationCase(
          selectedSpatialMappings.front(), *workload, *runtimeInput, artifacts);
      if (!resolved)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "cannot resolve selected CGRA runtime case: " +
                llvm::toString(resolved.takeError()));
      if (resolved->canonicalDataflow != dataflow)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "CGRA runtime case names a foreign canonical Dataflow");
      resolution.emplace(std::move(resolved->resolution));
      kind = RuntimeEvidenceKind::Cgra;
      mappedRuntimeInputs.push_back({*workload, *runtimeInput});
    } else {
      auto resolved = evaluation::models::resolveDfgSimulationCase(
          dataflow, *workload, *runtimeInput, artifacts);
      if (!resolved)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "cannot resolve DFG runtime case: " +
                llvm::toString(resolved.takeError()));
      resolution.emplace(std::move(*resolved));
      sourceRuntimeInputs.push_back({*workload, *runtimeInput});
    }
    auto strict = evaluation::importEvaluationEvidence(
        facts.evidence, *resolution, artifacts, blobs);
    if (!strict)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "runtime Evidence failed strict import: " +
              llvm::toString(strict.takeError()));
    if (strict->requestRef() != facts.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "strict runtime Evidence differs from its dependency projection");
    if (kind == RuntimeEvidenceKind::Cgra) {
      auto terminal =
          evaluation::models::classifyCompletedCgraSimulationEvidence(
              *strict, *resolution, artifacts, blobs);
      if (!terminal)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "cannot classify CGRA runtime terminal: " +
                llvm::toString(terminal.takeError()));
      if (*terminal !=
          evaluation::models::CgraSimulationEvidenceTerminal::Retired)
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "completed Application runtime contains a closed-wait CGRA "
            "execution");
    }
    auto execution = strictExecutionOutput(*strict);
    if (!execution)
      return execution.takeError();
    executionEvidence.push_back({facts.evidence, *execution, *workload,
                                 *runtimeInput, std::move(*resolution), kind});
  }
  const auto pairLess = [](const RuntimeInputPair &lhs,
                           const RuntimeInputPair &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  };
  const auto canonicalizePairs = [&](auto &pairs) {
    llvm::sort(pairs, pairLess);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  };
  canonicalizePairs(sourceRuntimeInputs);
  canonicalizePairs(mappedRuntimeInputs);
  std::vector<RuntimeInputPair> sourceBackedReplayInputs;
  sourceBackedReplayInputs.reserve(draft.sourceBackedReplayCases.size());
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       draft.sourceBackedReplayCases)
    sourceBackedReplayInputs.push_back({replay.workload, replay.runtimeInput});
  canonicalizePairs(sourceBackedReplayInputs);
  if (sourceRuntimeInputs.empty() ||
      sourceRuntimeInputs != mappedRuntimeInputs ||
      sourceRuntimeInputs != sourceBackedReplayInputs)
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "runtime Evidence does not join DFG and Spatial Mapping executions "
        "through the exact source-backed replay input set");
  std::vector<ArtifactRootReference> classifiedExecutionOutputs;
  for (const ExecutionEvidenceRecord &record : executionEvidence)
    classifiedExecutionOutputs.push_back(record.execution);
  llvm::sort(classifiedExecutionOutputs, artifactRootReferenceLess);
  if (classifiedExecutionOutputs != executionOutputs)
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "runtime Evidence has an unclassified SimulationExecution output");

  std::vector<ArtifactRootReference> verifiedOracleEvidence;
  for (const EvidenceFacts *facts : comparisonEvidence) {
    std::vector<const ExecutionEvidenceRecord *> compared;
    for (const ArtifactRootReference &reference : facts->requestReferences) {
      if (reference.schemaIdentity != sim::simulationExecutionSchema.identity ||
          reference.schemaVersion != sim::simulationExecutionSchema.version)
        continue;
      const auto record =
          llvm::find_if(executionEvidence, [&](const auto &row) {
            return row.execution == reference;
          });
      if (record == executionEvidence.end())
        return reject(
            ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
            "oracle Evidence names a foreign SimulationExecution");
      compared.push_back(&*record);
    }
    if (compared.size() != 2 || compared[0]->kind == compared[1]->kind)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "oracle Evidence does not compare one DFG and one CGRA execution");
    const ExecutionEvidenceRecord *dfg =
        compared[0]->kind == RuntimeEvidenceKind::Dfg ? compared[0]
                                                      : compared[1];
    const ExecutionEvidenceRecord *cgra =
        compared[0]->kind == RuntimeEvidenceKind::Cgra ? compared[0]
                                                       : compared[1];
    auto resolution = evaluation::models::resolveSimulationComparisonCase(
        dfg->execution, dfg->resolution, cgra->execution, cgra->resolution,
        artifacts, blobs);
    if (!resolution)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "cannot resolve SimulationComparison Evidence: " +
              llvm::toString(resolution.takeError()));
    auto strict = evaluation::importEvaluationEvidence(
        facts->evidence, *resolution, artifacts, blobs);
    if (!strict)
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "oracle Evidence failed strict SimulationComparison import: " +
              llvm::toString(strict.takeError()));
    if (strict->requestRef() != facts->projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed ||
        !llvm::is_contained(draft.oracleEvidence, facts->evidence))
      return reject(
          ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
          "strict SimulationComparison differs from the oracle Evidence set");
    verifiedOracleEvidence.push_back(facts->evidence);
  }
  llvm::sort(verifiedOracleEvidence, artifactRootReferenceLess);
  if (verifiedOracleEvidence != draft.oracleEvidence)
    return reject(
        ApplicationRuntimeManifestErrorReason::RuntimeEvidenceMismatch,
        "oracle Evidence contains a non-SimulationComparison result");

  if (draft.transitionGraph) {
    if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
            *draft.transitionGraph, artifacts, blobs))
      return reject(
          ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
          "resource-time transition graph failed independent replay: " +
              llvm::toString(std::move(error)));
    for (const pnr::ResourceTimeTransition &transition :
         draft.transitionGraph->transitions)
      if (!transition.safePoint ||
          transition.safePoint->kind !=
              pnr::ResourceTimeSafePointKind::Completion ||
          transition.safePoint->artifact != dataflow)
        return reject(
            ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
            "runtime manifest supports only canonical Dataflow root "
            "completion safe points");
    if (draft.transitionGraph->entry.mapping != draft.selectedMapping ||
        draft.transitionGraph->entry.deployment != draft.deployment)
      return reject(
          ApplicationRuntimeManifestErrorReason::TransitionGraphMismatch,
          "resource-time transition graph entry differs from the selected "
          "Mapping and Deployment");
  }
  return llvm::Error::success();
}

} // namespace

char ApplicationRuntimeManifestError::ID = 0;

void ApplicationRuntimeManifestError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code ApplicationRuntimeManifestError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef toString(ApplicationPairDecisionDisposition value) {
  switch (value) {
  case ApplicationPairDecisionDisposition::VerifiedAcceleration:
    return "verified_acceleration";
  case ApplicationPairDecisionDisposition::VerifiedFeasibleButNotBeneficial:
    return "verified_feasible_but_not_beneficial";
  case ApplicationPairDecisionDisposition::NoPromisingCandidate:
    return "no_promising_candidate";
  case ApplicationPairDecisionDisposition::ExactHardwareIncompatible:
    return "exact_hardware_incompatible";
  case ApplicationPairDecisionDisposition::MappingProofNotEstablished:
    return "mapping_proof_not_established";
  case ApplicationPairDecisionDisposition::CancelledOrTimeout:
    return "cancelled_or_timeout";
  case ApplicationPairDecisionDisposition::BudgetExhausted:
    return "budget_exhausted";
  case ApplicationPairDecisionDisposition::UnsupportedSemantic:
    return "unsupported_semantic";
  case ApplicationPairDecisionDisposition::ImplementationFailure:
    return "implementation_failure";
  case ApplicationPairDecisionDisposition::HardwareDseAlternative:
    return "hardware_dse_alternative";
  }
  llvm_unreachable("unknown application pair decision disposition");
}

llvm::Expected<ComponentViewDigest>
deriveApplicationPairIdentity(const ArtifactRootReference &sourceProgram,
                              const ArtifactRootReference &fabric,
                              const ArtifactRootReference &workload,
                              const ArtifactRootReference &runtimeInput) {
  std::vector<std::uint8_t> bytes;
  const std::array<ArtifactRootReference, 4> roots = {sourceProgram, fabric,
                                                      workload, runtimeInput};
  const auto appendU64 = [&](std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    bytes.push_back(static_cast<std::uint8_t>(value));
  };
  appendU64(roots.size());
  for (const ArtifactRootReference &root : roots) {
    const std::vector<std::uint8_t> encoded = encodeArtifactRootReference(root);
    appendU64(encoded.size());
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           applicationPairIdentityDescriptor.data()),
       applicationPairIdentityDescriptor.size()},
      bytes);
}

llvm::Expected<ApplicationRuntimeManifest>
ApplicationRuntimeManifest::get(ApplicationRuntimeManifestDraft draft,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  if (llvm::Error error =
          canonicalizeReplayCases(draft.sourceBackedReplayCases))
    return std::move(error);
  if (llvm::Error error = canonicalizeDigestSet(
          draft.selectedScheduleHintDigests, "selected schedule hints"))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeReferenceSet(draft.runtimeEvidence, "runtime Evidence"))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeReferenceSet(draft.oracleEvidence, "oracle Evidence"))
    return std::move(error);
  if (llvm::Error error = verifyManifestDraft(draft, artifacts, blobs))
    return std::move(error);
  if (draft.transitionGraph)
    canonicalizeGraphOrder(*draft.transitionGraph);
  const std::string text = serializeDraft(draft);
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
  return ApplicationRuntimeManifest(std::move(draft), std::move(bytes));
}

std::string serializeApplicationRuntimeManifest(
    const ApplicationRuntimeManifest &manifest) {
  const llvm::ArrayRef<std::uint8_t> bytes = manifest.canonicalBytes().bytes();
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<FinalizedApplicationRuntimeManifest>
publishApplicationRuntimeManifest(ApplicationRuntimeManifest manifest,
                                  const ArtifactStore &artifacts) {
  auto identity = artifacts.put(applicationRuntimeManifestSchema,
                                manifest.canonicalBytes());
  if (!identity)
    return identity.takeError();
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationRuntimeManifestSchema, manifest.canonicalBytes());
  if (*identity != expected)
    return reject(ApplicationRuntimeManifestErrorReason::NonCanonicalEncoding,
                  "ArtifactStore returned a foreign runtime manifest identity");
  ArtifactRootReference reference{
      applicationRuntimeManifestSchema.identity.str(),
      applicationRuntimeManifestSchema.version, std::move(*identity)};
  return FinalizedApplicationRuntimeManifest(std::move(reference),
                                             std::move(manifest));
}

llvm::Expected<FinalizedApplicationRuntimeManifest>
importApplicationRuntimeManifest(const ArtifactRootReference &reference,
                                 const ArtifactStore &artifacts,
                                 const BlobStore &blobs) {
  if (reference.schemaIdentity != applicationRuntimeManifestSchema.identity ||
      reference.schemaVersion != applicationRuntimeManifestSchema.version)
    return reject(ApplicationRuntimeManifestErrorReason::ForeignSchema,
                  "foreign Application runtime manifest reference schema");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  const llvm::ArrayRef<std::uint8_t> payload = bytes->bytes();
  const llvm::StringRef text(reinterpret_cast<const char *>(payload.data()),
                             payload.size());
  auto draft = parseDraft(text);
  if (!draft)
    return draft.takeError();
  auto manifest =
      ApplicationRuntimeManifest::get(std::move(*draft), artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  if (serializeApplicationRuntimeManifest(*manifest) != text)
    return reject(ApplicationRuntimeManifestErrorReason::NonCanonicalEncoding,
                  "Application runtime manifest JSON is not canonical");
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationRuntimeManifestSchema, manifest->canonicalBytes());
  if (expected != reference.artifact)
    return reject(ApplicationRuntimeManifestErrorReason::NonCanonicalEncoding,
                  "Application runtime manifest identity changed on import");
  return FinalizedApplicationRuntimeManifest(reference, std::move(*manifest));
}

} // namespace loom::application
