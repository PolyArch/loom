#include "Application/ResourceTimeExecution.h"

#include "Application/RuntimeManifest.h"
#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

constexpr llvm::StringLiteral kTraceEncoding{
    "loom.application.resource_time_execution_trace.1"};

llvm::Error reject(ApplicationResourceTimeExecutionErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ApplicationResourceTimeExecutionError>(reason,
                                                                 message.str());
}

llvm::Error malformed(const llvm::Twine &message) {
  return reject(ApplicationResourceTimeExecutionErrorReason::MalformedEncoding,
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

void writeRoot(llvm::json::OStream &json, llvm::StringRef key,
               const ArtifactRootReference &root) {
  json.attributeObject(
      key, [&] { writeArtifactRootReferenceJsonFields(json, root); });
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
  std::vector<ArtifactRootReference> result;
  result.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    const llvm::json::Object *root = value.getAsObject();
    if (!root)
      return malformed(context + " field '" + key +
                       "' must contain only objects");
    auto parsed = parseArtifactRootReferenceJson(*root);
    if (!parsed)
      return malformed(context + " field '" + key +
                       "' is invalid: " + llvm::toString(parsed.takeError()));
    result.push_back(std::move(*parsed));
  }
  return result;
}

void writeEndpoint(
    llvm::json::OStream &json, llvm::StringRef key,
    const pnr::ResourceTimeTransitionEndpointReference &endpoint) {
  json.attributeObject(key, [&] {
    writeRoot(json, "mapping", endpoint.mapping);
    json.attributeBegin("deployment");
    if (endpoint.deployment)
      writeArtifactRootReferenceJson(json, *endpoint.deployment);
    else
      json.value(nullptr);
    json.attributeEnd();
  });
}

llvm::Expected<pnr::ResourceTimeTransitionEndpointReference>
parseEndpoint(const llvm::json::Object &object, llvm::StringRef key,
              const llvm::Twine &context) {
  auto endpoint = requireObject(object, key, context);
  if (!endpoint)
    return endpoint.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **endpoint, {"mapping", "deployment"}, context + " endpoint"))
    return std::move(error);
  auto mapping = parseRoot(**endpoint, "mapping", context + " endpoint");
  if (!mapping)
    return mapping.takeError();
  const llvm::json::Value *deployment = (*endpoint)->get("deployment");
  if (!deployment)
    return malformed(context + " endpoint deployment is required");
  std::optional<ArtifactRootReference> parsedDeployment;
  if (!deployment->getAsNull()) {
    const llvm::json::Object *root = deployment->getAsObject();
    if (!root)
      return malformed(context + " endpoint deployment must be null or an "
                                 "object");
    auto parsed = parseArtifactRootReferenceJson(*root);
    if (!parsed)
      return malformed(context + " endpoint deployment is invalid: " +
                       llvm::toString(parsed.takeError()));
    parsedDeployment = std::move(*parsed);
  }
  return pnr::ResourceTimeTransitionEndpointReference{
      std::move(*mapping), std::move(parsedDeployment)};
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

template <typename Ref>
llvm::Expected<Ref> parseDataflowReference(const llvm::json::Value &value,
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
  return std::move(*reference);
}

void writeDataflowRootArray(
    llvm::json::OStream &json, llvm::StringRef key,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots) {
  json.attributeArray(key, [&] {
    for (const dataflow::RootThreadLaunchRef root : roots)
      writeDataflowReference(json, root.artifact, root);
  });
}

llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
parseDataflowRootArray(const llvm::json::Object &object, llvm::StringRef key,
                       const llvm::Twine &context) {
  auto values = requireArray(object, key, context);
  if (!values)
    return values.takeError();
  std::vector<dataflow::RootThreadLaunchRef> roots;
  roots.reserve((*values)->size());
  for (const llvm::json::Value &value : **values) {
    auto root = parseDataflowReference<dataflow::RootThreadLaunchRef>(
        value, context + " " + key + " root");
    if (!root)
      return root.takeError();
    roots.push_back(std::move(*root));
  }
  return roots;
}

template <typename T>
bool sameUnorderedValues(llvm::ArrayRef<T> lhs, llvm::ArrayRef<T> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::vector<bool> consumed(rhs.size(), false);
  for (const T &value : lhs) {
    std::optional<std::size_t> match;
    for (std::size_t index = 0; index != rhs.size(); ++index)
      if (!consumed[index] && rhs[index] == value) {
        match = index;
        break;
      }
    if (!match)
      return false;
    consumed[*match] = true;
  }
  return true;
}

bool sameAllocations(llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> lhs,
                     llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::vector<bool> consumed(rhs.size(), false);
  for (const pnr::ResourceTimeRegionAllocation &allocation : lhs) {
    std::optional<std::size_t> match;
    for (std::size_t index = 0; index != rhs.size(); ++index)
      if (!consumed[index] && rhs[index].region == allocation.region &&
          sameUnorderedValues(llvm::ArrayRef(rhs[index].resources),
                              llvm::ArrayRef(allocation.resources))) {
        match = index;
        break;
      }
    if (!match)
      return false;
    consumed[*match] = true;
  }
  return true;
}

bool sameSafePoint(
    const std::optional<pnr::ResourceTimeSafePointReference> &lhs,
    const std::optional<pnr::ResourceTimeSafePointReference> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (lhs->artifact == rhs->artifact && lhs->kind == rhs->kind);
}

bool sameTransition(const pnr::ResourceTimeTransition &lhs,
                    const pnr::ResourceTimeTransition &rhs) {
  return lhs.trigger == rhs.trigger &&
         sameSafePoint(lhs.safePoint, rhs.safePoint) &&
         lhs.parent == rhs.parent && lhs.child == rhs.child &&
         sameAllocations(lhs.beforeActive, rhs.beforeActive) &&
         sameAllocations(lhs.afterActive, rhs.afterActive) &&
         sameUnorderedValues(llvm::ArrayRef(lhs.completedBefore),
                             llvm::ArrayRef(rhs.completedBefore)) &&
         sameUnorderedValues(llvm::ArrayRef(lhs.beforeLiveWork),
                             llvm::ArrayRef(rhs.beforeLiveWork)) &&
         sameUnorderedValues(llvm::ArrayRef(lhs.afterLiveWork),
                             llvm::ArrayRef(rhs.afterLiveWork)) &&
         lhs.tokenLiveStateCorrespondence == rhs.tokenLiveStateCorrespondence &&
         lhs.resourceDeltaDigest == rhs.resourceDeltaDigest &&
         lhs.configurationDeltaDigest == rhs.configurationDeltaDigest &&
         lhs.routeDeltaDigest == rhs.routeDeltaDigest &&
         lhs.reprogrammingTimePicoseconds == rhs.reprogrammingTimePicoseconds &&
         lhs.migrationTimePicoseconds == rhs.migrationTimePicoseconds &&
         lhs.status == rhs.status;
}

bool sameGraph(const pnr::ResourceTimeTransitionGraph &lhs,
               const pnr::ResourceTimeTransitionGraph &rhs) {
  if (lhs.entry != rhs.entry || lhs.endpoints != rhs.endpoints ||
      lhs.transitions.size() != rhs.transitions.size())
    return false;
  for (std::size_t index = 0; index != lhs.transitions.size(); ++index)
    if (!sameTransition(lhs.transitions[index], rhs.transitions[index]))
      return false;
  return true;
}

llvm::Expected<std::pair<dataflow::RootThreadLaunchRef, bool>>
resolveLifecycleEvent(
    const runtime::ResourceTimeTransitionSelectionSession &selection,
    const dataflow::EventFamilyKey &event) {
  std::optional<std::pair<dataflow::RootThreadLaunchRef, bool>> resolved;
  for (const dataflow::RootThreadLaunchRef root : selection.mappedRoots()) {
    const bool start = event == dataflow::rootThreadStartEventFamily(root);
    const bool completion =
        event == dataflow::rootThreadCompletionEventFamily(root);
    if (!start && !completion)
      continue;
    if (resolved)
      return reject(
          ApplicationResourceTimeExecutionErrorReason::UnknownLifecycleEvent,
          "root lifecycle event resolves more than once");
    resolved = std::make_pair(root, start);
  }
  if (!resolved)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::UnknownLifecycleEvent,
        "root lifecycle event is outside the prepared Mapping root scope");
  return *resolved;
}

llvm::Error validateObservationPrefix(
    llvm::ArrayRef<ApplicationResourceTimeExecutionEvent> events,
    const sim::SystemRootLifecycleObservation &observation,
    dataflow::RootThreadLaunchRef root, bool start) {
  if (observation.occurrence == 0)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::InvalidOccurrence,
        "root lifecycle occurrence must be nonzero");
  if (!events.empty() &&
      sim::compareSystemEventCoordinates(events.back().observation.coordinate,
                                         observation.coordinate) >= 0)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::NonMonotonicCoordinate,
        "root lifecycle coordinate is not strictly increasing");
  const auto occurrence = llvm::find_if(events, [&](const auto &event) {
    return event.observation.occurrence == observation.occurrence;
  });
  if (start) {
    if (occurrence != events.end())
      return reject(
          ApplicationResourceTimeExecutionErrorReason::InvalidOccurrence,
          "root lifecycle occurrence is reused");
    return llvm::Error::success();
  }
  if (occurrence == events.end() || occurrence->root != root ||
      occurrence->outcome != ApplicationResourceTimeEventOutcome::RootStarted)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::OccurrenceMismatch,
        "root completion does not match its accepted start occurrence");
  return llvm::Error::success();
}

std::vector<ArtifactRootReference>
qorEvidence(const ApplicationRuntimeManifest &manifest) {
  std::vector<ArtifactRootReference> result(manifest.runtimeEvidence().begin(),
                                            manifest.runtimeEvidence().end());
  result.insert(result.end(), manifest.oracleEvidence().begin(),
                manifest.oracleEvidence().end());
  llvm::sort(result, artifactRootReferenceLess);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

struct TraceDraft final {
  ArtifactRootReference runtimeManifest;
  std::vector<ArtifactRootReference> qor;
  std::vector<ApplicationResourceTimeExecutionEvent> events;
};

struct ParsedTraceEvent final {
  sim::SystemRootLifecycleObservation observation;
  ApplicationResourceTimeEventOutcome outcome =
      ApplicationResourceTimeEventOutcome::RootStarted;
  pnr::ResourceTimeTransitionEndpointReference parent;
  pnr::ResourceTimeTransitionEndpointReference current;
  std::vector<dataflow::RootThreadLaunchRef> activeRoots;
  std::vector<dataflow::RootThreadLaunchRef> completedRoots;
};

struct ParsedTraceDraft final {
  ArtifactRootReference runtimeManifest;
  std::vector<ArtifactRootReference> qor;
  std::vector<ParsedTraceEvent> events;
};

std::string serializeTrace(const TraceDraft &draft) {
  std::string text;
  llvm::raw_string_ostream output(text);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", kTraceEncoding);
    writeRoot(json, "runtime_manifest", draft.runtimeManifest);
    writeRootArray(json, "qor_evidence", draft.qor);
    json.attributeArray("events", [&] {
      for (const ApplicationResourceTimeExecutionEvent &event : draft.events)
        json.object([&] {
          json.attributeBegin("event");
          writeDataflowReference(json, event.root.artifact,
                                 event.observation.event);
          json.attributeEnd();
          json.attribute("occurrence", event.observation.occurrence);
          json.attributeObject("coordinate", [&] {
            json.attribute("gem5_tick", event.observation.coordinate.gem5Tick);
            json.attribute("delta", event.observation.coordinate.delta);
          });
          json.attribute("outcome", applicationResourceTimeEventOutcomeSpelling(
                                        event.outcome));
          writeEndpoint(json, "parent", event.parent);
          writeEndpoint(json, "current", event.current);
          writeDataflowRootArray(json, "active_roots", event.activeRoots);
          writeDataflowRootArray(json, "completed_roots", event.completedRoots);
        });
    });
    json.attribute("joined", true);
  });
  return text;
}

llvm::Expected<ApplicationResourceTimeEventOutcome>
parseOutcome(llvm::StringRef spelling) {
  for (std::uint8_t ordinal = 0; ordinal != 3; ++ordinal) {
    const auto outcome =
        static_cast<ApplicationResourceTimeEventOutcome>(ordinal);
    if (applicationResourceTimeEventOutcomeSpelling(outcome) == spelling)
      return outcome;
  }
  return malformed("unknown resource-time event outcome '" + spelling + "'");
}

llvm::Expected<ParsedTraceDraft> parseTrace(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return malformed("trace is not JSON: " + llvm::toString(value.takeError()));
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return malformed("trace root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object,
          {"schema", "runtime_manifest", "qor_evidence", "events", "joined"},
          "trace"))
    return std::move(error);
  auto schema = requireString(*object, "schema", "trace");
  if (!schema)
    return schema.takeError();
  if (*schema != kTraceEncoding)
    return malformed("trace has an unknown encoding");
  auto manifest = parseRoot(*object, "runtime_manifest", "trace");
  if (!manifest)
    return manifest.takeError();
  auto qor = parseRootArray(*object, "qor_evidence", "trace");
  if (!qor)
    return qor.takeError();
  const llvm::json::Value *joined = object->get("joined");
  if (!joined || joined->getAsBoolean() != std::optional<bool>(true))
    return malformed("trace must be a joined execution");
  auto values = requireArray(*object, "events", "trace");
  if (!values)
    return values.takeError();
  std::vector<ParsedTraceEvent> events;
  events.reserve((*values)->size());
  for (const auto indexed : llvm::enumerate(**values)) {
    const llvm::Twine context =
        llvm::Twine("trace event ") + llvm::Twine(indexed.index());
    const llvm::json::Object *eventObject = indexed.value().getAsObject();
    if (!eventObject)
      return malformed(context + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *eventObject,
            {"event", "occurrence", "coordinate", "outcome", "parent",
             "current", "active_roots", "completed_roots"},
            context))
      return std::move(error);
    const llvm::json::Value *eventValue = eventObject->get("event");
    if (!eventValue)
      return malformed(context + " event is required");
    auto event =
        parseDataflowReference<dataflow::EventFamilyKey>(*eventValue, context);
    if (!event)
      return event.takeError();
    auto occurrence = requireUnsigned(*eventObject, "occurrence", context);
    if (!occurrence)
      return occurrence.takeError();
    auto coordinate = requireObject(*eventObject, "coordinate", context);
    if (!coordinate)
      return coordinate.takeError();
    if (llvm::Error error = rejectUnknownFields(
            **coordinate, {"gem5_tick", "delta"}, context + " coordinate"))
      return std::move(error);
    auto tick = requireUnsigned(**coordinate, "gem5_tick", context);
    if (!tick)
      return tick.takeError();
    auto delta = requireUnsigned(**coordinate, "delta", context);
    if (!delta)
      return delta.takeError();
    auto outcomeSpelling = requireString(*eventObject, "outcome", context);
    if (!outcomeSpelling)
      return outcomeSpelling.takeError();
    auto outcome = parseOutcome(*outcomeSpelling);
    if (!outcome)
      return outcome.takeError();
    auto parent = parseEndpoint(*eventObject, "parent", context);
    if (!parent)
      return parent.takeError();
    auto current = parseEndpoint(*eventObject, "current", context);
    if (!current)
      return current.takeError();
    auto active = parseDataflowRootArray(*eventObject, "active_roots", context);
    if (!active)
      return active.takeError();
    auto completed =
        parseDataflowRootArray(*eventObject, "completed_roots", context);
    if (!completed)
      return completed.takeError();
    events.push_back({{std::move(*event), *occurrence, {*tick, *delta}},
                      *outcome,
                      std::move(*parent),
                      std::move(*current),
                      std::move(*active),
                      std::move(*completed)});
  }
  return ParsedTraceDraft{std::move(*manifest), std::move(*qor),
                          std::move(events)};
}

llvm::Expected<std::vector<ApplicationResourceTimeExecutionEvent>>
replayTrace(ParsedTraceDraft draft, const ApplicationRuntimeManifest &manifest,
            const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (!manifest.transitionGraph())
    return reject(ApplicationResourceTimeExecutionErrorReason::ManifestMismatch,
                  "runtime manifest has no resource-time transition graph");
  if (draft.qor != qorEvidence(manifest))
    return reject(ApplicationResourceTimeExecutionErrorReason::ManifestMismatch,
                  "trace QoR references differ from the runtime manifest");
  const auto &graph = *manifest.transitionGraph();
  if (!graph.entry.deployment)
    return reject(ApplicationResourceTimeExecutionErrorReason::ManifestMismatch,
                  "runtime manifest transition entry has no Deployment");
  auto entry =
      deployment::importDeployment(*graph.entry.deployment, artifacts, blobs);
  if (!entry)
    return entry.takeError();
  auto selection = runtime::ResourceTimeTransitionSelectionSession::create(
      graph, *entry, artifacts, blobs);
  if (!selection)
    return selection.takeError();

  std::vector<ApplicationResourceTimeExecutionEvent> replayed;
  replayed.reserve(draft.events.size());
  for (ParsedTraceEvent &parsed : draft.events) {
    auto resolved = resolveLifecycleEvent(*selection, parsed.observation.event);
    if (!resolved)
      return resolved.takeError();
    const dataflow::RootThreadLaunchRef root = resolved->first;
    if (llvm::Error error = validateObservationPrefix(
            replayed, parsed.observation, root, resolved->second))
      return std::move(error);
    if (parsed.parent != selection->currentEndpoint())
      return reject(ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
                    "trace event parent differs from the replayed endpoint");
    std::optional<pnr::ResourceTimeTransition> transition;
    if (resolved->second) {
      if (parsed.outcome != ApplicationResourceTimeEventOutcome::RootStarted)
        return reject(
            ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
            "root start has a completion outcome");
      if (llvm::Error error = selection->startRoot(root))
        return std::move(error);
    } else {
      auto legal = selection->legalTransitionsForCompletion(root);
      if (!legal)
        return legal.takeError();
      if (parsed.outcome ==
          ApplicationResourceTimeEventOutcome::SelectedChild) {
        if (legal->size() != 1)
          return reject(
              ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
              "selected-child trace does not have one exact legal edge");
        auto selected = selection->completeRoot(root, legal->front().child);
        if (!selected)
          return selected.takeError();
        if (!*selected || !sameTransition(**selected, legal->front()))
          return reject(
              ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
              "selector replay returned another transition");
        transition = std::move(**selected);
      } else if (parsed.outcome ==
                 ApplicationResourceTimeEventOutcome::NoLegalTransition) {
        if (!legal->empty())
          return reject(
              ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
              "no-edge trace suppresses a legal transition");
        auto stayed = selection->completeRoot(root, std::nullopt);
        if (!stayed)
          return stayed.takeError();
        if (*stayed)
          return reject(
              ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
              "no-edge trace selected a transition");
      } else {
        return reject(
            ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
            "root completion has a start outcome");
      }
    }
    if (parsed.current != selection->currentEndpoint() ||
        parsed.activeRoots != selection->activeRoots() ||
        parsed.completedRoots != selection->completedRoots())
      return reject(ApplicationResourceTimeExecutionErrorReason::ReplayMismatch,
                    "trace state snapshot differs from selector replay");
    replayed.push_back({std::move(parsed.observation), root, parsed.outcome,
                        std::move(parsed.parent), std::move(parsed.current),
                        std::move(parsed.activeRoots),
                        std::move(parsed.completedRoots),
                        std::move(transition)});
  }
  if (llvm::Error error = selection->joinMappedRoots())
    return reject(ApplicationResourceTimeExecutionErrorReason::TraceNotJoined,
                  "trace does not complete the mapped root inventory: " +
                      llvm::toString(std::move(error)));
  return replayed;
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

} // namespace

char ApplicationResourceTimeExecutionError::ID = 0;

void ApplicationResourceTimeExecutionError::log(
    llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code
ApplicationResourceTimeExecutionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef applicationResourceTimeEventOutcomeSpelling(
    ApplicationResourceTimeEventOutcome outcome) {
  switch (outcome) {
  case ApplicationResourceTimeEventOutcome::RootStarted:
    return "root_started";
  case ApplicationResourceTimeEventOutcome::SelectedChild:
    return "selected_child";
  case ApplicationResourceTimeEventOutcome::NoLegalTransition:
    return "no_legal_transition";
  }
  llvm_unreachable("unknown Application resource-time event outcome");
}

llvm::Expected<ApplicationResourceTimeExecutionSession>
ApplicationResourceTimeExecutionSession::createPrepared(
    pnr::ResourceTimeTransitionGraph graph, runtime::LoadedDeployment &loaded,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto selection =
      runtime::ResourceTimeTransitionSelectionSession::createPrepared(
          std::move(graph), loaded, artifacts, blobs);
  if (!selection)
    return selection.takeError();
  return ApplicationResourceTimeExecutionSession(std::move(*selection));
}

llvm::Expected<ApplicationResourceTimeExecutionEvent>
ApplicationResourceTimeExecutionSession::apply(
    const sim::SystemRootLifecycleObservation &observation,
    runtime::LoadedDeployment &loaded) {
  auto resolved = resolveLifecycleEvent(selection_, observation.event);
  if (!resolved)
    return resolved.takeError();
  const dataflow::RootThreadLaunchRef root = resolved->first;
  const bool start = resolved->second;
  if (llvm::Error error =
          validateObservationPrefix(events_, observation, root, start))
    return std::move(error);

  const pnr::ResourceTimeTransitionEndpointReference parent =
      selection_.currentEndpoint();
  ApplicationResourceTimeEventOutcome outcome =
      ApplicationResourceTimeEventOutcome::RootStarted;
  std::optional<pnr::ResourceTimeTransition> transition;
  if (start) {
    if (llvm::Error error = selection_.startRoot(root))
      return std::move(error);
  } else {
    auto legal = selection_.legalTransitionsForCompletion(root);
    if (!legal)
      return legal.takeError();
    if (legal->size() > 1)
      return reject(
          ApplicationResourceTimeExecutionErrorReason::AmbiguousLegalTransition,
          "completion has more than one legal child and requires explicit "
          "runtime policy");
    if (legal->empty()) {
      auto stayed =
          selection_.completeRootAndActivate(root, std::nullopt, loaded);
      if (!stayed)
        return stayed.takeError();
      outcome = ApplicationResourceTimeEventOutcome::NoLegalTransition;
    } else {
      auto selected = selection_.completeRootAndActivate(
          root, legal->front().child, loaded);
      if (!selected)
        return selected.takeError();
      assert(*selected && sameTransition(**selected, legal->front()) &&
             "unique legal resource-time edge must be selected exactly");
      transition = std::move(**selected);
      outcome = ApplicationResourceTimeEventOutcome::SelectedChild;
    }
  }

  ApplicationResourceTimeExecutionEvent event{observation,
                                              root,
                                              outcome,
                                              parent,
                                              selection_.currentEndpoint(),
                                              selection_.activeRoots(),
                                              selection_.completedRoots(),
                                              std::move(transition)};
  events_.push_back(event);
  if (selection_.completedRoots().size() == selection_.mappedRoots().size())
    if (llvm::Error error = selection_.joinMappedRoots())
      return std::move(error);
  return event;
}

llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
publishApplicationResourceTimeExecutionTrace(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationResourceTimeExecutionSession &session,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (!session.joined())
    return reject(ApplicationResourceTimeExecutionErrorReason::TraceNotJoined,
                  "resource-time session has not joined every mapped root");
  const auto &graph = manifest.manifest().transitionGraph();
  if (!graph || !sameGraph(*graph, session.selection().graph()))
    return reject(
        ApplicationResourceTimeExecutionErrorReason::ManifestMismatch,
        "prepared session graph differs from its runtime manifest owner");
  TraceDraft draft{manifest.reference(), qorEvidence(manifest.manifest()),
                   std::vector<ApplicationResourceTimeExecutionEvent>(
                       session.events().begin(), session.events().end())};
  const std::string text = serializeTrace(draft);
  auto parsed = parseTrace(text);
  if (!parsed)
    return parsed.takeError();
  auto replayed = replayTrace(*parsed, manifest.manifest(), artifacts, blobs);
  if (!replayed)
    return replayed.takeError();
  TraceDraft strict{parsed->runtimeManifest, parsed->qor, std::move(*replayed)};
  if (serializeTrace(strict) != text)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::NonCanonicalEncoding,
        "resource-time trace failed independent canonical roundtrip");
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
  auto identity =
      artifacts.put(applicationResourceTimeExecutionTraceSchema, bytes);
  if (!identity)
    return identity.takeError();
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationResourceTimeExecutionTraceSchema, bytes);
  if (*identity != expected)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::NonCanonicalEncoding,
        "ArtifactStore returned a foreign trace identity");
  return importApplicationResourceTimeExecutionTrace(
      {applicationResourceTimeExecutionTraceSchema.identity.str(),
       applicationResourceTimeExecutionTraceSchema.version, *identity},
      artifacts, blobs);
}

llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
importApplicationResourceTimeExecutionTrace(
    const ArtifactRootReference &reference, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (reference.schemaIdentity !=
          applicationResourceTimeExecutionTraceSchema.identity ||
      reference.schemaVersion !=
          applicationResourceTimeExecutionTraceSchema.version)
    return reject(ApplicationResourceTimeExecutionErrorReason::ForeignSchema,
                  "foreign resource-time execution trace schema");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  const llvm::StringRef text = asText(bytes->bytes());
  auto draft = parseTrace(text);
  if (!draft)
    return draft.takeError();
  auto manifest = importApplicationRuntimeManifest(draft->runtimeManifest,
                                                   artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  auto events = replayTrace(*draft, manifest->manifest(), artifacts, blobs);
  if (!events)
    return events.takeError();
  TraceDraft strict{draft->runtimeManifest, draft->qor, *events};
  if (serializeTrace(strict) != text)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::NonCanonicalEncoding,
        "stored resource-time execution trace is not canonical");
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationResourceTimeExecutionTraceSchema, *bytes);
  if (expected != reference.artifact)
    return reject(
        ApplicationResourceTimeExecutionErrorReason::NonCanonicalEncoding,
        "resource-time execution trace identity changed on import");
  return FinalizedApplicationResourceTimeExecutionTrace(
      reference, std::move(draft->runtimeManifest), std::move(draft->qor),
      std::move(*events), std::move(*bytes));
}

} // namespace loom::application
