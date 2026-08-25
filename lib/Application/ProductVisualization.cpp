#include "Application/ProductVisualization.h"

#include "ADG/Export.h"
#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "BuildInternal.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {

namespace {

llvm::Error visualizationError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "loom_visualization_export_failed: " +
                                     message);
}

void canonicalize(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
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

bool sameProviderWork(const ApplicationMappingProviderWorkObservation &lhs,
                      const ApplicationMappingProviderWorkObservation &rhs) {
  return lhs.techMappingInvocations == rhs.techMappingInvocations &&
         lhs.spatialPnrInvocations == rhs.spatialPnrInvocations &&
         lhs.systemPnrInvocations == rhs.systemPnrInvocations &&
         lhs.techMappingDispatches == rhs.techMappingDispatches &&
         lhs.spatialPnrDispatches == rhs.spatialPnrDispatches &&
         lhs.systemPnrDispatches == rhs.systemPnrDispatches &&
         lhs.techMappingJournalReplays == rhs.techMappingJournalReplays &&
         lhs.spatialPnrJournalReplays == rhs.spatialPnrJournalReplays &&
         lhs.systemPnrJournalReplays == rhs.systemPnrJournalReplays;
}

bool providerWorkClosed(
    const ApplicationMappingProviderWorkObservation &work) {
  const auto tech = llvm::checkedAddUnsigned(
      work.techMappingDispatches, work.techMappingJournalReplays);
  const auto spatial = llvm::checkedAddUnsigned(
      work.spatialPnrDispatches, work.spatialPnrJournalReplays);
  const auto system = llvm::checkedAddUnsigned(
      work.systemPnrDispatches, work.systemPnrJournalReplays);
  return tech && *tech == work.techMappingInvocations && spatial &&
         *spatial == work.spatialPnrInvocations && system &&
         *system == work.systemPnrInvocations;
}

bool sameRepairEvidence(
    const ApplicationResourceTimeRepairEvidence &evidence,
    const ApplicationIncrementalMappingObservation &observation) {
  return sameUnorderedValues(llvm::ArrayRef(evidence.reopenedRoots),
                             llvm::ArrayRef(observation.reopenedRoots)) &&
         evidence.reuseDisposition == observation.reuseDisposition &&
         evidence.preservedTechMappings == observation.preservedTechMappings &&
         evidence.preservedSpatialMappings ==
             observation.preservedSpatialMappings &&
         evidence.repairedTechMappings == observation.repairedTechMappings &&
         evidence.repairedSpatialMappings ==
             observation.repairedSpatialMappings &&
         evidence.preservedSystemBindings ==
             observation.preservedSystemBindings &&
         evidence.reopenedSystemBindings ==
             observation.reopenedSystemBindings &&
         evidence.coldWallTimeNanoseconds ==
             observation.coldWallTimeNanoseconds &&
         evidence.incrementalWallTimeNanoseconds ==
             observation.incrementalWallTimeNanoseconds &&
         evidence.coldVerifierRetainedBytes ==
             observation.coldVerifierRetainedBytes &&
         evidence.incrementalVerifierRetainedBytes ==
             observation.incrementalVerifierRetainedBytes &&
         evidence.coldVerifierWork == observation.coldVerifierWork &&
         evidence.incrementalVerifierWork ==
             observation.incrementalVerifierWork &&
         sameProviderWork(evidence.coldProviderWork,
                          observation.coldProviderWork) &&
         sameProviderWork(evidence.incrementalProviderWork,
                          observation.incrementalProviderWork) &&
         evidence.coldDfgCycles == observation.coldDfgCycles &&
         evidence.coldCgraCycles == observation.coldCgraCycles &&
         evidence.incrementalDfgCycles == observation.incrementalDfgCycles &&
         evidence.incrementalCgraCycles == observation.incrementalCgraCycles;
}

llvm::Expected<mapping::SystemMappingImportSessionStatistics>
measureIndependentSystemMappingImport(const ArtifactRootReference &reference,
                                      const ArtifactIdentity &expectedDataflow,
                                      const ArtifactIdentity &expectedFabric,
                                      const ArtifactStore &artifacts) {
  mapping::SystemMappingImportSession session(
      artifacts, 1, mapping::SystemMappingImportSessionMode::New);
  auto imported = mapping::importSystemMapping(reference, artifacts);
  if (!imported)
    return imported.takeError();
  if (imported->view().dataflowIdentity() != expectedDataflow ||
      imported->view().fabricIdentity() != expectedFabric)
    return visualizationError(
        "independently replayed repair Mapping has foreign owners");
  return session.statistics();
}

llvm::Expected<std::vector<ArtifactIdentity>> deriveTechMappingIdentities(
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &artifacts) {
  std::vector<ArtifactIdentity> identities;
  for (const ArtifactRootReference &reference : spatialMappings) {
    auto spatial = mapping::importSpatialMapping(reference, artifacts);
    if (!spatial)
      return spatial.takeError();
    const ArtifactIdentity identity = spatial->view().techMappingIdentity();
    if (!llvm::is_contained(identities, identity))
      identities.push_back(identity);
  }
  return identities;
}

void writeReferenceArray(llvm::json::OStream &json, llvm::StringRef name,
                         llvm::ArrayRef<ArtifactRootReference> references) {
  json.attributeArray(name, [&] {
    for (const ArtifactRootReference &reference : references)
      json.object(
          [&] { writeArtifactRootReferenceJsonFields(json, reference); });
  });
}

void writeReference(llvm::json::OStream &json, llvm::StringRef name,
                    const ArtifactRootReference &reference) {
  json.attributeObject(
      name, [&] { writeArtifactRootReferenceJsonFields(json, reference); });
}

void writeAllocations(
    llvm::json::OStream &json, llvm::StringRef name,
    llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> allocations) {
  json.attributeArray(name, [&] {
    for (const pnr::ResourceTimeRegionAllocation &allocation : allocations) {
      json.object([&] {
        json.attribute("region_artifact",
                       formatArtifactIdentityHex(allocation.region.artifact));
        json.attribute("region_entity", allocation.region.entity.value());
        json.attributeArray("resources", [&] {
          for (const auto &resource : allocation.resources)
            json.value(llvm::toHex(fabric::canonicalFabricBytes(resource),
                                   /*LowerCase=*/true));
        });
      });
    }
  });
}

void writeProviderWork(
    llvm::json::OStream &json, llvm::StringRef name,
    const ApplicationMappingProviderWorkObservation &work) {
  json.attributeObject(name, [&] {
    json.attribute("tech_mapping_invocations", work.techMappingInvocations);
    json.attribute("spatial_pnr_invocations", work.spatialPnrInvocations);
    json.attribute("system_pnr_invocations", work.systemPnrInvocations);
    json.attribute("tech_mapping_dispatches", work.techMappingDispatches);
    json.attribute("spatial_pnr_dispatches", work.spatialPnrDispatches);
    json.attribute("system_pnr_dispatches", work.systemPnrDispatches);
    json.attribute("tech_mapping_journal_replays",
                   work.techMappingJournalReplays);
    json.attribute("spatial_pnr_journal_replays",
                   work.spatialPnrJournalReplays);
    json.attribute("system_pnr_journal_replays",
                   work.systemPnrJournalReplays);
  });
}

void writeSpectrumSummary(llvm::json::OStream &json, llvm::StringRef name,
                          const dse::VerifiedResourceTimeSpectrum &spectrum) {
  json.attributeObject(name, [&] {
    json.attribute("status", "verified");
    writeReference(json, "dataflow", spectrum.dataflow);
    writeReference(json, "fabric", spectrum.fabric);
    json.attributeArray("scenarios", [&] {
      for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
           spectrum.scenarios) {
        json.object([&] {
          json.attribute("ordinal", scenario.scenarioOrdinal);
          json.attribute("spectrum_class",
                         dse::toString(scenario.spectrumClass));
          json.attribute("peak_concurrent_regions",
                         scenario.peakConcurrentRegions);
          json.attribute("analytic_schedule_makespan_picoseconds",
                         scenario.makespanPicoseconds);
          writeReferenceArray(json, "system_mappings", scenario.systemMappings);
          json.attributeArray("states", [&] {
            for (const pnr::ResourceTimeScheduleState &state :
                 scenario.states) {
              const std::vector<std::uint8_t> event =
                  llvm::cantFail(dataflow::encodeDataflowReference(
                      spectrum.dataflow.artifact, state.event));
              json.object([&] {
                json.attribute("event", llvm::toHex(event, /*LowerCase=*/true));
                json.attribute("time_picoseconds", state.timePicoseconds);
                writeReference(json, "mapping", state.mapping);
                writeAllocations(json, "active", state.active);
              });
            }
          });
        });
      }
    });
  });
}

void writeSpectrumResult(llvm::json::OStream &json, llvm::StringRef name,
                         const dse::ResourceTimeSpectrumFunnelResult &result) {
  if (const auto *spectrum = std::get_if<dse::VerifiedResourceTimeSpectrum>(
          &result.verification)) {
    writeSpectrumSummary(json, name, *spectrum);
    return;
  }
  const auto &incomplete =
      std::get<dse::IncompleteResourceTimeSpectrum>(result.verification);
  json.attributeObject(name, [&] {
    json.attribute("status", "incomplete");
    json.attribute("reason", dse::resourceTimeSpectrumIncompleteReasonSpelling(
                                 incomplete.reason));
    json.attribute("diagnostic", incomplete.diagnostic);
    json.attribute("independently_imported_mapping_count",
                   incomplete.independentlyImportedMappingCount);
  });
}

llvm::Expected<std::vector<std::string>>
verifyResourceTimeEvidence(const ApplicationDeploymentArtifacts &application,
                           const PreparedApplicationBuild &prepared,
                           const ApplicationMappingExecution &execution,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs) {
  if (!application.resourceTimeTransitionGraph) {
    if (!application.resourceTimeTransitions.empty())
      return visualizationError("resource-time edge evidence has no finite "
                                "transition graph");
    return std::vector<std::string>{};
  }
  const pnr::ResourceTimeTransitionGraph &graph =
      *application.resourceTimeTransitionGraph;
  if (llvm::Error error =
          pnr::verifyResourceTimeTransitionGraph(graph, artifacts, blobs))
    return visualizationError("resource-time transition graph failed "
                              "independent closure: " +
                              llvm::toString(std::move(error)));
  const pnr::ResourceTimeTransitionEndpointReference expectedEntry{
      application.deployment.deployment().systemMapping(),
      application.deployment.reference()};
  if (graph.entry != expectedEntry ||
      !execution.execution.summary.selectedMapping ||
      *execution.execution.summary.selectedMapping != expectedEntry.mapping)
    return visualizationError("resource-time graph entry is not the selected "
                              "application Deployment");
  if (graph.transitions.size() != application.resourceTimeTransitions.size())
    return visualizationError("resource-time graph and edge evidence counts "
                              "disagree");
  for (const pnr::ResourceTimeTransition &graphEdge : graph.transitions)
    if (llvm::count_if(application.resourceTimeTransitions,
                       [&](const auto &evidence) {
                         return sameTransition(graphEdge, evidence.transition);
                       }) != 1)
      return visualizationError("resource-time graph edge does not have one "
                                "exact evidence record");

  std::vector<const ApplicationIncrementalMappingObservation *> pathSources;
  if (!graph.transitions.empty()) {
    if (!execution.provenance.resourceTimeMappingPath ||
        !execution.execution.summary.selectedPlanOrdinal)
      return visualizationError("resource-time graph has no ordered Mapping "
                                "path provenance");
    const ApplicationResourceTimeMappingPath &path =
        *execution.provenance.resourceTimeMappingPath;
    if (path.scheduleOwnerPlanOrdinal !=
            *execution.execution.summary.selectedPlanOrdinal ||
        path.scheduleOwnerPlanOrdinal >= prepared.mappingAlternatives.size() ||
        path.scheduleHintDigest !=
            prepared.mappingAlternatives[path.scheduleOwnerPlanOrdinal]
                .resourceTimeScheduleHintDigest ||
        path.observationOrdinals.size() != graph.transitions.size())
      return visualizationError("resource-time graph lost its exact schedule "
                                "path owner");
    ArtifactRootReference parentMapping = graph.entry.mapping;
    std::uint64_t parentPlanOrdinal = path.scheduleOwnerPlanOrdinal;
    for (std::size_t edgeOrdinal = 0;
         edgeOrdinal != graph.transitions.size(); ++edgeOrdinal) {
      const std::uint64_t observationOrdinal =
          path.observationOrdinals[edgeOrdinal];
      if (observationOrdinal >=
          execution.provenance.incrementalMappingObservations.size())
        return visualizationError("resource-time graph path has a foreign "
                                  "repair observation");
      const ApplicationIncrementalMappingObservation &source =
          execution.provenance
              .incrementalMappingObservations[observationOrdinal];
      const pnr::ResourceTimeTransition &edge = graph.transitions[edgeOrdinal];
      if (!source.verified || !source.childMapping ||
          source.parentPlanOrdinal != parentPlanOrdinal ||
          source.parentMapping != parentMapping ||
          edge.parent.mapping != source.parentMapping ||
          edge.child.mapping != *source.childMapping)
        return visualizationError("resource-time graph path does not match its "
                                  "ordered repair lineage");
      pathSources.push_back(&source);
      parentMapping = *source.childMapping;
      parentPlanOrdinal = source.childPlanOrdinal;
    }
  }

  std::vector<std::string> triggers;
  triggers.reserve(application.resourceTimeTransitions.size());
  for (const auto indexed :
       llvm::enumerate(application.resourceTimeTransitions)) {
    const ApplicationResourceTimeTransitionEvidence &evidence =
        indexed.value();
    if (!sameTransition(graph.transitions[indexed.index()],
                        evidence.transition))
      return visualizationError("resource-time edge evidence changed its "
                                "ordered graph position");
    if (llvm::Error error = pnr::verifyResourceTimeTransitionClosure(
            evidence.transition, artifacts, blobs))
      return visualizationError("resource-time transition evidence failed "
                                "independent closure: " +
                                llvm::toString(std::move(error)));
    for (const pnr::ResourceTimeTransitionEndpointReference *endpoint :
         {&evidence.transition.parent, &evidence.transition.child}) {
      if (!llvm::is_contained(graph.endpoints, *endpoint))
        return visualizationError("resource-time transition lost an exact "
                                  "endpoint catalog member");
    }
    if (llvm::count_if(graph.transitions, [&](const auto &candidate) {
          return sameTransition(candidate, evidence.transition);
        }) != 1)
      return visualizationError("resource-time edge evidence is not an exact "
                                "finite-graph member");
    const auto *parent = std::get_if<dse::VerifiedResourceTimeSpectrum>(
        &evidence.parentSpectrum.verification);
    const auto *child = std::get_if<dse::VerifiedResourceTimeSpectrum>(
        &evidence.childSpectrum.verification);
    if (!parent || !child)
      return visualizationError("resource-time transition retained an "
                                "unverified endpoint spectrum");
    auto parentMapping = mapping::importSystemMapping(
        evidence.transition.parent.mapping, artifacts);
    if (!parentMapping)
      return visualizationError("cannot independently import the parent "
                                "resource-time Mapping: " +
                                llvm::toString(parentMapping.takeError()));
    auto childMapping = mapping::importSystemMapping(
        evidence.transition.child.mapping, artifacts);
    if (!childMapping)
      return visualizationError("cannot independently import the child "
                                "resource-time Mapping: " +
                                llvm::toString(childMapping.takeError()));
    const ArtifactRootReference expectedDataflow{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        parentMapping->view().dataflowIdentity()};
    const ArtifactRootReference expectedFabric{
        fabric::fabricArtifactSchema.identity.str(),
        fabric::fabricArtifactSchema.version,
        parentMapping->view().fabricIdentity()};
    if (parent->dataflow != expectedDataflow ||
        parent->fabric != expectedFabric ||
        child->dataflow != expectedDataflow || child->fabric != expectedFabric)
      return visualizationError("resource-time spectrum names foreign "
                                "Dataflow or Fabric owners");

    auto canonicalDataflow =
        dataflow::importCanonicalDataflow(expectedDataflow, artifacts);
    if (!canonicalDataflow)
      return visualizationError("cannot independently import the resource-time "
                                "Dataflow: " +
                                llvm::toString(canonicalDataflow.takeError()));
    auto dataflowView = canonicalDataflow->view();
    if (!dataflowView)
      return visualizationError("cannot independently view the resource-time "
                                "Dataflow: " +
                                llvm::toString(dataflowView.takeError()));

    std::size_t parentMatches = 0;
    for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
         parent->scenarios) {
      if (scenario.states.empty() || scenario.systemMappings.empty())
        return visualizationError("parent spectrum scenario is empty");
      for (std::size_t mappingOrdinal = 0;
           mappingOrdinal != scenario.systemMappings.size(); ++mappingOrdinal) {
        const ArtifactRootReference &reference =
            scenario.systemMappings[mappingOrdinal];
        if (llvm::is_contained(llvm::ArrayRef(scenario.systemMappings)
                                   .take_front(mappingOrdinal),
                               reference) ||
            llvm::none_of(scenario.states, [&](const auto &state) {
              return state.mapping == reference;
            }))
          return visualizationError("parent spectrum Mapping inventory is not "
                                    "an exact state projection");
      }
      for (const pnr::ResourceTimeScheduleState &state : scenario.states) {
        if (!llvm::is_contained(scenario.systemMappings, state.mapping))
          return visualizationError("parent spectrum state is outside its "
                                    "Mapping inventory");
        if (llvm::Error error = dataflowView->validate(state.event))
          return visualizationError("parent spectrum state has a foreign "
                                    "Dataflow event: " +
                                    llvm::toString(std::move(error)));
      }
      std::size_t transitionOrdinal = 0;
      for (auto adjacent :
           llvm::zip(scenario.states, llvm::drop_begin(scenario.states))) {
        const pnr::ResourceTimeScheduleState &before = std::get<0>(adjacent);
        const pnr::ResourceTimeScheduleState &after = std::get<1>(adjacent);
        if (before.mapping == after.mapping)
          continue;
        if (transitionOrdinal >= scenario.transitions.transitions.size())
          return visualizationError("parent spectrum has an unbound Mapping "
                                    "change");
        const pnr::ResourceTimeTransition &candidate =
            scenario.transitions.transitions[transitionOrdinal++];
        if (candidate.parent.mapping != before.mapping ||
            candidate.child.mapping != after.mapping ||
            candidate.trigger != after.event ||
            !sameAllocations(candidate.beforeActive, before.active) ||
            !sameAllocations(candidate.afterActive, after.active))
          return visualizationError("parent spectrum transition does not bind "
                                    "its adjacent states");
        if (sameTransition(candidate, evidence.transition)) {
          ++parentMatches;
        }
      }
      if (transitionOrdinal != scenario.transitions.transitions.size())
        return visualizationError("parent spectrum has a transition without "
                                  "an adjacent Mapping change");
      if (!scenario.transitions.transitions.empty()) {
        if (!scenario.transitionGraph)
          return visualizationError("parent spectrum transition has no exact "
                                    "finite graph");
        if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
                *scenario.transitionGraph, artifacts, blobs))
          return visualizationError(
              "parent spectrum graph failed independent closure: " +
              llvm::toString(std::move(error)));
        if (scenario.transitionGraph->transitions.size() !=
            scenario.transitions.transitions.size())
          return visualizationError("parent spectrum graph and selected path "
                                    "counts disagree");
        for (const pnr::ResourceTimeTransition &graphEdge :
             scenario.transitionGraph->transitions)
          if (llvm::count_if(scenario.transitions.transitions,
                             [&](const auto &candidate) {
                               return sameTransition(graphEdge, candidate);
                             }) != 1)
            return visualizationError("parent spectrum graph does not match "
                                      "its exact selected path");
      } else if (scenario.transitionGraph)
        return visualizationError("parent spectrum has a graph without a "
                                  "Mapping transition");
    }
    if (parentMatches != 1)
      return visualizationError("parent spectrum does not bind one exact "
                                "resource-time edge and state boundary");

    bool childCarriesActiveWork = false;
    if (child->scenarios.empty())
      return visualizationError("child spectrum has no verified scenario");
    for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
         child->scenarios) {
      if (scenario.systemMappings.size() != 1 ||
          scenario.systemMappings.front() !=
              evidence.transition.child.mapping ||
          scenario.states.empty())
        return visualizationError("child spectrum is not restricted to the "
                                  "exact child Mapping");
      if (!scenario.transitions.transitions.empty() || scenario.transitionGraph)
        return visualizationError("single-Mapping child spectrum carries a "
                                  "foreign transition");
      for (const pnr::ResourceTimeScheduleState &state : scenario.states) {
        if (state.mapping != evidence.transition.child.mapping)
          return visualizationError("child spectrum state names another "
                                    "Mapping");
        if (llvm::Error error = dataflowView->validate(state.event))
          return visualizationError("child spectrum state has a foreign "
                                    "Dataflow event: " +
                                    llvm::toString(std::move(error)));
        childCarriesActiveWork |= !state.active.empty();
      }
    }
    if (!childCarriesActiveWork)
      return visualizationError("child spectrum carries no active state for "
                                "the exact child Mapping");
    if (!evidence.transition.safePoint)
      return visualizationError("verified resource-time transition lost its "
                                "safe point");
    if (evidence.repair.reopenedRoots.empty())
      return visualizationError("verified resource-time transition lost its "
                                "typed repair roots");
    for (std::size_t index = 0; index != evidence.repair.reopenedRoots.size();
         ++index)
      if (llvm::is_contained(
              llvm::ArrayRef(evidence.repair.reopenedRoots).take_front(index),
              evidence.repair.reopenedRoots[index]))
        return visualizationError(
            "verified resource-time transition repeats a repair root");
    if (evidence.repair.coldWallTimeNanoseconds == 0 ||
        evidence.repair.incrementalWallTimeNanoseconds == 0 ||
        evidence.repair.coldVerifierRetainedBytes == 0 ||
        evidence.repair.incrementalVerifierRetainedBytes == 0 ||
        evidence.repair.coldVerifierWork == 0 ||
        evidence.repair.incrementalVerifierWork == 0)
      return visualizationError("verified resource-time transition has no "
                                "paired cold/incremental measurement");
    if (!providerWorkClosed(evidence.repair.coldProviderWork) ||
        !providerWorkClosed(evidence.repair.incrementalProviderWork) ||
        evidence.repair.coldProviderWork.systemPnrInvocations == 0 ||
        evidence.repair.incrementalProviderWork.systemPnrInvocations == 0)
      return visualizationError("verified resource-time transition has an "
                                "unreconciled provider ledger");
    if ((!evidence.repair.coldDfgCycles && !evidence.repair.coldCgraCycles) ||
        (!evidence.repair.incrementalDfgCycles &&
         !evidence.repair.incrementalCgraCycles))
      return visualizationError("verified resource-time transition has no "
                                "paired runtime QoR measurement");

    const ApplicationIncrementalMappingObservation *source =
        pathSources[indexed.index()];
    if (!sameRepairEvidence(evidence.repair, *source))
      return visualizationError("resource-time repair evidence changed its "
                                "ordered source observation");
    if (!source->coldMapping ||
        source->disposition != dse::JointDesignAttemptDisposition::Verified ||
        source->incompleteReason ||
        source->parentPlanOrdinal >= prepared.mappingAlternatives.size() ||
        source->childPlanOrdinal >= prepared.mappingAlternatives.size())
      return visualizationError("resource-time repair source is incomplete");
    const PreparedApplicationMappingAlternative &parentAlternative =
        prepared.mappingAlternatives[source->parentPlanOrdinal];
    const PreparedApplicationMappingAlternative &childAlternative =
        prepared.mappingAlternatives[source->childPlanOrdinal];
    if (parentAlternative.resourceTimeScheduleHintDigest !=
            source->parentScheduleHintDigest ||
        childAlternative.resourceTimeScheduleHintDigest !=
            source->childScheduleHintDigest ||
        parentAlternative.dataflow != expectedDataflow ||
        childAlternative.dataflow != expectedDataflow ||
        parentAlternative.plan.pairOutputs.size() != 1 ||
        childAlternative.plan.pairOutputs.size() != 1 ||
        parentAlternative.plan.pairOutputs.front().pair.system !=
            expectedFabric ||
        childAlternative.plan.pairOutputs.front().pair.system !=
            expectedFabric ||
        source->childSystem != expectedFabric)
      return visualizationError("resource-time repair source lost its exact "
                                "plan and schedule lineage");
    auto derivedReopenedRoots = build_detail::deriveApplicationPartitionDelta(
        parentAlternative.plan, childAlternative.plan);
    if (!derivedReopenedRoots)
      return visualizationError(
          "cannot reproduce resource-time repair roots: " +
          llvm::toString(derivedReopenedRoots.takeError()));
    if (!sameUnorderedValues(llvm::ArrayRef(*derivedReopenedRoots),
                             llvm::ArrayRef(evidence.repair.reopenedRoots)))
      return visualizationError("resource-time repair roots disagree with "
                                "the exact partition delta");
    const auto pairedWallTime =
        llvm::checkedAddUnsigned(source->coldWallTimeNanoseconds,
                                 source->incrementalWallTimeNanoseconds);
    if (!pairedWallTime || source->wallTimeNanoseconds != *pairedWallTime)
      return visualizationError("resource-time repair wall-time observation "
                                "does not reconcile");
    if (source->coldDfgCycles.has_value() !=
            source->incrementalDfgCycles.has_value() ||
        source->coldCgraCycles.has_value() !=
            source->incrementalCgraCycles.has_value())
      return visualizationError("resource-time repair runtime measurements "
                                "use different QoR domains");

    const ApplicationMappingCandidateOutcome *childOutcome = nullptr;
    for (const ApplicationMappingCandidateOutcome &outcome :
         execution.candidateOutcomes) {
      if (outcome.planOrdinal != source->childPlanOrdinal ||
          outcome.resourceTimeScheduleHintDigest !=
              source->childScheduleHintDigest ||
          outcome.system != expectedFabric ||
          outcome.disposition != dse::JointDesignAttemptDisposition::Verified ||
          outcome.runtimeDisposition !=
              ApplicationMappingRuntimeDisposition::Completed ||
          !llvm::is_contained(outcome.systemMappings,
                              evidence.transition.child.mapping))
        continue;
      if (childOutcome)
        return visualizationError("resource-time repair has more than one "
                                  "exact child Mapping outcome");
      childOutcome = &outcome;
    }
    if (!childOutcome ||
        childOutcome->dfgCycles != evidence.repair.incrementalDfgCycles ||
        childOutcome->cgraCycles != evidence.repair.incrementalCgraCycles)
      return visualizationError("resource-time incremental QoR does not match "
                                "its exact Mapping outcome");

    auto coldImport = measureIndependentSystemMappingImport(
        *source->coldMapping, expectedDataflow.artifact,
        expectedFabric.artifact, artifacts);
    if (!coldImport)
      return visualizationError("cold Mapping replay failed: " +
                                llvm::toString(coldImport.takeError()));
    auto incrementalImport = measureIndependentSystemMappingImport(
        evidence.transition.child.mapping, expectedDataflow.artifact,
        expectedFabric.artifact, artifacts);
    if (!incrementalImport)
      return visualizationError("incremental Mapping replay failed: " +
                                llvm::toString(incrementalImport.takeError()));
    if (coldImport->retainedBytes !=
            evidence.repair.coldVerifierRetainedBytes ||
        coldImport->deterministicWork != evidence.repair.coldVerifierWork ||
        incrementalImport->retainedBytes !=
            evidence.repair.incrementalVerifierRetainedBytes ||
        incrementalImport->deterministicWork !=
            evidence.repair.incrementalVerifierWork)
      return visualizationError("resource-time repair verifier metrics do "
                                "not reproduce from exact Mapping roots");

    const auto &parentBindings = parentMapping->view().executionBindings();
    const auto &childBindings = childMapping->view().executionBindings();
    if (!sameUnorderedValues(parentBindings.spatialMappingImports(),
                             childBindings.spatialMappingImports()))
      return visualizationError("resource-time repair changed its immutable "
                                "SpatialMapping frontier");
    auto parentTech = deriveTechMappingIdentities(
        parentBindings.spatialMappingImports(), artifacts);
    if (!parentTech)
      return visualizationError("cannot derive parent TechMapping inventory: " +
                                llvm::toString(parentTech.takeError()));
    auto childTech = deriveTechMappingIdentities(
        childBindings.spatialMappingImports(), artifacts);
    if (!childTech)
      return visualizationError("cannot derive child TechMapping inventory: " +
                                llvm::toString(childTech.takeError()));
    if (!sameUnorderedValues(llvm::ArrayRef(*parentTech),
                             llvm::ArrayRef(*childTech)) ||
        evidence.repair.preservedSpatialMappings !=
            parentBindings.spatialMappingImports().size() ||
        evidence.repair.preservedTechMappings != parentTech->size() ||
        evidence.repair.repairedSpatialMappings != 0 ||
        evidence.repair.repairedTechMappings != 0)
      return visualizationError("resource-time Mapping preservation metrics "
                                "do not reconcile with exact artifacts");
    const std::uint64_t reopenedBindings =
        llvm::count_if(parentBindings.threadBindings(),
                       [&](const auto &row) {
                         return llvm::is_contained(
                             evidence.repair.reopenedRoots, row.key);
                       }) +
        llvm::count_if(parentBindings.graphBindings(), [&](const auto &row) {
          return llvm::is_contained(evidence.repair.reopenedRoots,
                                    row.key.rootThreadLaunch);
        });
    const std::uint64_t totalBindings = parentBindings.threadBindings().size() +
                                        parentBindings.graphBindings().size();
    if (evidence.repair.reopenedSystemBindings != reopenedBindings ||
        evidence.repair.preservedSystemBindings !=
            totalBindings - reopenedBindings)
      return visualizationError("resource-time repair binding metrics do not "
                                "reconcile with the parent Mapping");
    dse::JointMappingReuseDisposition expectedReuse =
        dse::JointMappingReuseDisposition::ColdFallback;
    if (evidence.repair.preservedTechMappings != 0 ||
        evidence.repair.preservedSpatialMappings != 0)
      expectedReuse = evidence.repair.repairedTechMappings != 0 ||
                              evidence.repair.repairedSpatialMappings != 0
                          ? dse::JointMappingReuseDisposition::LocalRepair
                          : dse::JointMappingReuseDisposition::Preserved;
    if (evidence.repair.reuseDisposition != expectedReuse)
      return visualizationError("resource-time repair disposition disagrees "
                                "with its Mapping accounting");
    for (const dataflow::RootThreadLaunchRef root :
         evidence.repair.reopenedRoots) {
      const auto allocationNamesRoot = [&](const auto &allocations) {
        return llvm::any_of(allocations, [&](const auto &allocation) {
          return allocation.region == root;
        });
      };
      if (!allocationNamesRoot(evidence.transition.beforeActive) &&
          !allocationNamesRoot(evidence.transition.afterActive) &&
          !llvm::is_contained(evidence.transition.completedBefore, root))
        return visualizationError("resource-time repair root is outside its "
                                  "event-relative transition cone");
    }
    auto trigger = dataflow::encodeDataflowReference(
        evidence.transition.safePoint->artifact.artifact,
        evidence.transition.trigger);
    if (!trigger)
      return visualizationError("cannot encode a resource-time trigger: " +
                                llvm::toString(trigger.takeError()));
    triggers.push_back(llvm::toHex(*trigger, /*LowerCase=*/true));
  }
  return triggers;
}

llvm::Error writeBundle(llvm::StringRef destination,
                        const fabric::FinalizedFabricRoot &system,
                        const PreparedApplicationBuild &prepared,
                        const ApplicationMappingExecution &execution,
                        const ApplicationDeploymentArtifacts &deployment,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  if (!execution.provenance.pairDecision)
    return visualizationError("completed Mapping has no pair decision");
  auto transitionTriggers = verifyResourceTimeEvidence(
      deployment, prepared, execution, artifacts, blobs);
  if (!transitionTriggers)
    return transitionTriggers.takeError();

  std::vector<ArtifactRootReference> structuredPrograms;
  std::vector<ArtifactRootReference> dataflows;
  for (const PreparedApplicationSoftware &software : prepared.software) {
    auto structured = frontend::importStructuredProgram(
        software.compilation.structuredProgram, artifacts);
    if (!structured)
      return visualizationError(
          "cannot strictly import a Structured Program: " +
          llvm::toString(structured.takeError()));
    auto importedDataflow = dataflow::importCanonicalDataflow(
        software.compilation.canonicalDataflow, artifacts);
    if (!importedDataflow)
      return visualizationError(
          "cannot strictly import a Canonical Dataflow: " +
          llvm::toString(importedDataflow.takeError()));
    structuredPrograms.push_back(software.compilation.structuredProgram);
    dataflows.push_back(software.compilation.canonicalDataflow);
  }
  canonicalize(structuredPrograms);
  canonicalize(dataflows);

  std::vector<ArtifactRootReference> systemMappings;
  std::vector<ArtifactRootReference> spatialMappings;
  std::vector<ArtifactRootReference> techMappings;
  std::vector<ArtifactRootReference> runtimeEvidence;
  for (const ApplicationMappingCandidateOutcome &outcome :
       execution.candidateOutcomes) {
    systemMappings.insert(systemMappings.end(), outcome.systemMappings.begin(),
                          outcome.systemMappings.end());
    runtimeEvidence.insert(runtimeEvidence.end(),
                           outcome.runtimeEvidence.begin(),
                           outcome.runtimeEvidence.end());
  }
  canonicalize(systemMappings);
  canonicalize(runtimeEvidence);
  for (const ArtifactRootReference &reference : systemMappings) {
    auto imported = mapping::importSystemMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a SystemMapping: " +
                                llvm::toString(imported.takeError()));
    if (imported->view().fabricIdentity() != system.reference().artifact)
      return visualizationError("SystemMapping names a foreign Fabric");
    spatialMappings.insert(
        spatialMappings.end(),
        imported->view().executionBindings().spatialMappingImports().begin(),
        imported->view().executionBindings().spatialMappingImports().end());
  }
  canonicalize(spatialMappings);
  for (const ArtifactRootReference &reference : spatialMappings) {
    auto imported = mapping::importSpatialMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a SpatialMapping: " +
                                llvm::toString(imported.takeError()));
    techMappings.push_back({mapping::mappingArtifactSchema.identity.str(),
                            mapping::mappingArtifactSchema.version,
                            imported->view().techMappingIdentity()});
  }
  canonicalize(techMappings);
  for (const ArtifactRootReference &reference : techMappings) {
    auto imported = mapping::importTechMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a TechMapping: " +
                                llvm::toString(imported.takeError()));
  }
  for (const ArtifactRootReference &reference : runtimeEvidence) {
    auto evidence = artifacts.get(reference);
    if (!evidence)
      return visualizationError("runtime Evidence reference is unavailable: " +
                                llvm::toString(evidence.takeError()));
  }

  llvm::SmallString<256> bundlePath(destination);
  llvm::sys::path::append(bundlePath, "viz.bundle.json");
  llvm::SmallString<256> temporaryModel(bundlePath);
  temporaryModel.append(".tmp-%%%%%%");
  auto temporary = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporary)
    return visualizationError("cannot create bundle output: " +
                              llvm::toString(temporary.takeError()));
  llvm::scope_exit discard([&] { llvm::consumeError(temporary->discard()); });
  {
    llvm::raw_fd_ostream output(temporary->FD, false);
    llvm::json::OStream json(output, 2);
    json.object([&] {
      json.attribute("schema", "loom.visualization_bundle");
      json.attribute("version", "1.2");
      json.attributeObject("fabric", [&] {
        writeArtifactRootReferenceJsonFields(json, system.reference());
      });
      writeReferenceArray(json, "structured_programs", structuredPrograms);
      writeReferenceArray(json, "canonical_dataflows", dataflows);
      writeReferenceArray(json, "tech_mappings", techMappings);
      writeReferenceArray(json, "spatial_mappings", spatialMappings);
      writeReferenceArray(json, "system_mappings", systemMappings);
      writeReferenceArray(json, "runtime_evidence", runtimeEvidence);
      json.attribute("pair_decision", projectApplicationPairDecisionJson(
                                          *execution.provenance.pairDecision));
      json.attributeObject("configuration_abi", [&] {
        writeArtifactRootReferenceJsonFields(json, deployment.configurationAbi);
      });
      writeReferenceArray(
          json, "configuration_images",
          deployment.deployment.deployment().configurationImages());
      json.attributeObject("deployment", [&] {
        writeArtifactRootReferenceJsonFields(json,
                                             deployment.deployment.reference());
      });
      if (deployment.resourceTimeSpectrum)
        writeSpectrumResult(json, "resource_time_spectrum",
                            *deployment.resourceTimeSpectrum);
      else
        json.attribute("resource_time_spectrum", nullptr);
      json.attributeArray("resource_time_endpoints", [&] {
        if (deployment.resourceTimeTransitionGraph)
          for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
               deployment.resourceTimeTransitionGraph->endpoints)
            json.object([&] {
              writeReference(json, "mapping", endpoint.mapping);
              writeReference(json, "deployment", *endpoint.deployment);
            });
      });
      json.attributeArray("resource_time_transitions", [&] {
        for (const auto indexed :
             llvm::enumerate(deployment.resourceTimeTransitions)) {
          const ApplicationResourceTimeTransitionEvidence &evidence =
              indexed.value();
          const pnr::ResourceTimeTransition &transition = evidence.transition;
          json.object([&] {
            json.attribute("trigger", (*transitionTriggers)[indexed.index()]);
            json.attributeObject("safe_point", [&] {
              writeReference(json, "artifact", transition.safePoint->artifact);
              json.attribute("kind", pnr::resourceTimeSafePointKindSpelling(
                                         transition.safePoint->kind));
            });
            json.attributeObject("parent", [&] {
              writeReference(json, "mapping", transition.parent.mapping);
              writeReference(json, "deployment", *transition.parent.deployment);
            });
            json.attributeObject("child", [&] {
              writeReference(json, "mapping", transition.child.mapping);
              writeReference(json, "deployment", *transition.child.deployment);
            });
            writeAllocations(json, "before_active", transition.beforeActive);
            writeAllocations(json, "after_active", transition.afterActive);
            json.attributeArray("completed_before", [&] {
              for (const auto root : transition.completedBefore) {
                json.object([&] {
                  json.attribute("artifact",
                                 formatArtifactIdentityHex(root.artifact));
                  json.attribute("entity", root.entity.value());
                });
              }
            });
            writeReferenceArray(json, "before_live_work",
                                transition.beforeLiveWork);
            writeReferenceArray(json, "after_live_work",
                                transition.afterLiveWork);
            if (transition.tokenLiveStateCorrespondence)
              writeReference(json, "token_live_state_correspondence",
                             *transition.tokenLiveStateCorrespondence);
            else
              json.attribute("token_live_state_correspondence", nullptr);
            json.attribute(
                "resource_delta",
                formatComponentViewDigestHex(*transition.resourceDeltaDigest));
            json.attribute("configuration_delta",
                           formatComponentViewDigestHex(
                               *transition.configurationDeltaDigest));
            json.attribute("route_delta", formatComponentViewDigestHex(
                                              *transition.routeDeltaDigest));
            json.attribute("reprogramming_time_picoseconds",
                           *transition.reprogrammingTimePicoseconds);
            json.attribute("migration_time_picoseconds",
                           *transition.migrationTimePicoseconds);
            json.attribute("status", pnr::resourceTimeTransitionStatusSpelling(
                                         transition.status));
            json.attributeObject("repair", [&] {
              json.attribute("mapping_reuse_disposition",
                             dse::jointMappingReuseDispositionSpelling(
                                 evidence.repair.reuseDisposition));
              json.attributeArray("reopened_roots", [&] {
                for (const dataflow::RootThreadLaunchRef root :
                     evidence.repair.reopenedRoots) {
                  json.object([&] {
                    json.attribute("artifact",
                                   formatArtifactIdentityHex(root.artifact));
                    json.attribute("entity", root.entity.value());
                  });
                }
              });
              json.attribute("preserved_tech_mappings",
                             evidence.repair.preservedTechMappings);
              json.attribute("preserved_spatial_mappings",
                             evidence.repair.preservedSpatialMappings);
              json.attribute("repaired_tech_mappings",
                             evidence.repair.repairedTechMappings);
              json.attribute("repaired_spatial_mappings",
                             evidence.repair.repairedSpatialMappings);
              json.attribute("preserved_system_bindings",
                             evidence.repair.preservedSystemBindings);
              json.attribute("reopened_system_bindings",
                             evidence.repair.reopenedSystemBindings);
              json.attribute("cold_wall_time_ns",
                             evidence.repair.coldWallTimeNanoseconds);
              json.attribute("incremental_wall_time_ns",
                             evidence.repair.incrementalWallTimeNanoseconds);
              json.attribute("cold_verifier_retained_bytes",
                             evidence.repair.coldVerifierRetainedBytes);
              json.attribute("incremental_verifier_retained_bytes",
                             evidence.repair.incrementalVerifierRetainedBytes);
              json.attribute("cold_verifier_work",
                             evidence.repair.coldVerifierWork);
              json.attribute("incremental_verifier_work",
                             evidence.repair.incrementalVerifierWork);
              writeProviderWork(json, "cold_provider_work",
                                evidence.repair.coldProviderWork);
              writeProviderWork(json, "incremental_provider_work",
                                evidence.repair.incrementalProviderWork);
              const auto optional = [&](llvm::StringRef name,
                                        std::optional<std::uint64_t> value) {
                if (value)
                  json.attribute(name, *value);
                else
                  json.attribute(name, nullptr);
              };
              optional("cold_dfg_cycles", evidence.repair.coldDfgCycles);
              optional("cold_cgra_cycles", evidence.repair.coldCgraCycles);
              optional("incremental_dfg_cycles",
                       evidence.repair.incrementalDfgCycles);
              optional("incremental_cgra_cycles",
                       evidence.repair.incrementalCgraCycles);
            });
            writeSpectrumResult(json, "parent_spectrum",
                                 evidence.parentSpectrum);
            writeSpectrumResult(json, "child_spectrum", evidence.childSpectrum);
          });
        }
      });
    });
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      return visualizationError("cannot write bundle output: " +
                                error.message());
    }
  }
  if (llvm::Error error = temporary->keep(bundlePath))
    return visualizationError("cannot publish bundle output: " +
                              llvm::toString(std::move(error)));
  discard.release();
  return llvm::Error::success();
}

} // namespace

llvm::Error exportProductVisualization(
    llvm::StringRef destination, const fabric::FinalizedFabricRoot &system,
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mapping,
    const ApplicationDeploymentArtifacts &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (destination.empty())
    return visualizationError("destination is empty");
  if (std::error_code error = llvm::sys::fs::create_directories(destination))
    return visualizationError("cannot create destination: " + error.message());
  if (!llvm::sys::fs::is_directory(destination))
    return visualizationError("destination is not a directory");

  llvm::SmallString<256> fabricBase(destination);
  llvm::sys::path::append(fabricBase, "fabric");
  if (llvm::Error error =
          adg::exportFabricDesign(system, artifacts, fabricBase))
    return visualizationError(llvm::toString(std::move(error)));
  return writeBundle(destination, system, prepared, mapping, deployment,
                     artifacts, blobs);
}

} // namespace loom::application
