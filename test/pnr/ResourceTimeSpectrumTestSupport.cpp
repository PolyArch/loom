#include "SystemCandidateStateTestSupport.h"

#include "DSE/ResourceTimeSpectrum.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <variant>

namespace loom::pnr::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "resource-time spectrum test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

::dataflow::EventFamilyKey startEvent(::dataflow::RootThreadLaunchRef root) {
  const ::dataflow::RootThreadBoundaryTransferRef transfer(
      ::dataflow::RootThreadStartTransferRef{root});
  return ::dataflow::EventFamilyKey(::dataflow::StaticTransferEventRef(
      ::dataflow::ConsumedTransferEventRef{::dataflow::CanonicalSinkTerminalRef(
          ::dataflow::RootThreadBoundarySinkRef{transfer})}));
}

::dataflow::EventFamilyKey
completionEvent(::dataflow::RootThreadLaunchRef root) {
  const ::dataflow::RootThreadBoundaryTransferRef transfer(
      ::dataflow::RootThreadCompletionTransferRef{root});
  return ::dataflow::EventFamilyKey(
      ::dataflow::StaticTransferEventRef(::dataflow::ProducedTransferEventRef{
          ::dataflow::CanonicalProducerTerminalRef(
              ::dataflow::RootThreadBoundarySourceRef{transfer})}));
}

std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> resourcesForRoot(
    ::dataflow::RootThreadLaunchRef root,
    const ::loom::mapping::SystemExecutionContextProjection &contexts) {
  std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
  const auto append = [&](::loom::fabric::AccCoreOccurrenceRef core) {
    resources.push_back(
        take(::loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
            ::loom::fabric::FabricInventoryOwnerRef::of(core))));
  };
  for (const auto &domain : contexts.instructionDomains)
    if (domain.root == root)
      append(domain.context.accCore);
  for (const auto &domain : contexts.spatialDomains)
    if (domain.graph.rootThreadLaunch == root)
      append(domain.context.accCore);
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  return resources;
}

} // namespace

void verifyResourceTimeSpectrumWorkflow(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::FinalizedSystemMapping &mapping,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    ArtifactStore &store) {
  require(roots.size() == 2,
          "fixture does not expose two mapped resource-time regions");
  const auto contexts = take(mapping::projectSystemExecutionContexts(
      dataflow, mapping.view().executionBindings()));
  const std::vector firstResources = resourcesForRoot(roots[0], contexts);
  const std::vector secondResources = resourcesForRoot(roots[1], contexts);
  require(!firstResources.empty() && !secondResources.empty(),
          "verified Mapping omitted a resource-time allocation");
  require(llvm::none_of(firstResources,
                        [&](const auto &resource) {
                          return llvm::is_contained(secondResources, resource);
                        }),
          "simultaneous fixture roots share an AccCore");

  const std::vector<::dataflow::RootThreadLaunchRef> scheduleRegions = {
      roots[0], roots[1]};
  const auto allocation = [&](std::size_t ordinal) {
    return ResourceTimeRegionAllocation{scheduleRegions[ordinal],
                                        ordinal == 0 ? firstResources
                                                     : secondResources};
  };

  ResourceTimeScheduleScenario spatial;
  spatial.executions = {{scheduleRegions[0], {}, 0, 0, 10},
                        {scheduleRegions[1], {}, 0, 0, 10}};
  spatial.states = {{mapping.reference(),
                     startEvent(roots[0]),
                     0,
                     {allocation(0), allocation(1)}},
                    {mapping.reference(), completionEvent(roots[0]), 10, {}}};
  spatial.makespanPicoseconds = 10;

  ResourceTimeScheduleScenario temporal;
  temporal.executions = {
      {scheduleRegions[0], {}, 0, 0, 10},
      {scheduleRegions[1],
       {{scheduleRegions[0], ResourceTimeReadinessKind::Completion}},
       10,
       10,
       20}};
  temporal.states = {
      {mapping.reference(), startEvent(roots[0]), 0, {allocation(0)}},
      {mapping.reference(), startEvent(roots[1]), 10, {allocation(1)}},
      {mapping.reference(), completionEvent(roots[1]), 20, {}}};
  temporal.makespanPicoseconds = 20;

  const ResourceTimeScheduleWitness witness{
      scheduleRegions,
      {spatial, temporal},
      1,
      2,
      ResourceTimeConcurrencyBoundStatus::Exact};
  const std::vector<dse::ResourceTimeRegionMapping> correspondence = {
      {roots[0], firstResources.size(), firstResources.size(),
       dse::ResourceTimeEstimateSupport::Exact,
       dse::ResourceTimeEstimateSupport::Exact, 1},
      {roots[1], secondResources.size(), secondResources.size(),
       dse::ResourceTimeEstimateSupport::Exact,
       dse::ResourceTimeEstimateSupport::Exact, 1}};
  const auto result =
      take(dse::verifyResourceTimeSpectrum(witness, correspondence, store));
  const auto *verified =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&result);
  require(verified && verified->scenarios.size() == 2,
          "real SystemMapping spectrum did not verify");
  require(verified->scenarios[0].spectrumClass ==
                  dse::PreMappingSpectrumClass::MaxSpatial &&
              verified->scenarios[1].spectrumClass ==
                  dse::PreMappingSpectrumClass::MaxTemporal,
          "real SystemMapping allocations did not establish both endpoints");

  auto unprovenBounds = witness;
  unprovenBounds.concurrencyBoundStatus =
      ResourceTimeConcurrencyBoundStatus::ProofNotEstablished;
  const auto unprovenResult = take(
      dse::verifyResourceTimeSpectrum(unprovenBounds, correspondence, store));
  const auto *unprovenVerified =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&unprovenResult);
  require(unprovenVerified &&
              llvm::all_of(unprovenVerified->scenarios,
                           [](const auto &row) {
                             return row.spectrumClass ==
                                    dse::PreMappingSpectrumClass::Intermediate;
                           }),
          "unproven concurrency bounds produced an endpoint label");

  ResourceTimeScheduleScenario singleton;
  singleton.executions = {{scheduleRegions[0], {}, 0, 0, 10}};
  singleton.states = {
      {mapping.reference(), startEvent(roots[0]), 0, {allocation(0)}},
      {mapping.reference(), completionEvent(roots[0]), 10, {}}};
  singleton.makespanPicoseconds = 10;
  const ResourceTimeScheduleWitness singletonWitness{
      {scheduleRegions[0]},
      {singleton},
      1,
      1,
      ResourceTimeConcurrencyBoundStatus::Exact};
  const auto singletonResult = take(dse::verifyResourceTimeSpectrum(
      singletonWitness, {correspondence.front()}, store));
  const auto *singletonVerified =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&singletonResult);
  require(singletonVerified && singletonVerified->scenarios.size() == 1 &&
              singletonVerified->scenarios.front().spectrumClass ==
                  dse::PreMappingSpectrumClass::Intermediate,
          "singleton coverage was incorrectly used as an endpoint proof");

  auto partitionedCorrespondence = correspondence;
  partitionedCorrespondence[0].logicalEpochCount = 2;
  const auto partitionedResult = take(dse::verifyResourceTimeSpectrum(
      witness, partitionedCorrespondence, store));
  const auto *partitionedVerified =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&partitionedResult);
  require(partitionedVerified &&
              partitionedVerified->scenarios[1].spectrumClass ==
                  dse::PreMappingSpectrumClass::MaxTemporal,
          "partition count changed an otherwise explicit temporal endpoint");

  auto mismatched = witness;
  mismatched.scenarios[1].states[0].active[0].resources = secondResources;
  const auto mismatchResult =
      take(dse::verifyResourceTimeSpectrum(mismatched, correspondence, store));
  const auto *incomplete =
      std::get_if<dse::IncompleteResourceTimeSpectrum>(&mismatchResult);
  require(
      incomplete &&
          incomplete->reason ==
              dse::ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
      "allocation mismatch was not kept as typed incomplete");

  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      mapping.view().fabricIdentity()};
  const auto modelDigest = take(dse::resourceTimeAnalyticModelSnapshotDigest());
  const dse::ResourceTimeInvocationKey invocation{
      mapping.reference(), dataflowReference,
      fabricReference,     mapping.reference(),
      mapping.reference(), modelDigest,
      modelDigest,         "main",
      std::nullopt};
  const std::vector<dse::ResourceTimeRegionFeature> features = {
      dse::ResourceTimeRegionFeature{
          roots[0],
          {},
          {dse::ResourceTimeSpeedupPoint{
              {firstResources.size()},
              10,
              std::nullopt,
              std::nullopt,
              0,
              0,
              0,
              dse::ResourceTimeEstimateSupport::Exact}},
          1,
          true,
          {}},
      dse::ResourceTimeRegionFeature{
          roots[1],
          {},
          {dse::ResourceTimeSpeedupPoint{
              {secondResources.size()},
              10,
              std::nullopt,
              std::nullopt,
              0,
              0,
              0,
              dse::ResourceTimeEstimateSupport::Exact}},
          1,
          true,
          {}}};
  dse::ResourceTimeFrontierPolicy frontierPolicy;
  frontierPolicy.availableResourceUnits = {firstResources.size() +
                                           secondResources.size()};
  frontierPolicy.maximumStatesGenerated = 256;
  frontierPolicy.maximumActionsGenerated = 1024;
  frontierPolicy.maximumStateCacheEntries = 256;
  frontierPolicy.beamWidth = 64;
  frontierPolicy.maximumFinalists = 8;
  const auto frontier = take(dse::exploreResourceTimeFrontier(
      invocation, {fabricReference}, features, frontierPolicy));
  const auto *completed =
      std::get_if<dse::CompletedResourceTimeFrontier>(&frontier);
  require(completed && !completed->finalists.empty(),
          "real Mapping fixture produced no schedule finalists");
  const std::vector<dse::ResourceTimeRegionResourceBound> bounds = {
      {roots[0], firstResources.size(), dse::ResourceTimeEstimateSupport::Exact,
       firstResources.size(), dse::ResourceTimeEstimateSupport::Exact},
      {roots[1], secondResources.size(),
       dse::ResourceTimeEstimateSupport::Exact, secondResources.size(),
       dse::ResourceTimeEstimateSupport::Exact}};
  const auto funnel = take(dse::verifyResourceTimeMappingFinalists(
      completed->finalists, features, bounds, {mapping.reference()}, store, {},
      completed->concurrencyBounds));
  const auto *funnelVerified =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&funnel.verification);
  require(funnelVerified && funnel.accounting.materializedScenarios != 0 &&
              funnel.accounting.verifiedScenarios ==
                  funnel.accounting.materializedScenarios &&
              funnel.accounting.mappingImportRequests != 0 &&
              funnel.accounting.mappingImportCacheMisses != 0 &&
              funnel.accounting.mappingImportCacheHits != 0,
          "bounded schedule funnel did not reach the independent Mapping "
          "verifier through its shared import session");

  auto unprovenMinimumBounds = bounds;
  for (auto &bound : unprovenMinimumBounds) {
    bound.minimumFeasibleResourceUnits = 0;
    bound.minimumSupport = dse::ResourceTimeEstimateSupport::Exact;
  }
  const auto unprovenMinimum = take(dse::verifyResourceTimeMappingFinalists(
      completed->finalists, features, unprovenMinimumBounds,
      {mapping.reference()}, store, {}, completed->concurrencyBounds));
  const auto *unprovenMinimumSpectrum =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(
          &unprovenMinimum.verification);
  require(unprovenMinimumSpectrum &&
              llvm::none_of(unprovenMinimumSpectrum->scenarios,
                            [](const auto &scenario) {
                              return scenario.spectrumClass ==
                                     dse::PreMappingSpectrumClass::MaxSpatial;
                            }),
          "observed one-core Mapping fabricated an exact minimum bound");

  ::loom::mapping::SystemMappingImportSession applicationSession(store, 8);
  const auto firstJoin = take(dse::verifyResourceTimeMappingFinalists(
      completed->finalists, features, bounds, {mapping.reference()}, store, {},
      completed->concurrencyBounds));
  const auto firstStats = applicationSession.statistics();
  const auto secondJoin = take(dse::verifyResourceTimeMappingFinalists(
      completed->finalists, features, bounds, {mapping.reference()}, store, {},
      completed->concurrencyBounds));
  const auto secondStats = applicationSession.statistics();
  require(std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
              firstJoin.verification) &&
              std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
                  secondJoin.verification) &&
              secondStats.cacheHits > firstStats.cacheHits,
          "application-local Spectrum joins did not reuse Mapping imports");

  auto misorderedBounds = bounds;
  std::swap(misorderedBounds[0], misorderedBounds[1]);
  auto rejectedBounds = dse::verifyResourceTimeMappingFinalists(
      completed->finalists, features, misorderedBounds, {mapping.reference()},
      store, {}, completed->concurrencyBounds);
  require(!rejectedBounds,
          "Spectrum accepted bounds that were not in canonical region order");
  llvm::consumeError(rejectedBounds.takeError());
}

} // namespace loom::pnr::test
