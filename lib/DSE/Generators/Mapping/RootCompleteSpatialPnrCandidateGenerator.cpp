#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"

#include "DSE/MappingCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/FabricTopologyQualityDiagnostic.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  TechMappingCandidatesInput,
  FabricInput,
  PhysicalTimingProfileInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(TechMappingCandidatesInput),
         "tech_mapping", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(PhysicalTimingProfileInput),
         "physical_timing_profile", PlanValueRole::CandidateSet,
         &::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "spatial_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation);

const CandidateGeneratorDescriptor descriptor{
    rootCompleteSpatialPnrCandidateGeneratorKind,
    "mapping.root_complete_spatial_pnr",
    "loom.mapping.root_complete_spatial_pnr.generator.v18",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    pnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
};

struct CachedDataflow final {
  ::dataflow::CanonicalDataflowArtifact artifact;
  ::dataflow::CanonicalDataflowProgramView view;
};

struct DataflowImportStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
};

struct ActiveProblemCacheStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t rankProjectionCount = 0;
  std::uint64_t rankProjectionNanoseconds = 0;
  std::uint64_t rankAssignmentAttempts = 0;
  std::uint64_t rankEndpointExpansions = 0;
  std::uint64_t rankNegotiationIterations = 0;
};

struct ActiveRouteRank final {
  ::loom::dse::ObjectiveVector objective;
  std::array<std::uint64_t, ::loom::resolvedPnrViolationKindCount> violations;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
};

struct PreparedTechCandidate final {
  std::size_t inputOrdinal = 0;
  ArtifactRootReference reference;
  ::loom::mapping::FinalizedTechMapping tech;
  CachedDataflow *dataflow = nullptr;
  ::loom::mapping::FinalizedSpatialMappingConstraintSet constraints;
  ::loom::pnr::FrozenSpatialPnrProblemHandle activeProblem;
  std::optional<::loom::pnr::SpatialCandidateInitializerPreference>
      preference;
  std::optional<ActiveRouteRank> routeRank;
};

void saturatingAdd(std::uint64_t &value, std::uint64_t amount) {
  value = amount > std::numeric_limits<std::uint64_t>::max() - value
              ? std::numeric_limits<std::uint64_t>::max()
              : value + amount;
}

template <typename T>
void accountArray(llvm::ArrayRef<T> values, std::uint64_t &bytes,
                  std::uint64_t &work) {
  const std::uint64_t count = values.size();
  const std::uint64_t elementBytes = sizeof(T);
  saturatingAdd(bytes,
                count > std::numeric_limits<std::uint64_t>::max() / elementBytes
                    ? std::numeric_limits<std::uint64_t>::max()
                    : count * elementBytes);
  saturatingAdd(work, count);
}

void accountRetainedDataflow(const CachedDataflow &cached,
                             DataflowImportStatistics &statistics) {
  std::uint64_t bytes = sizeof(CachedDataflow);
  std::uint64_t work = cached.artifact.canonicalBytes().bytes().size();
  saturatingAdd(bytes, cached.artifact.canonicalBytes().bytes().size());
  accountArray(cached.view.graphs(), bytes, work);
  accountArray(cached.view.actors(), bytes, work);
  accountArray(cached.view.rootThreadLaunches(), bytes, work);
  accountArray(cached.view.staticGraphLaunches(), bytes, work);
  accountArray(cached.view.logicalMemoryRoots(), bytes, work);
  saturatingAdd(statistics.retainedBytes, bytes);
  saturatingAdd(statistics.deterministicWork, work);
}

void emitDataflowImportStatistics(const DataflowImportStatistics &statistics) {
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::DerivedContext,
      [&](llvm::json::Object &fields) {
        fields["context_kind"] = "root_complete_spatial_dataflow_import";
        fields["cache_requests"] = statistics.requests;
        fields["cache_hits"] = statistics.hits;
        fields["cache_misses"] = statistics.misses;
        fields["construction_count"] = statistics.misses;
        fields["construction_time_ns"] = statistics.constructionNanoseconds;
        fields["retained_bytes"] = statistics.retainedBytes;
        fields["deterministic_work"] = statistics.deterministicWork;
      });
}

void emitActiveProblemCacheStatistics(
    const ActiveProblemCacheStatistics &statistics) {
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::DerivedContext,
      [&](llvm::json::Object &fields) {
        fields["context_kind"] = "root_complete_spatial_active_problem";
        fields["cache_requests"] = statistics.requests;
        fields["cache_hits"] = statistics.hits;
        fields["cache_misses"] = statistics.misses;
        fields["construction_count"] = statistics.misses;
        fields["construction_time_ns"] = statistics.constructionNanoseconds;
        fields["retained_bytes"] = statistics.retainedBytes;
        fields["deterministic_work"] = statistics.deterministicWork;
        fields["rank_projection_count"] = statistics.rankProjectionCount;
        fields["rank_projection_time_ns"] =
            statistics.rankProjectionNanoseconds;
        fields["rank_assignment_attempts"] =
            statistics.rankAssignmentAttempts;
        fields["rank_endpoint_expansions"] =
            statistics.rankEndpointExpansions;
        fields["rank_negotiation_iterations"] =
            statistics.rankNegotiationIterations;
      });
}

bool activeDemandRankLess(const PreparedTechCandidate &lhs,
                          const PreparedTechCandidate &rhs) {
  if (lhs.preference.has_value() != rhs.preference.has_value())
    return lhs.preference.has_value();
  if (lhs.preference && rhs.preference) {
    const auto lhsRank = std::tie(
        lhs.preference->residualExternalSinkCount,
        lhs.preference->topologyRefinementUnreachableSelectionCount,
        lhs.preference->topologyUnreachableSelectionCount,
        lhs.preference->topologyRefinementHopSum,
        lhs.preference->topologyHopSum,
        lhs.preference->maximumEndpointSelections,
        lhs.preference->maximumComputeOccurrenceSelections,
        lhs.preference->staticSchedulePressure);
    const auto rhsRank = std::tie(
        rhs.preference->residualExternalSinkCount,
        rhs.preference->topologyRefinementUnreachableSelectionCount,
        rhs.preference->topologyUnreachableSelectionCount,
        rhs.preference->topologyRefinementHopSum,
        rhs.preference->topologyHopSum,
        rhs.preference->maximumEndpointSelections,
        rhs.preference->maximumComputeOccurrenceSelections,
        rhs.preference->staticSchedulePressure);
    if (lhsRank != rhsRank)
      return lhsRank < rhsRank;
  }
  return artifactRootReferenceLess(lhs.reference, rhs.reference);
}

llvm::Expected<bool> activeRouteRankLess(const PreparedTechCandidate &lhs,
                                         const PreparedTechCandidate &rhs) {
  if (lhs.routeRank.has_value() != rhs.routeRank.has_value())
    return lhs.routeRank.has_value();
  if (lhs.routeRank && rhs.routeRank) {
    auto comparison = lhs.activeProblem->objectiveProgram().compareSelectedRank(
        lhs.routeRank->objective, {}, rhs.routeRank->objective, {});
    if (!comparison)
      return comparison.takeError();
    if (*comparison != 0)
      return *comparison < 0;
  }
  return activeDemandRankLess(lhs, rhs);
}

llvm::Error orderPreparedCandidates(
    std::vector<PreparedTechCandidate> &candidates) {
  for (std::size_t index = 1; index < candidates.size(); ++index) {
    PreparedTechCandidate candidate = std::move(candidates[index]);
    std::size_t insertion = index;
    while (insertion != 0) {
      auto before = activeRouteRankLess(candidate, candidates[insertion - 1]);
      if (!before)
        return before.takeError();
      if (!*before)
        break;
      candidates[insertion] = std::move(candidates[insertion - 1]);
      --insertion;
    }
    candidates[insertion] = std::move(candidate);
  }
  return llvm::Error::success();
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

void canonicalizeReferences(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

std::vector<CandidateGeneratorLineageEdge>
mechanicalLineage(llvm::ArrayRef<ArtifactRootReference> outputs) {
  std::vector<CandidateGeneratorLineageEdge> lineage;
  lineage.reserve(outputs.size());
  for (const ArtifactRootReference &output : outputs)
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
        CandidateGeneratorOutputSlotRef(0),
        output,
        {},
        {}});
  return lineage;
}

CompletedCandidateGeneratorResult
completed(std::vector<ArtifactRootReference> outputs) {
  canonicalizeReferences(outputs);
  auto lineage = mechanicalLineage(outputs);
  std::vector<CandidateGeneratorOutputBinding> bindings = {
      {CandidateGeneratorOutputSlotRef(0), std::move(outputs)}};
  return {std::move(bindings), std::move(lineage)};
}

IncompleteCandidateGeneratorResult
incomplete(CandidateGeneratorIncompleteReason reason,
           std::vector<ArtifactRootReference> outputs) {
  canonicalizeReferences(outputs);
  auto lineage = mechanicalLineage(outputs);
  std::vector<CandidateGeneratorOutputBinding> bindings = {
      {CandidateGeneratorOutputSlotRef(0), std::move(outputs)}};
  return {reason, std::move(bindings), std::move(lineage)};
}

llvm::Error
accumulateWorkSummary(llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> source,
                      std::vector<CandidateGeneratorWorkUnitSummary> &target) {
  if (source.size() != target.size())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "root-complete Spatial PnR work summaries have different widths");
  for (std::size_t ordinal = 0; ordinal != source.size(); ++ordinal) {
    if (!(source[ordinal].unit == target[ordinal].unit))
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "root-complete Spatial PnR work summaries have different units");
    if (source[ordinal].planned > std::numeric_limits<std::uint64_t>::max() -
                                      target[ordinal].planned ||
        source[ordinal].consumed > std::numeric_limits<std::uint64_t>::max() -
                                       target[ordinal].consumed)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "root-complete Spatial PnR work summary overflows u64");
    target[ordinal].planned += source[ordinal].planned;
    target[ordinal].consumed += source[ordinal].consumed;
  }
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation) {
  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      inputBindings[FabricInput].artifacts.front(), store);
  if (!fabric)
    return fabric.takeError();
  auto physicalTiming = ::loom::fabric::importFabricPhysicalTimingProfile(
      inputBindings[PhysicalTimingProfileInput].artifacts.front(),
      fabric->view(), store);
  if (!physicalTiming)
    return physicalTiming.takeError();
  auto derivedContexts = ::loom::pnr::buildFabricDerivedContextBundle(
      fabric->view(), *physicalTiming);
  if (!derivedContexts)
    return derivedContexts.takeError();
  ::loom::pnr::emitFabricDerivedContextStatistics(
      *derivedContexts, ::loom::mapping_debug::Stage::SpatialPnr, 0, 1, 0, 1);
  auto topology =
      ::loom::pnr::analyzeFabricTopologyQualityForDiagnostics(fabric->view());
  if (!topology)
    return topology.takeError();
  if (*topology)
    ::loom::pnr::emitFabricTopologyQuality(
        **topology, ::loom::mapping_debug::Stage::SpatialPnr);

  std::map<ArtifactRootReference, std::unique_ptr<CachedDataflow>,
           decltype(&artifactRootReferenceLess)>
      dataflowCache(&artifactRootReferenceLess);
  DataflowImportStatistics dataflowImportStatistics;
  auto emitDataflowImportStatisticsOnExit = llvm::scope_exit(
      [&] { emitDataflowImportStatistics(dataflowImportStatistics); });
  ActiveProblemCacheStatistics activeProblemCacheStatistics;
  auto emitActiveProblemCacheStatisticsOnExit = llvm::scope_exit([&] {
    emitActiveProblemCacheStatistics(activeProblemCacheStatistics);
  });
  std::vector<ArtifactRootReference> outputs;
  std::optional<CandidateGeneratorIncompleteReason> incompleteReason;
  const auto rememberIncomplete =
      [&](CandidateGeneratorIncompleteReason reason) {
        if (!incompleteReason ||
            reason == CandidateGeneratorIncompleteReason::CancelledOrTimeout ||
            (*incompleteReason !=
                 CandidateGeneratorIncompleteReason::CancelledOrTimeout &&
             reason == CandidateGeneratorIncompleteReason::ProofNotEstablished))
          incompleteReason = reason;
      };
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary =
      spatialPnrCandidateGeneratorWorkSummary({});
  const std::optional<std::uint64_t> maximumOutputs =
      invocation.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));

  std::vector<PreparedTechCandidate> preparedCandidates;
  preparedCandidates.reserve(
      inputBindings[TechMappingCandidatesInput].artifacts.size());
  for (const auto indexedTech :
       llvm::enumerate(inputBindings[TechMappingCandidatesInput].artifacts)) {
    if (invocation.executionControl().stopRequested()) {
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      break;
    }
    const std::size_t inputOrdinal = indexedTech.index();
    const ArtifactRootReference &techReference = indexedTech.value();
    auto tech = ::loom::mapping::importTechMapping(techReference, store);
    if (!tech)
      return tech.takeError();
    if (tech->view().fabricIdentity() != fabric->view().identity())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: TechMapping binds "
          "a foreign Fabric");

    ArtifactRootReference dataflowReference{
        ::dataflow::canonicalDataflowSchema.identity.str(),
        ::dataflow::canonicalDataflowSchema.version,
        tech->view().dataflowIdentity()};
    ++dataflowImportStatistics.requests;
    auto cached = dataflowCache.find(dataflowReference);
    if (cached == dataflowCache.end()) {
      const auto constructionBegin = std::chrono::steady_clock::now();
      auto artifact =
          ::dataflow::importCanonicalDataflow(dataflowReference, store);
      if (!artifact)
        return artifact.takeError();
      auto view = artifact->view();
      if (!view)
        return view.takeError();
      cached = dataflowCache
                   .emplace(dataflowReference,
                            std::make_unique<CachedDataflow>(CachedDataflow{
                                std::move(*artifact), std::move(*view)}))
                   .first;
      ++dataflowImportStatistics.misses;
      saturatingAdd(
          dataflowImportStatistics.constructionNanoseconds,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - constructionBegin)
                  .count()));
      accountRetainedDataflow(*cached->second, dataflowImportStatistics);
    } else {
      ++dataflowImportStatistics.hits;
    }
    if (cached->second->artifact.identity() != dataflowReference.artifact ||
        cached->second->view.identity() != dataflowReference.artifact)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: cached Dataflow "
          "identity does not match the exact TechMapping lineage");
    const ::dataflow::CanonicalDataflowProgramView &dataflow =
        cached->second->view;

    auto constraints =
        ::loom::mapping::finalizeEmptySpatialMappingConstraintSet(
            dataflow, tech->view(), fabric->view(), store);
    if (!constraints)
      return constraints.takeError();

    ++activeProblemCacheStatistics.requests;
    ++activeProblemCacheStatistics.misses;
    const auto activeProblemBegin = std::chrono::steady_clock::now();
    auto activeProblem = ::loom::pnr::freezeSpatialPnrProblem(
        dataflow, tech->view(), fabric->view(), *physicalTiming, *config,
        constraints->view(), &*derivedContexts);
    saturatingAdd(
        activeProblemCacheStatistics.constructionNanoseconds,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - activeProblemBegin)
                .count()));
    if (!activeProblem) {
      bool typedFailure = false;
      ::loom::pnr::SpatialPnrFreezeFailureKind failureKind =
          ::loom::pnr::SpatialPnrFreezeFailureKind::Invalid;
      std::string diagnostic;
      llvm::Error unhandled = llvm::handleErrors(
          activeProblem.takeError(),
          [&](const ::loom::pnr::SpatialPnrFreezeFailure &failure) {
            typedFailure = true;
            failureKind = failure.kind();
            diagnostic = errorMessage(failure);
          });
      if (unhandled)
        return std::move(unhandled);
      if (!typedFailure ||
          failureKind == ::loom::pnr::SpatialPnrFreezeFailureKind::Invalid)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "root_complete_spatial_pnr_generator_invalid: " + diagnostic);
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "active_problem_preparation";
            fields["closure_status"] = "proven_infeasible";
            fields["tech_mapping_input_ordinal"] =
                static_cast<std::uint64_t>(inputOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = diagnostic;
          });
      continue;
    }
    saturatingAdd(activeProblemCacheStatistics.retainedBytes,
                  (*activeProblem)->statistics().context.retainedBytes);
    saturatingAdd(activeProblemCacheStatistics.deterministicWork,
                  (*activeProblem)->statistics().context.deterministicWork);

    std::optional<::loom::pnr::SpatialCandidateInitializerPreference>
        preference;
    std::optional<ActiveRouteRank> routeRank;
    ::loom::pnr::SpatialPathFinderSeedWorkSummary rankWork;
    const auto rankBegin = std::chrono::steady_clock::now();
    auto seed = ::loom::pnr::createPathFinderSpatialSeed(*activeProblem, 0,
                                                         rankWork);
    saturatingAdd(
        activeProblemCacheStatistics.rankProjectionNanoseconds,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - rankBegin)
                .count()));
    ++activeProblemCacheStatistics.rankProjectionCount;
    saturatingAdd(activeProblemCacheStatistics.rankAssignmentAttempts,
                  rankWork.initializerAssignmentAttempts);
    saturatingAdd(activeProblemCacheStatistics.rankEndpointExpansions,
                  rankWork.endpointExpansions);
    saturatingAdd(activeProblemCacheStatistics.rankNegotiationIterations,
                  rankWork.negotiationIterations);
    if (seed) {
      preference = seed->initializerPreference;
      std::array<std::uint64_t, ::loom::resolvedPnrViolationKindCount>
          violations{};
      llvm::Error violationError = llvm::Error::success();
      for (std::uint32_t ordinal = 0; ordinal != violations.size(); ++ordinal) {
        auto value = ::loom::pnr::spatialMappingViolationValue(
            *seed->candidate,
            static_cast<::loom::ResolvedPnrViolationKind>(ordinal));
        if (!value) {
          violationError = value.takeError();
          break;
        }
        violations[ordinal] = *value;
      }
      if (violationError) {
        const std::string diagnostic =
            llvm::toString(std::move(violationError));
        ::loom::mapping_debug::emit(
            ::loom::mapping_debug::Level::Summary,
            ::loom::mapping_debug::Stage::SpatialPnr,
            ::loom::mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["failure_scope"] = "active_route_rank";
              fields["closure_status"] = "not_ranked";
              fields["tech_mapping_input_ordinal"] =
                  static_cast<std::uint64_t>(inputOrdinal);
              fields["tech_mapping"] =
                  formatArtifactIdentityHex(techReference.artifact);
              fields["diagnostic"] = diagnostic;
            });
      } else {
        auto objective =
            (*activeProblem)->objectiveProgram().evaluate(*seed->candidate);
        if (!objective)
          return objective.takeError();
        routeRank.emplace(ActiveRouteRank{
            std::move(*objective), violations, rankWork.endpointExpansions,
            rankWork.negotiationIterations});
      }
    } else {
      const std::string diagnostic = llvm::toString(seed.takeError());
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "active_demand_rank";
            fields["closure_status"] = "not_ranked";
            fields["tech_mapping_input_ordinal"] =
                static_cast<std::uint64_t>(inputOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = diagnostic;
          });
    }
    preparedCandidates.push_back(PreparedTechCandidate{
        inputOrdinal, techReference, std::move(*tech), cached->second.get(),
        std::move(*constraints), std::move(*activeProblem), preference,
        std::move(routeRank)});
  }

  ::loom::pnr::SpatialPnrGenerationAccounting rankAccounting;
  rankAccounting.seedAttemptSlots =
      activeProblemCacheStatistics.rankProjectionCount;
  rankAccounting.preparedSeeds =
      activeProblemCacheStatistics.rankProjectionCount;
  rankAccounting.initializerAssignmentAttempts =
      activeProblemCacheStatistics.rankAssignmentAttempts;
  rankAccounting.endpointExpansionSlots =
      activeProblemCacheStatistics.rankEndpointExpansions;
  rankAccounting.negotiationIterationSlots =
      activeProblemCacheStatistics.rankNegotiationIterations;
  if (llvm::Error error = accumulateWorkSummary(
          spatialPnrCandidateGeneratorWorkSummary(rankAccounting),
          workSummary))
    return std::move(error);

  if (llvm::Error error = orderPreparedCandidates(preparedCandidates))
    return std::move(error);
  for (const auto ranked : llvm::enumerate(preparedCandidates)) {
    const PreparedTechCandidate &candidate = ranked.value();
    ::loom::mapping_debug::emit(
        ::loom::mapping_debug::Level::Summary,
        ::loom::mapping_debug::Stage::SpatialPnr,
        ::loom::mapping_debug::Event::Candidate,
        [&](llvm::json::Object &fields) {
          fields["operation"] = "active_demand_rank";
          fields["rank"] = static_cast<std::uint64_t>(ranked.index());
          fields["tech_mapping_input_ordinal"] =
              static_cast<std::uint64_t>(candidate.inputOrdinal);
          fields["tech_mapping"] =
              formatArtifactIdentityHex(candidate.reference.artifact);
          fields["active_problem_key"] = llvm::toHex(
              candidate.activeProblem->cacheKey().bytes(),
              /*LowerCase=*/true);
          fields["rank_available"] = candidate.preference.has_value();
          fields["route_rank_available"] = candidate.routeRank.has_value();
          if (candidate.routeRank) {
            llvm::json::Array violations;
            for (const std::uint64_t value :
                 candidate.routeRank->violations)
              violations.push_back(value);
            fields["route_violation_values"] = std::move(violations);
            llvm::json::Array objectiveCodes;
            for (const std::uint64_t code :
                 candidate.routeRank->objective.codes())
              objectiveCodes.push_back(code);
            fields["route_objective_codes"] = std::move(objectiveCodes);
            fields["route_endpoint_expansions"] =
                candidate.routeRank->endpointExpansions;
            fields["route_negotiation_iterations"] =
                candidate.routeRank->negotiationIterations;
          }
          if (!candidate.preference)
            return;
          fields["residual_external_sink_count"] =
              candidate.preference->residualExternalSinkCount;
          fields["selected_register_fifo_transfer_count"] =
              candidate.preference->selectedRegisterFifoTransferCount;
          fields["topology_unreachable_selection_count"] =
              candidate.preference->topologyUnreachableSelectionCount;
          fields["topology_hop_sum"] =
              candidate.preference->topologyHopSum;
          fields["topology_refinement_unreachable_selection_count"] =
              candidate.preference
                  ->topologyRefinementUnreachableSelectionCount;
          fields["topology_refinement_hop_sum"] =
              candidate.preference->topologyRefinementHopSum;
          fields["maximum_compute_occurrence_selections"] =
              candidate.preference->maximumComputeOccurrenceSelections;
          fields["maximum_endpoint_selections"] =
              candidate.preference->maximumEndpointSelections;
          fields["static_schedule_pressure"] =
              candidate.preference->staticSchedulePressure;
        });
  }

  for (const auto rankedTech : llvm::enumerate(preparedCandidates)) {
    const std::size_t techOrdinal = rankedTech.index();
    PreparedTechCandidate &prepared = rankedTech.value();
    const ArtifactRootReference &techReference = prepared.reference;
    const ::dataflow::CanonicalDataflowProgramView &dataflow =
        prepared.dataflow->view;
    ++activeProblemCacheStatistics.requests;
    ++activeProblemCacheStatistics.hits;

    ::loom::pnr::SpatialPnrGenerationOutcome outcome =
        ::loom::pnr::generateSpatialMappings(
            {dataflow, prepared.tech.view(), fabric->view(), *physicalTiming,
             *config, prepared.constraints.view(), store,
             defaultCandidateWorkerCount(),
             invocation.executionControl(), &*derivedContexts,
             *topology ? &**topology : nullptr, prepared.activeProblem,
             false});
    const auto invocationWorkSummary = std::visit(
        [](const auto &value) {
          return spatialPnrCandidateGeneratorWorkSummary(value.accounting);
        },
        outcome);
    if (llvm::Error error =
            accumulateWorkSummary(invocationWorkSummary, workSummary))
      return std::move(error);
    if (auto *generated =
            std::get_if<::loom::pnr::GeneratedSpatialMappings>(&outcome)) {
      if (auto reason = pnrGenerationIncompleteReason(generated->termination))
        rememberIncomplete(*reason);
      outputs.insert(outputs.end(),
                     std::make_move_iterator(generated->candidates.begin()),
                     std::make_move_iterator(generated->candidates.end()));
      canonicalizeReferences(outputs);
      if (maximumOutputs && outputs.size() >= *maximumOutputs) {
        const bool droppedOutputs = outputs.size() > *maximumOutputs;
        if (droppedOutputs)
          outputs.erase(outputs.begin() +
                            static_cast<std::size_t>(*maximumOutputs),
                        outputs.end());
        const bool skippedTechMappings =
            techOrdinal + 1 != preparedCandidates.size();
        if (droppedOutputs || skippedTechMappings)
          rememberIncomplete(
              CandidateGeneratorIncompleteReason::SemanticLimitReached);
        if (droppedOutputs || skippedTechMappings)
          break;
      }
      continue;
    }
    if (const auto *infeasible =
            std::get_if<::loom::pnr::ProvenInfeasibleSpatialMapping>(
                &outcome)) {
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "invocation";
            fields["closure_status"] = "proven_infeasible";
            fields["tech_mapping_ordinal"] =
                static_cast<std::uint64_t>(techOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = infeasible->diagnostic;
          });
      continue;
    }
    if (const auto *partial =
            std::get_if<::loom::pnr::IncompleteSpatialPnrGeneration>(
                &outcome)) {
      const CandidateGeneratorIncompleteReason reason =
          partial->reason == ::loom::pnr::IncompleteSpatialPnrGenerationReason::
                                 SemanticLimitReached
              ? CandidateGeneratorIncompleteReason::SemanticLimitReached
              : CandidateGeneratorIncompleteReason::ProofNotEstablished;
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "invocation";
            fields["closure_status"] =
                partial->reason ==
                        ::loom::pnr::IncompleteSpatialPnrGenerationReason::
                            SemanticLimitReached
                    ? "semantic_limit_reached"
                    : "proof_not_established";
            fields["tech_mapping_ordinal"] =
                static_cast<std::uint64_t>(techOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = partial->diagnostic;
            fields["seed_attempts"] = partial->accounting.seedAttemptSlots;
            fields["prepared_seeds"] = partial->accounting.preparedSeeds;
            fields["exact_repair_invocations"] =
                partial->accounting.exactRepairInvocations;
            fields["exact_repair_region_decisions"] =
                partial->accounting.exactRepairRegionDecisions;
            fields["exact_repair_solver_calls"] =
                partial->accounting.exactRepairSolverCalls;
            fields["final_closure_attempts"] =
                partial->accounting.finalClosureAttempts;
          });
      rememberIncomplete(reason);
      continue;
    }
    if (auto *interrupted =
            std::get_if<::loom::pnr::InterruptedSpatialPnrGeneration>(
                &outcome)) {
      outputs.insert(outputs.end(),
                     std::make_move_iterator(interrupted->candidates.begin()),
                     std::make_move_iterator(interrupted->candidates.end()));
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      break;
    }
    if (std::holds_alternative<::loom::pnr::UnsupportedSpatialPnrGeneration>(
            outcome))
      return CandidateGeneratorProviderResult{
          incomplete(CandidateGeneratorIncompleteReason::Unsupported,
                     std::move(outputs)),
          std::move(workSummary)};
    if (const auto *invalid =
            std::get_if<::loom::pnr::InvalidSpatialPnrGeneration>(&outcome))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: " +
              invalid->diagnostic);
    const auto &internal =
        std::get<::loom::pnr::InternalSpatialPnrGeneration>(outcome);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "root_complete_spatial_pnr_generator_execution_failed: " +
            internal.diagnostic);
  }

  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        incomplete(*incompleteReason, std::move(outputs)),
        std::move(workSummary)};
  return CandidateGeneratorProviderResult{completed(std::move(outputs)),
                                          std::move(workSummary)};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeRootCompleteProvider}};

} // namespace

const CandidateGeneratorDescriptor &
rootCompleteSpatialPnrCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerRootCompleteSpatialPnrCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSpatialPnrCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> techMappingCandidates,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &physicalTimingProfile) {
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(TechMappingCandidatesInput),
       techMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(PhysicalTimingProfileInput),
       {physicalTimingProfile}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
