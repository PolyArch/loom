#include "DSE/PreMappingFrontier.h"

#include "DSE/Objective.h"
#include "DSE/Promotion.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral projectionDescriptor{
    "loom.pre_mapping.coordinate_projection.1.0"};
constexpr llvm::StringLiteral materializedProjectionDescriptor{
    "loom.pre_mapping.materialized_projection.1.0"};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "pre_mapping_frontier_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned index = 0; index != 8; ++index)
    bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

bool checkedAdd(std::uint64_t &destination, std::uint64_t value) {
  const std::optional<std::uint64_t> sum =
      llvm::checkedAddUnsigned(destination, value);
  if (!sum)
    return false;
  destination = *sum;
  return true;
}

using CandidateMeasureArray = std::array<std::uint64_t, 20>;

CandidateMeasureArray candidateMeasureValues(
    const PreMappingFrontierCandidate &candidate, std::uint64_t maximum) {
  const PreMappingCandidateProjection &projection = candidate.projection;
  return {
      projection.estimateSupport == PreMappingEstimateSupport::Supported
          ? 0ULL
          : 1ULL,
      projection.unknownCutPairCount,
      projection.cutUnknownObjectCount,
      projection.estimatedCutTrafficBytes.value_or(maximum),
      candidate.estimatedRuntimePicoseconds.value_or(maximum),
      projection.cutDependencyCount,
      projection.internalDependencyCount,
      projection.ownedRegionCount,
      projection.channelDepthLowerBound,
      projection.topologyCongestionProxy,
      projection.launchSynchronizationCost,
      projection.parallelismLowerBound,
      projection.producerRateLowerBound,
      projection.hostDynamicLeafExecutions,
      projection.maximumProducerFanout,
      projection.channelOpportunityCount,
      projection.consumerRateLowerBound,
      projection.reconfigurationLiveStateKnown ? 0ULL : 1ULL,
      projection.reconfigurationLiveStateKnown
          ? projection.reconfigurationLiveStateBytes
          : maximum,
      projection.hostDynamicActivations};
}

// This table is the sole bridge between the candidate-measure vector and the
// transient objective catalog below. The catalog has two dimensions for
// ownedRegionCount so both minimum- and maximum-ownership diversity remain
// explicit without maintaining a second hand-written field list.
constexpr std::array<std::uint8_t, 20> candidateObjectiveSources = {
    0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
constexpr std::array<bool, 20> candidateObjectiveMaximize = {
    false, false, false, false, false, false, true,  false, true,  false,
    false, false, false, false, false, true,  true,  false, false, false};

std::array<std::uint64_t, 20> candidateObjectiveKey(
    const PreMappingFrontierCandidate &candidate, std::uint64_t maximum) {
  const CandidateMeasureArray measures =
      candidateMeasureValues(candidate, maximum);
  std::array<std::uint64_t, 20> key{};
  for (std::size_t ordinal = 0; ordinal != key.size(); ++ordinal) {
    const std::uint64_t value = measures[candidateObjectiveSources[ordinal]];
    key[ordinal] = candidateObjectiveMaximize[ordinal] ? maximum - value : value;
  }
  return key;
}

std::vector<std::uint8_t>
rootKey(const frontend::StructuredEntityRef &root) {
  return frontend::encodeStructuredEntityRef(root);
}

llvm::Expected<PreMappingCandidateProjection> projectCoordinate(
    llvm::ArrayRef<std::size_t> selected,
    llvm::ArrayRef<frontend::StructuredEntityRef> roots,
    const frontend::analysis::StructuredProtocolDependencyProjection
        &dependencies,
    llvm::ArrayRef<PreMappingRootActivity> activity) {
  std::vector<bool> owned(roots.size(), false);
  for (std::size_t ordinal : selected) {
    if (ordinal >= roots.size() || owned[ordinal])
      return invalid("coordinate contains a foreign or duplicate root");
    owned[ordinal] = true;
  }

  std::map<std::vector<std::uint8_t>, std::size_t> ordinalByRoot;
  for (auto indexed : llvm::enumerate(roots))
    if (!ordinalByRoot.emplace(rootKey(indexed.value()), indexed.index()).second)
      return invalid("protocol root set is not unique");
  if (activity.size() != roots.size())
    return invalid("root activity is not total");

  auto placeholder = ComponentViewDigest::fromBytes(
      std::array<std::uint8_t, ComponentViewDigest::byteSize>{});
  if (!placeholder)
    return placeholder.takeError();
  PreMappingCandidateProjection result(*placeholder);
  result.ownedRegionCount = selected.size();
  result.hostRegionCount = roots.size() - selected.size();
  for (auto indexed : llvm::enumerate(activity)) {
    if (indexed.value().root != roots[indexed.index()])
      return invalid("root activity changed caller order");
    std::uint64_t &activations = owned[indexed.index()]
                                     ? result.ownedDynamicActivations
                                     : result.hostDynamicActivations;
    std::uint64_t &leaves = owned[indexed.index()]
                                ? result.ownedDynamicLeafExecutions
                                : result.hostDynamicLeafExecutions;
    if (!checkedAdd(activations, indexed.value().dynamicActivations) ||
        !checkedAdd(leaves, indexed.value().dynamicLeafExecutions))
      return invalid("coordinate activity overflows");
  }

  std::vector<std::uint64_t> fanout(roots.size(), 0);
  std::uint64_t cutTraffic = 0;
  bool cutTrafficKnown = true;
  for (const auto &relation : dependencies.relations) {
    const auto producer = ordinalByRoot.find(rootKey(relation.producer));
    const auto consumer = ordinalByRoot.find(rootKey(relation.consumer));
    if (producer == ordinalByRoot.end() || consumer == ordinalByRoot.end() ||
        producer->second == consumer->second)
      return invalid("dependency relation is outside its root set");
    const bool internal = owned[producer->second] && owned[consumer->second];
    const bool cut = owned[producer->second] != owned[consumer->second];
    using Knowledge =
        frontend::analysis::StructuredProtocolDependencyKnowledge;
    if (relation.knowledge == Knowledge::Unknown) {
      if (internal)
        ++result.unknownInternalPairCount;
      else if (cut)
        ++result.unknownCutPairCount;
      continue;
    }
    if (relation.knowledge == Knowledge::ProvenAbsent)
      continue;
    if (!relation.dependency)
      return invalid("present dependency has no exact payload");
    const auto &dependency = *relation.dependency;
    if (internal) {
      ++result.internalDependencyCount;
      ++result.channelOpportunityCount;
      if (!checkedAdd(result.internalKnownBytes,
                      dependency.knownSharedMemoryBytes) ||
          !checkedAdd(result.internalUnknownObjectCount,
                      dependency.unknownSharedMemoryObjectCount))
        return invalid("internal dependency projection overflows");
      ++fanout[producer->second];
      continue;
    }
    if (!cut)
      continue;
    ++result.cutDependencyCount;
    if (!checkedAdd(result.cutKnownBytes,
                    dependency.knownSharedMemoryBytes) ||
        !checkedAdd(result.cutUnknownObjectCount,
                    dependency.unknownSharedMemoryObjectCount))
      return invalid("cut dependency projection overflows");
    ++fanout[producer->second];
    if (dependency.unknownSharedMemoryObjectCount != 0) {
      cutTrafficKnown = false;
      continue;
    }
    const std::optional<std::uint64_t> traffic = llvm::checkedMulUnsigned(
        dependency.knownSharedMemoryBytes,
        activity[producer->second].dynamicActivations);
    if (!traffic || !checkedAdd(cutTraffic, *traffic))
      cutTrafficKnown = false;
  }
  result.maximumProducerFanout =
      fanout.empty() ? 0 : *std::max_element(fanout.begin(), fanout.end());
  if (cutTrafficKnown && result.unknownCutPairCount == 0)
    result.estimatedCutTrafficBytes = cutTraffic;
  if (result.estimatedCutTrafficBytes) {
    result.estimateSupport = PreMappingEstimateSupport::Supported;
    result.estimateConfidence =
        result.internalUnknownObjectCount == 0 &&
                result.unknownInternalPairCount == 0
            ? PreMappingEstimateConfidence::Low
            : PreMappingEstimateConfidence::None;
  }
  result.producerRateLowerBound = result.maximumProducerFanout == 0
                                      ? 0
                                      : std::max<std::uint64_t>(
                                            1, result.ownedDynamicActivations);
  result.consumerRateLowerBound = result.cutDependencyCount == 0
                                      ? 0
                                      : std::max<std::uint64_t>(
                                            1, result.hostDynamicActivations);
  result.channelDepthLowerBound =
      result.internalDependencyCount + result.maximumProducerFanout;
  result.launchSynchronizationCost = result.ownedRegionCount +
                                     result.cutDependencyCount;
  result.parallelismLowerBound = result.ownedRegionCount == 0
                                     ? 0
                                     : std::max<std::uint64_t>(
                                           1, result.ownedDynamicActivations);
  // One shared analytic pressure measure feeds the pre-Mapping beam. It
  // includes ordered-channel opportunities and depth lower bounds; it is not
  // a physical queue legality predicate and never triggers PnR by itself.
  result.topologyCongestionProxy =
      result.cutDependencyCount * (1 + result.maximumProducerFanout) +
      result.channelOpportunityCount + result.channelDepthLowerBound +
      result.unknownCutPairCount + result.unknownInternalPairCount;

  std::vector<std::uint8_t> bytes;
  appendU64(bytes, selected.size());
  for (std::size_t ordinal : selected) {
    std::vector<std::uint8_t> encoded = rootKey(roots[ordinal]);
    appendU64(bytes, encoded.size());
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  const std::array<std::uint64_t, 24> fields = {
      result.internalDependencyCount,
      result.internalKnownBytes,
      result.internalUnknownObjectCount,
      result.cutDependencyCount,
      result.cutKnownBytes,
      result.cutUnknownObjectCount,
      result.unknownInternalPairCount,
      result.unknownCutPairCount,
      result.channelOpportunityCount,
      result.maximumProducerFanout,
      result.ownedDynamicActivations,
      result.ownedDynamicLeafExecutions,
      result.hostDynamicActivations,
      result.hostDynamicLeafExecutions,
      result.estimatedCutTrafficBytes.value_or(0),
      result.estimatedCutTrafficBytes.has_value() ? 1ULL : 0ULL,
      result.producerRateLowerBound,
      result.consumerRateLowerBound,
      result.channelDepthLowerBound,
      result.launchSynchronizationCost,
      result.parallelismLowerBound,
      result.topologyCongestionProxy,
      result.reconfigurationLiveStateBytes,
      result.reconfigurationLiveStateKnown ? 1ULL : 0ULL};
  for (std::uint64_t field : fields)
    appendU64(bytes, field);
  auto identity = computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(projectionDescriptor.data()),
       projectionDescriptor.size()},
      bytes);
  if (!identity)
    return identity.takeError();
  result.identity = *identity;
  return result;
}

} // namespace

llvm::StringRef toString(PreMappingSpectrumSeedKind value) {
  switch (value) {
  case PreMappingSpectrumSeedKind::MaxSpatial:
    return "max_spatial";
  case PreMappingSpectrumSeedKind::MaxTemporal:
    return "max_temporal";
  case PreMappingSpectrumSeedKind::HighActivitySingleton:
    return "high_activity_singleton";
  case PreMappingSpectrumSeedKind::DependencyEdge:
    return "dependency_edge";
  case PreMappingSpectrumSeedKind::ProducerGroup:
    return "producer_group";
  case PreMappingSpectrumSeedKind::PipelineGroup:
    return "pipeline_group";
  case PreMappingSpectrumSeedKind::ConnectedComponent:
    return "connected_component";
  case PreMappingSpectrumSeedKind::CanonicalFallback:
    return "canonical_fallback";
  case PreMappingSpectrumSeedKind::Intermediate:
    return "intermediate";
  }
  llvm_unreachable("unknown pre-Mapping spectrum seed kind");
}

llvm::StringRef toString(PreMappingSpectrumEndpoint value) {
  switch (value) {
  case PreMappingSpectrumEndpoint::Automatic:
    return "automatic";
  case PreMappingSpectrumEndpoint::MaxTemporal:
    return "max_temporal";
  case PreMappingSpectrumEndpoint::MaxSpatial:
    return "max_spatial";
  case PreMappingSpectrumEndpoint::Intermediate:
    return "intermediate";
  }
  llvm_unreachable("unknown pre-Mapping spectrum endpoint");
}

llvm::StringRef toString(PreMappingScheduleIntent value) {
  switch (value) {
  case PreMappingScheduleIntent::Unconstrained:
    return "unconstrained";
  case PreMappingScheduleIntent::TemporalReuse:
    return "temporal_reuse_intent";
  case PreMappingScheduleIntent::SpatialParallel:
    return "spatial_parallel_intent";
  }
  llvm_unreachable("unknown pre-Mapping schedule intent");
}

llvm::StringRef toString(PreMappingSpectrumClass value) {
  switch (value) {
  case PreMappingSpectrumClass::MaxTemporal:
    return "max_temporal";
  case PreMappingSpectrumClass::MaxSpatial:
    return "max_spatial";
  case PreMappingSpectrumClass::Intermediate:
    return "intermediate";
  }
  llvm_unreachable("unknown pre-Mapping spectrum class");
}

llvm::StringRef toString(PreMappingLogicalDomainSupport value) {
  switch (value) {
  case PreMappingLogicalDomainSupport::Exact:
    return "exact";
  case PreMappingLogicalDomainSupport::Partial:
    return "partial";
  case PreMappingLogicalDomainSupport::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown pre-Mapping logical-domain support");
}

llvm::StringRef toString(PreMappingExactGateDisposition value) {
  switch (value) {
  case PreMappingExactGateDisposition::Admitted:
    return "admitted";
  case PreMappingExactGateDisposition::Rejected:
    return "rejected";
  }
  llvm_unreachable("unknown pre-Mapping exact-gate disposition");
}

llvm::StringRef toString(PreMappingEstimateSupport value) {
  switch (value) {
  case PreMappingEstimateSupport::Supported:
    return "supported";
  case PreMappingEstimateSupport::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown pre-Mapping estimate support");
}

llvm::StringRef toString(PreMappingEstimateConfidence value) {
  switch (value) {
  case PreMappingEstimateConfidence::None:
    return "none";
  case PreMappingEstimateConfidence::Low:
    return "low";
  case PreMappingEstimateConfidence::Calibrated:
    return "calibrated";
  case PreMappingEstimateConfidence::OutOfDistribution:
    return "out_of_distribution";
  }
  llvm_unreachable("unknown pre-Mapping estimate confidence");
}

std::uint64_t
PreMappingFrontierPolicy::beamWidth(std::size_t expansionDepth) const {
  if (beamWidthByExpansionDepth.empty())
    return 0;
  return beamWidthByExpansionDepth[
      std::min(expansionDepth, beamWidthByExpansionDepth.size() - 1)];
}

llvm::Expected<ComponentViewDigest> PreMappingFrontierPolicy::digest() const {
  std::vector<std::uint8_t> bytes;
  constexpr llvm::StringLiteral descriptor =
      "loom.pre_mapping.frontier_policy.2";
  bytes.insert(bytes.end(), descriptor.bytes_begin(), descriptor.bytes_end());
  const auto append = [&](std::uint64_t value) { appendU64(bytes, value); };
  append(budget.maximumSourceObservations);
  append(budget.maximumCoordinatesGenerated);
  append(budget.maximumProgramsMaterialized);
  append(budget.maximumAnalyticEvaluations);
  append(budget.maximumFunctionalReplays);
  append(budget.maximumDataflowPromotions);
  append(budget.maximumMappingPairs);
  append(beamWidthByExpansionDepth.size());
  for (std::uint64_t width : beamWidthByExpansionDepth)
    append(width);
  append(diversityCandidateCount);
  append(maximumExpansionDepth);
  append(maximumCompositionalGroups);
  append(static_cast<std::uint64_t>(spectrumEndpoint));
  append(static_cast<std::uint64_t>(stoppingPolicy));
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
       descriptor.size()},
      bytes);
}

llvm::Error
validatePreMappingFrontierPolicy(const PreMappingFrontierPolicy &policy) {
  const auto &budget = policy.budget;
  if (budget.maximumSourceObservations == 0 ||
      budget.maximumCoordinatesGenerated == 0 ||
      budget.maximumProgramsMaterialized == 0 ||
      budget.maximumAnalyticEvaluations == 0 ||
      budget.maximumFunctionalReplays == 0 ||
      budget.maximumDataflowPromotions == 0 ||
      budget.maximumMappingPairs == 0)
    return invalid("every work bound must be positive");
  if (policy.beamWidthByExpansionDepth.empty() ||
      llvm::is_contained(policy.beamWidthByExpansionDepth, 0ULL))
    return invalid("every expansion depth requires a positive beam width");
  if (policy.diversityCandidateCount == 0)
    return invalid("diversity candidate count must be positive");
  if (policy.maximumCompositionalGroups == 0)
    return invalid("compositional group bound must be positive");
  return llvm::Error::success();
}

PreMappingWorkAccounting
makePreMappingWorkAccounting(const PreMappingFrontierBudget &budget) {
  return {{budget.maximumSourceObservations, 0, 0},
          {budget.maximumCoordinatesGenerated, 0, 0},
          {budget.maximumProgramsMaterialized, 0, 0},
          {budget.maximumAnalyticEvaluations, 0, 0},
          {budget.maximumFunctionalReplays, 0, 0},
          {budget.maximumDataflowPromotions, 0, 0},
          {budget.maximumMappingPairs, 0, 0}};
}

llvm::Error validatePreMappingWorkAccounting(
    const PreMappingWorkAccounting &accounting) {
  const auto validate = [](const PreMappingWorkCounter &counter,
                           llvm::StringRef name) -> llvm::Error {
    if (counter.planned > counter.limit)
      return invalid(name + " planned work exceeds its limit");
    if (counter.reserved != counter.planned)
      return invalid(name + " reservation ledger does not match planned work");
    if (counter.consumed > counter.reserved ||
        counter.rejected > counter.reserved - counter.consumed ||
        counter.cancelled >
            counter.reserved - counter.consumed - counter.rejected ||
        counter.consumed >
            std::numeric_limits<std::uint64_t>::max() - counter.rejected ||
        counter.consumed + counter.rejected >
            std::numeric_limits<std::uint64_t>::max() - counter.cancelled ||
        counter.consumed + counter.rejected + counter.cancelled !=
            counter.reserved)
      return invalid(name + " settled work exceeds reservations");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validate(accounting.sourceObservations, "source_observations"))
    return error;
  if (llvm::Error error = validate(accounting.coordinates, "coordinates"))
    return error;
  if (llvm::Error error = validate(accounting.programMaterializations,
                                   "program_materializations"))
    return error;
  if (llvm::Error error =
          validate(accounting.analyticEvaluations, "analytic_evaluations"))
    return error;
  if (llvm::Error error =
          validate(accounting.functionalReplays, "functional_replays"))
    return error;
  if (llvm::Error error =
          validate(accounting.dataflowPromotions, "dataflow_promotions"))
    return error;
  return validate(accounting.mappingPairs, "mapping_pairs");
}

llvm::Expected<PreMappingFrontierSelection> selectPreMappingFrontier(
    llvm::ArrayRef<PreMappingFrontierCandidate> candidates,
    std::uint64_t maximumRetained, std::uint64_t diversityCandidateCount,
    PreMappingSpectrumEndpoint endpoint) {
  if (candidates.empty())
    return PreMappingFrontierSelection{};
  if (maximumRetained == 0 || diversityCandidateCount == 0)
    return invalid("frontier selection bounds must be positive");

  // A final Structured Artifact can be reached through several planning
  // coordinates. The Artifact is the downstream identity, while the
  // coordinate projection is only a heuristic representative. Select that
  // representative deterministically before constructing the central
  // objective set; retaining every duplicate here would either double-count
  // one Mapping candidate or require a second, synthetic Artifact identity.
  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  const auto representativeLess = [&](const PreMappingFrontierCandidate &lhs,
                                       const PreMappingFrontierCandidate &rhs) {
    const auto lhsKey = candidateObjectiveKey(lhs, maximum);
    const auto rhsKey = candidateObjectiveKey(rhs, maximum);
    if (lhsKey != rhsKey)
      return lhsKey < rhsKey;
    if (lhs.scheduleIntent != rhs.scheduleIntent)
      return static_cast<std::uint8_t>(lhs.scheduleIntent) <
             static_cast<std::uint8_t>(rhs.scheduleIntent);
    return lhs.projection.identity.bytes() < rhs.projection.identity.bytes();
  };
  std::map<ArtifactRootReference, const PreMappingFrontierCandidate *,
           decltype(&artifactRootReferenceLess)>
      unique(&artifactRootReferenceLess);
  for (const PreMappingFrontierCandidate &candidate : candidates) {
    auto [entry, inserted] = unique.emplace(candidate.candidate, &candidate);
    if (!inserted && representativeLess(candidate, *entry->second))
      entry->second = &candidate;
  }

  CandidateMeasureObjectiveCatalogs catalogs;
  const auto dimension = [](std::uint32_t source,
                            ResolvedObjectiveDirection direction) {
    return CandidateMeasureObjectiveDimension{
        source, direction, 0, std::numeric_limits<std::uint64_t>::max()};
  };
  catalogs.dimensions = {
      dimension(0, ResolvedObjectiveDirection::Minimize),
      dimension(1, ResolvedObjectiveDirection::Minimize),
      dimension(2, ResolvedObjectiveDirection::Minimize),
      dimension(3, ResolvedObjectiveDirection::Minimize),
      dimension(4, ResolvedObjectiveDirection::Minimize),
      dimension(5, ResolvedObjectiveDirection::Minimize),
      dimension(6, ResolvedObjectiveDirection::Maximize),
      dimension(7, ResolvedObjectiveDirection::Minimize),
      dimension(7, ResolvedObjectiveDirection::Maximize),
      dimension(8, ResolvedObjectiveDirection::Minimize),
      dimension(9, ResolvedObjectiveDirection::Minimize),
      dimension(10, ResolvedObjectiveDirection::Minimize),
      dimension(11, ResolvedObjectiveDirection::Minimize),
      dimension(12, ResolvedObjectiveDirection::Minimize),
      dimension(13, ResolvedObjectiveDirection::Minimize),
      dimension(14, ResolvedObjectiveDirection::Minimize),
      dimension(15, ResolvedObjectiveDirection::Maximize),
      dimension(16, ResolvedObjectiveDirection::Maximize),
      dimension(17, ResolvedObjectiveDirection::Minimize),
      dimension(18, ResolvedObjectiveDirection::Minimize),
      dimension(19, ResolvedObjectiveDirection::Minimize),
  };
  for (std::uint32_t ordinal = 0; ordinal != catalogs.dimensions.size();
       ++ordinal)
    catalogs.weightedLevels.push_back({{{ordinal, 1}}});
  catalogs.totalOrderings = {
      {{0, 1, 2, 3, 4, 5, 6}},
      {{0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19}},
      {{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19}},
      {{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19}},
  };
  auto program = ObjectiveProgram::getCandidateMeasures(catalogs);
  if (!program)
    return program.takeError();

  std::vector<ArtifactRootReference> roots;
  std::vector<CandidateObjectiveVector> objectives;
  roots.reserve(unique.size());
  objectives.reserve(unique.size());
  for (const auto &[root, candidate] : unique) {
    const CandidateMeasureArray measures =
        candidateMeasureValues(*candidate, maximum);
    ObjectiveVector vector = program->makeVector();
    if (llvm::Error error =
            program->evaluateCandidateMeasures(measures, vector))
      return std::move(error);
    roots.push_back(root);
    objectives.push_back({root, std::move(vector)});
  }
  auto candidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, roots);
  if (!candidateSet)
    return candidateSet.takeError();
  // The established exact/traffic dimensions define feasibility Pareto
  // fronts. The newly projected rate/depth/topology fields remain in the
  // central total ordering as calibrated tie-breakers; allowing their
  // optimistic lower bounds to define Pareto dominance would incorrectly
  // make an empty ownership candidate dominate useful work.
  const std::array<std::uint32_t, 7> paretoDimensions = {0, 1, 2, 3, 4,
                                                         5, 6};
  auto pareto = applyCandidateSelection(
      *candidateSet, roots, objectives,
      ParetoSelection{std::vector<std::uint32_t>(paretoDimensions.begin(),
                                                 paretoDimensions.end())},
      &*program);
  if (!pareto)
    return pareto.takeError();
  auto objectiveOrder = applyCandidateSelection(
      *candidateSet, *pareto, objectives,
      TopKSelection{0, maximumRetained}, &*program);
  if (!objectiveOrder)
    return objectiveOrder.takeError();
  auto allOrder = applyCandidateSelection(
      *candidateSet, roots, objectives,
      TopKSelection{0, maximumRetained}, &*program);
  if (!allOrder)
    return allOrder.takeError();
  auto minimumOwnership = applyCandidateSelection(
      *candidateSet, roots, objectives, TopKSelection{1, 1}, &*program);
  if (!minimumOwnership)
    return minimumOwnership.takeError();
  auto maximumOwnership = applyCandidateSelection(
      *candidateSet, roots, objectives, TopKSelection{2, 1}, &*program);
  if (!maximumOwnership)
    return maximumOwnership.takeError();

  PreMappingFrontierSelection result{std::move(*pareto), {}, {}};
  result.preferenceOrder.reserve(static_cast<std::size_t>(
      std::min<std::uint64_t>(maximumRetained, roots.size())));
  result.preferenceProjectionIdentities.reserve(result.preferenceOrder.capacity());
  const auto projectionIdentityFor = [&](const ArtifactRootReference &root) {
    auto found = unique.find(root);
    if (found == unique.end())
      return std::optional<ComponentViewDigest>{};
    return std::optional<ComponentViewDigest>{found->second->projection.identity};
  };
  const auto append = [&](const ArtifactRootReference &root) {
    if (result.preferenceOrder.size() == maximumRetained ||
        llvm::is_contained(result.preferenceOrder, root))
      return;
    auto projectionIdentity = projectionIdentityFor(root);
    if (!projectionIdentity)
      return;
    result.preferenceOrder.push_back(root);
    result.preferenceProjectionIdentities.push_back(*projectionIdentity);
  };
  const auto appendEndpointCandidate =
      [&](const PreMappingFrontierCandidate &candidate) {
        if (result.preferenceOrder.size() == maximumRetained ||
            llvm::is_contained(result.preferenceOrder, candidate.candidate))
          return;
        result.preferenceOrder.push_back(candidate.candidate);
        result.preferenceProjectionIdentities.push_back(
            candidate.projection.identity);
      };
  const auto endpointCandidate = [&]()
      -> const PreMappingFrontierCandidate * {
    const PreMappingFrontierCandidate *best = nullptr;
    for (const PreMappingFrontierCandidate &candidate : candidates) {
      bool matches = false;
      switch (endpoint) {
      case PreMappingSpectrumEndpoint::Automatic:
        break;
      case PreMappingSpectrumEndpoint::MaxTemporal:
        matches = candidate.verifiedSpectrum ==
                  PreMappingSpectrumClass::MaxTemporal;
        break;
      case PreMappingSpectrumEndpoint::MaxSpatial:
        matches = candidate.verifiedSpectrum ==
                  PreMappingSpectrumClass::MaxSpatial;
        break;
      case PreMappingSpectrumEndpoint::Intermediate:
        matches = candidate.verifiedSpectrum ==
                  PreMappingSpectrumClass::Intermediate;
        break;
      }
      if (matches && (!best || representativeLess(candidate, *best)))
        best = &candidate;
    }
    return best;
  };
  if (endpoint != PreMappingSpectrumEndpoint::Automatic) {
    const PreMappingFrontierCandidate *selected = endpointCandidate();
    if (!selected)
      return llvm::createStringError(
          std::make_error_code(std::errc::not_supported),
          "pre_mapping_spectrum_endpoint_unsupported: requested endpoint "
          "requires a verified SystemMapping schedule");
    appendEndpointCandidate(*selected);
  } else if (!objectiveOrder->empty()) {
    append(objectiveOrder->front());
  }
  std::uint64_t diversityRetained = result.preferenceOrder.size();
  const auto appendBestTemporal = [&]() {
    const PreMappingFrontierCandidate *best = nullptr;
    for (const auto &[root, candidate] : unique) {
      if (candidate->verifiedSpectrum !=
          PreMappingSpectrumClass::MaxTemporal)
        continue;
      if (!best || representativeLess(*candidate, *best))
        best = candidate;
    }
    if (!best)
      return false;
    const std::size_t before = result.preferenceOrder.size();
    append(best->candidate);
    return result.preferenceOrder.size() != before;
  };
  const auto appendBestTemporalHint = [&]() {
    const PreMappingFrontierCandidate *best = nullptr;
    for (const auto &[root, candidate] : unique) {
      (void)root;
      if (candidate->scheduleIntent !=
          PreMappingScheduleIntent::TemporalReuse)
        continue;
      if (!best || representativeLess(*candidate, *best))
        best = candidate;
    }
    if (!best)
      return false;
    const std::size_t before = result.preferenceOrder.size();
    append(best->candidate);
    return result.preferenceOrder.size() != before;
  };
  if (endpoint == PreMappingSpectrumEndpoint::Automatic &&
      diversityRetained < diversityCandidateCount) {
    const bool retainedTemporal = appendBestTemporal();
    diversityRetained += retainedTemporal;
    if (!retainedTemporal && diversityRetained < diversityCandidateCount)
      diversityRetained += appendBestTemporalHint();
  }
  if (diversityRetained < diversityCandidateCount &&
      !minimumOwnership->empty()) {
    const std::size_t before = result.preferenceOrder.size();
    append(minimumOwnership->front());
    diversityRetained += result.preferenceOrder.size() != before;
  }
  if (diversityRetained < diversityCandidateCount &&
      !maximumOwnership->empty())
    append(maximumOwnership->front());
  for (const ArtifactRootReference &root : *objectiveOrder)
    append(root);
  for (const ArtifactRootReference &root : *allOrder)
    append(root);
  return result;
}

llvm::Expected<PreMappingCoordinatePlan> buildPreMappingCoordinatePlan(
    llvm::ArrayRef<frontend::StructuredEntityRef> roots,
    const frontend::analysis::StructuredProtocolDependencyProjection
        &dependencies,
    llvm::ArrayRef<PreMappingRootActivity> activity,
    const PreMappingFrontierPolicy &policy,
    PreMappingWorkAccounting &accounting) {
  if (llvm::Error error = validatePreMappingFrontierPolicy(policy))
    return std::move(error);
  if (activity.size() != roots.size())
    return invalid("coordinate planning activity is not total");
  if (roots.empty())
    return invalid("coordinate planning requires a rooted workload domain");
  if (accounting.coordinates.limit !=
      policy.budget.maximumCoordinatesGenerated)
    return invalid("coordinate accounting has a foreign limit");

  std::map<std::vector<std::uint8_t>, std::size_t> ordinalByRoot;
  for (auto indexed : llvm::enumerate(roots)) {
    if (activity[indexed.index()].root != indexed.value())
      return invalid("coordinate planning activity changed root order");
    if (!ordinalByRoot.emplace(rootKey(indexed.value()), indexed.index()).second)
      return invalid("coordinate planning roots are not unique");
  }
  std::vector<std::vector<std::size_t>> producersByConsumer(roots.size());
  std::vector<std::vector<std::size_t>> undirected(roots.size());
  for (const auto &relation : dependencies.relations) {
    if (relation.knowledge != frontend::analysis::
                                  StructuredProtocolDependencyKnowledge::
                                      ProvenPresent)
      continue;
    const auto producer = ordinalByRoot.find(rootKey(relation.producer));
    const auto consumer = ordinalByRoot.find(rootKey(relation.consumer));
    if (producer == ordinalByRoot.end() || consumer == ordinalByRoot.end())
      return invalid("coordinate dependency is outside its root set");
    producersByConsumer[consumer->second].push_back(producer->second);
    undirected[producer->second].push_back(consumer->second);
    undirected[consumer->second].push_back(producer->second);
  }
  for (auto &adjacency : undirected) {
    llvm::sort(adjacency);
    adjacency.erase(std::unique(adjacency.begin(), adjacency.end()),
                    adjacency.end());
  }
  for (auto &producers : producersByConsumer) {
    llvm::sort(producers);
    producers.erase(std::unique(producers.begin(), producers.end()),
                    producers.end());
  }

  PreMappingCoordinatePlan result;
  using CoordinateKey =
      std::pair<std::vector<std::size_t>, PreMappingScheduleIntent>;
  std::set<CoordinateKey> seen;
  std::map<CoordinateKey, std::size_t> retained;
  const auto offer = [&](std::vector<std::size_t> ordinals,
                         PreMappingSpectrumSeedKind kind,
                         PreMappingScheduleIntent scheduleIntent =
                             PreMappingScheduleIntent::Unconstrained)
      -> llvm::Error {
    if (ordinals.empty() && !roots.empty())
      return llvm::Error::success();
    llvm::sort(ordinals);
    ordinals.erase(std::unique(ordinals.begin(), ordinals.end()),
                   ordinals.end());
    CoordinateKey key{ordinals, scheduleIntent};
    auto found = retained.find(key);
    if (found != retained.end()) {
      auto &kinds = result.coordinates[found->second].seedKinds;
      if (!llvm::is_contained(kinds, kind)) {
        kinds.push_back(kind);
        llvm::sort(kinds, [](auto lhs, auto rhs) {
          return static_cast<std::uint8_t>(lhs) <
                 static_cast<std::uint8_t>(rhs);
        });
      }
      return llvm::Error::success();
    }
    if (!seen.insert(key).second)
      return llvm::Error::success();
    // `eligibleCoordinateCount` describes the policy domain, not only the
    // portion admitted by the invocation budget. Keep it truthful so a
    // truncated plan can report the size of the omitted frontier.
    ++result.eligibleCoordinateCount;
    if (accounting.coordinates.planned == accounting.coordinates.limit) {
      result.truncated = true;
      return llvm::Error::success();
    }
    ++accounting.coordinates.planned;
    ++accounting.coordinates.reserved;
    auto projection = projectCoordinate(ordinals, roots, dependencies, activity);
    if (!projection)
      return projection.takeError();
    const std::size_t index = result.coordinates.size();
    retained.emplace(std::move(key), index);
    result.coordinates.push_back({std::move(ordinals), {kind},
                                  std::move(*projection), scheduleIntent,
                                  std::nullopt});
    ++accounting.coordinates.consumed;
    return llvm::Error::success();
  };

  std::vector<std::size_t> full;
  full.reserve(roots.size());
  for (std::size_t ordinal = 0; ordinal != roots.size(); ++ordinal)
    full.push_back(ordinal);
  std::vector<std::size_t> activityOrder = full;
  llvm::sort(activityOrder, [&](std::size_t lhs, std::size_t rhs) {
    if (activity[lhs].dynamicLeafExecutions !=
        activity[rhs].dynamicLeafExecutions)
      return activity[lhs].dynamicLeafExecutions >
             activity[rhs].dynamicLeafExecutions;
    if (activity[lhs].dynamicActivations != activity[rhs].dynamicActivations)
      return activity[lhs].dynamicActivations >
             activity[rhs].dynamicActivations;
    return lhs < rhs;
  });

  // The planner emits bounded ownership and schedule intents only. A full
  // root set is not an endpoint: active-set membership, per-region resource
  // allocation, and a verified SystemMapping schedule are required before a
  // candidate can be classified as MaxTemporal or MaxSpatial.
  if (llvm::Error error = offer(full, PreMappingSpectrumSeedKind::CanonicalFallback))
    return std::move(error);
  for (std::size_t index = 0; index != activityOrder.size(); ++index)
    if (llvm::Error error = offer(
            {activityOrder[index]}, PreMappingSpectrumSeedKind::Intermediate,
            PreMappingScheduleIntent::TemporalReuse))
      return std::move(error);
  if (llvm::Error error = offer(
          full, PreMappingSpectrumSeedKind::CanonicalFallback,
          PreMappingScheduleIntent::SpatialParallel))
    return std::move(error);

  for (auto indexed : llvm::enumerate(activityOrder))
    if (llvm::Error error = offer(
            {indexed.value()},
            indexed.index() == 0
                ? PreMappingSpectrumSeedKind::HighActivitySingleton
                : PreMappingSpectrumSeedKind::Intermediate,
            PreMappingScheduleIntent::SpatialParallel))
      return std::move(error);

  for (std::size_t consumer = 0; consumer != roots.size(); ++consumer) {
    const auto &producers = producersByConsumer[consumer];
    for (std::size_t producer : producers)
      if (llvm::Error error = offer(
              {producer, consumer},
              PreMappingSpectrumSeedKind::DependencyEdge))
        return std::move(error);
    if (producers.size() > 1) {
      if (llvm::Error error = offer(
              producers, PreMappingSpectrumSeedKind::ProducerGroup))
        return std::move(error);
      std::vector<std::size_t> pipeline = producers;
      pipeline.push_back(consumer);
      if (llvm::Error error = offer(
              std::move(pipeline),
              PreMappingSpectrumSeedKind::PipelineGroup))
        return std::move(error);
    }
  }

  // Bounded adjacency expansion supplies compositional candidates that are
  // not reducible to one dependency edge or one connected component. The
  // expansion is deterministic and stops at the explicit policy limits.
  std::uint64_t compositionalGroups = 0;
  for (std::size_t seed = 0; seed != roots.size() &&
                              compositionalGroups <
                                  policy.maximumCompositionalGroups;
       ++seed) {
    std::vector<std::size_t> group{seed};
    for (std::uint64_t depth = 1;
         depth <= policy.maximumExpansionDepth &&
         compositionalGroups < policy.maximumCompositionalGroups;
         ++depth) {
      std::vector<std::size_t> next = group;
      for (std::size_t member : group)
        next.insert(next.end(), undirected[member].begin(),
                    undirected[member].end());
      llvm::sort(next);
      next.erase(std::unique(next.begin(), next.end()), next.end());
      if (next == group)
        break;
      group = std::move(next);
      if (llvm::Error error =
              offer(group, PreMappingSpectrumSeedKind::Intermediate))
        return std::move(error);
      ++compositionalGroups;
    }
  }

  std::vector<bool> visited(roots.size(), false);
  for (std::size_t seed = 0; seed != roots.size(); ++seed) {
    if (visited[seed] || undirected[seed].empty())
      continue;
    std::vector<std::size_t> component;
    std::vector<std::size_t> frontier{seed};
    visited[seed] = true;
    for (std::size_t cursor = 0; cursor != frontier.size(); ++cursor) {
      const std::size_t member = frontier[cursor];
      component.push_back(member);
      for (std::size_t neighbor : undirected[member])
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          frontier.push_back(neighbor);
        }
    }
    if (llvm::Error error = offer(
            std::move(component),
            PreMappingSpectrumSeedKind::ConnectedComponent))
      return std::move(error);
  }

  std::vector<std::size_t> prefix;
  for (std::size_t ordinal = 0; ordinal != roots.size(); ++ordinal) {
    prefix.push_back(ordinal);
    if (llvm::Error error = offer(
            prefix, PreMappingSpectrumSeedKind::CanonicalFallback))
      return std::move(error);
  }
  return result;
}

llvm::Expected<PreMappingShadowRecall> evaluatePreMappingShadowRecall(
    std::size_t rootCount, const PreMappingCoordinatePlan &plan,
    std::size_t maximumRoots) {
  if (rootCount > maximumRoots)
    return invalid("shadow recall root domain exceeds its explicit bound");
  PreMappingShadowRecall result;
  std::set<std::vector<std::size_t>> generated;
  for (const PreMappingCoordinate &coordinate : plan.coordinates)
    generated.insert(coordinate.ownedProtocolOrdinals);
  const std::size_t subsetCount = std::size_t{1} << rootCount;
  for (std::size_t mask = 1; mask != subsetCount; ++mask) {
    std::vector<std::size_t> subset;
    for (std::size_t ordinal = 0; ordinal != rootCount; ++ordinal)
      if (mask & (std::size_t{1} << ordinal))
        subset.push_back(ordinal);
    ++result.eligibleSubsets;
    if (generated.count(subset))
      ++result.coveredSubsets;
    else
      result.missingSubsets.push_back(std::move(subset));
  }
  result.generatedSubsets = generated.size();
  return result;
}

llvm::Expected<PreMappingMaterializedProjection>
projectPreMappingMaterializedCandidate(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &system, llvm::StringRef entrySymbol) {
  auto placeholder = ComponentViewDigest::fromBytes(
      std::array<std::uint8_t, ComponentViewDigest::byteSize>{});
  if (!placeholder)
    return placeholder.takeError();
  PreMappingMaterializedProjection result(*placeholder);
  auto reachableRoots =
      dataflow.projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!reachableRoots)
    return reachableRoots.takeError();
  result.rootThreadLaunchCount = reachableRoots->size();
  std::vector<dataflow::GraphRef> reachableGraphs;
  std::vector<mlir::Operation *> reachableThreads;
  for (dataflow::RootThreadLaunchRef root : *reachableRoots) {
    auto launch = dataflow.resolve(root);
    if (!launch)
      return launch.takeError();
    if (!llvm::is_contained(reachableThreads, launch->callee))
      reachableThreads.push_back(launch->callee);
  }
  llvm::Error graphError = llvm::Error::success();
  dataflow.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    if (graphError ||
        !llvm::is_contained(*reachableRoots, launch.rootThreadLaunch))
      return;
    auto graph = dataflow.resolve(launch);
    if (!graph) {
      graphError = graph.takeError();
      return;
    }
    if (!llvm::is_contained(reachableGraphs, *graph))
      reachableGraphs.push_back(*graph);
  });
  if (graphError)
    return std::move(graphError);
  for (const dataflow::CanonicalLogicalMemoryRootView &memory :
       dataflow.logicalMemoryRoots()) {
    bool reachable = llvm::is_contained(reachableThreads, memory.op);
    if (!reachable) {
      if (auto graph = memory.op->getParentOfType<dataflow::GraphOp>()) {
        auto graphView = llvm::find_if(dataflow.graphs(), [&](const auto &view) {
          return view.op == graph.getOperation();
        });
        reachable = graphView != dataflow.graphs().end() &&
                    llvm::is_contained(reachableGraphs, graphView->ref);
      }
    }
    if (!reachable) {
      if (auto thread = memory.op->getParentOfType<dataflow::ThreadOp>())
        reachable = llvm::is_contained(reachableThreads,
                                       thread.getOperation());
    }
    result.logicalMemoryRootCount += reachable;
  }
  for (const dataflow::CanonicalActorView &actor : dataflow.actors()) {
    if (!llvm::is_contained(reachableGraphs, actor.graph))
      continue;
    ++result.actorCount;
    switch (actor.kind) {
    case dataflow::CanonicalDataflowActorKind::Compute:
      ++result.computeActorCount;
      break;
    case dataflow::CanonicalDataflowActorKind::Control:
      ++result.controlActorCount;
      break;
    case dataflow::CanonicalDataflowActorKind::Memory:
      ++result.memoryActorCount;
      break;
    }
    result.streamActorCount += llvm::isa<dataflow::StreamOp>(actor.op);
  }
  if (llvm::Error error = dataflow.forEachGraphEdge([&](const auto &,
                                                         const auto &consumer)
                                                     -> llvm::Error {
        std::optional<dataflow::GraphRef> graph;
        if (const auto *actor =
                std::get_if<dataflow::ActorTokenOperandRef>(&consumer)) {
          auto resolved = dataflow.resolve(actor->actor);
          if (!resolved)
            return resolved.takeError();
          graph = resolved->graph;
        } else {
          const auto &egress = std::get<dataflow::GraphEgressTokenRef>(consumer);
          std::visit([&](const auto &reference) { graph = reference.graph; },
                     egress);
        }
        if (!graph || !llvm::is_contained(reachableGraphs, *graph))
          return llvm::Error::success();
        if (result.graphEdgeCount == std::numeric_limits<std::uint64_t>::max())
          return invalid("materialized graph-edge count overflows");
        ++result.graphEdgeCount;
        return llvm::Error::success();
      }))
    return std::move(error);

  llvm::Error domainError = llvm::Error::success();
  dataflow.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    if (domainError ||
        !llvm::is_contained(*reachableRoots, launch.rootThreadLaunch))
      return;
    if (result.rootedGraphLaunchCount ==
        std::numeric_limits<std::uint64_t>::max()) {
      domainError = invalid("rooted graph count overflows");
      return;
    }
    ++result.rootedGraphLaunchCount;
    auto extents = dataflow.projectStaticDenseExtents(launch, entrySymbol);
    if (!extents) {
      domainError = extents.takeError();
      return;
    }
    if (!*extents) {
      ++result.unknownLogicalDomainCount;
      return;
    }
    std::uint64_t points = 1;
    for (std::uint64_t extent : **extents) {
      const std::optional<std::uint64_t> product =
          llvm::checkedMulUnsigned(points, extent);
      if (!product) {
        domainError = invalid("logical-domain point count overflows");
        return;
      }
      points = *product;
    }
    if (!checkedAdd(result.staticLogicalDomainPointCount, points))
      domainError = invalid("logical-domain point sum overflows");
  });
  if (domainError)
    return std::move(domainError);

  result.availableAccCoreCount =
      system.artifact().accCoreOccurrences().size();
  if (result.unknownLogicalDomainCount == 0) {
    result.logicalDomainSupport = PreMappingLogicalDomainSupport::Exact;
    if (result.availableAccCoreCount != 0) {
      const std::uint64_t quotient =
          result.staticLogicalDomainPointCount / result.availableAccCoreCount;
      const std::uint64_t remainder =
          result.staticLogicalDomainPointCount % result.availableAccCoreCount;
      result.minimumExecutionWaves = quotient + (remainder != 0);
      result.maximumParallelAccCoreCount =
          std::min(result.staticLogicalDomainPointCount,
                   result.availableAccCoreCount);
    }
  } else if (result.staticLogicalDomainPointCount != 0) {
    result.logicalDomainSupport = PreMappingLogicalDomainSupport::Partial;
  }

  result.temporalWitness.logicalEpochCount =
      result.staticLogicalDomainPointCount + result.unknownLogicalDomainCount;
  result.temporalWitness.launchCount = result.rootedGraphLaunchCount;
  result.temporalWitness.synchronizationCount =
      result.rootedGraphLaunchCount == 0 ? 0 : result.rootedGraphLaunchCount - 1;
  result.temporalWitness.accCoreOccupancy =
      result.maximumParallelAccCoreCount.value_or(0);
  result.temporalWitness.exact =
      result.logicalDomainSupport == PreMappingLogicalDomainSupport::Exact;

  result.systemTransportResourceCount = system.transportResources().size();
  for (fabric::SystemTransportResourceRef resource :
       system.transportResources()) {
    if (!checkedAdd(result.systemTransferPatternCount,
                    system.transferPatterns(resource).size()))
      return invalid("System transfer-pattern count overflows");
  }

  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), dataflow.identity().bytes().begin(),
               dataflow.identity().bytes().end());
  bytes.insert(bytes.end(), system.artifact().identity().bytes().begin(),
               system.artifact().identity().bytes().end());
  appendU64(bytes, entrySymbol.size());
  bytes.insert(bytes.end(), entrySymbol.bytes_begin(), entrySymbol.bytes_end());
  const std::array<std::uint64_t, 25> fields = {
      result.rootThreadLaunchCount,
      result.rootedGraphLaunchCount,
      result.staticLogicalDomainPointCount,
      result.unknownLogicalDomainCount,
      result.availableAccCoreCount,
      result.minimumExecutionWaves.value_or(0),
      result.minimumExecutionWaves.has_value(),
      result.maximumParallelAccCoreCount.value_or(0),
      result.maximumParallelAccCoreCount.has_value(),
      result.actorCount,
      result.computeActorCount,
      result.controlActorCount,
      result.memoryActorCount,
      result.graphEdgeCount,
      result.logicalMemoryRootCount,
      result.streamActorCount,
      result.systemTransportResourceCount,
      result.systemTransferPatternCount,
      result.temporalWitness.logicalEpochCount,
      result.temporalWitness.accCoreOccupancy,
      result.temporalWitness.launchCount,
      result.temporalWitness.synchronizationCount,
      result.temporalWitness.liveStateBytes,
      result.temporalWitness.liveStateKnown ? 1ULL : 0ULL,
      result.temporalWitness.exact ? 1ULL : 0ULL};
  for (std::uint64_t field : fields)
    appendU64(bytes, field);
  auto identity = computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           materializedProjectionDescriptor.data()),
       materializedProjectionDescriptor.size()},
      bytes);
  if (!identity)
    return identity.takeError();
  result.identity = *identity;
  return result;
}

} // namespace loom::dse
