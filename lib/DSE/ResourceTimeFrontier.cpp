#include "DSE/ResourceTimeFrontier.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <system_error>
#include <tuple>

namespace loom::dse {
namespace {

using MonotonicClock = std::chrono::steady_clock;

constexpr llvm::StringLiteral resourceTimeTransitionCacheDescriptor{
    "loom.dse.resource_time_transition_cache.1"};
constexpr llvm::StringLiteral resourceTimeAnalyticModelDescriptor{
    "loom.dse.resource_time_analytic_model.1"};
constexpr llvm::StringLiteral resourceTimeProjectionMemoDescriptor{
    "loom.dse.resource_time_projection_memo.1"};
constexpr llvm::StringLiteral resourceTimeExactMemoDescriptor{
    "loom.dse.resource_time_exact_frontier_memo.1"};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resource_time_frontier_invalid: " + message);
}

struct WorkTimer final {
  explicit WorkTimer(ResourceTimeWorkCounter &counter)
      : counter(counter), begin(MonotonicClock::now()) {}
  ~WorkTimer() {
    counter.elapsedNanoseconds +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            MonotonicClock::now() - begin)
            .count();
  }
  ResourceTimeWorkCounter &counter;
  MonotonicClock::time_point begin;
};

bool rootLess(::dataflow::RootThreadLaunchRef lhs,
              ::dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBlob(std::vector<std::uint8_t> &bytes,
                llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendRoot(std::vector<std::uint8_t> &bytes,
                const ArtifactRootReference &reference) {
  appendBlob(bytes, encodeArtifactRootReference(reference));
}

void appendDataflowRoot(std::vector<std::uint8_t> &bytes,
                        ::dataflow::RootThreadLaunchRef reference) {
  appendBlob(bytes, reference.artifact.bytes());
  appendU64(bytes, reference.entity.value());
}

void appendOptionalRoot(
    std::vector<std::uint8_t> &bytes,
    const std::optional<ArtifactRootReference> &reference) {
  bytes.push_back(reference ? 1 : 0);
  if (reference)
    appendRoot(bytes, *reference);
}

void appendOptionalU64(std::vector<std::uint8_t> &bytes,
                       std::optional<std::uint64_t> value) {
  bytes.push_back(value ? 1 : 0);
  if (value)
    appendU64(bytes, *value);
}

void appendString(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

void appendDigest(std::vector<std::uint8_t> &bytes,
                  const ComponentViewDigest &digest) {
  appendBlob(bytes, digest.bytes());
}

void appendResourceTimeInvocationKey(
    std::vector<std::uint8_t> &bytes,
    const ResourceTimeInvocationKey &invocation) {
  appendRoot(bytes, invocation.sourceLineage);
  appendRoot(bytes, invocation.dataflow);
  appendRoot(bytes, invocation.fabric);
  appendRoot(bytes, invocation.workload);
  appendRoot(bytes, invocation.runtimeInput);
  appendDigest(bytes, invocation.resolvedConfigDigest);
  appendDigest(bytes, invocation.modelSnapshotDigest);
  appendString(bytes, invocation.entrySymbol);
  appendOptionalU64(bytes, invocation.estimatedRuntimePicoseconds);
}

void appendResourceTimeFeatures(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions) {
  appendU64(bytes, resourceClasses.size());
  for (const ArtifactRootReference &resource : resourceClasses)
    appendRoot(bytes, resource);
  appendU64(bytes, regions.size());
  for (const ResourceTimeRegionFeature &region : regions) {
    appendDataflowRoot(bytes, region.region);
    appendU64(bytes, region.logicalEpochCount);
    bytes.push_back(region.allocationDomainExhaustive ? 1 : 0);
    appendU64(bytes, region.analyticFeatures.actorCount);
    appendU64(bytes, region.analyticFeatures.computeActorCount);
    appendU64(bytes, region.analyticFeatures.controlActorCount);
    appendU64(bytes, region.analyticFeatures.memoryActorCount);
    appendU64(bytes, region.analyticFeatures.graphCount);
    appendU64(bytes, region.analyticFeatures.launchSynchronizationCost);
    appendU64(bytes, region.analyticFeatures.parallelismLowerBound);
    appendU64(bytes, region.analyticFeatures.topologyCongestionProxy);
    appendU64(bytes, region.dependencies.size());
    for (const ResourceTimeDependencyFeature &dependency :
         region.dependencies) {
      appendDataflowRoot(bytes, dependency.producer);
      appendU64(bytes, static_cast<std::uint64_t>(dependency.readiness));
    }
    appendU64(bytes, region.speedupCurve.size());
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve) {
      appendU64(bytes, point.resourceUnits.size());
      for (std::uint64_t units : point.resourceUnits)
        appendU64(bytes, units);
      appendU64(bytes, point.executionTimePicoseconds);
      appendOptionalU64(bytes, point.firstTokenLatencyPicoseconds);
      appendOptionalU64(bytes, point.initiationIntervalPicoseconds);
      appendU64(bytes, point.hostTransferTimePicoseconds);
      appendU64(bytes, point.configurationTimePicoseconds);
      appendU64(bytes, point.liveStateMigrationTimePicoseconds);
      appendU64(bytes, static_cast<std::uint64_t>(point.support));
    }
  }
}

void appendResourceTimePolicy(std::vector<std::uint8_t> &bytes,
                              const ResourceTimeFrontierPolicy &policy) {
  appendU64(bytes, policy.availableResourceUnits.size());
  for (std::uint64_t units : policy.availableResourceUnits)
    appendU64(bytes, units);
  appendU64(bytes, policy.maximumStatesGenerated);
  appendU64(bytes, policy.maximumActionsGenerated);
  appendU64(bytes, policy.maximumStateCacheEntries);
  appendU64(bytes, policy.maximumRetainedBytes);
  appendU64(bytes, policy.beamWidth);
  appendU64(bytes, policy.maximumFinalists);
}

std::string exactFrontierMemoKey(
    const ResourceTimeInvocationKey &invocation,
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    const ResourceTimeFrontierPolicy &policy) {
  std::vector<std::uint8_t> bytes;
  appendString(bytes, resourceTimeExactMemoDescriptor);
  appendResourceTimeInvocationKey(bytes, invocation);
  appendResourceTimeFeatures(bytes, resourceClasses, regions);
  appendResourceTimePolicy(bytes, policy);
  return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

std::vector<std::uint8_t> canonicalAllocationBytes(
    const pnr::ResourceTimeRegionAllocation &allocation) {
  std::vector<std::vector<std::uint8_t>> resources;
  resources.reserve(allocation.resources.size());
  for (const auto &resource : allocation.resources)
    resources.push_back(fabric::canonicalFabricBytes(resource));
  llvm::sort(resources);
  std::vector<std::uint8_t> bytes;
  appendDataflowRoot(bytes, allocation.region);
  appendU64(bytes, resources.size());
  for (const auto &resource : resources)
    appendBlob(bytes, resource);
  return bytes;
}

void appendAllocations(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> allocations) {
  std::vector<std::vector<std::uint8_t>> encoded;
  encoded.reserve(allocations.size());
  for (const auto &allocation : allocations)
    encoded.push_back(canonicalAllocationBytes(allocation));
  llvm::sort(encoded);
  appendU64(bytes, encoded.size());
  for (const auto &allocation : encoded)
    appendBlob(bytes, allocation);
}

void appendRoots(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<ArtifactRootReference> roots) {
  std::vector<ArtifactRootReference> canonical(roots.begin(), roots.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  appendU64(bytes, canonical.size());
  for (const auto &root : canonical)
    appendRoot(bytes, root);
}

std::uint64_t allocationMagnitude(llvm::ArrayRef<std::uint64_t> allocation) {
  return std::accumulate(allocation.begin(), allocation.end(), 0ULL);
}

bool fits(llvm::ArrayRef<std::uint64_t> used,
          llvm::ArrayRef<std::uint64_t> requested,
          llvm::ArrayRef<std::uint64_t> available) {
  if (used.size() != requested.size() || used.size() != available.size())
    return false;
  for (std::size_t index = 0; index != used.size(); ++index)
    if (requested[index] > available[index] ||
        used[index] > available[index] - requested[index])
      return false;
  return true;
}

std::uint64_t pointDuration(const ResourceTimeSpeedupPoint &point) {
  const auto withHost = llvm::checkedAddUnsigned(
      point.executionTimePicoseconds, point.hostTransferTimePicoseconds);
  if (!withHost)
    return std::numeric_limits<std::uint64_t>::max();
  const auto withConfiguration = llvm::checkedAddUnsigned(
      *withHost, point.configurationTimePicoseconds);
  if (!withConfiguration)
    return std::numeric_limits<std::uint64_t>::max();
  return llvm::checkedAddUnsigned(*withConfiguration,
                                  point.liveStateMigrationTimePicoseconds)
      .value_or(std::numeric_limits<std::uint64_t>::max());
}

ResourceTimeEstimateSupport combineSupport(ResourceTimeEstimateSupport lhs,
                                           ResourceTimeEstimateSupport rhs) {
  const auto rank = [](ResourceTimeEstimateSupport support) {
    switch (support) {
    case ResourceTimeEstimateSupport::Exact:
      return 0;
    case ResourceTimeEstimateSupport::Calibrated:
      return 1;
    case ResourceTimeEstimateSupport::Analytic:
      return 2;
    case ResourceTimeEstimateSupport::OutOfDomain:
      return 3;
    case ResourceTimeEstimateSupport::Unsupported:
      return 4;
    }
    llvm_unreachable("unknown resource-time estimate support");
  };
  return rank(lhs) >= rank(rhs) ? lhs : rhs;
}

ResourceTimeEstimateConfidence confidenceForSupport(
    ResourceTimeEstimateSupport support) {
  switch (support) {
  case ResourceTimeEstimateSupport::Exact:
    return ResourceTimeEstimateConfidence::Calibrated;
  case ResourceTimeEstimateSupport::Calibrated:
    return ResourceTimeEstimateConfidence::Calibrated;
  case ResourceTimeEstimateSupport::Analytic:
    return ResourceTimeEstimateConfidence::Low;
  case ResourceTimeEstimateSupport::OutOfDomain:
    return ResourceTimeEstimateConfidence::OutOfDomain;
  case ResourceTimeEstimateSupport::Unsupported:
    return ResourceTimeEstimateConfidence::None;
  }
  llvm_unreachable("unknown resource-time estimate support");
}

std::uint8_t estimateSupportRank(ResourceTimeEstimateSupport support) {
  switch (support) {
  case ResourceTimeEstimateSupport::Exact:
    return 0;
  case ResourceTimeEstimateSupport::Calibrated:
    return 1;
  case ResourceTimeEstimateSupport::Analytic:
    return 2;
  case ResourceTimeEstimateSupport::OutOfDomain:
    return 3;
  case ResourceTimeEstimateSupport::Unsupported:
    return 4;
  }
  llvm_unreachable("unknown resource-time estimate support");
}

struct Dependency final {
  std::size_t producer = 0;
  std::size_t consumer = 0;
  pnr::ResourceTimeReadinessKind readiness =
      pnr::ResourceTimeReadinessKind::Completion;
};

struct FrozenInput final {
  std::vector<ResourceTimeRegionFeature> regions;
  std::vector<Dependency> dependencies;
  std::vector<std::vector<std::size_t>> outgoingDependencies;
  std::vector<std::vector<std::size_t>> incomingDependencies;
  std::vector<std::uint64_t> minimumDurations;
  std::vector<std::uint64_t> minimumResourceWork;
  std::vector<std::uint64_t> minimumSuccessorTails;
  std::vector<std::size_t> reverseTopologicalOrder;
};

struct ActiveRegion final {
  std::size_t region = 0;
  std::size_t point = 0;
  std::uint64_t completionTime = 0;
  std::optional<std::uint64_t> tokenTime;
  bool tokenPublished = false;
};

struct SearchState final {
  std::uint64_t time = 0;
  std::vector<bool> started;
  std::vector<bool> completed;
  std::vector<bool> dependencySatisfied;
  std::vector<std::uint64_t> satisfiedDependencyCount;
  std::vector<std::size_t> ready;
  std::vector<ActiveRegion> active;
  std::vector<std::uint64_t> usedResources;
  std::vector<ResourceTimeActionDelta> actions;
  std::vector<ResourceTimeHintState> snapshots;
  std::uint64_t lowerBound = 0;
  std::uint64_t minimumRemainingResourceWork = 0;
  bool lowerBoundInitialized = false;
  std::uint64_t peakConcurrentRegions = 0;
  std::uint64_t totalAllocatedResourceTime = 0;
  ResourceTimeEstimateSupport support = ResourceTimeEstimateSupport::Exact;
};

struct StateMemoEnvelope final {
  std::uint64_t minimumLowerBound = 0;
  std::uint64_t minimumPeakConcurrentRegions = 0;
  std::uint64_t maximumPeakConcurrentRegions = 0;
  std::uint64_t minimumAllocatedResourceTime = 0;
  std::uint8_t bestSupportRank = 0;
};

void sortOrdinals(std::vector<std::size_t> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::vector<std::uint64_t> stateMemoKey(const SearchState &state) {
  // This is a semantic future-state key, not a path key. Source/Dataflow,
  // Fabric, workload, runtime, resolved configuration, and model snapshot
  // are bound by ResourceTimeInvocationKey at the enclosing invocation. The
  // local key records the complete event-relative state: ready/live work,
  // completion and token readiness, per-region allocation, and current time.
  std::vector<std::uint64_t> key;
  key.reserve(5 + state.started.size() * 3 + state.active.size() * 5 +
              state.usedResources.size());
  key.push_back(state.time);
  key.push_back(state.minimumRemainingResourceWork);
  key.push_back(state.started.size());
  for (bool value : state.started)
    key.push_back(value ? 1 : 0);
  for (bool value : state.completed)
    key.push_back(value ? 1 : 0);
  key.push_back(state.dependencySatisfied.size());
  for (bool value : state.dependencySatisfied)
    key.push_back(value ? 1 : 0);
  key.push_back(state.satisfiedDependencyCount.size());
  key.insert(key.end(), state.satisfiedDependencyCount.begin(),
             state.satisfiedDependencyCount.end());
  key.push_back(state.ready.size());
  key.insert(key.end(), state.ready.begin(), state.ready.end());
  key.push_back(state.active.size());
  for (const ActiveRegion &active : state.active) {
    key.push_back(active.region);
    key.push_back(active.point);
    key.push_back(active.completionTime);
    key.push_back(active.tokenTime.value_or(
        std::numeric_limits<std::uint64_t>::max()));
    key.push_back(active.tokenPublished ? 1 : 0);
  }
  key.insert(key.end(), state.usedResources.begin(),
             state.usedResources.end());
  return key;
}

std::uint64_t retainedBytes(const SearchState &state,
                            llvm::ArrayRef<std::uint64_t> memoKey) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto add = [](std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > maximum - lhs ? maximum : lhs + rhs;
  };
  const auto product = [&](std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > maximum / rhs)
      return maximum;
    return static_cast<std::uint64_t>(lhs * rhs);
  };
  const auto sumSizes = [](std::size_t lhs, std::size_t rhs,
                           std::size_t &result) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
      return false;
    result = lhs + rhs;
    return true;
  };
  std::uint64_t bytes = sizeof(SearchState);
  bytes = add(bytes, sizeof(std::uint64_t) + sizeof(bool));
  bytes = add(bytes, product(memoKey.size(), sizeof(std::uint64_t)));
  bytes = add(bytes, product(state.started.size(), 3));
  bytes = add(bytes, state.dependencySatisfied.size());
  bytes = add(bytes, product(state.satisfiedDependencyCount.size(),
                             sizeof(std::uint64_t)));
  bytes = add(bytes, product(state.ready.size(), sizeof(std::size_t)));
  bytes = add(bytes, product(state.active.size(), sizeof(ActiveRegion)));
  bytes = add(bytes, product(state.usedResources.size(),
                             sizeof(std::uint64_t)));
  bytes = add(bytes, product(state.actions.size(),
                             sizeof(ResourceTimeActionDelta)));
  bytes = add(bytes, product(state.snapshots.size(),
                             sizeof(ResourceTimeHintState)));
  for (const ResourceTimeActionDelta &action : state.actions) {
    std::size_t rootCount = 0;
    if (!sumSizes(action.completedRegions.size(),
                  action.tokenReadyProducers.size(), rootCount) ||
        !sumSizes(rootCount, action.newlyReadyRegions.size(), rootCount))
      bytes = maximum;
    else
      bytes = add(bytes, product(rootCount,
                                 sizeof(::dataflow::RootThreadLaunchRef)));
  }
  for (const ResourceTimeHintState &snapshot : state.snapshots) {
    bytes = add(bytes, product(snapshot.active.size(),
                               sizeof(ResourceTimeHintAllocation)));
    std::size_t rootCount = 0;
    if (!sumSizes(snapshot.ready.size(), snapshot.completed.size(), rootCount))
      bytes = maximum;
    else
      bytes = add(bytes, product(rootCount,
                                 sizeof(::dataflow::RootThreadLaunchRef)));
    for (const ResourceTimeHintAllocation &allocation : snapshot.active)
      bytes = add(bytes, product(allocation.resourceUnits.size(),
                                 sizeof(std::uint64_t)));
  }
  return bytes;
}

std::uint64_t optimisticLowerBound(const FrozenInput &input,
                                   const SearchState &state,
                                   llvm::ArrayRef<std::uint64_t> capacity) {
  std::vector<std::uint64_t> remaining(input.regions.size(), 0);
  for (std::size_t region = 0; region != input.regions.size(); ++region) {
    if (state.completed[region])
      continue;
    const auto active = llvm::find_if(state.active, [&](const ActiveRegion &row) {
      return row.region == region;
    });
    remaining[region] =
        active == state.active.end()
            ? input.minimumDurations[region]
            : active->completionTime > state.time
                  ? active->completionTime - state.time
                  : 0;
  }

  std::vector<std::uint64_t> critical(input.regions.size(), 0);
  for (std::size_t region : input.reverseTopologicalOrder) {
    std::uint64_t successor = 0;
    for (std::size_t edge : input.outgoingDependencies[region])
      successor = std::max(successor,
                           critical[input.dependencies[edge].consumer]);
    critical[region] = llvm::checkedAddUnsigned(remaining[region], successor)
                           .value_or(std::numeric_limits<std::uint64_t>::max());
  }
  const std::uint64_t criticalPath =
      critical.empty() ? 0 : *std::max_element(critical.begin(), critical.end());

  std::uint64_t totalWork = 0;
  for (std::size_t region = 0; region != input.regions.size(); ++region) {
    if (state.completed[region])
      continue;
    const auto sum = llvm::checkedAddUnsigned(
        totalWork, input.minimumResourceWork[region]);
    if (!sum) {
      totalWork = std::numeric_limits<std::uint64_t>::max();
      break;
    }
    totalWork = *sum;
  }
  const std::uint64_t totalCapacity = allocationMagnitude(capacity);
  const std::uint64_t workBound =
      totalCapacity == 0
          ? std::numeric_limits<std::uint64_t>::max()
          : totalWork / totalCapacity + (totalWork % totalCapacity != 0);
  return llvm::checkedAddUnsigned(state.time,
                                  std::max(criticalPath, workBound))
      .value_or(std::numeric_limits<std::uint64_t>::max());
}

std::uint64_t incrementalLowerBound(
    const FrozenInput &input, const SearchState &parent,
    const SearchState &state, llvm::ArrayRef<std::uint64_t> capacity,
    llvm::ArrayRef<std::size_t> changedRegions) {
  std::uint64_t result = parent.lowerBound;
  const auto includeRegion = [&](std::size_t region) {
    if (region >= input.minimumSuccessorTails.size() ||
        state.completed[region])
      return;
    const auto tail = input.minimumSuccessorTails[region];
    const auto candidate = llvm::checkedAddUnsigned(state.time, tail);
    result = std::max(result, candidate.value_or(
                                  std::numeric_limits<std::uint64_t>::max()));
  };
  for (const ActiveRegion &active : state.active) {
    const std::uint64_t tail =
        input.minimumSuccessorTails[active.region] >=
                input.minimumDurations[active.region]
            ? input.minimumSuccessorTails[active.region] -
                  input.minimumDurations[active.region]
            : 0;
    const auto candidate = llvm::checkedAddUnsigned(
        active.completionTime, tail);
    result = std::max(result, candidate.value_or(
                                  std::numeric_limits<std::uint64_t>::max()));
  }
  for (std::size_t region : changedRegions)
    includeRegion(region);
  const std::uint64_t totalCapacity = allocationMagnitude(capacity);
  if (totalCapacity != 0) {
    const std::uint64_t workBound =
        state.minimumRemainingResourceWork / totalCapacity +
        (state.minimumRemainingResourceWork % totalCapacity != 0);
    const auto candidate = llvm::checkedAddUnsigned(state.time, workBound);
    result = std::max(result, candidate.value_or(
                                  std::numeric_limits<std::uint64_t>::max()));
  }
  return result;
}

ResourceTimeHintState makeSnapshot(const FrozenInput &input,
                                   const SearchState &state) {
  ResourceTimeHintState snapshot;
  snapshot.timePicoseconds = state.time;
  snapshot.optimisticMakespanLowerBoundPicoseconds = state.lowerBound;
  for (const ActiveRegion &active : state.active) {
    const ResourceTimeSpeedupPoint &point =
        input.regions[active.region].speedupCurve[active.point];
    snapshot.active.push_back(
        {input.regions[active.region].region,
         static_cast<std::uint64_t>(active.point), point.resourceUnits,
         active.completionTime});
  }
  llvm::sort(snapshot.active, [](const auto &lhs, const auto &rhs) {
    return rootLess(lhs.region, rhs.region);
  });
  for (std::size_t region : state.ready)
    snapshot.ready.push_back(input.regions[region].region);
  for (std::size_t region = 0; region != state.completed.size(); ++region)
    if (state.completed[region])
      snapshot.completed.push_back(input.regions[region].region);
  llvm::sort(snapshot.ready, rootLess);
  llvm::sort(snapshot.completed, rootLess);
  return snapshot;
}

std::vector<std::size_t> newlyReady(const FrozenInput &input,
                                    SearchState &state,
                                    llvm::ArrayRef<std::size_t> changedEdges) {
  std::vector<std::size_t> result;
  for (std::size_t edge : changedEdges) {
    if (state.dependencySatisfied[edge])
      continue;
    state.dependencySatisfied[edge] = true;
    const std::size_t consumer = input.dependencies[edge].consumer;
    ++state.satisfiedDependencyCount[consumer];
    if (!state.started[consumer] &&
        state.satisfiedDependencyCount[consumer] ==
            input.incomingDependencies[consumer].size()) {
      state.ready.push_back(consumer);
      result.push_back(consumer);
    }
  }
  sortOrdinals(state.ready);
  sortOrdinals(result);
  return result;
}

bool hintLess(const ResourceTimeScheduleHint &lhs,
              const ResourceTimeScheduleHint &rhs) {
  return std::tuple(estimateSupportRank(lhs.support),
                    lhs.estimatedMakespanPicoseconds,
                  lhs.optimisticMakespanLowerBoundPicoseconds,
                  lhs.totalAllocatedResourceTime,
                  lhs.peakConcurrentRegions) <
         std::tuple(estimateSupportRank(rhs.support),
                    rhs.estimatedMakespanPicoseconds,
                  rhs.optimisticMakespanLowerBoundPicoseconds,
                  rhs.totalAllocatedResourceTime,
                  rhs.peakConcurrentRegions);
}

ResourceTimeScheduleHint makeHint(SearchState state) {
  ResourceTimeScheduleHint hint;
  hint.actions = std::move(state.actions);
  hint.states = std::move(state.snapshots);
  hint.estimatedMakespanPicoseconds = state.time;
  hint.optimisticMakespanLowerBoundPicoseconds = state.lowerBound;
  hint.peakConcurrentRegions = state.peakConcurrentRegions;
  hint.totalAllocatedResourceTime = state.totalAllocatedResourceTime;
  hint.support = state.support;
  return hint;
}

std::vector<ResourceTimeScheduleHint> selectFinalists(
    std::vector<ResourceTimeScheduleHint> hints, std::uint64_t maximum) {
  llvm::sort(hints, hintLess);
  std::vector<ResourceTimeScheduleHint> selected;
  selected.reserve(std::min<std::size_t>(hints.size(), maximum));
  const auto append = [&](std::size_t ordinal) {
    if (ordinal >= hints.size() || selected.size() == maximum)
      return;
    const auto &candidate = hints[ordinal];
    const bool duplicate = llvm::any_of(selected, [&](const auto &existing) {
      return existing.actions == candidate.actions;
    });
    if (!duplicate)
      selected.push_back(candidate);
  };
  if (!hints.empty())
    append(0);
  if (selected.size() < maximum && !hints.empty()) {
    std::size_t temporal = 0;
    for (std::size_t index = 1; index != hints.size(); ++index)
      if (std::tie(hints[index].peakConcurrentRegions,
                   hints[index].estimatedMakespanPicoseconds,
                   hints[index].totalAllocatedResourceTime) <
          std::tie(hints[temporal].peakConcurrentRegions,
                   hints[temporal].estimatedMakespanPicoseconds,
                   hints[temporal].totalAllocatedResourceTime))
        temporal = index;
    append(temporal);
  }
  if (selected.size() < maximum && !hints.empty()) {
    std::size_t spatial = 0;
    for (std::size_t index = 1; index != hints.size(); ++index) {
      const bool betterConcurrency =
          hints[index].peakConcurrentRegions >
          hints[spatial].peakConcurrentRegions;
      const bool equalConcurrency =
          hints[index].peakConcurrentRegions ==
          hints[spatial].peakConcurrentRegions;
      const bool betterTie =
          std::tie(hints[index].totalAllocatedResourceTime,
                   hints[index].estimatedMakespanPicoseconds) <
          std::tie(hints[spatial].totalAllocatedResourceTime,
                   hints[spatial].estimatedMakespanPicoseconds);
      if (betterConcurrency || (equalConcurrency && betterTie))
        spatial = index;
    }
    append(spatial);
  }
  for (std::size_t index = 0; index != hints.size(); ++index)
    append(index);
  return selected;
}

bool stateLess(const SearchState &lhs, const SearchState &rhs) {
  return std::make_tuple(lhs.lowerBound, lhs.time,
                         lhs.totalAllocatedResourceTime,
                         lhs.peakConcurrentRegions, lhs.actions.size()) <
         std::make_tuple(rhs.lowerBound, rhs.time,
                         rhs.totalAllocatedResourceTime,
                         rhs.peakConcurrentRegions, rhs.actions.size());
}

llvm::Expected<FrozenInput> freezeInput(
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    const ResourceTimeFrontierPolicy &policy,
    std::optional<ResourceTimeFrontierInfeasibleReason> &infeasible,
    std::optional<ResourceTimeFrontierIncompleteReason> &unsupported) {
  if (resourceClasses.empty() || regions.empty())
    return invalid("resource classes and regions must be nonempty");
  if (resourceClasses.size() != policy.availableResourceUnits.size())
    return invalid("resource class and capacity vectors differ");
  for (std::size_t index = 0; index != resourceClasses.size(); ++index) {
    if (resourceClasses[index].schemaIdentity.empty())
      return invalid("resource class has an empty identity");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (resourceClasses[prior] == resourceClasses[index])
        return invalid("resource classes are not unique");
  }
  if (llvm::all_of(policy.availableResourceUnits,
                   [](std::uint64_t value) { return value == 0; }))
    return invalid("resource capacity is empty");

  FrozenInput input;
  input.regions.assign(regions.begin(), regions.end());
  input.outgoingDependencies.resize(regions.size());
  input.incomingDependencies.resize(regions.size());
  input.minimumDurations.resize(regions.size());
  input.minimumResourceWork.resize(regions.size());
  std::map<::dataflow::RootThreadLaunchRef, std::size_t,
           decltype(&rootLess)>
      ordinalByRegion(&rootLess);
  for (auto indexed : llvm::enumerate(regions)) {
    const ResourceTimeRegionFeature &region = indexed.value();
    if (region.speedupCurve.empty())
      return invalid("region speedup curve must be nonempty");
    if (!ordinalByRegion.emplace(region.region, indexed.index()).second)
      return invalid("region set is not unique");
    std::uint64_t minimumDuration = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t minimumWork = std::numeric_limits<std::uint64_t>::max();
    std::set<std::vector<std::uint64_t>> allocations;
    bool hasFittingPoint = false;
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve) {
      if (point.resourceUnits.size() != resourceClasses.size() ||
          allocationMagnitude(point.resourceUnits) == 0 ||
          point.executionTimePicoseconds == 0)
        return invalid("speedup point has an invalid resource or time vector");
      if (!allocations.insert(point.resourceUnits).second)
        return invalid("speedup curve contains a duplicate allocation");
      const std::uint64_t duration = pointDuration(point);
      if (duration == std::numeric_limits<std::uint64_t>::max())
        return invalid("speedup point duration overflows");
      if (point.firstTokenLatencyPicoseconds &&
          *point.firstTokenLatencyPicoseconds >
              point.executionTimePicoseconds)
        return invalid("first-token latency exceeds execution time");
      minimumDuration = std::min(minimumDuration, duration);
      const auto work = llvm::checkedMulUnsigned(
          duration, allocationMagnitude(point.resourceUnits));
      if (work)
        minimumWork = std::min(minimumWork, *work);
      std::vector<std::uint64_t> zero(resourceClasses.size(), 0);
      hasFittingPoint |= fits(zero, point.resourceUnits,
                              policy.availableResourceUnits);
    }
    if (!hasFittingPoint) {
      if (region.allocationDomainExhaustive)
        infeasible = ResourceTimeFrontierInfeasibleReason::ResourceCapacity;
      else
        unsupported =
            ResourceTimeFrontierIncompleteReason::ProofNotEstablished;
    }
    input.minimumDurations[indexed.index()] = minimumDuration;
    input.minimumResourceWork[indexed.index()] = minimumWork;
  }

  for (auto indexed : llvm::enumerate(regions)) {
    std::set<std::pair<std::size_t, pnr::ResourceTimeReadinessKind>> seen;
    for (const ResourceTimeDependencyFeature &dependency :
         indexed.value().dependencies) {
      const auto producer = ordinalByRegion.find(dependency.producer);
      if (producer == ordinalByRegion.end() ||
          producer->second == indexed.index())
        return invalid("dependency references a foreign or identical region");
      if (!seen.emplace(producer->second, dependency.readiness).second)
        return invalid("region has a duplicate dependency");
      const std::size_t edge = input.dependencies.size();
      input.dependencies.push_back(
          {producer->second, indexed.index(), dependency.readiness});
      input.outgoingDependencies[producer->second].push_back(edge);
      input.incomingDependencies[indexed.index()].push_back(edge);
      if (dependency.readiness == pnr::ResourceTimeReadinessKind::FifoToken) {
        const bool supported = llvm::all_of(
            regions[producer->second].speedupCurve, [](const auto &point) {
              return point.firstTokenLatencyPicoseconds.has_value();
            });
        if (!supported)
          unsupported = ResourceTimeFrontierIncompleteReason::Unsupported;
      }
    }
  }

  std::vector<std::uint64_t> indegree(regions.size(), 0);
  for (const Dependency &dependency : input.dependencies)
    ++indegree[dependency.consumer];
  std::vector<std::size_t> ready;
  for (std::size_t region = 0; region != regions.size(); ++region)
    if (indegree[region] == 0)
      ready.push_back(region);
  std::vector<std::size_t> topological;
  while (!ready.empty()) {
    const std::size_t region = ready.front();
    ready.erase(ready.begin());
    topological.push_back(region);
    for (std::size_t edge : input.outgoingDependencies[region]) {
      const Dependency &dependency = input.dependencies[edge];
      if (--indegree[dependency.consumer] == 0) {
        ready.push_back(dependency.consumer);
        sortOrdinals(ready);
      }
    }
  }
  if (topological.size() != regions.size()) {
    const bool unresolvedCycleHasFifo =
        llvm::any_of(input.dependencies, [&](const auto &edge) {
          return indegree[edge.producer] != 0 &&
                 indegree[edge.consumer] != 0 &&
                 edge.readiness == pnr::ResourceTimeReadinessKind::FifoToken;
        });
    if (unresolvedCycleHasFifo)
      unsupported = ResourceTimeFrontierIncompleteReason::Unsupported;
    else
      infeasible =
          ResourceTimeFrontierInfeasibleReason::CompletionDependencyCycle;
  }
  input.reverseTopologicalOrder.assign(topological.rbegin(),
                                       topological.rend());
  input.minimumSuccessorTails.assign(regions.size(), 0);
  for (std::size_t region : input.reverseTopologicalOrder) {
    std::uint64_t tail = input.minimumDurations[region];
    for (std::size_t edge : input.outgoingDependencies[region]) {
      const std::size_t consumer = input.dependencies[edge].consumer;
      const auto candidate = llvm::checkedAddUnsigned(
          input.minimumDurations[region],
          input.minimumSuccessorTails[consumer]);
      tail = std::max(tail, candidate.value_or(
                                std::numeric_limits<std::uint64_t>::max()));
    }
    input.minimumSuccessorTails[region] = tail;
  }
  return input;
}

void settleUnconsumed(ResourceTimeWorkCounter &counter, bool cancelled) {
  if (counter.reserved < counter.consumed)
    return;
  const std::uint64_t settled = counter.rejected + counter.cancelled;
  const std::uint64_t available = counter.reserved - counter.consumed;
  if (settled >= available)
    return;
  if (cancelled)
    counter.cancelled += available - settled;
  else
    counter.rejected += available - settled;
}

struct ResourceTimeCandidateScreening final {
  std::uint64_t lowerBoundPicoseconds = 0;
  std::uint64_t featureScore = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
  bool exactCapacityFailure = false;
};

llvm::Expected<ResourceTimeCandidateScreening> screenCandidate(
    const ResourceTimeMappingCandidateInput &candidate,
    const ResourceTimeFrontierPolicy &policy) {
  if (candidate.resourceClasses.empty() || candidate.regions.empty() ||
      candidate.resourceClasses.size() !=
          policy.availableResourceUnits.size())
    return invalid("resource-time screening inputs are not aligned");
  std::uint64_t totalCapacity = 0;
  for (std::uint64_t units : policy.availableResourceUnits) {
    const auto sum = llvm::checkedAddUnsigned(totalCapacity, units);
    if (!sum)
      return invalid("resource-time screening capacity overflows");
    totalCapacity = *sum;
  }
  if (totalCapacity == 0)
    return invalid("resource-time screening has no capacity");

  ResourceTimeCandidateScreening result;
  result.support = ResourceTimeEstimateSupport::Exact;
  std::uint64_t totalResourceWork = 0;
  std::uint64_t featureScore = 0;
  for (const ResourceTimeRegionFeature &region : candidate.regions) {
    if (region.speedupCurve.empty())
      return invalid("resource-time screening region has no speedup curve");
    bool hasFittingPoint = false;
    std::uint64_t minimumDuration =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t minimumWork =
        std::numeric_limits<std::uint64_t>::max();
    ResourceTimeEstimateSupport minimumDurationSupport =
        ResourceTimeEstimateSupport::Unsupported;
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve) {
      if (point.resourceUnits.size() != candidate.resourceClasses.size() ||
          allocationMagnitude(point.resourceUnits) == 0)
        return invalid("resource-time screening point is malformed");
      std::vector<std::uint64_t> unused(point.resourceUnits.size(), 0);
      hasFittingPoint |= fits(unused, point.resourceUnits,
                              policy.availableResourceUnits);
      const std::uint64_t duration = pointDuration(point);
      const auto work = llvm::checkedMulUnsigned(
          duration, allocationMagnitude(point.resourceUnits));
      if (!work)
        return invalid("resource-time screening work overflows");
      if (duration < minimumDuration ||
          (duration == minimumDuration &&
           estimateSupportRank(point.support) <
               estimateSupportRank(minimumDurationSupport))) {
        minimumDuration = duration;
        minimumDurationSupport = point.support;
      }
      minimumWork = std::min(minimumWork, *work);
    }
    if (!hasFittingPoint && region.allocationDomainExhaustive)
      result.exactCapacityFailure = true;
    if (!hasFittingPoint && !region.allocationDomainExhaustive)
      minimumDurationSupport = ResourceTimeEstimateSupport::Unsupported;
    result.support = combineSupport(result.support, minimumDurationSupport);
    result.lowerBoundPicoseconds =
        std::max(result.lowerBoundPicoseconds, minimumDuration);
    const auto total = llvm::checkedAddUnsigned(totalResourceWork, minimumWork);
    if (!total)
      return invalid("resource-time screening aggregate work overflows");
    totalResourceWork = *total;
    const auto feature = llvm::checkedAddUnsigned(
        region.analyticFeatures.launchSynchronizationCost,
        region.analyticFeatures.topologyCongestionProxy);
    if (!feature)
      return invalid("resource-time screening feature score overflows");
    const auto featureWithParallelism = llvm::checkedAddUnsigned(
        *feature, region.analyticFeatures.parallelismLowerBound);
    if (!featureWithParallelism)
      return invalid("resource-time screening feature score overflows");
    const auto aggregateFeature =
        llvm::checkedAddUnsigned(featureScore, *featureWithParallelism);
    if (!aggregateFeature)
      return invalid("resource-time screening feature score overflows");
    featureScore = *aggregateFeature;
  }
  const std::uint64_t resourceBound =
      totalResourceWork / totalCapacity +
      (totalResourceWork % totalCapacity != 0);
  result.lowerBoundPicoseconds =
      std::max(result.lowerBoundPicoseconds, resourceBound);
  result.featureScore = featureScore;
  return result;
}

} // namespace

llvm::StringRef
resourceTimeEstimateSupportSpelling(ResourceTimeEstimateSupport support) {
  switch (support) {
  case ResourceTimeEstimateSupport::Exact:
    return "exact";
  case ResourceTimeEstimateSupport::Analytic:
    return "analytic";
  case ResourceTimeEstimateSupport::Calibrated:
    return "calibrated";
  case ResourceTimeEstimateSupport::OutOfDomain:
    return "out_of_domain";
  case ResourceTimeEstimateSupport::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown resource-time estimate support");
}

llvm::StringRef resourceTimeEstimateConfidenceSpelling(
    ResourceTimeEstimateConfidence confidence) {
  switch (confidence) {
  case ResourceTimeEstimateConfidence::None:
    return "none";
  case ResourceTimeEstimateConfidence::Low:
    return "low";
  case ResourceTimeEstimateConfidence::Calibrated:
    return "calibrated";
  case ResourceTimeEstimateConfidence::OutOfDomain:
    return "out_of_domain";
  }
  llvm_unreachable("unknown resource-time estimate confidence");
}

llvm::StringRef resourceTimeFrontierIncompleteReasonSpelling(
    ResourceTimeFrontierIncompleteReason reason) {
  switch (reason) {
  case ResourceTimeFrontierIncompleteReason::BudgetExhausted:
    return "budget_exhausted";
  case ResourceTimeFrontierIncompleteReason::CancelledOrTimeout:
    return "cancelled_or_timeout";
  case ResourceTimeFrontierIncompleteReason::ProofNotEstablished:
    return "proof_not_established";
  case ResourceTimeFrontierIncompleteReason::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown resource-time frontier incomplete reason");
}

int incompleteReasonPriority(ResourceTimeFrontierIncompleteReason reason) {
  switch (reason) {
  case ResourceTimeFrontierIncompleteReason::CancelledOrTimeout:
    return 4;
  case ResourceTimeFrontierIncompleteReason::BudgetExhausted:
    return 3;
  case ResourceTimeFrontierIncompleteReason::Unsupported:
    return 2;
  case ResourceTimeFrontierIncompleteReason::ProofNotEstablished:
    return 1;
  }
  llvm_unreachable("unknown resource-time incomplete reason");
}

llvm::StringRef resourceTimeFrontierInfeasibleReasonSpelling(
    ResourceTimeFrontierInfeasibleReason reason) {
  switch (reason) {
  case ResourceTimeFrontierInfeasibleReason::CompletionDependencyCycle:
    return "completion_dependency_cycle";
  case ResourceTimeFrontierInfeasibleReason::ResourceCapacity:
    return "resource_capacity";
  }
  llvm_unreachable("unknown resource-time frontier infeasible reason");
}

llvm::StringRef resourceTimeCandidateFunnelDispositionSpelling(
    ResourceTimeCandidateFunnelDisposition disposition) {
  switch (disposition) {
  case ResourceTimeCandidateFunnelDisposition::Estimated:
    return "estimated";
  case ResourceTimeCandidateFunnelDisposition::SoundGateRejected:
    return "sound_gate_rejected";
  case ResourceTimeCandidateFunnelDisposition::Incomplete:
    return "incomplete";
  }
  llvm_unreachable("unknown resource-time funnel disposition");
}

llvm::Expected<ComponentViewDigest> resourceTimeAnalyticModelSnapshotDigest() {
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeAnalyticModelDescriptor.data()),
       resourceTimeAnalyticModelDescriptor.size()},
      {});
}

llvm::Expected<ComponentViewDigest>
deriveResourceTimeProjectionCacheKey(
    const ResourceTimeInvocationKey &invocation) {
  if (invocation.sourceLineage.schemaIdentity.empty() ||
      invocation.dataflow.schemaIdentity.empty() ||
      invocation.fabric.schemaIdentity.empty() ||
      invocation.workload.schemaIdentity.empty() ||
      invocation.runtimeInput.schemaIdentity.empty() ||
      invocation.entrySymbol.empty())
    return invalid("projection cache key contains an empty semantic input");
  std::vector<std::uint8_t> bytes;
  appendString(bytes, resourceTimeProjectionMemoDescriptor);
  appendResourceTimeInvocationKey(bytes, invocation);
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeProjectionMemoDescriptor.data()),
       resourceTimeProjectionMemoDescriptor.size()},
      bytes);
}

llvm::Expected<ResourceTimeDataflowProjection> projectResourceTimeDataflow(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    llvm::StringRef entrySymbol,
    std::optional<std::uint64_t> estimatedRuntimePicoseconds) {
  if (entrySymbol.empty())
    return invalid("resource-time projection requires an ABI entry symbol");
  auto reachable =
      dataflow.projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!reachable)
    return reachable.takeError();
  if (reachable->empty())
    return invalid("resource-time projection has no reachable root thread");
  llvm::sort(*reachable, rootLess);
  if (std::adjacent_find(reachable->begin(), reachable->end()) !=
      reachable->end())
    return invalid("resource-time projection has duplicate roots");
  const std::uint64_t availableAccCores =
      system.artifact().accCoreOccurrences().size();
  if (availableAccCores == 0)
    return invalid("resource-time projection has no AccCore capacity");

  std::vector<std::vector<::dataflow::RootedGraphLaunchRef>> launches(
      reachable->size());
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        const auto found = llvm::lower_bound(*reachable,
                                             launch.rootThreadLaunch, rootLess);
        if (found != reachable->end() && *found == launch.rootThreadLaunch)
          launches[static_cast<std::size_t>(found - reachable->begin())]
              .push_back(launch);
      });

  std::vector<std::uint64_t> weights(reachable->size(), 1);
  std::vector<std::uint64_t> maximumUseful(reachable->size(),
                                           availableAccCores);
  std::vector<std::uint64_t> logicalEpochCounts(reachable->size(), 0);
  std::vector<ResourceTimeRegionAnalyticFeatures> analyticFeatures(
      reachable->size());
  std::vector<ResourceTimeEstimateSupport> boundSupport(
      reachable->size(), ResourceTimeEstimateSupport::Unsupported);
  std::uint64_t totalWeight = 0;
  for (std::size_t ordinal = 0; ordinal != reachable->size(); ++ordinal) {
    std::uint64_t weight = 0;
    std::optional<std::uint64_t> pointCount;
    bool exactPoints = !launches[ordinal].empty();
    if (launches[ordinal].empty()) {
      auto logical = dataflow.projectRootThreadLogicalDomain((*reachable)[ordinal]);
      if (!logical)
        return logical.takeError();
      if (logical->coordinateRank == 0) {
        pointCount = 1;
        exactPoints = true;
      }
    }
    for (const ::dataflow::RootedGraphLaunchRef launch : launches[ordinal]) {
      auto graph = dataflow.resolve(launch);
      if (!graph)
        return graph.takeError();
      ++analyticFeatures[ordinal].graphCount;
      const std::uint64_t actors = llvm::count_if(
          dataflow.actors(),
          [&](const auto &actor) { return actor.graph == *graph; });
      analyticFeatures[ordinal].actorCount =
          llvm::checkedAddUnsigned(analyticFeatures[ordinal].actorCount,
                                   actors)
              .value_or(std::numeric_limits<std::uint64_t>::max());
      for (const auto &actor : dataflow.actors()) {
        if (actor.graph != *graph)
          continue;
        switch (actor.kind) {
        case ::dataflow::CanonicalDataflowActorKind::Compute:
          ++analyticFeatures[ordinal].computeActorCount;
          break;
        case ::dataflow::CanonicalDataflowActorKind::Control:
          ++analyticFeatures[ordinal].controlActorCount;
          break;
        case ::dataflow::CanonicalDataflowActorKind::Memory:
          ++analyticFeatures[ordinal].memoryActorCount;
          break;
        }
      }
      const auto addedWeight = llvm::checkedAddUnsigned(weight,
          std::max<std::uint64_t>(1, actors));
      if (!addedWeight)
        return invalid("resource-time region weight overflows");
      weight = *addedWeight;
      auto extents = dataflow.projectStaticDenseExtents(launch, entrySymbol);
      if (!extents)
        return extents.takeError();
      if (!*extents) {
        exactPoints = false;
        continue;
      }
      std::uint64_t points = 1;
      for (std::uint64_t extent : **extents) {
        auto product = llvm::checkedMulUnsigned(points, extent);
        if (!product)
          return invalid("resource-time logical-domain size overflows");
        points = *product;
      }
      if (pointCount && *pointCount != points)
        return invalid("one root has inconsistent static logical domains");
      pointCount = points;
    }
    weights[ordinal] = std::max<std::uint64_t>(1, weight);
    auto added = llvm::checkedAddUnsigned(totalWeight, weights[ordinal]);
    if (!added)
      return invalid("resource-time total region weight overflows");
    totalWeight = *added;
    if (exactPoints && pointCount && *pointCount != 0) {
      logicalEpochCounts[ordinal] = *pointCount;
      maximumUseful[ordinal] = std::min(*pointCount, availableAccCores);
      boundSupport[ordinal] = ResourceTimeEstimateSupport::Exact;
    }
  }

  std::vector<::dataflow::EventFamilyKey> boundaryEvents;
  boundaryEvents.reserve(reachable->size() * 2);
  for (const auto root : *reachable) {
    boundaryEvents.push_back(dataflow::rootThreadStartEventFamily(root));
    boundaryEvents.push_back(dataflow::rootThreadCompletionEventFamily(root));
  }
  auto causality =
      mapping::freezeMappingProgressModel(dataflow, boundaryEvents);
  if (!causality)
    return causality.takeError();

  ResourceTimeDataflowProjection result;
  result.acceleratedGraphCount = dataflow.graphs().size();
  result.acceleratedActorCount = dataflow.actors().size();
  result.resourceClasses.push_back(
      {fabric::fabricArtifactSchema.identity.str(),
       fabric::fabricArtifactSchema.version, system.artifact().identity()});
  result.availableResourceUnits.push_back(availableAccCores);
  result.regions.reserve(reachable->size());
  result.regionBounds.reserve(reachable->size());
  for (std::size_t ordinal = 0; ordinal != reachable->size(); ++ordinal) {
    ResourceTimeRegionFeature feature{(*reachable)[ordinal], {}, {},
                                      logicalEpochCounts[ordinal], false, {}};
    feature.allocationDomainExhaustive = true;
    feature.analyticFeatures = analyticFeatures[ordinal];
    feature.analyticFeatures.launchSynchronizationCost =
        feature.dependencies.size();
    feature.analyticFeatures.parallelismLowerBound =
        std::max<std::uint64_t>(1, logicalEpochCounts[ordinal]);
    feature.analyticFeatures.topologyCongestionProxy =
        feature.analyticFeatures.actorCount + feature.dependencies.size();
    for (std::size_t producer = 0; producer != reachable->size(); ++producer) {
      if (producer == ordinal)
        continue;
      auto completionPrecedes = mapping::mappingEventPrecedes(
          *causality,
          dataflow::rootThreadCompletionEventFamily((*reachable)[producer]),
          dataflow::rootThreadStartEventFamily((*reachable)[ordinal]));
      if (!completionPrecedes)
        return completionPrecedes.takeError();
      if (*completionPrecedes) {
        feature.dependencies.push_back(
            {(*reachable)[producer],
             pnr::ResourceTimeReadinessKind::Completion});
        continue;
      }
      auto startPrecedes = mapping::mappingEventPrecedes(
          *causality,
          dataflow::rootThreadStartEventFamily((*reachable)[producer]),
          dataflow::rootThreadStartEventFamily((*reachable)[ordinal]));
      if (!startPrecedes)
        return startPrecedes.takeError();
      if (*startPrecedes)
        feature.dependencies.push_back(
            {(*reachable)[producer],
             pnr::ResourceTimeReadinessKind::FifoToken});
    }
    llvm::sort(feature.dependencies, [](const auto &lhs, const auto &rhs) {
      return rootLess(lhs.producer, rhs.producer);
    });
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(
            estimatedRuntimePicoseconds.value_or(totalWeight)) *
        weights[ordinal];
    const std::uint64_t baseDuration = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(
               std::min<unsigned __int128>(
                   std::numeric_limits<std::uint64_t>::max(),
                   (scaled + totalWeight - 1) / totalWeight)));
    const ResourceTimeEstimateSupport estimateSupport =
        estimatedRuntimePicoseconds ? ResourceTimeEstimateSupport::Analytic
                                    : ResourceTimeEstimateSupport::Unsupported;
    for (std::uint64_t units = 1; units <= maximumUseful[ordinal]; ++units)
      feature.speedupCurve.push_back(
          {{units}, baseDuration / units + (baseDuration % units != 0),
           std::nullopt, std::nullopt, 0, 0, 0, estimateSupport});
    result.regions.push_back(std::move(feature));
    result.regionBounds.push_back({(*reachable)[ordinal],
                                   maximumUseful[ordinal],
                                   boundSupport[ordinal]});
  }
  return result;
}

std::uint64_t resourceTimeProjectionRetainedBytes(
    const ResourceTimeDataflowProjection &projection) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto add = [](std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > maximum - lhs ? maximum : lhs + rhs;
  };
  const auto product = [](std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > maximum / rhs)
      return maximum;
    return static_cast<std::uint64_t>(lhs * rhs);
  };
  std::uint64_t bytes = sizeof(ResourceTimeDataflowProjection);
  bytes = add(bytes, product(projection.resourceClasses.size(),
                             sizeof(ArtifactRootReference)));
  bytes = add(bytes, product(projection.availableResourceUnits.size(),
                             sizeof(std::uint64_t)));
  bytes = add(bytes, product(projection.regions.size(),
                             sizeof(ResourceTimeRegionFeature)));
  bytes = add(bytes, product(projection.regionBounds.size(),
                             sizeof(ResourceTimeRegionResourceBound)));
  for (const ResourceTimeRegionFeature &region : projection.regions) {
    bytes = add(bytes, product(region.dependencies.size(),
                               sizeof(ResourceTimeDependencyFeature)));
    bytes = add(bytes, product(region.speedupCurve.size(),
                               sizeof(ResourceTimeSpeedupPoint)));
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve)
      bytes = add(bytes, product(point.resourceUnits.size(),
                                 sizeof(std::uint64_t)));
  }
  return bytes;
}

llvm::Expected<ComponentViewDigest> deriveResourceTimeTransitionCacheKey(
    const pnr::ResourceTimeTransition &transition,
    const ResourceTimeTransitionCacheKeyInput &input) {
  if (llvm::Error error = pnr::validateResourceTimeTransition(transition))
    return std::move(error);
  const auto hasSchema = [](const ArtifactRootReference &reference,
                            const ArtifactSchemaDescriptor &schema) {
    return reference.schemaIdentity == schema.identity &&
           reference.schemaVersion == schema.version;
  };
  if (!hasSchema(transition.beforeMapping, mapping::mappingArtifactSchema) ||
      !hasSchema(transition.afterMapping, mapping::mappingArtifactSchema))
    return invalid("transition cache key has a non-Mapping endpoint");
  if (!hasSchema(input.parentDeployment, deployment::deploymentSchema) ||
      !hasSchema(input.childDeployment, deployment::deploymentSchema))
    return invalid("transition cache key has a non-Deployment endpoint");
  if (!hasSchema(input.constraints, mapping::mappingConstraintSetSchema))
    return invalid("transition cache key has a non-constraint root");
  if (!hasSchema(input.childTarget, fabric::fabricArtifactSchema))
    return invalid("transition cache key has a non-Fabric child target");
  if (!transition.resourceDeltaDigest ||
      !transition.configurationDeltaDigest || !transition.routeDeltaDigest)
    return invalid("transition cache key requires every derived delta");
  auto trigger = dataflow::encodeDataflowReference(transition.trigger);
  if (!trigger)
    return trigger.takeError();

  std::vector<std::uint8_t> bytes;
  appendBlob(bytes, *trigger);
  appendRoot(bytes, transition.safePoint);
  appendRoot(bytes, transition.beforeMapping);
  appendRoot(bytes, transition.afterMapping);
  appendRoot(bytes, input.parentDeployment);
  appendRoot(bytes, input.childDeployment);
  appendAllocations(bytes, transition.beforeActive);
  appendAllocations(bytes, transition.afterActive);
  appendRoots(bytes, transition.beforeLiveWork);
  appendRoots(bytes, transition.afterLiveWork);
  appendOptionalRoot(bytes, transition.tokenLiveStateCorrespondence);
  appendBlob(bytes, transition.resourceDeltaDigest->bytes());
  appendBlob(bytes, transition.configurationDeltaDigest->bytes());
  appendBlob(bytes, transition.routeDeltaDigest->bytes());
  appendRoot(bytes, input.constraints);
  appendBlob(bytes, input.algorithmIdentity.bytes());
  appendRoot(bytes, input.childTarget);
  appendBlob(bytes, input.scheduleDeltaDigest.bytes());
  appendBlob(bytes, input.hardwareDeltaDigest.bytes());
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeTransitionCacheDescriptor.data()),
       resourceTimeTransitionCacheDescriptor.size()},
      bytes);
}

llvm::Error validateResourceTimeFrontierAccounting(
    const ResourceTimeFrontierAccounting &accounting) {
  for (const auto *counter : {&accounting.sourceProjections,
                              &accounting.actions, &accounting.states,
                              &accounting.estimates, &accounting.finalists}) {
    if (counter->planned != counter->reserved)
      return invalid("planned and reserved work disagree");
    if (counter->consumed > counter->reserved ||
        counter->rejected > counter->reserved - counter->consumed ||
        counter->cancelled >
            counter->reserved - counter->consumed - counter->rejected ||
        counter->consumed + counter->rejected + counter->cancelled !=
            counter->reserved)
      return invalid("work ledger is not additively closed");
    if (counter->planned > counter->limit)
      return invalid("work ledger exceeds its limit");
  }
  auto admitted = llvm::checkedAddUnsigned(
      accounting.stateMemoMisses, accounting.stateMemoEnvelopeUpdates);
  if (!admitted || *admitted != accounting.states.consumed)
    return invalid("state memo admissions differ from consumed states");
  auto memoHits = llvm::checkedAddUnsigned(
      accounting.stateMemoEnvelopeUpdates,
      accounting.stateMemoDominatedStates);
  if (!memoHits || accounting.stateMemoHits != *memoHits)
    return invalid("state memo hit accounting is not closed");
  auto lowerBoundUpdates = llvm::checkedAddUnsigned(
      accounting.estimates.consumed, accounting.incrementalLowerBoundUpdates);
  if (!lowerBoundUpdates || *lowerBoundUpdates != accounting.states.consumed)
    return invalid("resource-time lower-bound update accounting is not closed");
  return llvm::Error::success();
}

llvm::Error validateResourceTimeMappingFunnelAccounting(
    const ResourceTimeMappingFunnelAccounting &accounting) {
  if (llvm::Error error =
          validateResourceTimeFrontierAccounting(accounting.frontierAccounting))
    return error;
  if (accounting.soundGateRejectedCandidates > accounting.generatedCandidates ||
      accounting.estimatedCandidates > accounting.generatedCandidates ||
      accounting.incompleteCandidates > accounting.generatedCandidates ||
      accounting.mappingFinalists > accounting.generatedCandidates)
    return invalid("resource-time funnel candidate counts exceed generation");
  if (accounting.screenedCandidates > accounting.generatedCandidates ||
      accounting.detailedFrontierCandidates >
          accounting.screenedCandidates ||
      accounting.successiveHalvingDeferredCandidates >
          accounting.screenedCandidates)
    return invalid("resource-time screening counts exceed their parent bound");
  auto evaluated = llvm::checkedAddUnsigned(
      accounting.estimatedCandidates, accounting.incompleteCandidates);
  if (evaluated)
    evaluated = llvm::checkedAddUnsigned(*evaluated,
                                         accounting.soundGateRejectedCandidates);
  auto promotedAndDeferred = llvm::checkedAddUnsigned(
      accounting.mappingFinalists, accounting.mappingCallsDeferredByModel);
  auto accounted = promotedAndDeferred
                      ? llvm::checkedAddUnsigned(
                            *promotedAndDeferred,
                            accounting.mappingCallsWithheldByIncomplete)
                      : std::nullopt;
  if (accounted)
    accounted = llvm::checkedAddUnsigned(*accounted,
                                         accounting.soundGateRejectedCandidates);
  if (!evaluated || !promotedAndDeferred || !accounted ||
      *accounted != *evaluated)
    return invalid("resource-time funnel promotion counts exceed evaluated "
                   "candidates");
  auto memoAttempts = llvm::checkedAddUnsigned(
      accounting.exactInvocationMemoHits, accounting.exactInvocationMemoMisses);
  if (memoAttempts)
    memoAttempts = llvm::checkedAddUnsigned(
        *memoAttempts,
        accounting.exactInvocationMemoCoalescedUncachedResults);
  if (!memoAttempts ||
      *memoAttempts != accounting.detailedFrontierCandidates)
    return invalid("resource-time exact memo attempts do not cover evaluated "
                   "detailed frontiers");
  auto detailedAndDeferred = llvm::checkedAddUnsigned(
      accounting.detailedFrontierCandidates,
      accounting.successiveHalvingDeferredCandidates);
  if (!detailedAndDeferred || *detailedAndDeferred != *evaluated)
    return invalid("resource-time successive-halving accounting is not "
                   "closed");
  if (accounting.exactInvocationMemoSingleFlightWaits <
          accounting.exactInvocationMemoCoalescedUncachedResults ||
      accounting.exactInvocationMemoSingleFlightWaits <
          accounting.exactInvocationMemoCancelledWaits)
    return invalid("resource-time exact memo wait accounting is inconsistent");
  auto projectionRequests = llvm::checkedAddUnsigned(
      accounting.dataflowProjectionCacheHits,
      accounting.dataflowProjectionCacheMisses);
  if (!projectionRequests ||
      *projectionRequests != accounting.dataflowProjectionRequests)
    return invalid("resource-time projection cache requests are not closed");
  if (accounting.dataflowProjectionCacheCapacityBypasses >
      accounting.dataflowProjectionCacheMisses)
    return invalid("resource-time projection cache bypasses exceed misses");
  if (accounting.dataflowProjectionCacheEntries >
      accounting.dataflowProjectionCacheMisses)
    return invalid("resource-time projection cache entries exceed misses");
  return llvm::Error::success();
}

llvm::Error accumulateResourceTimeWorkCounter(
    ResourceTimeWorkCounter &destination,
    const ResourceTimeWorkCounter &source) {
  const auto add = [](std::uint64_t &value, std::uint64_t increment,
                      llvm::StringRef name) -> llvm::Error {
    auto result = llvm::checkedAddUnsigned(value, increment);
    if (!result)
      return invalid("resource-time " + name + " work counter overflowed");
    value = *result;
    return llvm::Error::success();
  };
  if (llvm::Error error = add(destination.limit, source.limit, "limit"))
    return error;
  if (llvm::Error error =
          add(destination.planned, source.planned, "planned"))
    return error;
  if (llvm::Error error =
          add(destination.reserved, source.reserved, "reserved"))
    return error;
  if (llvm::Error error =
          add(destination.consumed, source.consumed, "consumed"))
    return error;
  if (llvm::Error error =
          add(destination.rejected, source.rejected, "rejected"))
    return error;
  if (llvm::Error error =
          add(destination.cancelled, source.cancelled, "cancelled"))
    return error;
  if (llvm::Error error = add(destination.elapsedNanoseconds,
                              source.elapsedNanoseconds, "elapsed"))
    return error;
  return llvm::Error::success();
}

llvm::Error accumulateResourceTimeFrontierAccounting(
    ResourceTimeFrontierAccounting &destination,
    const ResourceTimeFrontierAccounting &source) {
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.sourceProjections, source.sourceProjections))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.actions, source.actions))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.states, source.states))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.estimates, source.estimates))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.finalists, source.finalists))
    return error;
  auto add = [](std::uint64_t &value, std::uint64_t increment,
                llvm::StringRef name) -> llvm::Error {
    auto result = llvm::checkedAddUnsigned(value, increment);
    if (!result)
      return invalid("resource-time " + name + " accounting overflowed");
    value = *result;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          add(destination.stateMemoHits, source.stateMemoHits, "memo hits"))
    return error;
  if (llvm::Error error = add(destination.stateMemoMisses,
                              source.stateMemoMisses, "memo misses"))
    return error;
  if (llvm::Error error =
          add(destination.stateMemoEnvelopeUpdates,
              source.stateMemoEnvelopeUpdates, "memo envelope updates"))
    return error;
  if (llvm::Error error =
          add(destination.stateMemoDominatedStates,
              source.stateMemoDominatedStates, "memo dominated states"))
    return error;
  if (llvm::Error error = add(destination.statesPrunedByBeam,
                              source.statesPrunedByBeam, "beam pruning"))
    return error;
  if (llvm::Error error = add(destination.incrementalLowerBoundUpdates,
                              source.incrementalLowerBoundUpdates,
                              "incremental lower-bound updates"))
    return error;
  destination.maximumRetainedBytes =
      std::max(destination.maximumRetainedBytes,
               source.maximumRetainedBytes);
  return llvm::Error::success();
}

llvm::Expected<ResourceTimeMappingFunnel> selectResourceTimeMappingFinalists(
    llvm::ArrayRef<ResourceTimeMappingCandidateInput> candidates,
    const ResourceTimeFrontierPolicy &policy,
    ExecutionControlView executionControl,
    ResourceTimeFrontierSession *session) {
  if (candidates.empty() || policy.maximumMappingFinalists == 0 ||
      policy.maximumInvocationMemoEntries == 0 ||
      policy.maximumInvocationMemoBytes == 0)
    return invalid("resource-time Mapping funnel bounds must be positive");
  for (std::size_t index = 0; index != candidates.size(); ++index)
    for (std::size_t prior = 0; prior != index; ++prior) {
      if (candidates[prior].candidateIdentity ==
          candidates[index].candidateIdentity)
        return invalid("resource-time Mapping funnel has a duplicate semantic "
                       "candidate identity");
      if (candidates[prior].inputPreferenceRank ==
          candidates[index].inputPreferenceRank)
        return invalid("resource-time Mapping funnel has a duplicate input "
                       "preference rank");
    }

  const auto begin = MonotonicClock::now();
  ResourceTimeMappingFunnel result;
  result.accounting.generatedCandidates = candidates.size();
  result.evaluations.reserve(candidates.size());
  std::unique_ptr<ResourceTimeFrontierSession> localSession;
  if (!session) {
    localSession = std::make_unique<ResourceTimeFrontierSession>(
        policy.maximumInvocationMemoEntries,
        policy.maximumInvocationMemoBytes);
    session = localSession.get();
  }
  struct ScreenedCandidate final {
    std::size_t index = 0;
    ResourceTimeCandidateScreening screening;
  };
  std::vector<ScreenedCandidate> screened;
  screened.reserve(candidates.size());
  for (auto indexed : llvm::enumerate(candidates)) {
    if (executionControl.stopRequested()) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      break;
    }
    auto screening = screenCandidate(indexed.value(), policy);
    if (!screening)
      return screening.takeError();
    screened.push_back({indexed.index(), std::move(*screening)});
    ++result.accounting.screenedCandidates;
  }

  const auto screenedLess = [&](std::size_t lhs, std::size_t rhs) {
    const auto &left = screened[lhs];
    const auto &right = screened[rhs];
    const auto &leftCandidate = candidates[left.index];
    const auto &rightCandidate = candidates[right.index];
    const auto leftKey = std::tuple(
        estimateSupportRank(left.screening.support),
        left.screening.lowerBoundPicoseconds,
        left.screening.featureScore,
        leftCandidate.maximumUsefulResourceUnits,
        leftCandidate.candidateIdentity.bytes());
    const auto rightKey = std::tuple(
        estimateSupportRank(right.screening.support),
        right.screening.lowerBoundPicoseconds,
        right.screening.featureScore,
        rightCandidate.maximumUsefulResourceUnits,
        rightCandidate.candidateIdentity.bytes());
    return leftKey < rightKey;
  };
  std::vector<std::size_t> ranked(screened.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  llvm::sort(ranked, screenedLess);
  std::vector<std::size_t> promotionOrder;
  promotionOrder.reserve(ranked.size());
  const auto appendPromotion = [&](std::size_t screenedOrdinal) {
    if (screenedOrdinal >= screened.size() ||
        llvm::is_contained(promotionOrder, screenedOrdinal))
      return;
    promotionOrder.push_back(screenedOrdinal);
  };
  if (!ranked.empty())
    appendPromotion(ranked.front());
  if (!ranked.empty()) {
    const auto minimumCoverage = *std::min_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.acceleratedRegionCount,
                            left.acceleratedGraphCount,
                            left.acceleratedActorCount,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.acceleratedRegionCount,
                            right.acceleratedGraphCount,
                            right.acceleratedActorCount,
                            right.candidateIdentity.bytes());
        });
    const auto maximumCoverage = *std::max_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.acceleratedRegionCount,
                            left.acceleratedGraphCount,
                            left.acceleratedActorCount,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.acceleratedRegionCount,
                            right.acceleratedGraphCount,
                            right.acceleratedActorCount,
                            right.candidateIdentity.bytes());
        });
    appendPromotion(minimumCoverage);
    appendPromotion(maximumCoverage);
    const auto maximumConcentration = *std::max_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.maximumUsefulResourceUnits,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.maximumUsefulResourceUnits,
                            right.candidateIdentity.bytes());
        });
    appendPromotion(maximumConcentration);
    const auto canonical = *std::min_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          return candidates[screened[lhs].index].candidateIdentity.bytes() <
                 candidates[screened[rhs].index].candidateIdentity.bytes();
        });
    appendPromotion(canonical);
  }
  for (std::size_t screenedOrdinal : ranked)
    appendPromotion(screenedOrdinal);

  std::vector<std::optional<ResourceTimeCandidateFunnelEvaluation>>
      evaluations(candidates.size());
  std::uint64_t detailedWithHint = 0;
  const auto evaluateDetailed = [&](std::size_t screenedOrdinal)
      -> llvm::Expected<bool> {
    const ScreenedCandidate &screenedCandidate = screened[screenedOrdinal];
    const ResourceTimeMappingCandidateInput &candidate =
        candidates[screenedCandidate.index];
    const std::string memoKey = exactFrontierMemoKey(
        candidate.invocation, candidate.resourceClasses, candidate.regions,
        policy);
    auto lookup = session->lookupOrCompute(
        memoKey,
        [&]() {
          return exploreResourceTimeFrontier(
              candidate.invocation, candidate.resourceClasses,
              candidate.regions, policy, executionControl);
        },
        executionControl);
    if (!lookup)
      return lookup.takeError();
    result.accounting.exactInvocationMemoHits += lookup->cacheHit;
    result.accounting.exactInvocationMemoMisses += lookup->cacheMiss;
    result.accounting.exactInvocationMemoSingleFlightWaits += lookup->waited;
    result.accounting.exactInvocationMemoCoalescedUncachedResults +=
        lookup->coalescedUncachedResult;
    result.accounting.exactInvocationMemoCancelledWaits +=
        lookup->cancelledWait;
    result.accounting.exactInvocationMemoCapacityBypasses +=
        lookup->capacityBypass;
    if (lookup->cancelledWait) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      return false;
    }
    const ResourceTimeFrontierOutcome *outcome = lookup->outcome.get();
    if (!outcome)
      return invalid("resource-time frontier memo produced no outcome");
    ResourceTimeCandidateFunnelEvaluation evaluation{
        candidate.candidateIdentity,
        candidate.inputPreferenceRank,
        candidate.acceleratedRegionCount,
        candidate.acceleratedGraphCount,
        candidate.acceleratedActorCount,
        candidate.maximumUsefulResourceUnits,
        ResourceTimeCandidateFunnelDisposition::Incomplete,
        screenedCandidate.screening.lowerBoundPicoseconds,
        screenedCandidate.screening.featureScore,
        screenedCandidate.screening.support,
        confidenceForSupport(screenedCandidate.screening.support),
        true,
        std::nullopt,
        std::nullopt,
        {},
        std::nullopt,
        std::nullopt,
        {}};
    if (auto *completed =
            std::get_if<CompletedResourceTimeFrontier>(outcome)) {
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::Estimated;
      evaluation.concurrencyBounds = completed->concurrencyBounds;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = completed->accounting;
      evaluation.retainedHints = completed->finalists;
      if (!evaluation.retainedHints.empty())
        evaluation.bestHint = completed->finalists.front();
      ++result.accounting.estimatedCandidates;
    } else if (auto *incomplete =
                   std::get_if<IncompleteResourceTimeFrontier>(
                       outcome)) {
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::Incomplete;
      evaluation.incompleteReason = incomplete->reason;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = incomplete->accounting;
      evaluation.retainedHints = incomplete->retainedFinalists;
      if (!evaluation.retainedHints.empty())
        evaluation.bestHint = incomplete->retainedFinalists.front();
      ++result.accounting.incompleteCandidates;
      if (!result.incompleteReason ||
          incompleteReasonPriority(incomplete->reason) >
              incompleteReasonPriority(*result.incompleteReason))
        result.incompleteReason = incomplete->reason;
    } else {
      const auto &infeasible =
          std::get<ProvenInfeasibleResourceTimeFrontier>(*outcome);
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::SoundGateRejected;
      evaluation.infeasibleReason = infeasible.reason;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = infeasible.accounting;
      ++result.accounting.soundGateRejectedCandidates;
      ++result.accounting.mappingCallsAvoidedBySoundGate;
    }
    ++result.accounting.detailedFrontierCandidates;
    detailedWithHint += evaluation.bestHint.has_value();
    evaluations[screenedCandidate.index] = std::move(evaluation);
    if (llvm::Error error = accumulateResourceTimeFrontierAccounting(
            result.accounting.frontierAccounting,
            evaluations[screenedCandidate.index]->frontierAccounting))
      return std::move(error);
    if (result.incompleteReason ==
        ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
      return false;
    return true;
  };

  // Exact no-fit candidates are cheap necessary-condition checks and do not
  // consume a detailed-survivor slot. Every other candidate advances in the
  // deterministic screening/diversity order until enough real Mapping
  // finalists have a schedule hint. Remaining candidates stay analytic
  // estimates and are never called infeasible.
  for (std::size_t screenedOrdinal : promotionOrder) {
    if (executionControl.stopRequested()) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      break;
    }
    const ScreenedCandidate &screenedCandidate = screened[screenedOrdinal];
    if (evaluations[screenedCandidate.index])
      continue;
    if (!screenedCandidate.screening.exactCapacityFailure &&
        detailedWithHint >= policy.maximumMappingFinalists)
      continue;
    auto keepGoing = evaluateDetailed(screenedOrdinal);
    if (!keepGoing)
      return keepGoing.takeError();
    if (!*keepGoing)
      break;
  }

  if (result.incompleteReason !=
      ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
    for (const ScreenedCandidate &screenedCandidate : screened) {
      if (evaluations[screenedCandidate.index])
        continue;
      const auto &candidate = candidates[screenedCandidate.index];
      evaluations[screenedCandidate.index] =
          ResourceTimeCandidateFunnelEvaluation{
              candidate.candidateIdentity,
              candidate.inputPreferenceRank,
              candidate.acceleratedRegionCount,
              candidate.acceleratedGraphCount,
              candidate.acceleratedActorCount,
              candidate.maximumUsefulResourceUnits,
              ResourceTimeCandidateFunnelDisposition::Estimated,
              screenedCandidate.screening.lowerBoundPicoseconds,
              screenedCandidate.screening.featureScore,
              screenedCandidate.screening.support,
              confidenceForSupport(screenedCandidate.screening.support),
              false,
              std::nullopt,
              std::nullopt,
              {},
              std::nullopt,
              std::nullopt,
              {}};
      ++result.accounting.estimatedCandidates;
      ++result.accounting.successiveHalvingDeferredCandidates;
    }
  for (auto &evaluation : evaluations)
    if (evaluation)
      result.evaluations.push_back(std::move(*evaluation));

  if (result.incompleteReason ==
      ResourceTimeFrontierIncompleteReason::CancelledOrTimeout) {
    result.truncated = true;
  }

  std::vector<const ResourceTimeCandidateFunnelEvaluation *> admissible;
  std::uint64_t modelEligibleCandidates = 0;
  std::uint64_t withheldByIncomplete = 0;
  std::uint64_t deferredByScreening = 0;
  for (const auto &evaluation : result.evaluations) {
    if (evaluation.disposition ==
        ResourceTimeCandidateFunnelDisposition::SoundGateRejected)
      continue;
    if (!evaluation.detailedFrontierEvaluated) {
      ++deferredByScreening;
      continue;
    }
    // A timeout is a terminal incomplete checkpoint for this invocation. A
    // budget-bounded candidate may still promote a retained hint, but no
    // candidate without an explicit hint may trigger real Mapping work.
    if (!evaluation.bestHint ||
        (evaluation.incompleteReason &&
         *evaluation.incompleteReason ==
             ResourceTimeFrontierIncompleteReason::CancelledOrTimeout))
      {
      ++withheldByIncomplete;
      continue;
    }
    ++modelEligibleCandidates;
    admissible.push_back(&evaluation);
  }
  const auto candidateLess = [](const auto *lhs, const auto *rhs) {
    if (lhs->bestHint.has_value() != rhs->bestHint.has_value())
      return lhs->bestHint.has_value();
    if (lhs->bestHint) {
      if (hintLess(*lhs->bestHint, *rhs->bestHint))
        return true;
      if (hintLess(*rhs->bestHint, *lhs->bestHint))
        return false;
    }
    return lhs->candidateIdentity.bytes() < rhs->candidateIdentity.bytes();
  };
  llvm::sort(admissible, candidateLess);
  const std::uint64_t limit =
      std::min<std::uint64_t>(policy.maximumMappingFinalists,
                              admissible.size());
  const auto append = [&](const ResourceTimeCandidateFunnelEvaluation *value) {
    if (!value || result.preferenceOrder.size() == limit ||
        llvm::is_contained(result.preferenceOrder, value->candidateIdentity))
      return;
    result.preferenceOrder.push_back(value->candidateIdentity);
  };
  if (!admissible.empty())
    append(admissible.front());
  const ResourceTimeCandidateFunnelEvaluation *minimumConcurrency = nullptr;
  const ResourceTimeCandidateFunnelEvaluation *maximumConcurrency = nullptr;
  for (const auto *candidate : admissible) {
    if (!candidate->bestHint)
      continue;
    if (!minimumConcurrency ||
        std::tie(candidate->bestHint->peakConcurrentRegions,
                 candidate->bestHint->estimatedMakespanPicoseconds) <
            std::tie(minimumConcurrency->bestHint->peakConcurrentRegions,
                     minimumConcurrency->bestHint
                         ->estimatedMakespanPicoseconds))
      minimumConcurrency = candidate;
    if (!maximumConcurrency ||
        candidate->bestHint->peakConcurrentRegions >
            maximumConcurrency->bestHint->peakConcurrentRegions ||
        (candidate->bestHint->peakConcurrentRegions ==
             maximumConcurrency->bestHint->peakConcurrentRegions &&
         candidate->bestHint->estimatedMakespanPicoseconds <
             maximumConcurrency->bestHint->estimatedMakespanPicoseconds))
      maximumConcurrency = candidate;
  }
  append(minimumConcurrency);
  append(maximumConcurrency);
  const ResourceTimeCandidateFunnelEvaluation *minimumCoverage = nullptr;
  const ResourceTimeCandidateFunnelEvaluation *maximumCoverage = nullptr;
  for (const auto *candidate : admissible) {
    if (!minimumCoverage ||
        std::tie(candidate->acceleratedRegionCount,
                 candidate->acceleratedGraphCount,
                 candidate->acceleratedActorCount,
                 candidate->inputPreferenceRank) <
            std::tie(minimumCoverage->acceleratedRegionCount,
                     minimumCoverage->acceleratedGraphCount,
                     minimumCoverage->acceleratedActorCount,
                     minimumCoverage->inputPreferenceRank))
      minimumCoverage = candidate;
    if (!maximumCoverage ||
        candidate->acceleratedRegionCount >
            maximumCoverage->acceleratedRegionCount ||
        (candidate->acceleratedRegionCount ==
             maximumCoverage->acceleratedRegionCount &&
         candidate->acceleratedGraphCount >
             maximumCoverage->acceleratedGraphCount) ||
        (candidate->acceleratedRegionCount ==
             maximumCoverage->acceleratedRegionCount &&
         candidate->acceleratedGraphCount ==
             maximumCoverage->acceleratedGraphCount &&
         candidate->acceleratedActorCount >
             maximumCoverage->acceleratedActorCount))
      maximumCoverage = candidate;
  }
  append(minimumCoverage);
  append(maximumCoverage);
  const ResourceTimeCandidateFunnelEvaluation *maximumConcentration = nullptr;
  for (const auto *candidate : admissible)
    if (!maximumConcentration ||
        candidate->maximumUsefulResourceUnits >
            maximumConcentration->maximumUsefulResourceUnits ||
        (candidate->maximumUsefulResourceUnits ==
             maximumConcentration->maximumUsefulResourceUnits &&
         candidateLess(candidate, maximumConcentration)))
      maximumConcentration = candidate;
  append(maximumConcentration);
  if (!admissible.empty()) {
    const auto canonical = *std::min_element(
        admissible.begin(), admissible.end(), [](const auto *lhs,
                                                 const auto *rhs) {
          return lhs->candidateIdentity.bytes() <
                 rhs->candidateIdentity.bytes();
        });
    append(canonical);
  }
  for (const auto *candidate : admissible)
    append(candidate);
  // Keep the analytic order for promotion. Input preference is a stable
  // tie-break inside the model order, not an authority that can undo the
  // cheap-to-expensive ranking before a real Mapping dispatch.
  llvm::sort(result.preferenceOrder,
             [&](const ComponentViewDigest &lhs,
                 const ComponentViewDigest &rhs) {
               const auto left = llvm::find_if(
                   admissible, [&](const auto *candidate) {
                     return candidate->candidateIdentity == lhs;
                   });
               const auto right = llvm::find_if(
                   admissible, [&](const auto *candidate) {
                     return candidate->candidateIdentity == rhs;
                   });
               if (left == admissible.end() || right == admissible.end())
                 return lhs.bytes() < rhs.bytes();
               if (candidateLess(*left, *right))
                 return true;
               if (candidateLess(*right, *left))
                 return false;
               return (*left)->inputPreferenceRank <
                      (*right)->inputPreferenceRank;
             });
  result.accounting.mappingFinalists = result.preferenceOrder.size();
  result.accounting.mappingCallsDeferredByModel =
      deferredByScreening +
      modelEligibleCandidates - result.preferenceOrder.size();
  result.accounting.mappingCallsWithheldByIncomplete = withheldByIncomplete;
  const ResourceTimeFrontierSessionStatistics sessionStatistics =
      session->statistics();
  result.accounting.exactInvocationMemoEntries = sessionStatistics.entryCount;
  result.accounting.exactInvocationMemoRetainedBytes =
      sessionStatistics.retainedBytes;
  result.truncated = result.truncated ||
                     result.preferenceOrder.size() < admissible.size() ||
                     result.evaluations.size() < candidates.size() ||
                     result.accounting.successiveHalvingDeferredCandidates !=
                         0;
  result.accounting.elapsedNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          MonotonicClock::now() - begin)
          .count();
  if (llvm::Error error =
          validateResourceTimeMappingFunnelAccounting(result.accounting))
    return std::move(error);
  return result;
}

llvm::Expected<ResourceTimeFrontierOutcome> exploreResourceTimeFrontier(
    const ResourceTimeInvocationKey &invocation,
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    const ResourceTimeFrontierPolicy &policy,
    ExecutionControlView executionControl) {
  if (invocation.sourceLineage.schemaIdentity.empty() ||
      invocation.dataflow.schemaIdentity.empty() ||
      invocation.fabric.schemaIdentity.empty() ||
      invocation.workload.schemaIdentity.empty() ||
      invocation.runtimeInput.schemaIdentity.empty() ||
      invocation.entrySymbol.empty())
    return invalid("invocation key contains an empty semantic input");
  for (const ResourceTimeRegionFeature &region : regions)
    if (region.region.artifact != invocation.dataflow.artifact)
      return invalid("resource-time region has a foreign Dataflow owner");
  if (policy.maximumStatesGenerated == 0 ||
      policy.maximumActionsGenerated == 0 ||
      policy.maximumStateCacheEntries == 0 ||
      policy.maximumRetainedBytes == 0 || policy.beamWidth == 0 ||
      policy.maximumFinalists == 0 || policy.maximumMappingFinalists == 0)
    return invalid("frontier policy limits must be positive");

  ResourceTimeFrontierAccounting accounting;
  accounting.sourceProjections.limit = 1;
  accounting.actions.limit = policy.maximumActionsGenerated;
  accounting.states.limit = policy.maximumStatesGenerated;
  accounting.estimates.limit = policy.maximumStatesGenerated;
  accounting.finalists.limit = policy.maximumFinalists;
  ++accounting.sourceProjections.planned;
  ++accounting.sourceProjections.reserved;
  std::optional<ResourceTimeFrontierInfeasibleReason> infeasible;
  std::optional<ResourceTimeFrontierIncompleteReason> unsupported;
  llvm::Expected<FrozenInput> frozen = [&]() {
    WorkTimer timer(accounting.sourceProjections);
    return freezeInput(resourceClasses, regions, policy, infeasible,
                       unsupported);
  }();
  if (!frozen)
    return frozen.takeError();
  ++accounting.sourceProjections.consumed;
  if (infeasible) {
    if (llvm::Error error = validateResourceTimeFrontierAccounting(accounting))
      return std::move(error);
    return ResourceTimeFrontierOutcome{ProvenInfeasibleResourceTimeFrontier{
        invocation, *infeasible, std::move(accounting)}};
  }
  if (unsupported) {
    if (llvm::Error error = validateResourceTimeFrontierAccounting(accounting))
      return std::move(error);
    return ResourceTimeFrontierOutcome{IncompleteResourceTimeFrontier{
        invocation, *unsupported, {}, std::move(accounting)}};
  }

  SearchState initial;
  initial.started.assign(regions.size(), false);
  initial.completed.assign(regions.size(), false);
  initial.dependencySatisfied.assign(frozen->dependencies.size(), false);
  initial.satisfiedDependencyCount.assign(regions.size(), 0);
  initial.usedResources.assign(resourceClasses.size(), 0);
  for (std::size_t region = 0; region != regions.size(); ++region) {
    if (frozen->incomingDependencies[region].empty())
      initial.ready.push_back(region);
    initial.minimumRemainingResourceWork = llvm::checkedAddUnsigned(
        initial.minimumRemainingResourceWork,
        frozen->minimumResourceWork[region])
                                             .value_or(
                                                 std::numeric_limits<
                                                     std::uint64_t>::max());
  }
  initial.snapshots.push_back(makeSnapshot(*frozen, initial));

  std::map<std::vector<std::uint64_t>, StateMemoEnvelope> memo;
  std::uint64_t retained = 0;
  const auto admitState = [&](SearchState state,
                              std::vector<SearchState> &destination) {
    if (accounting.states.planned == accounting.states.limit)
      return false;
    ++accounting.states.planned;
    ++accounting.states.reserved;
    if (executionControl.stopRequested()) {
      ++accounting.states.cancelled;
      return false;
    }
    std::vector<std::uint64_t> key = stateMemoKey(state);
    const auto existing = memo.find(key);
    bool envelopeUpdate = false;
    if (existing != memo.end()) {
      ++accounting.stateMemoHits;
      StateMemoEnvelope &envelope = existing->second;
      envelopeUpdate =
          state.lowerBound < envelope.minimumLowerBound ||
          state.peakConcurrentRegions <
              envelope.minimumPeakConcurrentRegions ||
          state.peakConcurrentRegions >
              envelope.maximumPeakConcurrentRegions ||
          state.totalAllocatedResourceTime <
              envelope.minimumAllocatedResourceTime ||
          estimateSupportRank(state.support) < envelope.bestSupportRank;
      if (!envelopeUpdate) {
        ++accounting.stateMemoDominatedStates;
        ++accounting.states.rejected;
        return true;
      }
    }
    const std::uint64_t bytes = retainedBytes(state, key);
    if ((existing == memo.end() &&
         memo.size() == policy.maximumStateCacheEntries) ||
        bytes > policy.maximumRetainedBytes -
                    std::min(policy.maximumRetainedBytes, retained)) {
      if (existing != memo.end())
        ++accounting.stateMemoDominatedStates;
      ++accounting.states.rejected;
      return false;
    }
    if (existing == memo.end()) {
      memo.emplace(std::move(key),
                   StateMemoEnvelope{state.lowerBound,
                                     state.peakConcurrentRegions,
                                     state.peakConcurrentRegions,
                                     state.totalAllocatedResourceTime,
                                     estimateSupportRank(state.support)});
      ++accounting.stateMemoMisses;
    } else {
      StateMemoEnvelope &envelope = existing->second;
      envelope.minimumLowerBound =
          std::min(envelope.minimumLowerBound, state.lowerBound);
      envelope.minimumPeakConcurrentRegions =
          std::min(envelope.minimumPeakConcurrentRegions,
                   state.peakConcurrentRegions);
      envelope.maximumPeakConcurrentRegions =
          std::max(envelope.maximumPeakConcurrentRegions,
                   state.peakConcurrentRegions);
      envelope.minimumAllocatedResourceTime =
          std::min(envelope.minimumAllocatedResourceTime,
                   state.totalAllocatedResourceTime);
      envelope.bestSupportRank =
          std::min(envelope.bestSupportRank,
                   estimateSupportRank(state.support));
      ++accounting.stateMemoEnvelopeUpdates;
    }
    retained += bytes;
    accounting.maximumRetainedBytes =
        std::max(accounting.maximumRetainedBytes, retained);
    ++accounting.states.consumed;
    if (!state.lowerBoundInitialized) {
      ++accounting.estimates.planned;
      ++accounting.estimates.reserved;
      WorkTimer timer(accounting.estimates);
      state.lowerBound = optimisticLowerBound(
          *frozen, state, policy.availableResourceUnits);
      state.lowerBoundInitialized = true;
      ++accounting.estimates.consumed;
    } else if (!state.actions.empty())
      ++accounting.incrementalLowerBoundUpdates;
    if (!state.snapshots.empty())
      state.snapshots.back().optimisticMakespanLowerBoundPicoseconds =
          state.lowerBound;
    destination.push_back(std::move(state));
    return true;
  };

  std::vector<SearchState> frontier;
  if (!admitState(std::move(initial), frontier)) {
    const bool stopped = executionControl.stopRequested();
    settleUnconsumed(accounting.actions, stopped);
    settleUnconsumed(accounting.states, stopped);
    settleUnconsumed(accounting.estimates, stopped);
    settleUnconsumed(accounting.finalists, stopped);
    if (llvm::Error error = validateResourceTimeFrontierAccounting(accounting))
      return std::move(error);
    return ResourceTimeFrontierOutcome{IncompleteResourceTimeFrontier{
        invocation,
        stopped ? ResourceTimeFrontierIncompleteReason::CancelledOrTimeout
                : ResourceTimeFrontierIncompleteReason::BudgetExhausted,
        {}, std::move(accounting)}};
  }

  std::vector<ResourceTimeScheduleHint> terminal;
  bool budgetExhausted = false;
  bool cancelled = false;
  while (!frontier.empty()) {
    std::vector<SearchState> next;
    for (const SearchState &state : frontier) {
      if (executionControl.stopRequested()) {
        cancelled = true;
        break;
      }
      if (llvm::all_of(state.completed, [](bool value) { return value; })) {
        terminal.push_back(makeHint(state));
        continue;
      }

      bool generatedAction = false;
      for (std::size_t region : state.ready) {
        const ResourceTimeRegionFeature &feature = frozen->regions[region];
        for (auto indexedPoint : llvm::enumerate(feature.speedupCurve)) {
          const ResourceTimeSpeedupPoint &point = indexedPoint.value();
          if (!fits(state.usedResources, point.resourceUnits,
                    policy.availableResourceUnits))
            continue;
          if (accounting.actions.planned == accounting.actions.limit) {
            budgetExhausted = true;
            break;
          }
          ++accounting.actions.planned;
          ++accounting.actions.reserved;
          SearchState child = state;
          child.ready.erase(
              llvm::find(child.ready, region));
          child.started[region] = true;
          for (std::size_t resource = 0;
               resource != child.usedResources.size(); ++resource)
            child.usedResources[resource] += point.resourceUnits[resource];
          const std::uint64_t duration = pointDuration(point);
          const auto completion = llvm::checkedAddUnsigned(state.time, duration);
          if (!completion)
            return invalid("schedule completion time overflows");
          std::optional<std::uint64_t> tokenTime;
          if (point.firstTokenLatencyPicoseconds) {
            const auto configurationAndState = llvm::checkedAddUnsigned(
                point.configurationTimePicoseconds,
                point.liveStateMigrationTimePicoseconds);
            if (!configurationAndState)
              return invalid("first-token prefix time overflows");
            const auto prefix = llvm::checkedAddUnsigned(
                *configurationAndState, point.hostTransferTimePicoseconds);
            if (!prefix)
              return invalid("first-token prefix time overflows");
            const auto latency = llvm::checkedAddUnsigned(
                *prefix, *point.firstTokenLatencyPicoseconds);
            if (!latency)
              return invalid("first-token time overflows");
            tokenTime = llvm::checkedAddUnsigned(state.time, *latency);
            if (!tokenTime)
              return invalid("first-token event time overflows");
          }
          child.active.push_back(
              {region, indexedPoint.index(), *completion, tokenTime, false});
          llvm::sort(child.active, [](const auto &lhs, const auto &rhs) {
            return lhs.region < rhs.region;
          });
          child.peakConcurrentRegions = std::max<std::uint64_t>(
              child.peakConcurrentRegions, child.active.size());
          const auto resourceTime = llvm::checkedMulUnsigned(
              duration, allocationMagnitude(point.resourceUnits));
          if (!resourceTime)
            return invalid("allocated resource-time overflows");
          const auto total = llvm::checkedAddUnsigned(
              child.totalAllocatedResourceTime, *resourceTime);
          if (!total)
            return invalid("total allocated resource-time overflows");
          child.totalAllocatedResourceTime = *total;
          child.support = combineSupport(child.support, point.support);
          ResourceTimeActionDelta delta;
          delta.kind = ResourceTimeActionKind::AdmitRegion;
          delta.admittedRegion = feature.region;
          delta.speedupPointOrdinal = indexedPoint.index();
          delta.beforeTimePicoseconds = state.time;
          delta.afterTimePicoseconds = state.time;
          child.actions.push_back(std::move(delta));
          const std::array<std::size_t, 1> changedRegions = {region};
          child.lowerBound = incrementalLowerBound(
              *frozen, state, child, policy.availableResourceUnits,
              changedRegions);
          child.lowerBoundInitialized = true;
          child.snapshots.push_back(makeSnapshot(*frozen, child));
          ++accounting.actions.consumed;
          generatedAction = true;
          if (!admitState(std::move(child), next)) {
            cancelled = executionControl.stopRequested();
            budgetExhausted = !cancelled;
          }
        }
        if (budgetExhausted)
          break;
      }
      if (budgetExhausted)
        break;

      if (!state.active.empty()) {
        if (accounting.actions.planned == accounting.actions.limit) {
          budgetExhausted = true;
          break;
        }
        ++accounting.actions.planned;
        ++accounting.actions.reserved;
        SearchState child = state;
        std::uint64_t eventTime = std::numeric_limits<std::uint64_t>::max();
        for (const ActiveRegion &active : child.active) {
          eventTime = std::min(eventTime, active.completionTime);
          if (active.tokenTime && !active.tokenPublished)
            eventTime = std::min(eventTime, *active.tokenTime);
        }
        if (eventTime <= state.time)
          return invalid("resource-time event did not advance time");
        child.time = eventTime;
        ResourceTimeActionDelta delta;
        delta.kind = ResourceTimeActionKind::AdvanceEvent;
        delta.beforeTimePicoseconds = state.time;
        delta.afterTimePicoseconds = eventTime;
        std::vector<std::size_t> changedEdges;
        for (ActiveRegion &active : child.active) {
          if (active.tokenTime && !active.tokenPublished &&
              *active.tokenTime == eventTime) {
            active.tokenPublished = true;
            delta.tokenReadyProducers.push_back(
                frozen->regions[active.region].region);
            for (std::size_t edge : frozen->outgoingDependencies[active.region])
              if (frozen->dependencies[edge].readiness ==
                  pnr::ResourceTimeReadinessKind::FifoToken)
                changedEdges.push_back(edge);
          }
          if (active.completionTime == eventTime) {
            child.completed[active.region] = true;
            delta.completedRegions.push_back(
                frozen->regions[active.region].region);
            for (std::size_t edge : frozen->outgoingDependencies[active.region])
              if (frozen->dependencies[edge].readiness ==
                  pnr::ResourceTimeReadinessKind::Completion)
                changedEdges.push_back(edge);
            const ResourceTimeSpeedupPoint &point =
                frozen->regions[active.region].speedupCurve[active.point];
            const std::uint64_t completedWork =
                frozen->minimumResourceWork[active.region];
            if (child.minimumRemainingResourceWork !=
                std::numeric_limits<std::uint64_t>::max()) {
              if (completedWork > child.minimumRemainingResourceWork)
                return invalid("remaining resource-work underflowed");
              child.minimumRemainingResourceWork -= completedWork;
            }
            for (std::size_t resource = 0;
                 resource != child.usedResources.size(); ++resource)
              child.usedResources[resource] -= point.resourceUnits[resource];
          }
        }
        child.active.erase(
            std::remove_if(child.active.begin(), child.active.end(),
                           [&](const ActiveRegion &active) {
                             return active.completionTime == eventTime;
                           }),
            child.active.end());
        sortOrdinals(changedEdges);
        const std::vector<std::size_t> newlyReadyRegions =
            newlyReady(*frozen, child, changedEdges);
        for (std::size_t region : newlyReadyRegions)
          delta.newlyReadyRegions.push_back(frozen->regions[region].region);
        llvm::sort(delta.completedRegions, rootLess);
        llvm::sort(delta.tokenReadyProducers, rootLess);
        llvm::sort(delta.newlyReadyRegions, rootLess);
        child.actions.push_back(std::move(delta));
        child.lowerBound = incrementalLowerBound(
            *frozen, state, child, policy.availableResourceUnits,
            newlyReadyRegions);
        child.lowerBoundInitialized = true;
        child.snapshots.push_back(makeSnapshot(*frozen, child));
        ++accounting.actions.consumed;
        generatedAction = true;
        if (!admitState(std::move(child), next)) {
          cancelled = executionControl.stopRequested();
          budgetExhausted = !cancelled;
        }
      }
      if (!generatedAction && !llvm::all_of(
                                  state.completed,
                                  [](bool value) { return value; }))
        budgetExhausted = true;
      if (budgetExhausted)
        break;
    }
    if (cancelled || budgetExhausted)
      break;
    llvm::sort(next, stateLess);
    if (next.size() > policy.beamWidth) {
      accounting.statesPrunedByBeam += next.size() - policy.beamWidth;
      next.resize(static_cast<std::size_t>(policy.beamWidth));
    }
    frontier = std::move(next);
  }

  std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds;
  if (!terminal.empty() && !budgetExhausted && !cancelled &&
      accounting.statesPrunedByBeam == 0) {
    std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum = 0;
    for (const ResourceTimeScheduleHint &hint : terminal) {
      minimum = std::min(minimum, hint.peakConcurrentRegions);
      maximum = std::max(maximum, hint.peakConcurrentRegions);
    }
    concurrencyBounds = ResourceTimeConcurrencyBounds{
        minimum, maximum, ResourceTimeEstimateSupport::Exact};
  }
  std::vector<ResourceTimeScheduleHint> finalists =
      selectFinalists(std::move(terminal), policy.maximumFinalists);
  accounting.finalists.planned = finalists.size();
  accounting.finalists.reserved = finalists.size();
  accounting.finalists.consumed = finalists.size();
  settleUnconsumed(accounting.actions, cancelled);
  settleUnconsumed(accounting.states, cancelled);
  settleUnconsumed(accounting.estimates, cancelled);
  settleUnconsumed(accounting.finalists, cancelled);
  if (llvm::Error error = validateResourceTimeFrontierAccounting(accounting))
    return std::move(error);
  if (cancelled)
    return ResourceTimeFrontierOutcome{IncompleteResourceTimeFrontier{
        invocation, ResourceTimeFrontierIncompleteReason::CancelledOrTimeout,
        std::move(finalists), std::move(accounting)}};
  if (budgetExhausted || finalists.empty())
    return ResourceTimeFrontierOutcome{IncompleteResourceTimeFrontier{
        invocation,
        budgetExhausted
            ? ResourceTimeFrontierIncompleteReason::BudgetExhausted
            : ResourceTimeFrontierIncompleteReason::ProofNotEstablished,
        std::move(finalists), std::move(accounting)}};
  return ResourceTimeFrontierOutcome{CompletedResourceTimeFrontier{
      invocation, std::move(finalists),
      accounting.statesPrunedByBeam == 0, std::move(concurrencyBounds),
      std::move(accounting)}};
}

} // namespace loom::dse
