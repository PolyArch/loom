#include "DSE/ResourceTimeFrontier.h"

#include "ResourceTimeFrontierInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace loom::dse {
namespace {
using namespace detail;

using MonotonicClock = std::chrono::steady_clock;

llvm::Error invalid(const llvm::Twine &message) {
  return invalidResourceTimeFrontier(message);
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

struct StateMemoPoint final {
  std::uint64_t lowerBound = 0;
  std::uint64_t peakConcurrentRegions = 0;
  std::uint64_t totalAllocatedResourceTime = 0;
  std::uint8_t supportRank = 0;
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
    key.push_back(
        active.tokenTime.value_or(std::numeric_limits<std::uint64_t>::max()));
    key.push_back(active.tokenPublished ? 1 : 0);
  }
  key.insert(key.end(), state.usedResources.begin(), state.usedResources.end());
  return key;
}

std::uint64_t stateRetainedBytes(const SearchState &state) {
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
  bytes = add(bytes, product(state.started.size(), 3));
  bytes = add(bytes, state.dependencySatisfied.size());
  bytes = add(bytes, product(state.satisfiedDependencyCount.size(),
                             sizeof(std::uint64_t)));
  bytes = add(bytes, product(state.ready.size(), sizeof(std::size_t)));
  bytes = add(bytes, product(state.active.size(), sizeof(ActiveRegion)));
  bytes =
      add(bytes, product(state.usedResources.size(), sizeof(std::uint64_t)));
  bytes = add(bytes,
              product(state.actions.size(), sizeof(ResourceTimeActionDelta)));
  bytes = add(bytes,
              product(state.snapshots.size(), sizeof(ResourceTimeHintState)));
  for (const ResourceTimeActionDelta &action : state.actions) {
    std::size_t rootCount = 0;
    if (!sumSizes(action.completedRegions.size(),
                  action.tokenReadyProducers.size(), rootCount) ||
        !sumSizes(rootCount, action.newlyReadyRegions.size(), rootCount))
      bytes = maximum;
    else
      bytes = add(bytes,
                  product(rootCount, sizeof(::dataflow::RootThreadLaunchRef)));
  }
  for (const ResourceTimeHintState &snapshot : state.snapshots) {
    bytes = add(bytes, product(snapshot.active.size(),
                               sizeof(ResourceTimeHintAllocation)));
    std::size_t rootCount = 0;
    if (!sumSizes(snapshot.ready.size(), snapshot.completed.size(), rootCount))
      bytes = maximum;
    else
      bytes = add(bytes,
                  product(rootCount, sizeof(::dataflow::RootThreadLaunchRef)));
    for (const ResourceTimeHintAllocation &allocation : snapshot.active)
      bytes = add(bytes, product(allocation.resourceUnits.size(),
                                 sizeof(std::uint64_t)));
  }
  return bytes;
}

std::uint64_t hintRetainedBytes(const ResourceTimeScheduleHint &hint) {
  SearchState state;
  state.actions = hint.actions;
  state.snapshots = hint.states;
  return stateRetainedBytes(state);
}

std::uint64_t memoRetainedBytes(llvm::ArrayRef<std::uint64_t> key,
                                llvm::ArrayRef<StateMemoPoint> points) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto product = [&](std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > maximum / rhs)
      return maximum;
    return static_cast<std::uint64_t>(lhs * rhs);
  };
  const std::uint64_t keyBytes = product(key.size(), sizeof(std::uint64_t));
  const std::uint64_t pointBytes =
      product(points.size(), sizeof(StateMemoPoint));
  constexpr std::uint64_t fixedBytes = sizeof(std::vector<std::uint64_t>) +
                                       sizeof(std::vector<StateMemoPoint>) + 64;
  if (keyBytes > maximum - fixedBytes ||
      pointBytes > maximum - fixedBytes - keyBytes)
    return maximum;
  return fixedBytes + keyBytes + pointBytes;
}

std::uint64_t
retainedHintBytes(llvm::ArrayRef<ResourceTimeScheduleHint> hints) {
  std::uint64_t bytes = 0;
  for (const ResourceTimeScheduleHint &hint : hints) {
    const std::uint64_t row = hintRetainedBytes(hint);
    if (row > std::numeric_limits<std::uint64_t>::max() - bytes)
      return std::numeric_limits<std::uint64_t>::max();
    bytes += row;
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
    const auto active =
        llvm::find_if(state.active, [&](const ActiveRegion &row) {
          return row.region == region;
        });
    remaining[region] = active == state.active.end()
                            ? input.minimumDurations[region]
                        : active->completionTime > state.time
                            ? active->completionTime - state.time
                            : 0;
  }

  std::vector<std::uint64_t> critical(input.regions.size(), 0);
  for (std::size_t region : input.reverseTopologicalOrder) {
    std::uint64_t successor = 0;
    for (std::size_t edge : input.outgoingDependencies[region])
      successor =
          std::max(successor, critical[input.dependencies[edge].consumer]);
    critical[region] = llvm::checkedAddUnsigned(remaining[region], successor)
                           .value_or(std::numeric_limits<std::uint64_t>::max());
  }
  const std::uint64_t criticalPath =
      critical.empty() ? 0
                       : *std::max_element(critical.begin(), critical.end());

  std::uint64_t totalWork = 0;
  for (std::size_t region = 0; region != input.regions.size(); ++region) {
    if (state.completed[region])
      continue;
    const auto sum =
        llvm::checkedAddUnsigned(totalWork, input.minimumResourceWork[region]);
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
  return llvm::checkedAddUnsigned(state.time, std::max(criticalPath, workBound))
      .value_or(std::numeric_limits<std::uint64_t>::max());
}

std::uint64_t
incrementalLowerBound(const FrozenInput &input, const SearchState &parent,
                      const SearchState &state,
                      llvm::ArrayRef<std::uint64_t> capacity,
                      llvm::ArrayRef<std::size_t> changedRegions) {
  std::uint64_t result = parent.lowerBound;
  const auto includeRegion = [&](std::size_t region) {
    if (region >= input.minimumSuccessorTails.size() || state.completed[region])
      return;
    const auto tail = input.minimumSuccessorTails[region];
    const auto candidate = llvm::checkedAddUnsigned(state.time, tail);
    result = std::max(
        result, candidate.value_or(std::numeric_limits<std::uint64_t>::max()));
  };
  for (const ActiveRegion &active : state.active) {
    const std::uint64_t tail =
        input.minimumSuccessorTails[active.region] >=
                input.minimumDurations[active.region]
            ? input.minimumSuccessorTails[active.region] -
                  input.minimumDurations[active.region]
            : 0;
    const auto candidate =
        llvm::checkedAddUnsigned(active.completionTime, tail);
    result = std::max(
        result, candidate.value_or(std::numeric_limits<std::uint64_t>::max()));
  }
  for (std::size_t region : changedRegions)
    includeRegion(region);
  const std::uint64_t totalCapacity = allocationMagnitude(capacity);
  if (totalCapacity != 0) {
    const std::uint64_t workBound =
        state.minimumRemainingResourceWork / totalCapacity +
        (state.minimumRemainingResourceWork % totalCapacity != 0);
    const auto candidate = llvm::checkedAddUnsigned(state.time, workBound);
    result = std::max(
        result, candidate.value_or(std::numeric_limits<std::uint64_t>::max()));
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
    snapshot.active.push_back({input.regions[active.region].region,
                               static_cast<std::uint64_t>(active.point),
                               point.resourceUnits, active.completionTime});
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

bool temporalHintLess(const ResourceTimeScheduleHint &lhs,
                      const ResourceTimeScheduleHint &rhs) {
  return std::tie(lhs.peakConcurrentRegions, lhs.estimatedMakespanPicoseconds,
                  lhs.totalAllocatedResourceTime) <
         std::tie(rhs.peakConcurrentRegions, rhs.estimatedMakespanPicoseconds,
                  rhs.totalAllocatedResourceTime);
}

bool spatialHintLess(const ResourceTimeScheduleHint &lhs,
                     const ResourceTimeScheduleHint &rhs) {
  if (lhs.peakConcurrentRegions != rhs.peakConcurrentRegions)
    return lhs.peakConcurrentRegions > rhs.peakConcurrentRegions;
  return std::tie(lhs.totalAllocatedResourceTime,
                  lhs.estimatedMakespanPicoseconds) <
         std::tie(rhs.totalAllocatedResourceTime,
                  rhs.estimatedMakespanPicoseconds);
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

std::vector<ResourceTimeScheduleHint>
selectFinalists(std::vector<ResourceTimeScheduleHint> hints,
                std::uint64_t maximum) {
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
      const bool betterConcurrency = hints[index].peakConcurrentRegions >
                                     hints[spatial].peakConcurrentRegions;
      const bool equalConcurrency = hints[index].peakConcurrentRegions ==
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

void retainBoundedTerminalHint(std::vector<ResourceTimeScheduleHint> &retained,
                               ResourceTimeScheduleHint hint,
                               std::uint64_t maximumFinalists) {
  retained.push_back(std::move(hint));
  const std::size_t objectiveLimit = static_cast<std::size_t>(
      std::min<std::uint64_t>(maximumFinalists, retained.size()));
  std::vector<const ResourceTimeScheduleHint *> objective;
  objective.reserve(retained.size());
  for (const ResourceTimeScheduleHint &candidate : retained)
    objective.push_back(&candidate);
  llvm::sort(objective, [](const auto *lhs, const auto *rhs) {
    return hintLess(*lhs, *rhs);
  });

  std::vector<ResourceTimeScheduleHint> bounded;
  bounded.reserve(objectiveLimit + 2);
  const auto append = [&](const ResourceTimeScheduleHint *candidate) {
    if (!candidate || llvm::any_of(bounded, [&](const auto &existing) {
          return existing.actions == candidate->actions;
        }))
      return;
    bounded.push_back(*candidate);
  };
  for (std::size_t index = 0; index != objectiveLimit; ++index)
    append(objective[index]);
  append(*std::min_element(objective.begin(), objective.end(),
                           [](const auto *lhs, const auto *rhs) {
                             return temporalHintLess(*lhs, *rhs);
                           }));
  append(*std::min_element(objective.begin(), objective.end(),
                           [](const auto *lhs, const auto *rhs) {
                             return spatialHintLess(*lhs, *rhs);
                           }));
  retained = std::move(bounded);
}

bool stateLess(const SearchState &lhs, const SearchState &rhs) {
  return std::make_tuple(lhs.lowerBound, lhs.time,
                         lhs.totalAllocatedResourceTime,
                         lhs.peakConcurrentRegions, lhs.actions.size()) <
         std::make_tuple(rhs.lowerBound, rhs.time,
                         rhs.totalAllocatedResourceTime,
                         rhs.peakConcurrentRegions, rhs.actions.size());
}

llvm::Expected<FrozenInput>
freezeInput(llvm::ArrayRef<ArtifactRootReference> resourceClasses,
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
  std::map<::dataflow::RootThreadLaunchRef, std::size_t, decltype(&rootLess)>
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
          *point.firstTokenLatencyPicoseconds > point.executionTimePicoseconds)
        return invalid("first-token latency exceeds execution time");
      minimumDuration = std::min(minimumDuration, duration);
      const auto work = llvm::checkedMulUnsigned(
          duration, allocationMagnitude(point.resourceUnits));
      if (work)
        minimumWork = std::min(minimumWork, *work);
      std::vector<std::uint64_t> zero(resourceClasses.size(), 0);
      hasFittingPoint |=
          fits(zero, point.resourceUnits, policy.availableResourceUnits);
    }
    if (!hasFittingPoint) {
      if (region.allocationDomainExhaustive)
        infeasible = ResourceTimeFrontierInfeasibleReason::ResourceCapacity;
      else
        unsupported = ResourceTimeFrontierIncompleteReason::ProofNotEstablished;
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
          return indegree[edge.producer] != 0 && indegree[edge.consumer] != 0 &&
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
      const auto candidate =
          llvm::checkedAddUnsigned(input.minimumDurations[region],
                                   input.minimumSuccessorTails[consumer]);
      tail = std::max(
          tail, candidate.value_or(std::numeric_limits<std::uint64_t>::max()));
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
  accounting.estimates.limit = 1;
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
    initial.minimumRemainingResourceWork =
        llvm::checkedAddUnsigned(initial.minimumRemainingResourceWork,
                                 frozen->minimumResourceWork[region])
            .value_or(std::numeric_limits<std::uint64_t>::max());
  }
  initial.snapshots.push_back(makeSnapshot(*frozen, initial));

  std::map<std::vector<std::uint64_t>, std::vector<StateMemoPoint>> memo;
  std::uint64_t memoBytes = 0;
  std::uint64_t terminalBytes = 0;
  const auto admitState = [&](SearchState state,
                              std::vector<SearchState> &destination,
                              std::uint64_t &destinationBytes,
                              std::uint64_t otherRetainedBytes) {
    if (accounting.states.planned == accounting.states.limit)
      return false;
    ++accounting.states.planned;
    ++accounting.states.reserved;
    if (executionControl.stopRequested()) {
      ++accounting.states.cancelled;
      return false;
    }
    if (!state.lowerBoundInitialized) {
      if (accounting.estimates.planned == accounting.estimates.limit) {
        ++accounting.states.rejected;
        return false;
      }
      ++accounting.estimates.planned;
      ++accounting.estimates.reserved;
      {
        WorkTimer timer(accounting.estimates);
        state.lowerBound =
            optimisticLowerBound(*frozen, state, policy.availableResourceUnits);
      }
      state.lowerBoundInitialized = true;
      ++accounting.estimates.consumed;
    }
    if (!state.snapshots.empty())
      state.snapshots.back().optimisticMakespanLowerBoundPicoseconds =
          state.lowerBound;

    std::vector<std::uint64_t> key = stateMemoKey(state);
    auto existing = memo.find(key);
    const bool memoHit = existing != memo.end();
    StateMemoPoint point{state.lowerBound, state.peakConcurrentRegions,
                         state.totalAllocatedResourceTime,
                         estimateSupportRank(state.support)};
    std::vector<StateMemoPoint> points;
    std::uint64_t oldMemoEntryBytes = 0;
    if (memoHit) {
      ++accounting.stateMemoHits;
      oldMemoEntryBytes = memoRetainedBytes(existing->first, existing->second);
      const auto dominates = [](const StateMemoPoint &lhs,
                                const StateMemoPoint &rhs) {
        return lhs.peakConcurrentRegions == rhs.peakConcurrentRegions &&
               lhs.lowerBound <= rhs.lowerBound &&
               lhs.totalAllocatedResourceTime <=
                   rhs.totalAllocatedResourceTime &&
               lhs.supportRank <= rhs.supportRank;
      };
      if (llvm::any_of(existing->second, [&](const StateMemoPoint &candidate) {
            return dominates(candidate, point);
          })) {
        ++accounting.stateMemoDominatedStates;
        ++accounting.states.rejected;
        return true;
      }
      points = existing->second;
      points.erase(std::remove_if(points.begin(), points.end(),
                                  [&](const StateMemoPoint &candidate) {
                                    return dominates(point, candidate);
                                  }),
                   points.end());
      points.push_back(point);
    } else {
      ++accounting.stateMemoMisses;
      points.push_back(point);
    }

    const std::uint64_t newMemoEntryBytes = memoRetainedBytes(key, points);
    const std::uint64_t prospectiveMemoBytes =
        oldMemoEntryBytes > memoBytes ||
                newMemoEntryBytes > std::numeric_limits<std::uint64_t>::max() -
                                        (memoBytes - oldMemoEntryBytes)
            ? std::numeric_limits<std::uint64_t>::max()
            : memoBytes - oldMemoEntryBytes + newMemoEntryBytes;
    const std::uint64_t stateBytes = stateRetainedBytes(state);
    std::uint64_t prospectiveRetained = prospectiveMemoBytes;
    for (std::uint64_t bytes :
         {otherRetainedBytes, destinationBytes, terminalBytes, stateBytes}) {
      if (bytes >
          std::numeric_limits<std::uint64_t>::max() - prospectiveRetained) {
        prospectiveRetained = std::numeric_limits<std::uint64_t>::max();
        break;
      }
      prospectiveRetained += bytes;
    }
    if ((!memoHit && memo.size() >= policy.maximumStateCacheEntries) ||
        prospectiveRetained > policy.maximumRetainedBytes) {
      if (memoHit)
        ++accounting.stateMemoHitCapacityRejections;
      else
        ++accounting.stateMemoMissCapacityRejections;
      ++accounting.states.rejected;
      return false;
    }
    if (!memoHit) {
      memo.emplace(std::move(key), std::move(points));
    } else {
      existing->second = std::move(points);
      ++accounting.stateMemoParetoInsertions;
    }
    memoBytes = prospectiveMemoBytes;
    destinationBytes += stateBytes;
    accounting.maximumRetainedBytes =
        std::max(accounting.maximumRetainedBytes, prospectiveRetained);
    ++accounting.states.consumed;
    destination.push_back(std::move(state));
    return true;
  };

  std::vector<SearchState> frontier;
  std::uint64_t frontierBytes = 0;
  if (!admitState(std::move(initial), frontier, frontierBytes, 0)) {
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
        {},
        std::move(accounting)}};
  }

  std::vector<ResourceTimeScheduleHint> terminal;
  bool budgetExhausted = false;
  bool cancelled = false;
  while (!frontier.empty()) {
    std::vector<SearchState> next;
    std::uint64_t nextBytes = 0;
    for (const SearchState &state : frontier) {
      if (executionControl.stopRequested()) {
        cancelled = true;
        break;
      }
      if (llvm::all_of(state.completed, [](bool value) { return value; })) {
        ++accounting.terminalHintsGenerated;
        std::vector<ResourceTimeScheduleHint> retained = terminal;
        retainBoundedTerminalHint(retained, makeHint(state),
                                  policy.maximumFinalists);
        const std::uint64_t retainedBytes = retainedHintBytes(retained);
        std::uint64_t prospectiveRetained = memoBytes;
        for (std::uint64_t bytes : {frontierBytes, nextBytes, retainedBytes}) {
          if (bytes >
              std::numeric_limits<std::uint64_t>::max() - prospectiveRetained) {
            prospectiveRetained = std::numeric_limits<std::uint64_t>::max();
            break;
          }
          prospectiveRetained += bytes;
        }
        if (prospectiveRetained > policy.maximumRetainedBytes) {
          budgetExhausted = true;
          break;
        }
        terminal = std::move(retained);
        terminalBytes = retainedBytes;
        accounting.maximumRetainedBytes =
            std::max(accounting.maximumRetainedBytes, prospectiveRetained);
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
          child.ready.erase(llvm::find(child.ready, region));
          child.started[region] = true;
          for (std::size_t resource = 0; resource != child.usedResources.size();
               ++resource)
            child.usedResources[resource] += point.resourceUnits[resource];
          const std::uint64_t duration = pointDuration(point);
          const auto completion =
              llvm::checkedAddUnsigned(state.time, duration);
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
          ++accounting.incrementalLowerBoundUpdates;
          ++accounting.actions.consumed;
          generatedAction = true;
          if (!admitState(std::move(child), next, nextBytes, frontierBytes)) {
            cancelled = executionControl.stopRequested();
            budgetExhausted = !cancelled;
            break;
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
        child.lowerBound = incrementalLowerBound(*frozen, state, child,
                                                 policy.availableResourceUnits,
                                                 newlyReadyRegions);
        child.lowerBoundInitialized = true;
        child.snapshots.push_back(makeSnapshot(*frozen, child));
        ++accounting.incrementalLowerBoundUpdates;
        ++accounting.actions.consumed;
        generatedAction = true;
        if (!admitState(std::move(child), next, nextBytes, frontierBytes)) {
          cancelled = executionControl.stopRequested();
          budgetExhausted = !cancelled;
        }
      }
      if (!generatedAction &&
          !llvm::all_of(state.completed, [](bool value) { return value; }))
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
      nextBytes = 0;
      for (const SearchState &state : next) {
        const std::uint64_t bytes = stateRetainedBytes(state);
        nextBytes =
            bytes > std::numeric_limits<std::uint64_t>::max() - nextBytes
                ? std::numeric_limits<std::uint64_t>::max()
                : nextBytes + bytes;
      }
    }
    frontier = std::move(next);
    frontierBytes = nextBytes;
  }

  std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds;
  accounting.terminalHintsRetained = terminal.size();
  accounting.terminalHintsPruned =
      accounting.terminalHintsGenerated - accounting.terminalHintsRetained;
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
      invocation, std::move(finalists), accounting.statesPrunedByBeam == 0,
      std::move(concurrencyBounds), std::move(accounting)}};
}

} // namespace loom::dse
