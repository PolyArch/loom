#include "SpatialTagColoring.h"

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

constexpr std::uint64_t exactColoringWorkLimit = UINT64_C(1048576);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial tag coloring problem: " + message);
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  const llvm::APInt left = lhs.zext(width);
  const llvm::APInt right = rhs.zext(width);
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

llvm::APInt canonicalUnsigned(const llvm::APInt &value) {
  return value.zextOrTrunc(std::max(1u, value.getActiveBits()));
}

bool unsignedLess(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return compareUnsigned(lhs, rhs) < 0;
}

void normalizeValues(std::vector<llvm::APInt> &values) {
  for (llvm::APInt &value : values)
    value = canonicalUnsigned(value);
  llvm::sort(values, unsignedLess);
  values.erase(std::unique(values.begin(), values.end(),
                           [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                             return compareUnsigned(lhs, rhs) == 0;
                           }),
               values.end());
}

llvm::Expected<llvm::APInt> nextUnsigned(const llvm::APInt &value) {
  if (value.getBitWidth() == std::numeric_limits<unsigned>::max())
    return invalid("Physical Tag candidate width overflows");
  llvm::APInt next =
      value.isAllOnes() ? value.zext(value.getBitWidth() + 1) : value;
  ++next;
  return canonicalUnsigned(next);
}

class DisjointSet final {
public:
  explicit DisjointSet(std::size_t size) : parent_(size) {
    std::iota(parent_.begin(), parent_.end(), PnrIndex{0});
  }

  PnrIndex find(PnrIndex value) {
    while (parent_[value] != value) {
      parent_[value] = parent_[parent_[value]];
      value = parent_[value];
    }
    return value;
  }

  void unite(PnrIndex lhs, PnrIndex rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parent_[rhs] = lhs;
  }

private:
  std::vector<PnrIndex> parent_;
};

enum class ExactColoringResult : std::uint8_t {
  Solved,
  Unsolvable,
  WorkLimit,
};

class ColoringState final {
public:
  static llvm::Expected<ColoringState>
  create(const SpatialTagColoringProblemView &problem,
         llvm::ArrayRef<SpatialTagColoringVertexIdentity> identities,
         const SpatialTagColoringCache *previous) {
    const std::size_t vertexCount = problem.vertices.size();
    if (vertexCount > getPnrIndexMax())
      return invalid("vertex inventory exceeds PnrIndex");
    if ((previous || !identities.empty()) && identities.size() != vertexCount)
      return invalid("coloring-cache identities do not match the vertices");
    if (problem.vertexDomainOffsets.size() != vertexCount + 1 ||
        problem.vertexIntervalOffsets.size() != vertexCount + 1 ||
        problem.vertexDomainOffsets.empty() ||
        problem.vertexIntervalOffsets.empty() ||
        problem.vertexDomainOffsets.front() != 0 ||
        problem.vertexIntervalOffsets.front() != 0 ||
        problem.vertexDomainOffsets.back() != problem.vertexDomains.size() ||
        problem.vertexIntervalOffsets.back() != problem.intervals.size())
      return invalid("vertex CSR dimensions are inconsistent");
    for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
      if (problem.vertexDomainOffsets[vertex] >
              problem.vertexDomainOffsets[vertex + 1] ||
          problem.vertexDomainOffsets[vertex + 1] >
              problem.vertexDomains.size() ||
          problem.vertexIntervalOffsets[vertex] >
              problem.vertexIntervalOffsets[vertex + 1] ||
          problem.vertexIntervalOffsets[vertex + 1] > problem.intervals.size())
        return invalid("vertex CSR offsets are not canonical");

    ColoringState state(problem, identities);
    state.conflictVertices_.resize(vertexCount);
    state.pressure_.assign(vertexCount, 0);
    DisjointSet components(vertexCount);
    for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex) {
      if (problem.vertices[vertex].tagWidthBits == 0)
        return invalid("vertex has a zero-width Physical Tag");
      const auto domains = state.domains(vertex);
      PnrIndex previous = 0;
      bool first = true;
      for (PnrIndex domain : domains) {
        if (domain >= problem.domainCount || (!first && domain <= previous))
          return invalid("vertex domains are not canonical and in range");
        first = false;
        previous = domain;
        auto &members = state.domainVertices_[domain];
        if (!members.empty()) {
          if (problem.vertices[members.front()].tagWidthBits !=
              problem.vertices[vertex].tagWidthBits)
            return invalid("one match domain contains different tag widths");
        }
        members.push_back(vertex);
      }

      const auto intervals = state.intervals(vertex);
      if (!problem.vertices[vertex].restricted && !intervals.empty())
        return invalid("unrestricted vertex carries restriction intervals");
      if (problem.vertices[vertex].restricted && intervals.empty())
        continue;
      for (std::size_t ordinal = 0; ordinal < intervals.size(); ++ordinal) {
        const auto &interval = intervals[ordinal];
        if (compareUnsigned(interval.lower, interval.upper) >= 0)
          return invalid("tag restriction interval is empty or reversed");
        if (ordinal != 0 &&
            compareUnsigned(intervals[ordinal - 1].upper, interval.lower) > 0)
          return invalid("tag restriction intervals overlap");
      }
    }
    if (problem.vertexConflictOffsets.empty()) {
      if (!problem.vertexConflicts.empty())
        return invalid("conflict incidence has no CSR offsets");
      for (const auto &members : state.domainVertices_)
        for (std::size_t lhs = 0; lhs != members.size(); ++lhs)
          for (std::size_t rhs = lhs + 1; rhs != members.size(); ++rhs) {
            state.conflictVertices_[members[lhs]].push_back(members[rhs]);
            state.conflictVertices_[members[rhs]].push_back(members[lhs]);
          }
    } else {
      if (problem.vertexConflictOffsets.size() != vertexCount + 1 ||
          problem.vertexConflictOffsets.front() != 0 ||
          problem.vertexConflictOffsets.back() !=
              problem.vertexConflicts.size())
        return invalid("conflict CSR dimensions are inconsistent");
      for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex) {
        const PnrIndex begin = problem.vertexConflictOffsets[vertex];
        const PnrIndex end = problem.vertexConflictOffsets[vertex + 1];
        if (begin > end || end > problem.vertexConflicts.size())
          return invalid("conflict CSR offsets are not canonical");
        auto &neighbors = state.conflictVertices_[vertex];
        neighbors.assign(problem.vertexConflicts.begin() + begin,
                         problem.vertexConflicts.begin() + end);
        PnrIndex previous = 0;
        bool first = true;
        for (PnrIndex neighbor : neighbors) {
          if (neighbor >= vertexCount || neighbor == vertex ||
              (!first && neighbor <= previous))
            return invalid("conflict adjacency is not canonical");
          first = false;
          previous = neighbor;
        }
      }
      for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex)
        for (PnrIndex neighbor : state.conflictVertices_[vertex])
          if (!std::binary_search(state.conflictVertices_[neighbor].begin(),
                                  state.conflictVertices_[neighbor].end(),
                                  vertex))
            return invalid("conflict adjacency is not symmetric");
    }
    for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex) {
      state.pressure_[vertex] = state.conflictVertices_[vertex].size();
      for (PnrIndex neighbor : state.conflictVertices_[vertex])
        components.unite(vertex, neighbor);
    }
    std::map<PnrIndex, std::vector<PnrIndex>> componentMap;
    for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex)
      componentMap[components.find(vertex)].push_back(vertex);
    state.components_.reserve(componentMap.size());
    for (auto &[root, members] : componentMap) {
      (void)root;
      state.components_.push_back(std::move(members));
    }
    if (!identities.empty()) {
      std::vector<SpatialTagColoringVertexIdentity> uniqueIdentities(
          identities.begin(), identities.end());
      llvm::sort(uniqueIdentities);
      if (std::adjacent_find(uniqueIdentities.begin(),
                             uniqueIdentities.end()) != uniqueIdentities.end())
        return invalid("coloring-cache identities are not unique");
    }
    if (previous)
      for (const auto &component : previous->components) {
        if (component.identities.empty())
          return invalid("cached coloring component has no identity");
        const auto inserted = state.previousComponents_.emplace(
            component.identities.front(), &component);
        if (!inserted.second)
          return invalid("cached coloring components have duplicate keys");
      }
    return state;
  }

  llvm::Expected<SpatialTagColoringResult> color() {
    if (!identities_.empty())
      result_.cache.components.reserve(components_.size());
    for (const auto &component : components_) {
      std::optional<SpatialTagColoringComponentCache> componentCache;
      if (!identities_.empty()) {
        auto built = buildComponentCache(component);
        if (!built)
          return built.takeError();
        componentCache = std::move(*built);
        auto reused = reuseComponent(component, *componentCache);
        if (!reused)
          return reused.takeError();
        if (*reused) {
          result_.cache.components.push_back(std::move(*componentCache));
          continue;
        }
        componentCache->exactWorkBefore = exactWork_;
        result_.recomputedIdentities.insert(result_.recomputedIdentities.end(),
                                            componentCache->identities.begin(),
                                            componentCache->identities.end());
      }
      const std::uint64_t unassignedBefore = result_.unassignedCount;
      const std::uint64_t conflictsBefore = result_.conflictCount;
      bool solvedExactly = false;
      if (component.size() <= spatialTagExactColoringVertexLimit) {
        std::vector<std::vector<llvm::APInt>> candidates;
        candidates.reserve(component.size());
        for (PnrIndex vertex : component) {
          auto values = allowedPrefix(vertex, component.size());
          if (!values)
            return values.takeError();
          candidates.push_back(std::move(*values));
        }
        auto exact = exactColor(component, candidates, exactWork_);
        if (!exact)
          return exact.takeError();
        if (*exact == ExactColoringResult::Solved)
          solvedExactly = true;
      }
      if (!solvedExactly)
        if (llvm::Error error = heuristicColor(component))
          return std::move(error);
      if (componentCache) {
        componentCache->exactWorkAfter = exactWork_;
        componentCache->unassignedCount =
            result_.unassignedCount - unassignedBefore;
        componentCache->conflictCount = result_.conflictCount - conflictsBefore;
        componentCache->values.reserve(component.size());
        for (PnrIndex vertex : component)
          componentCache->values.push_back(result_.values[vertex]);
        result_.cache.components.push_back(std::move(*componentCache));
      }
    }
    return std::move(result_);
  }

private:
  explicit ColoringState(
      const SpatialTagColoringProblemView &problem,
      llvm::ArrayRef<SpatialTagColoringVertexIdentity> identities)
      : problem_(problem), result_{std::vector<std::optional<llvm::APInt>>(
                                       problem.vertices.size()),
                                   {},
                                   0,
                                   0,
                                   {}},
        colored_(problem.vertices.size(), 0),
        saturationValueCounts_(problem.vertices.size()),
        domainVertices_(problem.domainCount), identities_(identities) {}

  llvm::Expected<SpatialTagColoringComponentCache>
  buildComponentCache(llvm::ArrayRef<PnrIndex> component) const {
    SpatialTagColoringComponentCache cache;
    cache.identities.reserve(component.size());
    cache.vertices.reserve(component.size());
    cache.domainOffsets.reserve(component.size() + 1);
    cache.intervalOffsets.reserve(component.size() + 1);
    cache.conflictOffsets.reserve(component.size() + 1);
    cache.domainOffsets.push_back(0);
    cache.intervalOffsets.push_back(0);
    cache.conflictOffsets.push_back(0);
    for (PnrIndex vertex : component) {
      cache.identities.push_back(identities_[vertex]);
      cache.vertices.push_back(problem_.vertices[vertex]);
      const auto localDomains = domains(vertex);
      cache.domains.insert(cache.domains.end(), localDomains.begin(),
                           localDomains.end());
      cache.domainOffsets.push_back(
          static_cast<PnrIndex>(cache.domains.size()));
      const auto localIntervals = intervals(vertex);
      cache.intervals.insert(cache.intervals.end(), localIntervals.begin(),
                             localIntervals.end());
      cache.intervalOffsets.push_back(
          static_cast<PnrIndex>(cache.intervals.size()));
      for (PnrIndex neighbor : conflictVertices_[vertex]) {
        const auto found = llvm::lower_bound(component, neighbor);
        if (found == component.end() || *found != neighbor)
          return invalid("coloring component is not conflict-closed");
        cache.conflicts.push_back(
            static_cast<PnrIndex>(found - component.begin()));
      }
      cache.conflictOffsets.push_back(
          static_cast<PnrIndex>(cache.conflicts.size()));
    }
    return cache;
  }

  static bool sameComponentInput(const SpatialTagColoringComponentCache &lhs,
                                 const SpatialTagColoringComponentCache &rhs) {
    return lhs.identities == rhs.identities && lhs.vertices == rhs.vertices &&
           lhs.domainOffsets == rhs.domainOffsets &&
           lhs.domains == rhs.domains &&
           lhs.intervalOffsets == rhs.intervalOffsets &&
           lhs.intervals == rhs.intervals &&
           lhs.conflictOffsets == rhs.conflictOffsets &&
           lhs.conflicts == rhs.conflicts;
  }

  llvm::Expected<bool> reuseComponent(llvm::ArrayRef<PnrIndex> component,
                                      SpatialTagColoringComponentCache &cache) {
    const auto found = previousComponents_.find(cache.identities.front());
    if (found == previousComponents_.end())
      return false;
    const SpatialTagColoringComponentCache &previous = *found->second;
    if (!sameComponentInput(cache, previous) ||
        previous.exactWorkBefore != exactWork_)
      return false;
    if (previous.values.size() != component.size() ||
        previous.exactWorkAfter < previous.exactWorkBefore ||
        previous.exactWorkAfter > exactColoringWorkLimit)
      return invalid("cached coloring component has inconsistent output");

    const std::uint64_t unassignedBefore = result_.unassignedCount;
    const std::uint64_t conflictsBefore = result_.conflictCount;
    for (auto [local, vertex] : llvm::enumerate(component))
      if (llvm::Error error = assign(vertex, previous.values[local]))
        return std::move(error);
    if (result_.unassignedCount - unassignedBefore !=
            previous.unassignedCount ||
        result_.conflictCount - conflictsBefore != previous.conflictCount)
      return invalid("cached coloring component summary is inconsistent");
    exactWork_ = previous.exactWorkAfter;
    cache.values = previous.values;
    cache.exactWorkBefore = previous.exactWorkBefore;
    cache.exactWorkAfter = previous.exactWorkAfter;
    cache.unassignedCount = previous.unassignedCount;
    cache.conflictCount = previous.conflictCount;
    return true;
  }

  llvm::ArrayRef<PnrIndex> domains(PnrIndex vertex) const {
    return problem_.vertexDomains.slice(
        problem_.vertexDomainOffsets[vertex],
        problem_.vertexDomainOffsets[vertex + 1] -
            problem_.vertexDomainOffsets[vertex]);
  }

  llvm::ArrayRef<SpatialTagColoringInterval> intervals(PnrIndex vertex) const {
    return problem_.intervals.slice(problem_.vertexIntervalOffsets[vertex],
                                    problem_.vertexIntervalOffsets[vertex + 1] -
                                        problem_.vertexIntervalOffsets[vertex]);
  }

  llvm::Expected<std::vector<llvm::APInt>>
  allowedPrefix(PnrIndex vertex, std::size_t limit) const {
    std::vector<llvm::APInt> result;
    result.reserve(limit);
    const auto appendRange = [&](llvm::APInt candidate,
                                 const llvm::APInt *upper) -> llvm::Error {
      candidate = canonicalUnsigned(candidate);
      while (result.size() != limit &&
             (!upper || compareUnsigned(candidate, *upper) < 0) &&
             ::fabric::isRepresentablePhysicalTagValue(
                 problem_.vertices[vertex].tagWidthBits, candidate)) {
        result.push_back(candidate);
        auto next = nextUnsigned(candidate);
        if (!next)
          return next.takeError();
        candidate = std::move(*next);
      }
      return llvm::Error::success();
    };
    if (!problem_.vertices[vertex].restricted) {
      if (llvm::Error error = appendRange(llvm::APInt(1, 0), nullptr))
        return std::move(error);
    } else {
      for (const auto &interval : intervals(vertex)) {
        if (llvm::Error error = appendRange(interval.lower, &interval.upper))
          return std::move(error);
        if (result.size() == limit)
          break;
      }
    }
    normalizeValues(result);
    return result;
  }

  bool isFree(PnrIndex vertex, const llvm::APInt &value) const {
    return llvm::all_of(conflictVertices_[vertex], [&](PnrIndex neighbor) {
      return !colored_[neighbor] || !result_.values[neighbor] ||
             compareUnsigned(*result_.values[neighbor], value) != 0;
    });
  }

  std::uint64_t conflictCost(PnrIndex vertex, const llvm::APInt &value) const {
    return llvm::count_if(conflictVertices_[vertex], [&](PnrIndex neighbor) {
      return colored_[neighbor] && result_.values[neighbor] &&
             compareUnsigned(*result_.values[neighbor], value) == 0;
    });
  }

  llvm::Error assign(PnrIndex vertex, const std::optional<llvm::APInt> &value) {
    if (colored_[vertex])
      return invalid("vertex is assigned more than once");
    colored_[vertex] = 1;
    result_.values[vertex] = value;
    if (!value) {
      if (result_.unassignedCount == std::numeric_limits<std::uint64_t>::max())
        return invalid("unassigned count overflows u64");
      ++result_.unassignedCount;
      return llvm::Error::success();
    }
    const std::uint64_t conflicts = conflictCost(vertex, *value);
    if (conflicts >
        std::numeric_limits<std::uint64_t>::max() - result_.conflictCount)
      return invalid("tag conflict count overflows u64");
    result_.conflictCount += conflicts;
    for (PnrIndex neighbor : conflictVertices_[vertex]) {
      PnrIndex &count = saturationValueCounts_[neighbor][*value];
      if (count >= getInvalidPnrIndex() - PnrIndex{1})
        return invalid("tag saturation incidence overflows PnrIndex");
      ++count;
    }
    return llvm::Error::success();
  }

  void unassign(PnrIndex vertex) noexcept {
    assert(colored_[vertex] && result_.values[vertex]);
    const llvm::APInt value = *result_.values[vertex];
    const std::uint64_t conflicts = conflictCost(vertex, value);
    assert(conflicts <= result_.conflictCount);
    result_.conflictCount -= conflicts;
    for (PnrIndex neighbor : conflictVertices_[vertex]) {
      auto saturation = saturationValueCounts_[neighbor].find(value);
      assert(saturation != saturationValueCounts_[neighbor].end() &&
             saturation->second != 0);
      if (saturation->second == 1)
        saturationValueCounts_[neighbor].erase(saturation);
      else
        --saturation->second;
    }
    result_.values[vertex].reset();
    colored_[vertex] = 0;
  }

  std::size_t saturation(PnrIndex vertex) const {
    return saturationValueCounts_[vertex].size();
  }

  PnrIndex
  selectVertex(llvm::ArrayRef<PnrIndex> component,
               llvm::ArrayRef<std::vector<llvm::APInt>> candidates = {}) const {
    PnrIndex selected = getInvalidPnrIndex();
    std::size_t selectedSaturation = 0;
    for (PnrIndex vertex : component) {
      if (colored_[vertex])
        continue;
      const std::size_t currentSaturation = saturation(vertex);
      const std::size_t currentCandidateCount =
          candidates.empty()
              ? std::numeric_limits<std::size_t>::max()
              : candidates[static_cast<std::size_t>(
                               llvm::lower_bound(component, vertex) -
                               component.begin())]
                    .size();
      const std::size_t selectedCandidateCount =
          selected == getInvalidPnrIndex() || candidates.empty()
              ? std::numeric_limits<std::size_t>::max()
              : candidates[static_cast<std::size_t>(
                               llvm::lower_bound(component, selected) -
                               component.begin())]
                    .size();
      if (selected == getInvalidPnrIndex() ||
          currentSaturation > selectedSaturation ||
          (currentSaturation == selectedSaturation &&
           pressure_[vertex] > pressure_[selected]) ||
          (currentSaturation == selectedSaturation &&
           pressure_[vertex] == pressure_[selected] &&
           currentCandidateCount < selectedCandidateCount) ||
          (currentSaturation == selectedSaturation &&
           pressure_[vertex] == pressure_[selected] &&
           currentCandidateCount == selectedCandidateCount &&
           domains(vertex).size() > domains(selected).size()) ||
          (currentSaturation == selectedSaturation &&
           pressure_[vertex] == pressure_[selected] &&
           currentCandidateCount == selectedCandidateCount &&
           domains(vertex).size() == domains(selected).size() &&
           vertex < selected)) {
        selected = vertex;
        selectedSaturation = currentSaturation;
      }
    }
    return selected;
  }

  llvm::Expected<ExactColoringResult>
  exactColor(llvm::ArrayRef<PnrIndex> component,
             llvm::ArrayRef<std::vector<llvm::APInt>> candidates,
             std::uint64_t &work) {
    const PnrIndex vertex = selectVertex(component, candidates);
    if (vertex == getInvalidPnrIndex())
      return ExactColoringResult::Solved;
    const std::size_t local = static_cast<std::size_t>(
        llvm::lower_bound(component, vertex) - component.begin());
    for (const llvm::APInt &value : candidates[local]) {
      if (!isFree(vertex, value))
        continue;
      if (work == exactColoringWorkLimit)
        return ExactColoringResult::WorkLimit;
      ++work;
      if (llvm::Error error = assign(vertex, value))
        return std::move(error);
      auto nested = exactColor(component, candidates, work);
      if (!nested) {
        unassign(vertex);
        return nested.takeError();
      }
      if (*nested == ExactColoringResult::Solved)
        return *nested;
      unassign(vertex);
      if (*nested == ExactColoringResult::WorkLimit)
        return *nested;
    }
    return ExactColoringResult::Unsolvable;
  }

  llvm::Expected<std::optional<llvm::APInt>>
  chooseHeuristicValue(PnrIndex vertex) const {
    std::optional<llvm::APInt> best;
    std::uint64_t bestCost = std::numeric_limits<std::uint64_t>::max();
    const auto considerRange = [&](llvm::APInt candidate,
                                   const llvm::APInt *upper) -> llvm::Error {
      candidate = canonicalUnsigned(candidate);
      while ((!upper || compareUnsigned(candidate, *upper) < 0) &&
             ::fabric::isRepresentablePhysicalTagValue(
                 problem_.vertices[vertex].tagWidthBits, candidate)) {
        const std::uint64_t cost = conflictCost(vertex, candidate);
        if (cost < bestCost) {
          best = candidate;
          bestCost = cost;
          if (cost == 0)
            return llvm::Error::success();
        }
        auto next = nextUnsigned(candidate);
        if (!next)
          return next.takeError();
        candidate = std::move(*next);
      }
      return llvm::Error::success();
    };
    if (!problem_.vertices[vertex].restricted) {
      if (llvm::Error error = considerRange(llvm::APInt(1, 0), nullptr))
        return std::move(error);
    } else {
      for (const auto &interval : intervals(vertex)) {
        if (llvm::Error error = considerRange(interval.lower, &interval.upper))
          return std::move(error);
        if (bestCost == 0)
          break;
      }
    }
    return best;
  }

  llvm::Error heuristicColor(llvm::ArrayRef<PnrIndex> component) {
    while (true) {
      const PnrIndex vertex = selectVertex(component);
      if (vertex == getInvalidPnrIndex())
        return llvm::Error::success();
      auto value = chooseHeuristicValue(vertex);
      if (!value)
        return value.takeError();
      if (llvm::Error error = assign(vertex, *value))
        return error;
    }
  }

  SpatialTagColoringProblemView problem_;
  SpatialTagColoringResult result_;
  std::vector<std::uint8_t> colored_;
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> saturationValueCounts_;
  std::vector<std::vector<PnrIndex>> domainVertices_;
  std::vector<std::vector<PnrIndex>> conflictVertices_;
  std::vector<std::vector<PnrIndex>> components_;
  std::vector<std::size_t> pressure_;
  llvm::ArrayRef<SpatialTagColoringVertexIdentity> identities_;
  std::map<SpatialTagColoringVertexIdentity,
           const SpatialTagColoringComponentCache *>
      previousComponents_;
  std::uint64_t exactWork_ = 0;
};

} // namespace

llvm::Expected<SpatialTagColoringResult>
loom::pnr::detail::colorSpatialTagInterference(
    const SpatialTagColoringProblemView &problem,
    llvm::ArrayRef<SpatialTagColoringVertexIdentity> identities,
    const SpatialTagColoringCache *previous) {
  auto state = ColoringState::create(problem, identities, previous);
  if (!state)
    return state.takeError();
  return state->color();
}
