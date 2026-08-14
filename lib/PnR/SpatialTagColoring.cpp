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
  create(const SpatialTagColoringProblemView &problem) {
    const std::size_t vertexCount = problem.vertices.size();
    if (vertexCount > getPnrIndexMax())
      return invalid("vertex inventory exceeds PnrIndex");
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

    ColoringState state(problem);
    state.domainVertices_.resize(problem.domainCount);
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
          components.unite(members.front(), vertex);
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
    for (const auto &members : state.domainVertices_)
      for (PnrIndex vertex : members) {
        const std::size_t increment = members.size() - 1;
        state.pressure_[vertex] =
            increment > std::numeric_limits<std::size_t>::max() -
                            state.pressure_[vertex]
                ? std::numeric_limits<std::size_t>::max()
                : state.pressure_[vertex] + increment;
      }
    std::map<PnrIndex, std::vector<PnrIndex>> componentMap;
    for (PnrIndex vertex = 0; vertex < vertexCount; ++vertex)
      componentMap[components.find(vertex)].push_back(vertex);
    state.components_.reserve(componentMap.size());
    for (auto &[root, members] : componentMap) {
      (void)root;
      state.components_.push_back(std::move(members));
    }
    return state;
  }

  llvm::Expected<SpatialTagColoringResult> color() {
    for (const auto &component : components_) {
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
          continue;
      }
      if (llvm::Error error = heuristicColor(component))
        return std::move(error);
    }
    return std::move(result_);
  }

private:
  explicit ColoringState(const SpatialTagColoringProblemView &problem)
      : problem_(problem), result_{std::vector<std::optional<llvm::APInt>>(
                                       problem.vertices.size()),
                                   0, 0},
        colored_(problem.vertices.size(), 0), occupancy_(problem.domainCount) {}

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
    return llvm::all_of(domains(vertex), [&](PnrIndex domain) {
      return occupancy_[domain].lookup(value) == 0;
    });
  }

  std::uint64_t conflictCost(PnrIndex vertex, const llvm::APInt &value) const {
    return llvm::count_if(domains(vertex), [&](PnrIndex domain) {
      return occupancy_[domain].lookup(value) != 0;
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
    for (PnrIndex domain : domains(vertex)) {
      PnrIndex &count = occupancy_[domain][*value];
      if (count >= getInvalidPnrIndex() - PnrIndex{1})
        return invalid("tag match-domain occupancy overflows PnrIndex");
      ++count;
    }
    return llvm::Error::success();
  }

  void unassign(PnrIndex vertex) noexcept {
    assert(colored_[vertex] && result_.values[vertex]);
    const llvm::APInt value = *result_.values[vertex];
    for (PnrIndex domain : domains(vertex)) {
      auto found = occupancy_[domain].find(value);
      assert(found != occupancy_[domain].end() && found->second != 0);
      if (found->second > 1) {
        assert(result_.conflictCount != 0);
        --result_.conflictCount;
        --found->second;
      } else {
        occupancy_[domain].erase(found);
      }
    }
    result_.values[vertex].reset();
    colored_[vertex] = 0;
  }

  std::size_t saturation(PnrIndex vertex) const {
    std::vector<llvm::APInt> values;
    for (PnrIndex domain : domains(vertex))
      for (const auto &entry : occupancy_[domain])
        values.push_back(entry.first);
    normalizeValues(values);
    return values.size();
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
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy_;
  std::vector<std::vector<PnrIndex>> domainVertices_;
  std::vector<std::vector<PnrIndex>> components_;
  std::vector<std::size_t> pressure_;
  std::uint64_t exactWork_ = 0;
};

} // namespace

llvm::Expected<SpatialTagColoringResult>
loom::pnr::detail::colorSpatialTagInterference(
    const SpatialTagColoringProblemView &problem) {
  auto state = ColoringState::create(problem);
  if (!state)
    return state.takeError();
  return state->color();
}
