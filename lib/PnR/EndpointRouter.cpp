#include "PnR/EndpointRouter.h"

#include "SpatialPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <tuple>
#include <utility>

using namespace loom::pnr;

char EndpointRouteSearchFailure::ID;

llvm::StringRef loom::pnr::stringifyEndpointRouteSearchFailureKind(
    EndpointRouteSearchFailureKind kind) {
  switch (kind) {
  case EndpointRouteSearchFailureKind::Invalid:
    return "invalid";
  case EndpointRouteSearchFailureKind::ArithmeticOverflow:
    return "arithmetic_overflow";
  case EndpointRouteSearchFailureKind::Unreachable:
    return "unreachable";
  case EndpointRouteSearchFailureKind::WorkLimit:
    return "work_limit";
  }
  llvm_unreachable("invalid endpoint route search failure kind");
}

EndpointRouteSearchFailure::EndpointRouteSearchFailure(
    EndpointRouteSearchFailureKind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void EndpointRouteSearchFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code EndpointRouteSearchFailure::convertToErrorCode() const {
  switch (kind_) {
  case EndpointRouteSearchFailureKind::Invalid:
    return std::make_error_code(std::errc::invalid_argument);
  case EndpointRouteSearchFailureKind::ArithmeticOverflow:
    return std::make_error_code(std::errc::result_out_of_range);
  case EndpointRouteSearchFailureKind::Unreachable:
    return std::make_error_code(std::errc::host_unreachable);
  case EndpointRouteSearchFailureKind::WorkLimit:
    return std::make_error_code(std::errc::operation_canceled);
  }
  llvm_unreachable("invalid endpoint route search failure kind");
}

EndpointRoutingGraphView loom::pnr::endpointRoutingGraphView(
    const FrozenEndpointRoutingTopology &topology) {
  return {static_cast<PnrIndex>(topology.endpoints().size()),
          topology.arcs(),
          topology.arcSources(),
          topology.adjacencyOffsets(),
          topology.reverseAdjacencyOffsets(),
          topology.reverseArcOrdinals(),
          topology.traversalReplicationGroups()};
}

namespace {

constexpr PnrIndex invalidIndex = std::numeric_limits<PnrIndex>::max();
constexpr std::size_t heuristicCacheByteBudget = 16 * 1024 * 1024;
constexpr std::size_t maximumHeuristicCacheEntryCount = 1024;
constexpr std::uint64_t hashOffsetBasis = UINT64_C(14695981039346656037);
constexpr std::uint64_t hashPrime = UINT64_C(1099511628211);

void hashWord(std::uint64_t &hash, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte) {
    hash ^= value & UINT64_C(0xff);
    hash *= hashPrime;
    value >>= 8;
  }
}

void saturatingIncrement(std::uint64_t &value) {
  if (value != std::numeric_limits<std::uint64_t>::max())
    ++value;
}

template <typename... Parts> std::string renderMessage(Parts &&...parts) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  (stream << ... << parts);
  return message;
}

template <typename... Parts>
llvm::Error failure(EndpointRouteSearchFailureKind kind, Parts &&...parts) {
  return llvm::make_error<EndpointRouteSearchFailure>(
      kind, renderMessage("endpoint route search: ", parts...));
}

template <typename... Parts> llvm::Error invalid(Parts &&...parts) {
  return failure(EndpointRouteSearchFailureKind::Invalid,
                 std::forward<Parts>(parts)...);
}

template <typename... Parts> llvm::Error overflow(Parts &&...parts) {
  return failure(EndpointRouteSearchFailureKind::ArithmeticOverflow,
                 std::forward<Parts>(parts)...);
}

llvm::Expected<RouteCost> addFiniteCost(RouteCost lhs, RouteCost rhs,
                                        llvm::StringRef operation) {
  if (lhs == routeCostInfinity || rhs == routeCostInfinity ||
      lhs > maxFiniteRouteCost - rhs)
    return overflow(operation, " exceeds the largest finite route cost");
  return lhs + rhs;
}

llvm::Expected<RouteCost>
arcSearchCost(const EndpointRouteSearchRequest &request, PnrIndex arc,
              bool current) {
  const RouteCost resourceCost =
      current ? request.currentArcCosts[arc] : request.lowerBoundArcCosts[arc];
  if (!request.physicalTimingEnabled)
    return resourceCost;
  auto timingCost = detail::physicalTimingDrivenTraversalCost(
      request.arcTimingDelayQuanta[arc], request.requiredTimingQuanta,
      request.timingCriticality);
  if (!timingCost) {
    llvm::consumeError(timingCost.takeError());
    return overflow("physical traversal cost exceeds the largest finite route "
                    "cost");
  }
  return addFiniteCost(resourceCost, *timingCost,
                       current ? "current physical arc cost"
                               : "lower-bound physical arc cost");
}

bool isCanonicalEndpointSet(llvm::ArrayRef<PnrIndex> endpoints) {
  return std::adjacent_find(endpoints.begin(), endpoints.end(),
                            std::greater_equal<PnrIndex>()) == endpoints.end();
}

void advanceGeneration(std::vector<std::uint64_t> &epochs,
                       std::uint64_t &generation) {
  ++generation;
  if (generation != 0)
    return;
  std::fill(epochs.begin(), epochs.end(), 0);
  generation = 1;
}

} // namespace

llvm::Error
EndpointRouteSearchScratch::prepare(EndpointRoutingGraphView graph) {
  if constexpr (sizeof(PnrIndex) >= sizeof(std::size_t)) {
    if (graph.endpointCount >
        static_cast<PnrIndex>(std::numeric_limits<std::size_t>::max() - 1))
      return invalid("endpoint count cannot form a host-sized CSR row table");
  }
  const std::size_t endpointCount =
      static_cast<std::size_t>(graph.endpointCount);
  if (static_cast<PnrIndex>(endpointCount) != graph.endpointCount)
    return invalid("endpoint count is not representable by host size_t");
  if (graph.adjacencyOffsets.size() != endpointCount + 1 ||
      graph.reverseAdjacencyOffsets.size() != endpointCount + 1)
    return invalid("forward and reverse CSR offsets must contain V + 1 rows");
  if (graph.arcSources.size() != graph.arcs.size() ||
      graph.reverseArcOrdinals.size() != graph.arcs.size())
    return invalid("arc source and reverse-arc tables must contain E rows");
  if (graph.adjacencyOffsets.empty() || graph.adjacencyOffsets.front() != 0 ||
      graph.adjacencyOffsets.back() != graph.arcs.size() ||
      graph.reverseAdjacencyOffsets.front() != 0 ||
      graph.reverseAdjacencyOffsets.back() != graph.arcs.size())
    return invalid("CSR offsets must span the exact arc table");

  for (PnrIndex endpoint = 0; endpoint < graph.endpointCount; ++endpoint) {
    const PnrIndex begin = graph.adjacencyOffsets[endpoint];
    const PnrIndex end = graph.adjacencyOffsets[endpoint + 1];
    if (begin > end || end > graph.arcs.size())
      return invalid("forward CSR offsets are not monotonic at endpoint ",
                     endpoint);
    for (PnrIndex arc = begin; arc < end; ++arc) {
      if (graph.arcSources[arc] != endpoint)
        return invalid("arc ", arc,
                       " is not owned by its forward CSR endpoint");
      if (graph.arcs[arc].target >= graph.endpointCount)
        return invalid("arc ", arc, " has an out-of-range target");
      if (graph.arcs[arc].traversal >= graph.traversalReplicationGroups.size())
        return invalid("arc ", arc, " has an out-of-range traversal");
    }

    const PnrIndex reverseBegin = graph.reverseAdjacencyOffsets[endpoint];
    const PnrIndex reverseEnd = graph.reverseAdjacencyOffsets[endpoint + 1];
    if (reverseBegin > reverseEnd || reverseEnd > graph.arcs.size())
      return invalid("reverse CSR offsets are not monotonic at endpoint ",
                     endpoint);
    PnrIndex previousArc = invalidIndex;
    for (PnrIndex offset = reverseBegin; offset < reverseEnd; ++offset) {
      const PnrIndex arc = graph.reverseArcOrdinals[offset];
      if (arc >= graph.arcs.size())
        return invalid("reverse CSR contains an out-of-range arc ordinal");
      if (graph.arcs[arc].target != endpoint)
        return invalid("reverse CSR arc ", arc,
                       " is indexed under the wrong target endpoint");
      if (previousArc != invalidIndex && previousArc >= arc)
        return invalid("reverse CSR arc ordinals are not canonical");
      previousArc = arc;
    }
  }

  graph_ = graph;
  heuristics_.assign(endpointCount, 0);
  if (endpointCount > std::numeric_limits<std::size_t>::max() / 2 ||
      graph.endpointCount > std::numeric_limits<PnrIndex>::max() / 2)
    return invalid("endpoint count cannot form the required-state product");
  const std::size_t searchStateCount = endpointCount * 2;
  distances_.assign(searchStateCount, 0);
  priorities_.assign(searchStateCount, 0);
  predecessorArcs_.assign(searchStateCount, invalidIndex);
  predecessorStates_.assign(searchStateCount, invalidIndex);
  heuristicEpochs_.assign(endpointCount, 0);
  distanceEpochs_.assign(searchStateCount, 0);
  targetEpochs_.assign(endpointCount, 0);
  sourceEpochs_.assign(endpointCount, 0);
  targetPreferenceRanks_.assign(endpointCount, 0);
  targetRequiresTraversal_.assign(endpointCount, 0);
  sourceReplicationGroups_.assign(endpointCount, getInvalidPnrIndex());
  heap_.clear();
  heap_.reserve(searchStateCount);
  heapPositions_.assign(searchStateCount, invalidIndex);
  path_.clear();
  path_.reserve(searchStateCount);
  timingLabels_.clear();
  timingLabels_.reserve(searchStateCount);
  timingStateLabels_.clear();
  timingStateLabels_.resize(searchStateCount);
  timingHeap_.clear();
  timingHeap_.reserve(searchStateCount);
  heuristicCache_.clear();
  heuristicCacheTargets_.clear();
  heuristicCacheEligibility_.clear();
  heuristicCacheDistances_.clear();
  const std::size_t traversalWordCount =
      graph.traversalReplicationGroups.size() / 64 +
      (graph.traversalReplicationGroups.size() % 64 != 0);
  const bool cacheShapeFits =
      endpointCount != 0 &&
      endpointCount <=
          heuristicCacheByteBudget / (sizeof(RouteCost) + sizeof(PnrIndex)) &&
      traversalWordCount <= heuristicCacheByteBudget / sizeof(std::uint64_t);
  if (cacheShapeFits) {
    const std::size_t maximumKeyBytes =
        endpointCount * sizeof(PnrIndex) +
        traversalWordCount * sizeof(std::uint64_t);
    const std::size_t estimatedEntryBytes =
        endpointCount * sizeof(RouteCost) + maximumKeyBytes;
    const std::size_t entryCount =
        std::min(maximumHeuristicCacheEntryCount,
                 heuristicCacheByteBudget /
                     std::max<std::size_t>(estimatedEntryBytes, 1));
    const bool targetStorageFits =
        entryCount <= std::numeric_limits<std::size_t>::max() /
                          std::max<std::size_t>(endpointCount, 1);
    const bool eligibilityStorageFits =
        entryCount <= std::numeric_limits<std::size_t>::max() /
                          std::max<std::size_t>(traversalWordCount, 1);
    if (targetStorageFits && eligibilityStorageFits) {
      heuristicCache_.resize(entryCount);
      heuristicCacheTargets_.resize(entryCount * endpointCount);
      heuristicCacheEligibility_.resize(entryCount * traversalWordCount);
      heuristicCacheDistances_.resize(entryCount * endpointCount);
    }
  }
  heuristicCacheTraversalWordCount_ = traversalWordCount;
  activeCachedHeuristics_ = nullptr;
  heuristicGeneration_ = 0;
  searchGeneration_ = 0;
  targetGeneration_ = 0;
  sourceGeneration_ = 0;
  endpointExpansionCount_ = 0;
  heuristicCacheHitCount_ = 0;
  heuristicBuildCount_ = 0;
  prepared_ = true;
  return llvm::Error::success();
}

void EndpointRouteSearchScratch::resetHeap() {
  for (PnrIndex endpoint : heap_)
    heapPositions_[endpoint] = invalidIndex;
  heap_.clear();
}

bool EndpointRouteSearchScratch::heapLess(PnrIndex lhs, PnrIndex rhs) const {
  if (heapMode_ == HeapMode::ReverseDistance) {
    if (heuristics_[lhs] != heuristics_[rhs])
      return heuristics_[lhs] < heuristics_[rhs];
    return lhs < rhs;
  }
  if (priorities_[lhs] != priorities_[rhs])
    return priorities_[lhs] < priorities_[rhs];
  const PnrIndex lhsEndpoint = searchEndpoint(lhs);
  const PnrIndex rhsEndpoint = searchEndpoint(rhs);
  if (heuristic(lhsEndpoint) != heuristic(rhsEndpoint))
    return heuristic(lhsEndpoint) < heuristic(rhsEndpoint);
  return lhs < rhs;
}

void EndpointRouteSearchScratch::heapSwap(std::size_t lhs, std::size_t rhs) {
  std::swap(heap_[lhs], heap_[rhs]);
  heapPositions_[heap_[lhs]] = static_cast<PnrIndex>(lhs);
  heapPositions_[heap_[rhs]] = static_cast<PnrIndex>(rhs);
}

void EndpointRouteSearchScratch::siftUp(std::size_t position) {
  while (position != 0) {
    const std::size_t parent = (position - 1) / 2;
    if (!heapLess(heap_[position], heap_[parent]))
      break;
    heapSwap(position, parent);
    position = parent;
  }
}

void EndpointRouteSearchScratch::siftDown(std::size_t position) {
  while (true) {
    const std::size_t left = position * 2 + 1;
    if (left >= heap_.size())
      return;
    const std::size_t right = left + 1;
    std::size_t minimum = left;
    if (right < heap_.size() && heapLess(heap_[right], heap_[left]))
      minimum = right;
    if (!heapLess(heap_[minimum], heap_[position]))
      return;
    heapSwap(position, minimum);
    position = minimum;
  }
}

void EndpointRouteSearchScratch::insertOrDecrease(PnrIndex endpoint) {
  const PnrIndex position = heapPositions_[endpoint];
  if (position != invalidIndex) {
    siftUp(static_cast<std::size_t>(position));
    return;
  }
  heapPositions_[endpoint] = static_cast<PnrIndex>(heap_.size());
  heap_.push_back(endpoint);
  siftUp(heap_.size() - 1);
}

PnrIndex EndpointRouteSearchScratch::popMinimum() {
  assert(!heap_.empty());
  const PnrIndex minimum = heap_.front();
  if (heap_.size() == 1) {
    heap_.pop_back();
    heapPositions_[minimum] = invalidIndex;
    return minimum;
  }
  heapSwap(0, heap_.size() - 1);
  heap_.pop_back();
  heapPositions_[minimum] = invalidIndex;
  siftDown(0);
  return minimum;
}

PnrIndex EndpointRouteSearchScratch::peekMinimum() const {
  assert(!heap_.empty());
  return heap_.front();
}

void EndpointRouteSearchScratch::beginHeuristicGeneration() {
  advanceGeneration(heuristicEpochs_, heuristicGeneration_);
}

void EndpointRouteSearchScratch::beginSearchGeneration() {
  advanceGeneration(distanceEpochs_, searchGeneration_);
}

void EndpointRouteSearchScratch::beginTargetGeneration() {
  advanceGeneration(targetEpochs_, targetGeneration_);
}

void EndpointRouteSearchScratch::beginSourceGeneration() {
  advanceGeneration(sourceEpochs_, sourceGeneration_);
}

RouteCost EndpointRouteSearchScratch::heuristic(PnrIndex endpoint) const {
  if (activeCachedHeuristics_)
    return activeCachedHeuristics_[endpoint];
  if (heuristicEpochs_[endpoint] != heuristicGeneration_)
    return routeCostInfinity;
  return heuristics_[endpoint];
}

RouteCost EndpointRouteSearchScratch::distance(PnrIndex state) const {
  if (distanceEpochs_[state] != searchGeneration_)
    return routeCostInfinity;
  return distances_[state];
}

PnrIndex EndpointRouteSearchScratch::searchState(PnrIndex endpoint,
                                                 bool requirementMet) const {
  assert(endpoint < graph_.endpointCount);
  return endpoint + (requirementMet ? graph_.endpointCount : 0);
}

PnrIndex EndpointRouteSearchScratch::searchEndpoint(PnrIndex state) const {
  assert(graph_.endpointCount != 0 &&
         state < graph_.endpointCount * static_cast<PnrIndex>(2));
  return state < graph_.endpointCount ? state : state - graph_.endpointCount;
}

bool EndpointRouteSearchScratch::searchRequirementMet(PnrIndex state) const {
  assert(graph_.endpointCount != 0 &&
         state < graph_.endpointCount * static_cast<PnrIndex>(2));
  return state >= graph_.endpointCount;
}

bool EndpointRouteSearchScratch::isTarget(PnrIndex endpoint) const {
  return targetEpochs_[endpoint] == targetGeneration_;
}

bool EndpointRouteSearchScratch::targetRequiresTraversal(
    PnrIndex endpoint) const {
  assert(isTarget(endpoint));
  return targetRequiresTraversal_[endpoint] != 0;
}

bool EndpointRouteSearchScratch::isSource(PnrIndex endpoint) const {
  return sourceEpochs_[endpoint] == sourceGeneration_;
}

PnrIndex
EndpointRouteSearchScratch::targetPreferenceRank(PnrIndex endpoint) const {
  assert(isTarget(endpoint));
  return targetPreferenceRanks_[endpoint];
}

bool EndpointRouteSearchScratch::arcEligible(
    PnrIndex arc, const EndpointRouteSearchRequest &request,
    bool enforceSourceReplication) const {
  if (graph_.arcs[arc].payloadCapacityBits < request.requiredPayloadWidthBits ||
      graph_.arcs[arc].tagCapacityBits < request.requiredTagWidthBits)
    return false;
  const PnrIndex traversal = graph_.arcs[arc].traversal;
  if (!request.eligibleTraversalBits.empty() &&
      (traversal / 64 >= request.eligibleTraversalBits.size() ||
       (request.eligibleTraversalBits[traversal / 64] &
        (std::uint64_t{1} << (traversal % 64))) == 0))
    return false;
  const PnrIndex source = graph_.arcSources[arc];
  if (!enforceSourceReplication || !isSource(source))
    return true;
  const PnrIndex required = sourceReplicationGroups_[source];
  return required == getInvalidPnrIndex() ||
         graph_.traversalReplicationGroups[graph_.arcs[arc].traversal] ==
             required;
}

llvm::Error EndpointRouteSearchScratch::buildHeuristic(
    const EndpointRouteSearchRequest &request) {
  activeCachedHeuristics_ = nullptr;
  saturatingIncrement(heuristicBuildCount_);
  resetHeap();
  heapMode_ = HeapMode::ReverseDistance;
  beginHeuristicGeneration();
  for (PnrIndex target : request.targetEndpoints) {
    heuristics_[target] = 0;
    heuristicEpochs_[target] = heuristicGeneration_;
    insertOrDecrease(target);
  }

  while (!heap_.empty()) {
    const PnrIndex endpoint = popMinimum();
    const RouteCost endpointCost = heuristics_[endpoint];
    const PnrIndex begin = graph_.reverseAdjacencyOffsets[endpoint];
    const PnrIndex end = graph_.reverseAdjacencyOffsets[endpoint + 1];
    for (PnrIndex offset = begin; offset < end; ++offset) {
      const PnrIndex arc = graph_.reverseArcOrdinals[offset];
      if (!arcEligible(arc, request, false))
        continue;
      const PnrIndex predecessor = graph_.arcSources[arc];
      auto arcCost = arcSearchCost(request, arc, false);
      if (!arcCost)
        return arcCost.takeError();
      auto candidate = addFiniteCost(endpointCost, *arcCost,
                                     "reverse lower-bound distance");
      if (!candidate)
        return candidate.takeError();
      if (*candidate >= heuristic(predecessor))
        continue;
      heuristics_[predecessor] = *candidate;
      heuristicEpochs_[predecessor] = heuristicGeneration_;
      insertOrDecrease(predecessor);
    }
  }
  return llvm::Error::success();
}

std::uint64_t EndpointRouteSearchScratch::heuristicCacheKeyHash(
    const EndpointRouteSearchRequest &request) const {
  assert(request.lowerBoundCostRevision);
  std::uint64_t hash = hashOffsetBasis;
  hashWord(hash, *request.lowerBoundCostRevision);
  hashWord(hash, request.requiredPayloadWidthBits);
  hashWord(hash, request.requiredTagWidthBits);
  hashWord(hash, request.targetEndpoints.size());
  for (PnrIndex endpoint : request.targetEndpoints)
    hashWord(hash, endpoint);
  hashWord(hash, request.eligibleTraversalBits.size());
  for (std::uint64_t word : request.eligibleTraversalBits)
    hashWord(hash, word);
  return hash;
}

bool EndpointRouteSearchScratch::heuristicCacheKeyEquals(
    const HeuristicCacheEntry &entry, const EndpointRouteSearchRequest &request,
    std::uint64_t keyHash, std::size_t slot) const {
  const std::size_t endpointCount =
      static_cast<std::size_t>(graph_.endpointCount);
  const auto targets =
      llvm::ArrayRef(heuristicCacheTargets_)
          .slice(slot * endpointCount, entry.targetEndpointCount);
  const auto eligibility = llvm::ArrayRef(heuristicCacheEligibility_)
                               .slice(slot * heuristicCacheTraversalWordCount_,
                                      entry.eligibleTraversalWordCount);
  return entry.populated && entry.keyHash == keyHash &&
         entry.lowerBoundCostData == request.lowerBoundArcCosts.data() &&
         entry.lowerBoundCostSize == request.lowerBoundArcCosts.size() &&
         entry.lowerBoundCostRevision == *request.lowerBoundCostRevision &&
         entry.requiredPayloadWidthBits == request.requiredPayloadWidthBits &&
         entry.requiredTagWidthBits == request.requiredTagWidthBits &&
         targets == request.targetEndpoints &&
         eligibility == request.eligibleTraversalBits;
}

bool EndpointRouteSearchScratch::loadCachedHeuristic(
    const EndpointRouteSearchRequest &request) {
  activeCachedHeuristics_ = nullptr;
  if (!request.lowerBoundCostRevision || heuristicCache_.empty())
    return false;
  const std::uint64_t keyHash = heuristicCacheKeyHash(request);
  const std::size_t slot = keyHash % heuristicCache_.size();
  HeuristicCacheEntry &entry = heuristicCache_[slot];
  if (!heuristicCacheKeyEquals(entry, request, keyHash, slot))
    return false;
  activeCachedHeuristics_ =
      heuristicCacheDistances_.data() +
      slot * static_cast<std::size_t>(graph_.endpointCount);
  saturatingIncrement(heuristicCacheHitCount_);
  return true;
}

void EndpointRouteSearchScratch::storeCachedHeuristic(
    const EndpointRouteSearchRequest &request) {
  if (!request.lowerBoundCostRevision || heuristicCache_.empty())
    return;
  const std::uint64_t keyHash = heuristicCacheKeyHash(request);
  const std::size_t slot = keyHash % heuristicCache_.size();
  HeuristicCacheEntry &entry = heuristicCache_[slot];
  entry.populated = false;
  entry.lowerBoundCostData = request.lowerBoundArcCosts.data();
  entry.lowerBoundCostSize = request.lowerBoundArcCosts.size();
  entry.lowerBoundCostRevision = *request.lowerBoundCostRevision;
  entry.keyHash = keyHash;
  entry.requiredPayloadWidthBits = request.requiredPayloadWidthBits;
  entry.requiredTagWidthBits = request.requiredTagWidthBits;
  entry.targetEndpointCount = request.targetEndpoints.size();
  entry.eligibleTraversalWordCount = request.eligibleTraversalBits.size();
  const std::size_t endpointCount =
      static_cast<std::size_t>(graph_.endpointCount);
  llvm::copy(request.targetEndpoints,
             heuristicCacheTargets_.begin() + slot * endpointCount);
  llvm::copy(request.eligibleTraversalBits,
             heuristicCacheEligibility_.begin() +
                 slot * heuristicCacheTraversalWordCount_);
  RouteCost *distances = heuristicCacheDistances_.data() + slot * endpointCount;
  for (PnrIndex endpoint = 0; endpoint != graph_.endpointCount; ++endpoint)
    distances[endpoint] = heuristic(endpoint);
  entry.populated = true;
}

llvm::Expected<EndpointRouteSearchResult>
EndpointRouteSearchScratch::searchTimingAware(
    const EndpointRouteSearchRequest &request) {
  const PnrIndex invalidLabel = getInvalidPnrIndex();
  for (auto &labels : timingStateLabels_)
    labels.clear();
  timingLabels_.clear();
  timingHeap_.clear();

  const auto key = [&](PnrIndex label) {
    const TimingSearchLabel &value = timingLabels_[label];
    return std::make_tuple(value.distance, value.endpoint, value.requirementMet,
                           value.arrivalQuanta, label);
  };
  const auto heapWorse = [&](PnrIndex lhs, PnrIndex rhs) {
    return key(lhs) > key(rhs);
  };
  const auto push = [&](PnrIndex label) {
    timingHeap_.push_back(label);
    std::push_heap(timingHeap_.begin(), timingHeap_.end(), heapWorse);
  };
  const auto pop = [&]() {
    std::pop_heap(timingHeap_.begin(), timingHeap_.end(), heapWorse);
    const PnrIndex result = timingHeap_.back();
    timingHeap_.pop_back();
    return result;
  };
  const auto addLabel =
      [&](PnrIndex endpoint, bool requirementMet, std::uint64_t arrival,
          RouteCost distance, PnrIndex predecessorLabel,
          PnrIndex predecessorArc) -> llvm::Expected<std::optional<PnrIndex>> {
    const PnrIndex state = searchState(endpoint, requirementMet);
    if (state >= timingStateLabels_.size())
      return invalid("physical timing search state is out of range");
    for (PnrIndex existingOrdinal : timingStateLabels_[state]) {
      const TimingSearchLabel &existing = timingLabels_[existingOrdinal];
      if (existing.active && existing.arrivalQuanta <= arrival &&
          existing.distance <= distance)
        return std::optional<PnrIndex>();
    }
    for (PnrIndex existingOrdinal : timingStateLabels_[state]) {
      TimingSearchLabel &existing = timingLabels_[existingOrdinal];
      if (existing.active && arrival <= existing.arrivalQuanta &&
          distance <= existing.distance)
        existing.active = false;
    }
    if (timingLabels_.size() >= std::numeric_limits<PnrIndex>::max())
      return overflow("physical timing label domain exceeds PnrIndex");
    const PnrIndex ordinal = static_cast<PnrIndex>(timingLabels_.size());
    timingLabels_.push_back({endpoint, predecessorLabel, predecessorArc,
                             arrival, distance, requirementMet, true});
    timingStateLabels_[state].push_back(ordinal);
    push(ordinal);
    return std::optional<PnrIndex>(ordinal);
  };

  beginSourceGeneration();
  for (auto [ordinal, source] : llvm::enumerate(request.sourceEndpoints)) {
    sourceEpochs_[source] = sourceGeneration_;
    sourceReplicationGroups_[source] = request.sourceReplicationGroups[ordinal];
    const std::uint64_t sourceArrival =
        request.sourceTimingArrivalQuanta[ordinal];
    const std::uint64_t sourceExcess =
        sourceArrival > request.requiredTimingQuanta
            ? sourceArrival - request.requiredTimingQuanta
            : 0;
    auto initialCost = detail::physicalTimingDrivenNegativeSlackCost(
        sourceExcess, request.requiredTimingQuanta, request.timingCriticality);
    if (!initialCost) {
      llvm::consumeError(initialCost.takeError());
      return overflow("physical timing source slack penalty exceeds the "
                      "largest finite route cost");
    }
    auto inserted = addLabel(source, false, sourceArrival, *initialCost,
                             invalidLabel, invalidLabel);
    if (!inserted)
      return inserted.takeError();
  }

  PnrIndex bestTargetLabel = invalidLabel;
  RouteCost bestCost = routeCostInfinity;
  std::uint64_t bestTargetArrival = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t expansions = 0;
  while (!timingHeap_.empty()) {
    while (!timingHeap_.empty() && !timingLabels_[timingHeap_.front()].active)
      (void)pop();
    if (timingHeap_.empty())
      break;
    if (bestTargetLabel != invalidLabel &&
        timingLabels_[timingHeap_.front()].distance > bestCost)
      break;
    if (expansions == request.endpointExpansionLimit)
      return failure(
          EndpointRouteSearchFailureKind::WorkLimit,
          "endpoint expansion limit reached before timing optimality proof");
    const PnrIndex labelOrdinal = pop();
    const TimingSearchLabel label = timingLabels_[labelOrdinal];
    if (!label.active)
      continue;
    if (endpointExpansionCount_ == std::numeric_limits<std::uint64_t>::max())
      return overflow("cumulative endpoint expansion count overflows u64");
    ++expansions;
    ++endpointExpansionCount_;

    if (isTarget(label.endpoint) &&
        (!targetRequiresTraversal(label.endpoint) || label.requirementMet)) {
      const auto targetPosition =
          llvm::lower_bound(request.targetEndpoints, label.endpoint);
      if (targetPosition == request.targetEndpoints.end() ||
          *targetPosition != label.endpoint)
        return invalid("physical timing target has no request ordinal");
      const std::size_t targetOrdinal =
          targetPosition - request.targetEndpoints.begin();
      const std::uint64_t terminalDelay =
          request.targetTimingDelayQuanta[targetOrdinal];
      if (terminalDelay >
          std::numeric_limits<std::uint64_t>::max() - label.arrivalQuanta)
        return overflow("physical timing target arrival exceeds u64");
      const std::uint64_t terminalArrival = label.arrivalQuanta + terminalDelay;
      const std::uint64_t oldExcess =
          label.arrivalQuanta > request.requiredTimingQuanta
              ? label.arrivalQuanta - request.requiredTimingQuanta
              : 0;
      const std::uint64_t terminalExcess =
          terminalArrival > request.requiredTimingQuanta
              ? terminalArrival - request.requiredTimingQuanta
              : 0;
      auto terminalPenalty = detail::physicalTimingDrivenNegativeSlackCost(
          terminalExcess - oldExcess, request.requiredTimingQuanta,
          request.timingCriticality);
      if (!terminalPenalty) {
        llvm::consumeError(terminalPenalty.takeError());
        return overflow("physical timing target slack penalty exceeds the "
                        "largest finite route cost");
      }
      auto targetCost = addFiniteCost(label.distance, *terminalPenalty,
                                      "physical timing target distance");
      if (!targetCost)
        return targetCost.takeError();
      if (*targetCost < bestCost ||
          (*targetCost == bestCost &&
           (bestTargetLabel == invalidLabel ||
            std::make_tuple(targetPreferenceRank(label.endpoint),
                            label.endpoint, terminalArrival, labelOrdinal) <
                std::make_tuple(targetPreferenceRank(
                                    timingLabels_[bestTargetLabel].endpoint),
                                timingLabels_[bestTargetLabel].endpoint,
                                bestTargetArrival, bestTargetLabel)))) {
        bestTargetLabel = labelOrdinal;
        bestCost = *targetCost;
        bestTargetArrival = terminalArrival;
      }
      continue;
    }

    const PnrIndex begin = graph_.adjacencyOffsets[label.endpoint];
    const PnrIndex end = graph_.adjacencyOffsets[label.endpoint + 1];
    for (PnrIndex arc = begin; arc < end; ++arc) {
      if (!arcEligible(arc, request, true))
        continue;
      const PnrIndex successor = graph_.arcs[arc].target;
      if (request.forbidSourceReentry && isSource(successor) &&
          successor != label.endpoint)
        continue;
      if (request.arcTimingDelayQuanta[arc] >
          std::numeric_limits<std::uint64_t>::max() - label.arrivalQuanta)
        return overflow("physical timing arrival exceeds u64");
      const std::uint64_t reached =
          label.arrivalQuanta + request.arcTimingDelayQuanta[arc];
      const std::uint64_t oldExcess =
          label.arrivalQuanta > request.requiredTimingQuanta
              ? label.arrivalQuanta - request.requiredTimingQuanta
              : 0;
      const std::uint64_t newExcess =
          reached > request.requiredTimingQuanta
              ? reached - request.requiredTimingQuanta
              : 0;
      auto penalty = detail::physicalTimingDrivenNegativeSlackCost(
          newExcess - oldExcess, request.requiredTimingQuanta,
          request.timingCriticality);
      if (!penalty) {
        llvm::consumeError(penalty.takeError());
        return overflow("physical timing slack penalty exceeds the largest "
                        "finite route cost");
      }
      auto arcCost = arcSearchCost(request, arc, true);
      if (!arcCost)
        return arcCost.takeError();
      auto distance = addFiniteCost(label.distance, *arcCost,
                                    "timing-aware forward distance");
      if (!distance)
        return distance.takeError();
      distance =
          addFiniteCost(*distance, *penalty, "timing-aware slack distance");
      if (!distance)
        return distance.takeError();
      const PnrIndex traversal = graph_.arcs[arc].traversal;
      const bool selectsRequired =
          !request.requiredTraversalBits.empty() &&
          traversal / 64 < request.requiredTraversalBits.size() &&
          (request.requiredTraversalBits[traversal / 64] &
           (std::uint64_t{1} << (traversal % 64))) != 0;
      const std::uint64_t successorArrival =
          request.arcTimingRegisteredDestination[arc] ? 0 : reached;
      auto inserted =
          addLabel(successor, label.requirementMet || selectsRequired,
                   successorArrival, *distance, labelOrdinal, arc);
      if (!inserted)
        return inserted.takeError();
    }
  }

  if (bestTargetLabel == invalidLabel)
    return failure(EndpointRouteSearchFailureKind::Unreachable,
                   "no eligible timing-aware route connects the endpoint "
                   "sets");

  path_.clear();
  PnrIndex label = bestTargetLabel;
  while (timingLabels_[label].predecessorLabel != invalidLabel) {
    if (timingLabels_[label].predecessorArc == invalidLabel)
      return invalid("physical timing predecessor chain has no arc");
    path_.push_back(timingLabels_[label].predecessorArc);
    label = timingLabels_[label].predecessorLabel;
    if (label >= timingLabels_.size())
      return invalid("physical timing predecessor chain is out of range");
  }
  const PnrIndex source = timingLabels_[label].endpoint;
  if (!isSource(source))
    return invalid("physical timing predecessor chain has no source");
  std::reverse(path_.begin(), path_.end());
  return EndpointRouteSearchResult{
      source, timingLabels_[bestTargetLabel].endpoint, bestCost, path_};
}

llvm::Expected<EndpointRouteSearchResult>
EndpointRouteSearchScratch::search(const EndpointRouteSearchRequest &request) {
  if (!prepared_)
    return invalid("scratch must be prepared before search");
  if (request.sourceEndpoints.empty() || request.targetEndpoints.empty())
    return invalid("source and target endpoint sets must be nonempty");
  if (request.sourceReplicationGroups.size() !=
          request.sourceEndpoints.size() ||
      request.targetPreferenceRanks.size() != request.targetEndpoints.size())
    return invalid(
        "source replication and target preference arrays must match their "
        "endpoint domains");
  if (!request.targetRequiresTraversal.empty() &&
      request.targetRequiresTraversal.size() != request.targetEndpoints.size())
    return invalid(
        "target traversal requirements must match the target endpoint domain");
  if (!isCanonicalEndpointSet(request.sourceEndpoints) ||
      !isCanonicalEndpointSet(request.targetEndpoints))
    return invalid("source and target endpoint sets must be sorted and unique");
  if (request.lowerBoundArcCosts.size() != graph_.arcs.size() ||
      request.currentArcCosts.size() != graph_.arcs.size())
    return invalid("lower-bound and current cost arrays must contain E rows");
  const bool timingAware = request.physicalTimingEnabled;
  if (!timingAware &&
      (!request.arcTimingRegisteredDestination.empty() ||
       !request.sourceTimingArrivalQuanta.empty() ||
       !request.targetTimingDelayQuanta.empty() ||
       request.requiredTimingQuanta != 0 || request.timingCriticality != 0))
    return invalid("partial physical timing search input is not allowed");
  if (timingAware &&
      (request.arcTimingDelayQuanta.size() != graph_.arcs.size() ||
       request.arcTimingRegisteredDestination.size() != graph_.arcs.size() ||
       request.sourceTimingArrivalQuanta.size() !=
           request.sourceEndpoints.size() ||
       request.targetTimingDelayQuanta.size() !=
           request.targetEndpoints.size() ||
       request.requiredTimingQuanta == 0))
    return invalid("physical timing search arrays have the wrong domain");
  if (timingAware)
    for (PnrIndex arc = 0; arc < graph_.arcs.size(); ++arc) {
      if (request.arcTimingDelayQuanta[arc] == 0)
        return invalid("physical timing search has a zero-delay arc");
      if (request.arcTimingRegisteredDestination[arc] > 1)
        return invalid("physical timing boundary flag is not boolean");
    }
  const std::size_t traversalWords =
      (graph_.traversalReplicationGroups.size() + 63) / 64;
  if (!request.eligibleTraversalBits.empty() &&
      request.eligibleTraversalBits.size() != traversalWords)
    return invalid("eligible traversal mask has the wrong width");
  if (!request.requiredTraversalBits.empty() &&
      request.requiredTraversalBits.size() != traversalWords)
    return invalid("required traversal mask has the wrong width");
  if (!request.eligibleTraversalBits.empty() &&
      graph_.traversalReplicationGroups.size() % 64 != 0) {
    const std::uint64_t paddingMask = ~(
        (std::uint64_t{1} << (graph_.traversalReplicationGroups.size() % 64)) -
        1);
    if ((request.eligibleTraversalBits.back() & paddingMask) != 0)
      return invalid("eligible traversal mask has nonzero padding");
  }
  if (!request.requiredTraversalBits.empty() &&
      graph_.traversalReplicationGroups.size() % 64 != 0) {
    const std::uint64_t paddingMask = ~(
        (std::uint64_t{1} << (graph_.traversalReplicationGroups.size() % 64)) -
        1);
    if ((request.requiredTraversalBits.back() & paddingMask) != 0)
      return invalid("required traversal mask has nonzero padding");
  }
  if (!request.requiredTraversalBits.empty()) {
    bool hasRequiredTraversal = false;
    for (std::uint64_t word : request.requiredTraversalBits)
      hasRequiredTraversal |= word != 0;
    if (!hasRequiredTraversal)
      return invalid("required traversal mask names no traversal");
  }
  bool anyTargetRequiresTraversal = false;
  for (std::uint8_t required : request.targetRequiresTraversal) {
    if (required > 1)
      return invalid("target traversal requirement is not boolean");
    anyTargetRequiresTraversal |= required != 0;
  }
  if (anyTargetRequiresTraversal && request.requiredTraversalBits.empty())
    return invalid(
        "a target traversal requirement has no required traversal mask");
  if (!anyTargetRequiresTraversal && !request.requiredTraversalBits.empty())
    return invalid("required traversal mask has no target requirement");
  if (request.endpointExpansionLimit == 0)
    return invalid("endpoint expansion limit must be positive");
  for (PnrIndex endpoint : request.sourceEndpoints)
    if (endpoint >= graph_.endpointCount)
      return invalid("source endpoint is out of range: ", endpoint);
  for (PnrIndex endpoint : request.targetEndpoints)
    if (endpoint >= graph_.endpointCount)
      return invalid("target endpoint is out of range: ", endpoint);
  for (PnrIndex arc = 0; arc < graph_.arcs.size(); ++arc) {
    const RouteCost lower = request.lowerBoundArcCosts[arc];
    const RouteCost current = request.currentArcCosts[arc];
    if (lower == routeCostInfinity || current == routeCostInfinity)
      return invalid("arc costs must be finite");
    if (current < lower)
      return invalid("current arc cost is below its admissible lower bound");
  }

  beginTargetGeneration();
  for (auto [ordinal, target] : llvm::enumerate(request.targetEndpoints)) {
    targetEpochs_[target] = targetGeneration_;
    targetPreferenceRanks_[target] = request.targetPreferenceRanks[ordinal];
    targetRequiresTraversal_[target] =
        request.targetRequiresTraversal.empty()
            ? 0
            : request.targetRequiresTraversal[ordinal];
  }
  if (timingAware)
    return searchTimingAware(request);
  if (!loadCachedHeuristic(request)) {
    if (llvm::Error error = buildHeuristic(request))
      return std::move(error);
    storeCachedHeuristic(request);
  }

  resetHeap();
  heapMode_ = HeapMode::ForwardAStar;
  beginSearchGeneration();
  beginSourceGeneration();
  for (auto [source, replicationGroup] : llvm::zip_equal(
           request.sourceEndpoints, request.sourceReplicationGroups)) {
    sourceEpochs_[source] = sourceGeneration_;
    sourceReplicationGroups_[source] = replicationGroup;
    const RouteCost lowerBound = heuristic(source);
    if (lowerBound == routeCostInfinity)
      continue;
    const PnrIndex state = searchState(source, false);
    distances_[state] = 0;
    priorities_[state] = lowerBound;
    predecessorArcs_[state] = invalidIndex;
    predecessorStates_[state] = invalidIndex;
    distanceEpochs_[state] = searchGeneration_;
    insertOrDecrease(state);
  }

  PnrIndex bestTargetState = invalidIndex;
  RouteCost bestCost = routeCostInfinity;
  std::uint64_t expansions = 0;
  while (!heap_.empty()) {
    const PnrIndex next = peekMinimum();
    if (bestTargetState != invalidIndex && priorities_[next] > bestCost)
      break;
    if (expansions == request.endpointExpansionLimit)
      return failure(
          EndpointRouteSearchFailureKind::WorkLimit,
          "endpoint expansion limit reached before optimality proof");
    const PnrIndex state = popMinimum();
    const PnrIndex endpoint = searchEndpoint(state);
    if (endpointExpansionCount_ == std::numeric_limits<std::uint64_t>::max())
      return overflow("cumulative endpoint expansion count overflows u64");
    ++expansions;
    ++endpointExpansionCount_;
    const RouteCost endpointDistance = distances_[state];
    const bool requirementMet = searchRequirementMet(state);
    if (isTarget(endpoint) &&
        (!targetRequiresTraversal(endpoint) || requirementMet)) {
      if (endpointDistance < bestCost ||
          (endpointDistance == bestCost &&
           (bestTargetState == invalidIndex ||
            std::make_tuple(targetPreferenceRank(endpoint), endpoint) <
                std::make_tuple(
                    targetPreferenceRank(searchEndpoint(bestTargetState)),
                    searchEndpoint(bestTargetState))))) {
        bestTargetState = state;
        bestCost = endpointDistance;
      }
      continue;
    }

    const PnrIndex begin = graph_.adjacencyOffsets[endpoint];
    const PnrIndex end = graph_.adjacencyOffsets[endpoint + 1];
    for (PnrIndex arc = begin; arc < end; ++arc) {
      if (!arcEligible(arc, request, true))
        continue;
      const PnrIndex successor = graph_.arcs[arc].target;
      if (request.forbidSourceReentry && isSource(successor) &&
          successor != endpoint)
        continue;
      const RouteCost successorHeuristic = heuristic(successor);
      if (successorHeuristic == routeCostInfinity)
        continue;
      auto arcCost = arcSearchCost(request, arc, true);
      if (!arcCost)
        return arcCost.takeError();
      auto candidateDistance =
          addFiniteCost(endpointDistance, *arcCost, "forward distance");
      if (!candidateDistance)
        return candidateDistance.takeError();
      const PnrIndex traversal = graph_.arcs[arc].traversal;
      const bool selectsRequired =
          !request.requiredTraversalBits.empty() &&
          traversal / 64 < request.requiredTraversalBits.size() &&
          (request.requiredTraversalBits[traversal / 64] &
           (std::uint64_t{1} << (traversal % 64))) != 0;
      const PnrIndex successorState =
          searchState(successor, requirementMet || selectsRequired);
      if (*candidateDistance >= distance(successorState))
        continue;
      auto candidatePriority = addFiniteCost(
          *candidateDistance, successorHeuristic, "A-star priority");
      if (!candidatePriority)
        return candidatePriority.takeError();
      distances_[successorState] = *candidateDistance;
      priorities_[successorState] = *candidatePriority;
      predecessorArcs_[successorState] = arc;
      predecessorStates_[successorState] = state;
      distanceEpochs_[successorState] = searchGeneration_;
      insertOrDecrease(successorState);
    }
  }

  if (bestTargetState == invalidIndex) {
    const std::size_t targetRequiringTraversalCount =
        llvm::count_if(request.targetRequiresTraversal,
                       [](std::uint8_t required) { return required != 0; });
    const std::size_t sourceTargetOverlapCount =
        llvm::count_if(request.targetEndpoints, [&](PnrIndex target) {
          return llvm::binary_search(request.sourceEndpoints, target);
        });
    std::size_t eligibleRequiredTraversalCount = 0;
    std::size_t eligibleTraversalCount = 0;
    for (std::size_t traversal = 0;
         traversal != graph_.traversalReplicationGroups.size(); ++traversal) {
      const std::uint64_t bit = std::uint64_t{1} << (traversal % 64);
      if (request.eligibleTraversalBits.empty() ||
          (request.eligibleTraversalBits[traversal / 64] & bit) != 0)
        ++eligibleTraversalCount;
      if (traversal / 64 < request.requiredTraversalBits.size() &&
          (request.requiredTraversalBits[traversal / 64] & bit) != 0 &&
          (request.eligibleTraversalBits.empty() ||
           (request.eligibleTraversalBits[traversal / 64] & bit) != 0))
        ++eligibleRequiredTraversalCount;
    }
    const std::size_t heuristicReachableSourceCount =
        llvm::count_if(request.sourceEndpoints, [&](PnrIndex source) {
          return heuristic(source) != routeCostInfinity;
        });
    return failure(
        EndpointRouteSearchFailureKind::Unreachable,
        "no eligible route connects the endpoint sets (source_count=",
        request.sourceEndpoints.size(),
        ", target_count=", request.targetEndpoints.size(),
        ", source_target_overlap_count=", sourceTargetOverlapCount,
        ", target_requiring_traversal_count=", targetRequiringTraversalCount,
        ", heuristic_reachable_source_count=", heuristicReachableSourceCount,
        ", eligible_traversal_count=", eligibleTraversalCount,
        ", eligible_required_traversal_count=", eligibleRequiredTraversalCount,
        ", first_source=", request.sourceEndpoints.front(),
        ", first_target=", request.targetEndpoints.front(),
        ", first_target_requires_traversal=",
        request.targetRequiresTraversal.empty()
            ? 0
            : request.targetRequiresTraversal.front(),
        ")");
  }

  path_.clear();
  PnrIndex state = bestTargetState;
  while (predecessorStates_[state] != invalidIndex) {
    const PnrIndex arc = predecessorArcs_[state];
    if (arc == invalidIndex)
      return invalid("predecessor chain does not reach a source endpoint");
    path_.push_back(arc);
    state = predecessorStates_[state];
  }
  const PnrIndex source = searchEndpoint(state);
  const PnrIndex target = searchEndpoint(bestTargetState);
  if (!isSource(source))
    return invalid("predecessor chain terminates outside the source set");
  std::reverse(path_.begin(), path_.end());
  return EndpointRouteSearchResult{source, target, bestCost, path_};
}

std::size_t EndpointRouteSearchScratch::retainedStorageBytes() const {
  std::size_t cacheBytes =
      heuristicCache_.capacity() * sizeof(HeuristicCacheEntry) +
      heuristicCacheTargets_.capacity() * sizeof(PnrIndex) +
      heuristicCacheEligibility_.capacity() * sizeof(std::uint64_t) +
      heuristicCacheDistances_.capacity() * sizeof(RouteCost);
  std::size_t timingBytes =
      timingLabels_.capacity() * sizeof(TimingSearchLabel) +
      timingStateLabels_.capacity() * sizeof(std::vector<PnrIndex>) +
      timingHeap_.capacity() * sizeof(PnrIndex);
  for (const auto &labels : timingStateLabels_)
    timingBytes += labels.capacity() * sizeof(PnrIndex);
  return cacheBytes + heuristics_.capacity() * sizeof(RouteCost) +
         distances_.capacity() * sizeof(RouteCost) +
         priorities_.capacity() * sizeof(RouteCost) +
         predecessorArcs_.capacity() * sizeof(PnrIndex) +
         predecessorStates_.capacity() * sizeof(PnrIndex) +
         heuristicEpochs_.capacity() * sizeof(std::uint64_t) +
         distanceEpochs_.capacity() * sizeof(std::uint64_t) +
         targetEpochs_.capacity() * sizeof(std::uint64_t) +
         sourceEpochs_.capacity() * sizeof(std::uint64_t) +
         targetPreferenceRanks_.capacity() * sizeof(PnrIndex) +
         targetRequiresTraversal_.capacity() * sizeof(std::uint8_t) +
         sourceReplicationGroups_.capacity() * sizeof(PnrIndex) +
         heap_.capacity() * sizeof(PnrIndex) +
         heapPositions_.capacity() * sizeof(PnrIndex) +
         path_.capacity() * sizeof(PnrIndex) + timingBytes;
}
