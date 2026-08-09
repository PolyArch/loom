#include "PnR/EndpointRouter.h"

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
  distances_.assign(endpointCount, 0);
  priorities_.assign(endpointCount, 0);
  predecessorArcs_.assign(endpointCount, invalidIndex);
  heuristicEpochs_.assign(endpointCount, 0);
  distanceEpochs_.assign(endpointCount, 0);
  targetEpochs_.assign(endpointCount, 0);
  sourceEpochs_.assign(endpointCount, 0);
  targetPreferenceRanks_.assign(endpointCount, 0);
  sourceReplicationGroups_.assign(endpointCount, getInvalidPnrIndex());
  heap_.clear();
  heap_.reserve(endpointCount);
  heapPositions_.assign(endpointCount, invalidIndex);
  path_.clear();
  path_.reserve(endpointCount);
  heuristicGeneration_ = 0;
  searchGeneration_ = 0;
  targetGeneration_ = 0;
  sourceGeneration_ = 0;
  endpointExpansionCount_ = 0;
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
  if (heuristics_[lhs] != heuristics_[rhs])
    return heuristics_[lhs] < heuristics_[rhs];
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
  if (heuristicEpochs_[endpoint] != heuristicGeneration_)
    return routeCostInfinity;
  return heuristics_[endpoint];
}

RouteCost EndpointRouteSearchScratch::distance(PnrIndex endpoint) const {
  if (distanceEpochs_[endpoint] != searchGeneration_)
    return routeCostInfinity;
  return distances_[endpoint];
}

bool EndpointRouteSearchScratch::isTarget(PnrIndex endpoint) const {
  return targetEpochs_[endpoint] == targetGeneration_;
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
      auto candidate =
          addFiniteCost(endpointCost, request.lowerBoundArcCosts[arc],
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
  if (!isCanonicalEndpointSet(request.sourceEndpoints) ||
      !isCanonicalEndpointSet(request.targetEndpoints))
    return invalid("source and target endpoint sets must be sorted and unique");
  if (request.lowerBoundArcCosts.size() != graph_.arcs.size() ||
      request.currentArcCosts.size() != graph_.arcs.size())
    return invalid("lower-bound and current cost arrays must contain E rows");
  const std::size_t traversalWords =
      (graph_.traversalReplicationGroups.size() + 63) / 64;
  if (!request.eligibleTraversalBits.empty() &&
      request.eligibleTraversalBits.size() != traversalWords)
    return invalid("eligible traversal mask has the wrong width");
  if (!request.eligibleTraversalBits.empty() &&
      graph_.traversalReplicationGroups.size() % 64 != 0) {
    const std::uint64_t paddingMask = ~(
        (std::uint64_t{1} << (graph_.traversalReplicationGroups.size() % 64)) -
        1);
    if ((request.eligibleTraversalBits.back() & paddingMask) != 0)
      return invalid("eligible traversal mask has nonzero padding");
  }
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
  for (auto [target, rank] : llvm::zip_equal(request.targetEndpoints,
                                             request.targetPreferenceRanks)) {
    targetEpochs_[target] = targetGeneration_;
    targetPreferenceRanks_[target] = rank;
  }
  if (llvm::Error error = buildHeuristic(request))
    return std::move(error);

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
    distances_[source] = 0;
    priorities_[source] = lowerBound;
    predecessorArcs_[source] = invalidIndex;
    distanceEpochs_[source] = searchGeneration_;
    insertOrDecrease(source);
  }

  PnrIndex bestTarget = invalidIndex;
  RouteCost bestCost = routeCostInfinity;
  std::uint64_t expansions = 0;
  while (!heap_.empty()) {
    const PnrIndex next = peekMinimum();
    if (bestTarget != invalidIndex && priorities_[next] > bestCost)
      break;
    if (expansions == request.endpointExpansionLimit)
      return failure(
          EndpointRouteSearchFailureKind::WorkLimit,
          "endpoint expansion limit reached before optimality proof");
    const PnrIndex endpoint = popMinimum();
    if (endpointExpansionCount_ == std::numeric_limits<std::uint64_t>::max())
      return overflow("cumulative endpoint expansion count overflows u64");
    ++expansions;
    ++endpointExpansionCount_;
    const RouteCost endpointDistance = distances_[endpoint];
    if (isTarget(endpoint)) {
      if (endpointDistance < bestCost ||
          (endpointDistance == bestCost &&
           (bestTarget == invalidIndex ||
            std::make_tuple(targetPreferenceRank(endpoint), endpoint) <
                std::make_tuple(targetPreferenceRank(bestTarget),
                                bestTarget)))) {
        bestTarget = endpoint;
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
      const RouteCost successorHeuristic = heuristic(successor);
      if (successorHeuristic == routeCostInfinity)
        continue;
      auto candidateDistance = addFiniteCost(
          endpointDistance, request.currentArcCosts[arc], "forward distance");
      if (!candidateDistance)
        return candidateDistance.takeError();
      if (*candidateDistance >= distance(successor))
        continue;
      auto candidatePriority = addFiniteCost(
          *candidateDistance, successorHeuristic, "A-star priority");
      if (!candidatePriority)
        return candidatePriority.takeError();
      distances_[successor] = *candidateDistance;
      priorities_[successor] = *candidatePriority;
      predecessorArcs_[successor] = arc;
      distanceEpochs_[successor] = searchGeneration_;
      insertOrDecrease(successor);
    }
  }

  if (bestTarget == invalidIndex)
    return failure(
        EndpointRouteSearchFailureKind::Unreachable,
        "no eligible route connects the endpoint sets (source_count=",
        request.sourceEndpoints.size(),
        ", target_count=", request.targetEndpoints.size(),
        ", first_source=", request.sourceEndpoints.front(),
        ", first_target=", request.targetEndpoints.front(), ")");

  path_.clear();
  PnrIndex endpoint = bestTarget;
  while (!isSource(endpoint)) {
    const PnrIndex arc = predecessorArcs_[endpoint];
    if (arc == invalidIndex)
      return invalid("predecessor chain does not reach a source endpoint");
    path_.push_back(arc);
    endpoint = graph_.arcSources[arc];
  }
  std::reverse(path_.begin(), path_.end());
  return EndpointRouteSearchResult{endpoint, bestTarget, bestCost, path_};
}

std::size_t EndpointRouteSearchScratch::retainedStorageBytes() const {
  return heuristics_.capacity() * sizeof(RouteCost) +
         distances_.capacity() * sizeof(RouteCost) +
         priorities_.capacity() * sizeof(RouteCost) +
         predecessorArcs_.capacity() * sizeof(PnrIndex) +
         heuristicEpochs_.capacity() * sizeof(std::uint64_t) +
         distanceEpochs_.capacity() * sizeof(std::uint64_t) +
         targetEpochs_.capacity() * sizeof(std::uint64_t) +
         sourceEpochs_.capacity() * sizeof(std::uint64_t) +
         targetPreferenceRanks_.capacity() * sizeof(PnrIndex) +
         sourceReplicationGroups_.capacity() * sizeof(PnrIndex) +
         heap_.capacity() * sizeof(PnrIndex) +
         heapPositions_.capacity() * sizeof(PnrIndex) +
         path_.capacity() * sizeof(PnrIndex);
}
