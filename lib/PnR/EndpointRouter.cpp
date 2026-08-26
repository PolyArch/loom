#include "PnR/EndpointRouter.h"

#include "SpatialPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <limits>
#include <tuple>
#include <utility>

using namespace loom::pnr;

char EndpointRouteSearchFailure::ID;

namespace {

std::uint64_t nextEndpointRouteInputOwnerIdentity() {
  static std::atomic<std::uint64_t> nextIdentity{1};
  const std::uint64_t identity =
      nextIdentity.fetch_add(1, std::memory_order_relaxed);
  if (identity == 0)
    llvm::report_fatal_error(
        "EndpointRouter input owner identity domain exhausted");
  return identity;
}

} // namespace

struct EndpointRouteInputRevision::State final {
  explicit State(std::uint64_t ownerIdentity) : generation{ownerIdentity, 0} {}

  Generation generation;
};

EndpointRouteInputRevisionOwner::EndpointRouteInputRevisionOwner()
    : state_(std::make_shared<EndpointRouteInputRevision::State>(
          nextEndpointRouteInputOwnerIdentity())) {}

EndpointRouteInputRevisionOwner::EndpointRouteInputRevisionOwner(
    EndpointRouteInputRevisionOwner &&other) noexcept = default;

EndpointRouteInputRevisionOwner::~EndpointRouteInputRevisionOwner() = default;

EndpointRouteInputRevision EndpointRouteInputRevisionOwner::revision() const & {
  if (!state_)
    return EndpointRouteInputRevision({}, {});
  return EndpointRouteInputRevision(
      std::weak_ptr<const EndpointRouteInputRevision::State>(state_),
      state_->generation);
}

llvm::Error EndpointRouteInputRevisionOwner::advance() {
  if (!state_)
    return llvm::make_error<llvm::StringError>(
        "cannot advance a moved EndpointRouter input revision owner",
        std::make_error_code(std::errc::invalid_argument));
  if (state_->generation.revision == std::numeric_limits<std::uint64_t>::max())
    return llvm::make_error<llvm::StringError>(
        "EndpointRouter input revision exceeds uint64_t",
        std::make_error_code(std::errc::result_out_of_range));
  ++state_->generation.revision;
  return llvm::Error::success();
}

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
constexpr std::size_t heuristicCacheByteBudget = 512 * 1024 * 1024;
constexpr std::size_t maximumHeuristicCacheEntryCount = 4096;
constexpr std::uint32_t compactHeuristicInfinity =
    std::numeric_limits<std::uint32_t>::max();
constexpr llvm::StringLiteral endpointHeuristicAlgorithmIdentity =
    "loom.pnr.endpoint_lower_bound_heuristic.6";

void updateDigestWord(llvm::SHA256 &digest, std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (unsigned byte = 0; byte != 8; ++byte) {
    bytes[7 - byte] = static_cast<std::uint8_t>(value);
    value >>= 8;
  }
  digest.update(bytes);
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
computeArcSearchCost(const EndpointRouteSearchRequest &request, PnrIndex arc,
                     bool current, RouteCost physicalTimingCost) {
  const RouteCost resourceCost =
      current ? request.currentArcCosts[arc] : request.lowerBoundArcCosts[arc];
  if (!request.physicalTimingEnabled)
    return resourceCost;
  return addFiniteCost(resourceCost, physicalTimingCost,
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
  if (graph.arcs.size() >
      (std::numeric_limits<std::size_t>::max() - searchStateCount) / 2)
    return invalid("routing graph cannot bound its radix queue storage");
  routeQueueEntries_.resize(graph.arcs.size() * 2 + searchStateCount);
  routeQueueEntryCount_ = 0;
  routeQueueMinimumHeap_.clear();
  routeQueueMinimumHeap_.reserve(searchStateCount);
  resetRouteQueue();
  path_.clear();
  path_.reserve(searchStateCount);
  timingLabels_.clear();
  timingLabels_.reserve(searchStateCount);
  timingStateLabelHeads_.assign(searchStateCount, invalidIndex);
  timingStateLabelEpochs_.assign(searchStateCount, 0);
  timingHeap_.clear();
  timingHeap_.reserve(searchStateCount);
  timingArcCosts_.assign(graph.arcs.size(), 0);
  timingArcCostEpochs_.assign(graph.arcs.size(), 0);
  heuristicCache_.clear();
  heuristicCacheIndex_.clear();
  eligibleTraversalMaskSnapshot_.clear();
  eligibleTraversalMaskDigest_ = {};
  eligibleTraversalMaskDigestValid_ = false;
  heuristicCacheDistanceByteBudget_ = 0;
  heuristicCacheDistanceBytes_ = 0;
  if (endpointCount != 0 &&
      endpointCount <= heuristicCacheByteBudget / sizeof(std::uint32_t)) {
    constexpr std::size_t indexEntryBytes =
        sizeof(std::array<std::uint8_t, 32>) + sizeof(std::size_t) +
        sizeof(void *) * 4;
    const std::size_t distanceBytes = endpointCount * sizeof(std::uint32_t);
    const std::size_t entryBytes =
        distanceBytes + sizeof(HeuristicCacheEntry) + indexEntryBytes;
    std::size_t entryCount = std::min(maximumHeuristicCacheEntryCount,
                                      heuristicCacheByteBudget /
                                          std::max<std::size_t>(entryBytes, 1));
    if (entryCount != 0) {
      heuristicCache_.resize(entryCount);
      heuristicCacheIndex_.reserve(entryCount);
      const std::size_t fixedBytes =
          heuristicCache_.capacity() *
          (sizeof(HeuristicCacheEntry) + indexEntryBytes);
      if (fixedBytes < heuristicCacheByteBudget)
        heuristicCacheDistanceByteBudget_ =
            heuristicCacheByteBudget - fixedBytes;
    }
  }
  activeCachedHeuristic_ = nullptr;
  heuristicCacheUseEpoch_ = 0;
  heuristicGeneration_ = 0;
  searchGeneration_ = 0;
  targetGeneration_ = 0;
  sourceGeneration_ = 0;
  timingLabelGeneration_ = 0;
  timingArcCostGeneration_ = 0;
  endpointExpansionCount_ = 0;
  heuristicCacheHitCount_ = 0;
  heuristicBuildCount_ = 0;
  forwardHeuristicQueryCount_ = 0;
  forwardHeuristicUnreachableCount_ = 0;
  heuristicCacheEvictionCount_ = 0;
  arcCostValidationScanCount_ = 0;
  physicalTimingValidationScanCount_ = 0;
  validatedArcCosts_ = {};
  validatedPhysicalTiming_ = {};
  prepared_ = true;
  return llvm::Error::success();
}

void EndpointRouteSearchScratch::resetRouteQueue() {
  routeQueueBucketHeads_.fill(std::numeric_limits<std::size_t>::max());
  routeQueueEntryCount_ = 0;
  routeQueueMinimumHeap_.clear();
  routeQueueLastKey_ = 0;
}

bool EndpointRouteSearchScratch::routeQueueEntryCurrent(
    const RouteQueueEntry &entry) const {
  if (heapMode_ == HeapMode::ReverseDistance) {
    return entry.state < graph_.endpointCount &&
           heuristicEpochs_[entry.state] == heuristicGeneration_ &&
           heuristics_[entry.state] == entry.key;
  }
  return entry.state < priorities_.size() &&
         distanceEpochs_[entry.state] == searchGeneration_ &&
         priorities_[entry.state] == entry.key;
}

bool EndpointRouteSearchScratch::routeQueueTieWorse(
    const RouteQueueEntry &lhs, const RouteQueueEntry &rhs) const {
  if (lhs.key != rhs.key)
    return lhs.key > rhs.key;
  if (heapMode_ == HeapMode::ReverseDistance)
    return lhs.state > rhs.state;
  const PnrIndex lhsEndpoint = searchEndpoint(lhs.state);
  const PnrIndex rhsEndpoint = searchEndpoint(rhs.state);
  const RouteCost lhsHeuristic = heuristic(lhsEndpoint);
  const RouteCost rhsHeuristic = heuristic(rhsEndpoint);
  if (lhsHeuristic != rhsHeuristic)
    return lhsHeuristic > rhsHeuristic;
  return lhs.state > rhs.state;
}

bool EndpointRouteSearchScratch::refillRouteQueueMinimumBucket() {
  constexpr std::size_t invalidEntry = std::numeric_limits<std::size_t>::max();
  while (true) {
    std::size_t bucket = 1;
    while (bucket != routeQueueBucketHeads_.size() &&
           routeQueueBucketHeads_[bucket] == invalidEntry)
      ++bucket;
    if (bucket == routeQueueBucketHeads_.size())
      return false;

    const std::size_t head = routeQueueBucketHeads_[bucket];
    routeQueueBucketHeads_[bucket] = invalidEntry;
    RouteCost minimumKey = routeCostInfinity;
    for (std::size_t entry = head; entry != invalidEntry;
         entry = routeQueueEntries_[entry].next)
      if (routeQueueEntryCurrent(routeQueueEntries_[entry]))
        minimumKey = std::min(minimumKey, routeQueueEntries_[entry].key);
    if (minimumKey == routeCostInfinity)
      continue;
    routeQueueLastKey_ = minimumKey;

    for (std::size_t entry = head; entry != invalidEntry;) {
      RouteQueueEntry &record = routeQueueEntries_[entry];
      const std::size_t next = record.next;
      if (routeQueueEntryCurrent(record)) {
        const RouteCost difference = record.key ^ routeQueueLastKey_;
        const std::size_t destination =
            difference == 0 ? 0 : 64 - llvm::countl_zero(difference);
        if (destination == 0) {
          if (heapMode_ == HeapMode::ReverseDistance) {
            record.next = routeQueueBucketHeads_[0];
            routeQueueBucketHeads_[0] = entry;
          } else {
            routeQueueMinimumHeap_.push_back(entry);
            const auto worse = [&](std::size_t lhs, std::size_t rhs) {
              return routeQueueTieWorse(routeQueueEntries_[lhs],
                                        routeQueueEntries_[rhs]);
            };
            std::push_heap(routeQueueMinimumHeap_.begin(),
                           routeQueueMinimumHeap_.end(), worse);
          }
        } else {
          record.next = routeQueueBucketHeads_[destination];
          routeQueueBucketHeads_[destination] = entry;
        }
      }
      entry = next;
    }
    if (heapMode_ == HeapMode::ReverseDistance) {
      if (routeQueueBucketHeads_[0] != invalidEntry)
        return true;
    } else if (!routeQueueMinimumHeap_.empty()) {
      return true;
    }
  }
}

bool EndpointRouteSearchScratch::routeQueueEmpty() {
  constexpr std::size_t invalidEntry = std::numeric_limits<std::size_t>::max();
  const auto worse = [&](std::size_t lhs, std::size_t rhs) {
    return routeQueueTieWorse(routeQueueEntries_[lhs], routeQueueEntries_[rhs]);
  };
  while (true) {
    if (heapMode_ == HeapMode::ReverseDistance) {
      while (routeQueueBucketHeads_[0] != invalidEntry &&
             !routeQueueEntryCurrent(
                 routeQueueEntries_[routeQueueBucketHeads_[0]]))
        routeQueueBucketHeads_[0] =
            routeQueueEntries_[routeQueueBucketHeads_[0]].next;
      if (routeQueueBucketHeads_[0] != invalidEntry)
        return false;
    } else {
      while (!routeQueueMinimumHeap_.empty() &&
             !routeQueueEntryCurrent(
                 routeQueueEntries_[routeQueueMinimumHeap_.front()])) {
        std::pop_heap(routeQueueMinimumHeap_.begin(),
                      routeQueueMinimumHeap_.end(), worse);
        routeQueueMinimumHeap_.pop_back();
      }
      if (!routeQueueMinimumHeap_.empty())
        return false;
    }
    if (!refillRouteQueueMinimumBucket())
      return true;
  }
}

void EndpointRouteSearchScratch::insertOrDecrease(PnrIndex endpoint) {
  const RouteCost key = heapMode_ == HeapMode::ReverseDistance
                            ? heuristics_[endpoint]
                            : priorities_[endpoint];
  assert(key != routeCostInfinity && key >= routeQueueLastKey_);
  const RouteCost difference = key ^ routeQueueLastKey_;
  const std::size_t bucket =
      difference == 0 ? 0 : 64 - llvm::countl_zero(difference);
  assert(routeQueueEntryCount_ < routeQueueEntries_.size());
  const std::size_t entry = routeQueueEntryCount_++;
  routeQueueEntries_[entry] = {key, endpoint, routeQueueBucketHeads_[bucket]};
  if (bucket != 0) {
    routeQueueBucketHeads_[bucket] = entry;
    return;
  }
  if (heapMode_ == HeapMode::ReverseDistance) {
    routeQueueBucketHeads_[0] = entry;
    return;
  }
  routeQueueMinimumHeap_.push_back(entry);
  const auto worse = [&](std::size_t lhs, std::size_t rhs) {
    return routeQueueTieWorse(routeQueueEntries_[lhs], routeQueueEntries_[rhs]);
  };
  std::push_heap(routeQueueMinimumHeap_.begin(), routeQueueMinimumHeap_.end(),
                 worse);
}

PnrIndex EndpointRouteSearchScratch::popMinimum() {
  if (heapMode_ == HeapMode::ReverseDistance) {
    constexpr std::size_t invalidEntry =
        std::numeric_limits<std::size_t>::max();
    while (routeQueueBucketHeads_[0] != invalidEntry) {
      const std::size_t entry = routeQueueBucketHeads_[0];
      routeQueueBucketHeads_[0] = routeQueueEntries_[entry].next;
      if (routeQueueEntryCurrent(routeQueueEntries_[entry]))
        return routeQueueEntries_[entry].state;
    }
    llvm_unreachable("reverse-distance queue is empty");
  }
  assert(!routeQueueMinimumHeap_.empty());
  const auto worse = [&](std::size_t lhs, std::size_t rhs) {
    return routeQueueTieWorse(routeQueueEntries_[lhs], routeQueueEntries_[rhs]);
  };
  std::pop_heap(routeQueueMinimumHeap_.begin(), routeQueueMinimumHeap_.end(),
                worse);
  const std::size_t entry = routeQueueMinimumHeap_.back();
  routeQueueMinimumHeap_.pop_back();
  return routeQueueEntries_[entry].state;
}

PnrIndex EndpointRouteSearchScratch::peekMinimum() {
  assert(!routeQueueMinimumHeap_.empty());
  return routeQueueEntries_[routeQueueMinimumHeap_.front()].state;
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
  if (activeCachedHeuristic_)
    return cachedHeuristic(*activeCachedHeuristic_, endpoint);
  if (heuristicEpochs_[endpoint] != heuristicGeneration_)
    return routeCostInfinity;
  return heuristics_[endpoint];
}

RouteCost
EndpointRouteSearchScratch::queryForwardHeuristic(PnrIndex endpoint) {
  saturatingIncrement(forwardHeuristicQueryCount_);
  const RouteCost value = heuristic(endpoint);
  if (value == routeCostInfinity)
    saturatingIncrement(forwardHeuristicUnreachableCount_);
  return value;
}

RouteCost
EndpointRouteSearchScratch::cachedHeuristic(const HeuristicCacheEntry &entry,
                                            PnrIndex endpoint) const {
  assert(endpoint < entry.scaledDistances.size());
  const std::uint32_t scaled = entry.scaledDistances[endpoint];
  if (scaled != compactHeuristicInfinity)
    return static_cast<RouteCost>(scaled) << entry.scaleShift;
  const auto found =
      llvm::lower_bound(entry.wideDistances, endpoint,
                        [](const HeuristicCacheWideDistance &value,
                           PnrIndex key) { return value.endpoint < key; });
  return found != entry.wideDistances.end() && found->endpoint == endpoint
             ? found->distance
             : routeCostInfinity;
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

llvm::Expected<RouteCost> EndpointRouteSearchScratch::searchArcCost(
    const EndpointRouteSearchRequest &request, PnrIndex arc, bool current) {
  if (!request.physicalTimingEnabled)
    return computeArcSearchCost(request, arc, current, 0);
  if (timingArcCostEpochs_[arc] != timingArcCostGeneration_) {
    auto timingCost = detail::physicalTimingDrivenTraversalCost(
        request.arcTimingDelayQuanta[arc], request.requiredTimingQuanta,
        request.timingCriticality);
    if (!timingCost) {
      llvm::consumeError(timingCost.takeError());
      return overflow(
          "physical traversal cost exceeds the largest finite route "
          "cost");
    }
    timingArcCosts_[arc] = *timingCost;
    timingArcCostEpochs_[arc] = timingArcCostGeneration_;
  }
  return computeArcSearchCost(request, arc, current, timingArcCosts_[arc]);
}

llvm::Error EndpointRouteSearchScratch::buildHeuristic(
    const EndpointRouteSearchRequest &request) {
  activeCachedHeuristic_ = nullptr;
  saturatingIncrement(heuristicBuildCount_);
  resetRouteQueue();
  heapMode_ = HeapMode::ReverseDistance;
  beginHeuristicGeneration();
  for (PnrIndex target : request.targetEndpoints) {
    heuristics_[target] = 0;
    heuristicEpochs_[target] = heuristicGeneration_;
    insertOrDecrease(target);
  }

  while (!routeQueueEmpty()) {
    const PnrIndex endpoint = popMinimum();
    const RouteCost endpointCost = heuristics_[endpoint];
    const PnrIndex begin = graph_.reverseAdjacencyOffsets[endpoint];
    const PnrIndex end = graph_.reverseAdjacencyOffsets[endpoint + 1];
    for (PnrIndex offset = begin; offset < end; ++offset) {
      const PnrIndex arc = graph_.reverseArcOrdinals[offset];
      if (!arcEligible(arc, request, false))
        continue;
      const PnrIndex predecessor = graph_.arcSources[arc];
      auto arcCost = searchArcCost(request, arc, false);
      if (!arcCost)
        return arcCost.takeError();
      auto candidate =
          addFiniteCost(endpointCost, *arcCost, "reverse lower-bound distance");
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

bool EndpointRouteSearchScratch::revisionIsCurrent(
    const EndpointRouteInputRevision &revision) const {
  const std::shared_ptr<const EndpointRouteInputRevision::State> state =
      revision.state_.lock();
  return state && revision.generation_.ownerIdentity != 0 &&
         state->generation == revision.generation_;
}

bool EndpointRouteSearchScratch::arcCostsAlreadyValidated(
    const EndpointRouteSearchRequest &request) const {
  if (!validatedArcCosts_.populated || !request.lowerBoundArcCostRevision ||
      !request.currentArcCostRevision ||
      !revisionIsCurrent(*request.lowerBoundArcCostRevision) ||
      !revisionIsCurrent(*request.currentArcCostRevision))
    return false;
  return validatedArcCosts_.lowerBoundGeneration ==
             request.lowerBoundArcCostRevision->generation_ &&
         validatedArcCosts_.currentGeneration ==
             request.currentArcCostRevision->generation_;
}

void EndpointRouteSearchScratch::rememberValidatedArcCosts(
    const EndpointRouteSearchRequest &request) {
  assert(request.lowerBoundArcCostRevision && request.currentArcCostRevision);
  assert(revisionIsCurrent(*request.lowerBoundArcCostRevision) &&
         revisionIsCurrent(*request.currentArcCostRevision));
  validatedArcCosts_ = {
      request.lowerBoundArcCostRevision->generation_,
      request.currentArcCostRevision->generation_,
      true,
  };
}

bool EndpointRouteSearchScratch::physicalTimingAlreadyValidated(
    const EndpointRouteSearchRequest &request) const {
  if (!validatedPhysicalTiming_.populated || !request.physicalTimingRevision ||
      !revisionIsCurrent(*request.physicalTimingRevision))
    return false;
  return validatedPhysicalTiming_.generation ==
         request.physicalTimingRevision->generation_;
}

void EndpointRouteSearchScratch::rememberValidatedPhysicalTiming(
    const EndpointRouteSearchRequest &request) {
  assert(request.physicalTimingRevision &&
         revisionIsCurrent(*request.physicalTimingRevision));
  validatedPhysicalTiming_ = {
      request.physicalTimingRevision->generation_,
      true,
  };
}

bool EndpointRouteSearchScratch::heuristicInputsAreCacheable(
    const EndpointRouteSearchRequest &request) const {
  if (!request.lowerBoundArcCostRevision ||
      !revisionIsCurrent(*request.lowerBoundArcCostRevision))
    return false;
  return !request.physicalTimingEnabled ||
         (request.physicalTimingRevision &&
          revisionIsCurrent(*request.physicalTimingRevision));
}

std::array<std::uint8_t, 32>
EndpointRouteSearchScratch::heuristicCacheKeyDigest(
    const EndpointRouteSearchRequest &request) const {
  assert(heuristicInputsAreCacheable(request));
  llvm::SHA256 digest;
  digest.update(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(
          endpointHeuristicAlgorithmIdentity.data()),
      endpointHeuristicAlgorithmIdentity.size()));
  updateDigestWord(
      digest, request.lowerBoundArcCostRevision->generation_.ownerIdentity);
  updateDigestWord(digest,
                   request.lowerBoundArcCostRevision->generation_.revision);
  updateDigestWord(digest, request.requiredPayloadWidthBits);
  updateDigestWord(digest, request.requiredTagWidthBits);
  updateDigestWord(digest, request.physicalTimingEnabled ? 1 : 0);
  if (request.physicalTimingEnabled) {
    updateDigestWord(digest,
                     request.physicalTimingRevision->generation_.ownerIdentity);
    updateDigestWord(digest,
                     request.physicalTimingRevision->generation_.revision);
  }
  updateDigestWord(digest, request.requiredTimingQuanta);
  updateDigestWord(digest, request.timingCriticality);
  updateDigestWord(digest, request.targetEndpoints.size());
  for (PnrIndex endpoint : request.targetEndpoints)
    updateDigestWord(digest, endpoint);
  updateDigestWord(digest, request.eligibleTraversalBits.size());
  digest.update(eligibleTraversalMaskDigest(request));
  return digest.final();
}

std::array<std::uint8_t, 32>
EndpointRouteSearchScratch::eligibleTraversalMaskDigest(
    const EndpointRouteSearchRequest &request) const {
  const auto mask = request.eligibleTraversalBits;
  const bool unchanged =
      eligibleTraversalMaskDigestValid_ &&
      eligibleTraversalMaskSnapshot_.size() == mask.size() &&
      std::equal(eligibleTraversalMaskSnapshot_.begin(),
                 eligibleTraversalMaskSnapshot_.end(), mask.begin());
  if (unchanged)
    return eligibleTraversalMaskDigest_;

  eligibleTraversalMaskSnapshot_.assign(mask.begin(), mask.end());
  llvm::SHA256 digest;
  for (std::uint64_t word : mask)
    updateDigestWord(digest, word);
  eligibleTraversalMaskDigest_ = digest.final();
  eligibleTraversalMaskDigestValid_ = true;
  return eligibleTraversalMaskDigest_;
}

bool EndpointRouteSearchScratch::loadCachedHeuristic(
    const EndpointRouteSearchRequest &request) {
  activeCachedHeuristic_ = nullptr;
  if (!heuristicInputsAreCacheable(request) || heuristicCache_.empty())
    return false;
  const auto digest = heuristicCacheKeyDigest(request);
  const auto indexed = heuristicCacheIndex_.find(digest);
  if (indexed == heuristicCacheIndex_.end() ||
      indexed->second >= heuristicCache_.size())
    return false;
  HeuristicCacheEntry &entry = heuristicCache_[indexed->second];
  if (!entry.populated || entry.keyDigest != digest ||
      entry.scaledDistances.size() != graph_.endpointCount)
    return false;
  saturatingIncrement(heuristicCacheUseEpoch_);
  entry.lastUse = heuristicCacheUseEpoch_;
  activeCachedHeuristic_ = &entry;
  saturatingIncrement(heuristicCacheHitCount_);
  return true;
}

std::size_t EndpointRouteSearchScratch::heuristicCacheEntryDistanceBytes(
    const HeuristicCacheEntry &entry) const {
  return entry.scaledDistances.capacity() * sizeof(std::uint32_t) +
         entry.wideDistances.capacity() * sizeof(HeuristicCacheWideDistance);
}

void EndpointRouteSearchScratch::evictHeuristicCacheEntry(std::size_t slot) {
  assert(slot < heuristicCache_.size());
  HeuristicCacheEntry &entry = heuristicCache_[slot];
  const std::size_t retained = heuristicCacheEntryDistanceBytes(entry);
  assert(retained <= heuristicCacheDistanceBytes_);
  heuristicCacheDistanceBytes_ -= retained;
  if (entry.populated) {
    heuristicCacheIndex_.erase(entry.keyDigest);
    saturatingIncrement(heuristicCacheEvictionCount_);
  }
  std::vector<std::uint32_t>().swap(entry.scaledDistances);
  std::vector<HeuristicCacheWideDistance>().swap(entry.wideDistances);
  entry = HeuristicCacheEntry{};
}

void EndpointRouteSearchScratch::storeCachedHeuristic(
    const EndpointRouteSearchRequest &request) {
  if (!heuristicInputsAreCacheable(request) || heuristicCache_.empty())
    return;
  const auto digest = heuristicCacheKeyDigest(request);
  std::size_t selected = 0;
  for (std::size_t slot = 0; slot != heuristicCache_.size(); ++slot) {
    if (!heuristicCache_[slot].populated) {
      selected = slot;
      break;
    }
    if (heuristicCache_[slot].lastUse < heuristicCache_[selected].lastUse)
      selected = slot;
  }
  evictHeuristicCacheEntry(selected);
  HeuristicCacheEntry &entry = heuristicCache_[selected];
  entry.keyDigest = digest;
  unsigned commonShift = std::numeric_limits<RouteCost>::digits;
  for (PnrIndex endpoint = 0; endpoint != graph_.endpointCount; ++endpoint) {
    const RouteCost value = heuristic(endpoint);
    if (value != 0 && value != routeCostInfinity)
      commonShift = std::min(commonShift,
                             static_cast<unsigned>(llvm::countr_zero(value)));
  }
  if (commonShift == std::numeric_limits<RouteCost>::digits)
    commonShift = 0;
  entry.scaleShift = static_cast<std::uint8_t>(commonShift);
  entry.scaledDistances.resize(graph_.endpointCount, compactHeuristicInfinity);
  for (PnrIndex endpoint = 0; endpoint != graph_.endpointCount; ++endpoint) {
    const RouteCost value = heuristic(endpoint);
    if (value == routeCostInfinity)
      continue;
    const RouteCost scaled = value >> commonShift;
    if (scaled < compactHeuristicInfinity) {
      entry.scaledDistances[endpoint] = static_cast<std::uint32_t>(scaled);
      continue;
    }
    entry.wideDistances.push_back({endpoint, value});
  }
  const std::size_t entryBytes = heuristicCacheEntryDistanceBytes(entry);
  if (entryBytes > heuristicCacheDistanceByteBudget_) {
    std::vector<std::uint32_t>().swap(entry.scaledDistances);
    std::vector<HeuristicCacheWideDistance>().swap(entry.wideDistances);
    entry = HeuristicCacheEntry{};
    return;
  }
  while (heuristicCacheDistanceBytes_ >
         heuristicCacheDistanceByteBudget_ - entryBytes) {
    std::size_t victim = heuristicCache_.size();
    for (std::size_t slot = 0; slot != heuristicCache_.size(); ++slot) {
      if (slot == selected || !heuristicCache_[slot].populated)
        continue;
      if (victim == heuristicCache_.size() ||
          heuristicCache_[slot].lastUse < heuristicCache_[victim].lastUse)
        victim = slot;
    }
    if (victim == heuristicCache_.size()) {
      std::vector<std::uint32_t>().swap(entry.scaledDistances);
      std::vector<HeuristicCacheWideDistance>().swap(entry.wideDistances);
      entry = HeuristicCacheEntry{};
      return;
    }
    evictHeuristicCacheEntry(victim);
  }
  heuristicCacheDistanceBytes_ += entryBytes;
  saturatingIncrement(heuristicCacheUseEpoch_);
  entry.lastUse = heuristicCacheUseEpoch_;
  entry.populated = true;
  const bool indexed = heuristicCacheIndex_.emplace(digest, selected).second;
  (void)indexed;
  assert(indexed);
}

llvm::Expected<EndpointRouteSearchResult>
EndpointRouteSearchScratch::searchTimingAware(
    const EndpointRouteSearchRequest &request) {
  const PnrIndex invalidLabel = getInvalidPnrIndex();
  advanceGeneration(timingStateLabelEpochs_, timingLabelGeneration_);
  timingLabels_.clear();
  timingHeap_.clear();

  const auto key = [&](PnrIndex label) {
    const TimingSearchLabel &value = timingLabels_[label];
    return std::make_tuple(value.priority, heuristic(value.endpoint),
                           value.endpoint, value.requirementMet,
                           value.arrivalQuanta, value.distance, label);
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
    if (state >= timingStateLabelHeads_.size())
      return invalid("physical timing search state is out of range");
    PnrIndex existingOrdinal =
        timingStateLabelEpochs_[state] == timingLabelGeneration_
            ? timingStateLabelHeads_[state]
            : invalidLabel;
    for (; existingOrdinal != invalidLabel;
         existingOrdinal = timingLabels_[existingOrdinal].nextStateLabel) {
      const TimingSearchLabel &existing = timingLabels_[existingOrdinal];
      if (existing.active && existing.arrivalQuanta <= arrival &&
          existing.distance <= distance)
        return std::optional<PnrIndex>();
    }
    existingOrdinal = timingStateLabelEpochs_[state] == timingLabelGeneration_
                          ? timingStateLabelHeads_[state]
                          : invalidLabel;
    for (; existingOrdinal != invalidLabel;
         existingOrdinal = timingLabels_[existingOrdinal].nextStateLabel) {
      TimingSearchLabel &existing = timingLabels_[existingOrdinal];
      if (existing.active && arrival <= existing.arrivalQuanta &&
          distance <= existing.distance)
        existing.active = false;
    }
    if (timingLabels_.size() >= std::numeric_limits<PnrIndex>::max())
      return overflow("physical timing label domain exceeds PnrIndex");
    const RouteCost lowerBound = queryForwardHeuristic(endpoint);
    if (lowerBound == routeCostInfinity)
      return std::optional<PnrIndex>();
    auto priority =
        addFiniteCost(distance, lowerBound, "timing-aware A-star priority");
    if (!priority)
      return priority.takeError();
    const PnrIndex ordinal = static_cast<PnrIndex>(timingLabels_.size());
    const PnrIndex oldHead =
        timingStateLabelEpochs_[state] == timingLabelGeneration_
            ? timingStateLabelHeads_[state]
            : invalidLabel;
    timingLabels_.push_back({endpoint, predecessorLabel, predecessorArc,
                             oldHead, arrival, distance, *priority,
                             requirementMet, true});
    timingStateLabelHeads_[state] = ordinal;
    timingStateLabelEpochs_[state] = timingLabelGeneration_;
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
    if (bestTargetLabel != invalidLabel) {
      const RouteCost nextLowerBound =
          timingLabels_[timingHeap_.front()].priority;
      if (nextLowerBound > bestCost)
        break;
    }
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
      auto arcCost = searchArcCost(request, arc, true);
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
      (!request.arcTimingDelayQuanta.empty() ||
       !request.arcTimingRegisteredDestination.empty() ||
       !request.sourceTimingArrivalQuanta.empty() ||
       !request.targetTimingDelayQuanta.empty() ||
       request.physicalTimingRevision || request.requiredTimingQuanta != 0 ||
       request.timingCriticality != 0))
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
  if (timingAware && !physicalTimingAlreadyValidated(request)) {
    validatedPhysicalTiming_ = {};
    saturatingIncrement(physicalTimingValidationScanCount_);
    for (PnrIndex arc = 0; arc < graph_.arcs.size(); ++arc) {
      if (request.arcTimingDelayQuanta[arc] == 0)
        return invalid("physical timing search has a zero-delay arc");
      if (request.arcTimingRegisteredDestination[arc] > 1)
        return invalid("physical timing boundary flag is not boolean");
    }
    if (request.physicalTimingRevision &&
        revisionIsCurrent(*request.physicalTimingRevision))
      rememberValidatedPhysicalTiming(request);
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
  if (!arcCostsAlreadyValidated(request)) {
    validatedArcCosts_ = {};
    saturatingIncrement(arcCostValidationScanCount_);
    for (PnrIndex arc = 0; arc < graph_.arcs.size(); ++arc) {
      const RouteCost lower = request.lowerBoundArcCosts[arc];
      const RouteCost current = request.currentArcCosts[arc];
      if (lower == routeCostInfinity || current == routeCostInfinity)
        return invalid("arc costs must be finite");
      if (current < lower)
        return invalid("current arc cost is below its admissible lower bound");
    }
    if (request.lowerBoundArcCostRevision && request.currentArcCostRevision &&
        revisionIsCurrent(*request.lowerBoundArcCostRevision) &&
        revisionIsCurrent(*request.currentArcCostRevision))
      rememberValidatedArcCosts(request);
  }

  advanceGeneration(timingArcCostEpochs_, timingArcCostGeneration_);
  beginTargetGeneration();
  for (auto [ordinal, target] : llvm::enumerate(request.targetEndpoints)) {
    targetEpochs_[target] = targetGeneration_;
    targetPreferenceRanks_[target] = request.targetPreferenceRanks[ordinal];
    targetRequiresTraversal_[target] =
        request.targetRequiresTraversal.empty()
            ? 0
            : request.targetRequiresTraversal[ordinal];
  }
  if (!loadCachedHeuristic(request)) {
    if (llvm::Error error = buildHeuristic(request))
      return std::move(error);
    storeCachedHeuristic(request);
  }
  if (timingAware)
    return searchTimingAware(request);

  resetRouteQueue();
  heapMode_ = HeapMode::ForwardAStar;
  beginSearchGeneration();
  beginSourceGeneration();
  for (auto [source, replicationGroup] : llvm::zip_equal(
           request.sourceEndpoints, request.sourceReplicationGroups)) {
    sourceEpochs_[source] = sourceGeneration_;
    sourceReplicationGroups_[source] = replicationGroup;
    const RouteCost lowerBound = queryForwardHeuristic(source);
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
  while (!routeQueueEmpty()) {
    const PnrIndex next = peekMinimum();
    if (bestTargetState != invalidIndex) {
      if (priorities_[next] > bestCost)
        break;
    }
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
      const RouteCost successorHeuristic = queryForwardHeuristic(successor);
      if (successorHeuristic == routeCostInfinity)
        continue;
      auto arcCost = searchArcCost(request, arc, true);
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

std::size_t EndpointRouteSearchScratch::heuristicCacheRetainedBytes() const {
  std::size_t cacheBytes =
      heuristicCache_.capacity() * sizeof(HeuristicCacheEntry);
  for (const HeuristicCacheEntry &entry : heuristicCache_)
    cacheBytes += heuristicCacheEntryDistanceBytes(entry);
  cacheBytes +=
      heuristicCacheIndex_.size() * (sizeof(std::array<std::uint8_t, 32>) +
                                     sizeof(std::size_t) + sizeof(void *) * 3);
  cacheBytes +=
      eligibleTraversalMaskSnapshot_.capacity() * sizeof(std::uint64_t);
  return cacheBytes;
}

std::size_t EndpointRouteSearchScratch::retainedStorageBytes() const {
  const std::size_t cacheBytes = heuristicCacheRetainedBytes();
  std::size_t timingBytes =
      timingLabels_.capacity() * sizeof(TimingSearchLabel) +
      timingStateLabelHeads_.capacity() * sizeof(PnrIndex) +
      timingStateLabelEpochs_.capacity() * sizeof(std::uint64_t) +
      timingHeap_.capacity() * sizeof(PnrIndex) +
      timingArcCosts_.capacity() * sizeof(RouteCost) +
      timingArcCostEpochs_.capacity() * sizeof(std::uint64_t);
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
         routeQueueEntries_.capacity() * sizeof(RouteQueueEntry) +
         routeQueueMinimumHeap_.capacity() * sizeof(std::size_t) +
         path_.capacity() * sizeof(PnrIndex) + timingBytes;
}
