#ifndef LOOM_PNR_ENDPOINTROUTER_H
#define LOOM_PNR_ENDPOINTROUTER_H

#include "PnR/EndpointRoutingTopology.h"
#include "PnR/RoutingNegotiation.h"
#include "PnR/SpatialPnrWorkLedger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace loom::pnr {

class EndpointRouteInputRevisionOwner;

/// Proof that one owner-backed EndpointRouter input is unchanged. The token
/// observes stable owner control across moves without extending the lifetime
/// of either the owner or the borrowed input view.
class EndpointRouteInputRevision final {
public:
  EndpointRouteInputRevision(const EndpointRouteInputRevision &) = default;
  EndpointRouteInputRevision &
  operator=(const EndpointRouteInputRevision &) = default;

private:
  struct Generation final {
    std::uint64_t ownerIdentity = 0;
    std::uint64_t revision = 0;

    bool operator==(const Generation &other) const {
      return ownerIdentity == other.ownerIdentity && revision == other.revision;
    }
  };

  struct State;

  friend class EndpointRouteInputRevisionOwner;
  friend class EndpointRouteSearchScratch;

  EndpointRouteInputRevision(std::weak_ptr<const State> state,
                             Generation generation)
      : state_(std::move(state)), generation_(generation) {}

  std::weak_ptr<const State> state_;
  Generation generation_;
};

/// Semantic owner for one independently changing EndpointRouter input.
class EndpointRouteInputRevisionOwner final {
public:
  EndpointRouteInputRevisionOwner();
  EndpointRouteInputRevisionOwner(const EndpointRouteInputRevisionOwner &) =
      delete;
  EndpointRouteInputRevisionOwner &
  operator=(const EndpointRouteInputRevisionOwner &) = delete;
  EndpointRouteInputRevisionOwner(
      EndpointRouteInputRevisionOwner &&other) noexcept;
  EndpointRouteInputRevisionOwner &
  operator=(EndpointRouteInputRevisionOwner &&) = delete;
  ~EndpointRouteInputRevisionOwner();

  EndpointRouteInputRevision revision() const &;
  EndpointRouteInputRevision revision() && = delete;
  llvm::Error advance();

private:
  friend class EndpointRouteSearchScratch;

  std::shared_ptr<EndpointRouteInputRevision::State> state_;
};

struct EndpointRoutingGraphView final {
  PnrIndex endpointCount = 0;
  llvm::ArrayRef<EndpointRoutingArc> arcs;
  llvm::ArrayRef<PnrIndex> arcSources;
  llvm::ArrayRef<PnrIndex> adjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals;
  llvm::ArrayRef<PnrIndex> traversalReplicationGroups;
};

EndpointRoutingGraphView
endpointRoutingGraphView(const FrozenEndpointRoutingTopology &topology);

enum class EndpointRouteSearchFailureKind {
  Invalid,
  ArithmeticOverflow,
  Unreachable,
  WorkLimit,
};

llvm::StringRef
stringifyEndpointRouteSearchFailureKind(EndpointRouteSearchFailureKind kind);

class EndpointRouteSearchFailure final
    : public llvm::ErrorInfo<EndpointRouteSearchFailure> {
public:
  static char ID;

  EndpointRouteSearchFailure(EndpointRouteSearchFailureKind kind,
                             std::string message);

  EndpointRouteSearchFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  EndpointRouteSearchFailureKind kind_;
  std::string message_;
};

struct EndpointRouteSearchRequest final {
  llvm::ArrayRef<PnrIndex> sourceEndpoints;
  llvm::ArrayRef<PnrIndex> sourceReplicationGroups;
  llvm::ArrayRef<PnrIndex> targetEndpoints;
  llvm::ArrayRef<PnrIndex> targetPreferenceRanks;
  llvm::ArrayRef<RouteCost> lowerBoundArcCosts;
  llvm::ArrayRef<RouteCost> currentArcCosts;
  std::uint32_t requiredPayloadWidthBits = 0;
  std::uint32_t requiredTagWidthBits = 0;
  std::uint64_t endpointExpansionLimit = 0;
  /// Empty means every traversal is eligible beyond width/replication checks.
  /// Otherwise this is the worker-local dense mask for the exact route probe.
  llvm::ArrayRef<std::uint64_t> eligibleTraversalBits;
  /// Stable input tokens enable validated worker-local reuse. Each owner must
  /// advance before the corresponding input view or any value in it changes.
  std::optional<EndpointRouteInputRevision> lowerBoundArcCostRevision;
  std::optional<EndpointRouteInputRevision> currentArcCostRevision;
  /// Empty means that no traversal class is mandatory. Otherwise the selected
  /// path must contain at least one traversal named by this dense mask. The
  /// search carries this one monotonic predicate as a two-state product graph;
  /// it does not change the routing topology or traversal costs.
  llvm::ArrayRef<std::uint64_t> requiredTraversalBits;
  /// A constrained branch search may begin at any node of an existing route
  /// tree, but may not leave one source and re-enter another source: doing so
  /// would satisfy a path predicate before the branch's actual divergence.
  bool forbidSourceReentry = false;
  /// Empty means no target requires the monotonic traversal predicate.
  /// Otherwise this canonical target-parallel array contains only zero or
  /// one, and a target is eligible only after the predicate is satisfied when
  /// its entry is one.
  llvm::ArrayRef<std::uint8_t> targetRequiresTraversal;
  /// Optional exact physical timing product. When present, every timing field
  /// is complete: arc delays and registered-destination flags are parallel to
  /// `arcs`, source arrivals are parallel to `sourceEndpoints`, target-local
  /// delays are parallel to `targetEndpoints`, and the required delay is
  /// positive. Search then retains nondominated
  /// `(endpoint, requirement, combinational arrival)` labels so negative slack
  /// and registered boundaries participate in path selection rather than
  /// being scored only after a route is chosen.
  bool physicalTimingEnabled = false;
  std::optional<EndpointRouteInputRevision> physicalTimingRevision;
  llvm::ArrayRef<std::uint64_t> arcTimingDelayQuanta;
  llvm::ArrayRef<std::uint8_t> arcTimingRegisteredDestination;
  llvm::ArrayRef<std::uint64_t> sourceTimingArrivalQuanta;
  llvm::ArrayRef<std::uint64_t> targetTimingDelayQuanta;
  std::uint64_t requiredTimingQuanta = 0;
  std::uint64_t timingCriticality = 0;
};

struct EndpointRouteSearchResult final {
  PnrIndex source = 0;
  PnrIndex target = 0;
  RouteCost cost = 0;
  llvm::ArrayRef<PnrIndex> forwardArcs;
};

class EndpointRouteSearchScratch final {
public:
  EndpointRouteSearchScratch() = default;
  EndpointRouteSearchScratch(const EndpointRouteSearchScratch &) = delete;
  EndpointRouteSearchScratch &
  operator=(const EndpointRouteSearchScratch &) = delete;
  EndpointRouteSearchScratch(EndpointRouteSearchScratch &&) = delete;
  EndpointRouteSearchScratch &operator=(EndpointRouteSearchScratch &&) = delete;
  ~EndpointRouteSearchScratch() = default;

  llvm::Error prepare(EndpointRoutingGraphView graph,
                      SpatialPnrWorkLedgerView workLedger = {});
  llvm::Expected<EndpointRouteSearchResult>
  search(const EndpointRouteSearchRequest &request);
  std::uint64_t endpointExpansionCount() const {
    return endpointExpansionCount_;
  }
  std::uint64_t heuristicCacheHitCount() const {
    return heuristicCacheHitCount_;
  }
  std::uint64_t heuristicBuildCount() const { return heuristicBuildCount_; }
  std::uint64_t forwardHeuristicQueryCount() const {
    return forwardHeuristicQueryCount_;
  }
  std::uint64_t forwardHeuristicUnreachableCount() const {
    return forwardHeuristicUnreachableCount_;
  }
  std::uint64_t heuristicCacheEvictionCount() const {
    return heuristicCacheEvictionCount_;
  }
  std::uint64_t heuristicComposeCount() const { return heuristicComposeCount_; }
  std::uint64_t arcCostValidationScanCount() const {
    return arcCostValidationScanCount_;
  }
  std::uint64_t physicalTimingValidationScanCount() const {
    return physicalTimingValidationScanCount_;
  }
  std::size_t heuristicCacheEntryCount() const {
    return heuristicCacheIndex_.size();
  }
  std::size_t heuristicCacheRetainedBytes() const;
  std::size_t retainedStorageBytes() const;

private:
  struct RouteQueueEntry final {
    RouteCost key = 0;
    PnrIndex state = 0;
    std::size_t next = std::numeric_limits<std::size_t>::max();
  };

  enum class HeapMode {
    ReverseDistance,
    ForwardAStar,
  };

  void resetRouteQueue();
  bool routeQueueEmpty();
  bool routeQueueEntryCurrent(const RouteQueueEntry &entry) const;
  bool routeQueueTieWorse(const RouteQueueEntry &lhs,
                          const RouteQueueEntry &rhs) const;
  bool refillRouteQueueMinimumBucket();
  void insertOrDecrease(PnrIndex endpoint);
  PnrIndex popMinimum();
  PnrIndex peekMinimum();

  void beginHeuristicGeneration();
  void beginSearchGeneration();
  void beginTargetGeneration();
  void beginSourceGeneration();
  RouteCost heuristic(PnrIndex endpoint) const;
  RouteCost queryForwardHeuristic(PnrIndex endpoint);
  RouteCost distance(PnrIndex searchState) const;
  PnrIndex searchState(PnrIndex endpoint, bool requirementMet) const;
  PnrIndex searchEndpoint(PnrIndex searchState) const;
  bool searchRequirementMet(PnrIndex searchState) const;
  bool isTarget(PnrIndex endpoint) const;
  bool targetRequiresTraversal(PnrIndex endpoint) const;
  bool isSource(PnrIndex endpoint) const;
  PnrIndex targetPreferenceRank(PnrIndex endpoint) const;
  llvm::Expected<RouteCost>
  searchArcCost(const EndpointRouteSearchRequest &request, PnrIndex arc,
                bool current);
  bool arcEligible(PnrIndex arc, const EndpointRouteSearchRequest &request,
                   bool enforceSourceReplication) const;
  llvm::Error buildHeuristic(const EndpointRouteSearchRequest &request);
  /// Composes the multi-target heuristic as the elementwise minimum of
  /// cached singleton-target rows: the reverse shortest distance to a target
  /// set is exactly the minimum of the singleton distances, so composition
  /// reproduces buildHeuristic() value for value. Builds and caches missing
  /// singleton rows first; returns false when the request is outside the
  /// composition bounds or a singleton row could not be retained.
  llvm::Expected<bool>
  composeHeuristicFromSingletons(const EndpointRouteSearchRequest &request);
  llvm::Expected<EndpointRouteSearchResult>
  searchTimingAware(const EndpointRouteSearchRequest &request);
  bool loadCachedHeuristic(const EndpointRouteSearchRequest &request);
  void storeCachedHeuristic(const EndpointRouteSearchRequest &request);
  bool revisionIsCurrent(const EndpointRouteInputRevision &revision) const;
  bool
  arcCostsAlreadyValidated(const EndpointRouteSearchRequest &request) const;
  void rememberValidatedArcCosts(const EndpointRouteSearchRequest &request);
  bool physicalTimingAlreadyValidated(
      const EndpointRouteSearchRequest &request) const;
  void
  rememberValidatedPhysicalTiming(const EndpointRouteSearchRequest &request);
  bool
  heuristicInputsAreCacheable(const EndpointRouteSearchRequest &request) const;

  struct HeuristicCacheWideDistance final {
    PnrIndex endpoint = 0;
    RouteCost distance = 0;
  };

  struct HeuristicCacheEntry final {
    std::array<std::uint8_t, 32> keyDigest{};
    std::vector<std::uint32_t> scaledDistances;
    std::vector<HeuristicCacheWideDistance> wideDistances;
    std::uint64_t lastUse = 0;
    std::uint8_t scaleShift = 0;
    bool populated = false;
  };

  struct ValidatedArcCosts final {
    EndpointRouteInputRevision::Generation lowerBoundGeneration;
    EndpointRouteInputRevision::Generation currentGeneration;
    bool populated = false;
  };

  struct ValidatedPhysicalTiming final {
    EndpointRouteInputRevision::Generation generation;
    bool populated = false;
  };

  struct HeuristicCacheDigestHash final {
    std::size_t
    operator()(const std::array<std::uint8_t, 32> &digest) const noexcept {
      std::uint64_t hash = 1469598103934665603ULL;
      for (std::uint8_t byte : digest) {
        hash ^= byte;
        hash *= 1099511628211ULL;
      }
      return static_cast<std::size_t>(hash);
    }
  };

  RouteCost cachedHeuristic(const HeuristicCacheEntry &entry,
                            PnrIndex endpoint) const;
  std::size_t
  heuristicCacheEntryDistanceBytes(const HeuristicCacheEntry &entry) const;
  void evictHeuristicCacheEntry(std::size_t slot);

  struct TimingSearchLabel final {
    PnrIndex endpoint = 0;
    PnrIndex predecessorLabel = 0;
    PnrIndex predecessorArc = 0;
    PnrIndex nextStateLabel = 0;
    std::uint64_t arrivalQuanta = 0;
    RouteCost distance = 0;
    RouteCost priority = 0;
    bool requirementMet = false;
    bool active = false;
  };

  std::array<std::uint8_t, 32>
  heuristicCacheKeyDigest(const EndpointRouteSearchRequest &request) const;
  std::array<std::uint8_t, 32>
  eligibleTraversalMaskDigest(const EndpointRouteSearchRequest &request) const;

  EndpointRoutingGraphView graph_;
  std::vector<RouteCost> heuristics_;
  std::vector<RouteCost> distances_;
  std::vector<RouteCost> priorities_;
  std::vector<PnrIndex> predecessorArcs_;
  std::vector<PnrIndex> predecessorStates_;
  std::vector<std::uint64_t> heuristicEpochs_;
  std::vector<std::uint64_t> distanceEpochs_;
  std::vector<std::uint64_t> targetEpochs_;
  std::vector<std::uint64_t> sourceEpochs_;
  std::vector<PnrIndex> targetPreferenceRanks_;
  std::vector<std::uint8_t> targetRequiresTraversal_;
  std::vector<PnrIndex> sourceReplicationGroups_;
  std::array<std::size_t, 65> routeQueueBucketHeads_{};
  std::vector<RouteQueueEntry> routeQueueEntries_;
  std::size_t routeQueueEntryCount_ = 0;
  std::vector<std::size_t> routeQueueMinimumHeap_;
  RouteCost routeQueueLastKey_ = 0;
  std::vector<PnrIndex> path_;
  std::vector<TimingSearchLabel> timingLabels_;
  std::vector<PnrIndex> timingStateLabelHeads_;
  std::vector<std::uint64_t> timingStateLabelEpochs_;
  std::vector<PnrIndex> timingHeap_;
  std::vector<RouteCost> timingArcCosts_;
  std::vector<std::uint64_t> timingArcCostEpochs_;
  std::vector<HeuristicCacheEntry> heuristicCache_;
  std::unordered_map<std::array<std::uint8_t, 32>, std::size_t,
                     HeuristicCacheDigestHash>
      heuristicCacheIndex_;
  const HeuristicCacheEntry *activeCachedHeuristic_ = nullptr;
  std::size_t heuristicCacheDistanceByteBudget_ = 0;
  std::size_t heuristicCacheDistanceBytes_ = 0;
  std::uint64_t heuristicCacheUseEpoch_ = 0;
  mutable std::vector<std::uint64_t> eligibleTraversalMaskSnapshot_;
  mutable std::array<std::uint8_t, 32> eligibleTraversalMaskDigest_{};
  mutable bool eligibleTraversalMaskDigestValid_ = false;
  std::uint64_t heuristicGeneration_ = 0;
  std::uint64_t searchGeneration_ = 0;
  std::uint64_t targetGeneration_ = 0;
  std::uint64_t sourceGeneration_ = 0;
  std::uint64_t timingLabelGeneration_ = 0;
  std::uint64_t timingArcCostGeneration_ = 0;
  std::uint64_t endpointExpansionCount_ = 0;
  std::uint64_t heuristicCacheHitCount_ = 0;
  std::uint64_t heuristicBuildCount_ = 0;
  std::uint64_t forwardHeuristicQueryCount_ = 0;
  std::uint64_t forwardHeuristicUnreachableCount_ = 0;
  std::uint64_t heuristicCacheEvictionCount_ = 0;
  std::uint64_t heuristicComposeCount_ = 0;
  std::uint64_t arcCostValidationScanCount_ = 0;
  std::uint64_t physicalTimingValidationScanCount_ = 0;
  SpatialPnrWorkLedgerView workLedger_;
  ValidatedArcCosts validatedArcCosts_;
  ValidatedPhysicalTiming validatedPhysicalTiming_;
  HeapMode heapMode_ = HeapMode::ReverseDistance;
  bool prepared_ = false;
};

} // namespace loom::pnr

#endif // LOOM_PNR_ENDPOINTROUTER_H
