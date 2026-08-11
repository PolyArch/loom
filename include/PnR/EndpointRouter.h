#ifndef LOOM_PNR_ENDPOINTROUTER_H
#define LOOM_PNR_ENDPOINTROUTER_H

#include "PnR/EndpointRoutingTopology.h"
#include "PnR/RoutingNegotiation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace loom::pnr {

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
  /// Enables exact worker-local heuristic reuse. The owner must increment the
  /// revision whenever any lower-bound arc cost changes in place.
  std::optional<std::uint64_t> lowerBoundCostRevision;
  /// Empty means that no traversal class is mandatory. Otherwise the selected
  /// path must contain at least one traversal named by this dense mask. The
  /// search carries this one monotonic predicate as a two-state product graph;
  /// it does not change the routing topology or traversal costs.
  llvm::ArrayRef<std::uint64_t> requiredTraversalBits;
  /// A constrained branch search may begin at any node of an existing route
  /// tree, but may not leave one source and re-enter another source: doing so
  /// would satisfy a path predicate before the branch's actual divergence.
  bool forbidSourceReentry = false;
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

  llvm::Error prepare(EndpointRoutingGraphView graph);
  llvm::Expected<EndpointRouteSearchResult>
  search(const EndpointRouteSearchRequest &request);
  std::uint64_t endpointExpansionCount() const {
    return endpointExpansionCount_;
  }
  std::uint64_t heuristicCacheHitCount() const {
    return heuristicCacheHitCount_;
  }
  std::uint64_t heuristicBuildCount() const { return heuristicBuildCount_; }
  std::size_t retainedStorageBytes() const;

private:
  enum class HeapMode {
    ReverseDistance,
    ForwardAStar,
  };

  void resetHeap();
  bool heapLess(PnrIndex lhs, PnrIndex rhs) const;
  void heapSwap(std::size_t lhs, std::size_t rhs);
  void siftUp(std::size_t position);
  void siftDown(std::size_t position);
  void insertOrDecrease(PnrIndex endpoint);
  PnrIndex popMinimum();
  PnrIndex peekMinimum() const;

  void beginHeuristicGeneration();
  void beginSearchGeneration();
  void beginTargetGeneration();
  void beginSourceGeneration();
  RouteCost heuristic(PnrIndex endpoint) const;
  RouteCost distance(PnrIndex searchState) const;
  PnrIndex searchState(PnrIndex endpoint, bool requirementMet) const;
  PnrIndex searchEndpoint(PnrIndex searchState) const;
  bool searchRequirementMet(PnrIndex searchState) const;
  bool isTarget(PnrIndex endpoint) const;
  bool isSource(PnrIndex endpoint) const;
  PnrIndex targetPreferenceRank(PnrIndex endpoint) const;
  bool arcEligible(PnrIndex arc, const EndpointRouteSearchRequest &request,
                   bool enforceSourceReplication) const;
  llvm::Error buildHeuristic(const EndpointRouteSearchRequest &request);
  bool loadCachedHeuristic(const EndpointRouteSearchRequest &request);
  void storeCachedHeuristic(const EndpointRouteSearchRequest &request);

  struct HeuristicCacheEntry final {
    const RouteCost *lowerBoundCostData = nullptr;
    std::size_t lowerBoundCostSize = 0;
    std::uint64_t lowerBoundCostRevision = 0;
    std::uint64_t keyHash = 0;
    std::uint32_t requiredPayloadWidthBits = 0;
    std::uint32_t requiredTagWidthBits = 0;
    std::size_t targetEndpointCount = 0;
    std::size_t eligibleTraversalWordCount = 0;
    bool populated = false;
  };

  std::uint64_t
  heuristicCacheKeyHash(const EndpointRouteSearchRequest &request) const;
  bool heuristicCacheKeyEquals(const HeuristicCacheEntry &entry,
                               const EndpointRouteSearchRequest &request,
                               std::uint64_t keyHash, std::size_t slot) const;

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
  std::vector<PnrIndex> sourceReplicationGroups_;
  std::vector<PnrIndex> heap_;
  std::vector<PnrIndex> heapPositions_;
  std::vector<PnrIndex> path_;
  std::vector<HeuristicCacheEntry> heuristicCache_;
  std::vector<PnrIndex> heuristicCacheTargets_;
  std::vector<std::uint64_t> heuristicCacheEligibility_;
  std::vector<RouteCost> heuristicCacheDistances_;
  std::size_t heuristicCacheTraversalWordCount_ = 0;
  const RouteCost *activeCachedHeuristics_ = nullptr;
  std::uint64_t heuristicGeneration_ = 0;
  std::uint64_t searchGeneration_ = 0;
  std::uint64_t targetGeneration_ = 0;
  std::uint64_t sourceGeneration_ = 0;
  std::uint64_t endpointExpansionCount_ = 0;
  std::uint64_t heuristicCacheHitCount_ = 0;
  std::uint64_t heuristicBuildCount_ = 0;
  HeapMode heapMode_ = HeapMode::ReverseDistance;
  bool prepared_ = false;
};

} // namespace loom::pnr

#endif // LOOM_PNR_ENDPOINTROUTER_H
