#ifndef LOOM_PNR_ENDPOINTROUTER_H
#define LOOM_PNR_ENDPOINTROUTER_H

#include "PnR/RoutingNegotiation.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace loom::pnr {

struct EndpointRoutingGraphView final {
  PnrIndex endpointCount = 0;
  llvm::ArrayRef<FrozenSpatialRoutingArc> arcs;
  llvm::ArrayRef<PnrIndex> arcSources;
  llvm::ArrayRef<PnrIndex> adjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals;
  llvm::ArrayRef<PnrIndex> traversalReplicationGroups;
};

EndpointRoutingGraphView
endpointRoutingGraphView(const FrozenSpatialRoutingGraph &graph);

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
  RouteCost distance(PnrIndex endpoint) const;
  bool isTarget(PnrIndex endpoint) const;
  bool isSource(PnrIndex endpoint) const;
  PnrIndex targetPreferenceRank(PnrIndex endpoint) const;
  bool arcEligible(PnrIndex arc, const EndpointRouteSearchRequest &request,
                   bool enforceSourceReplication) const;
  llvm::Error buildHeuristic(const EndpointRouteSearchRequest &request);

  EndpointRoutingGraphView graph_;
  std::vector<RouteCost> heuristics_;
  std::vector<RouteCost> distances_;
  std::vector<RouteCost> priorities_;
  std::vector<PnrIndex> predecessorArcs_;
  std::vector<std::uint64_t> heuristicEpochs_;
  std::vector<std::uint64_t> distanceEpochs_;
  std::vector<std::uint64_t> targetEpochs_;
  std::vector<std::uint64_t> sourceEpochs_;
  std::vector<PnrIndex> targetPreferenceRanks_;
  std::vector<PnrIndex> sourceReplicationGroups_;
  std::vector<PnrIndex> heap_;
  std::vector<PnrIndex> heapPositions_;
  std::vector<PnrIndex> path_;
  std::uint64_t heuristicGeneration_ = 0;
  std::uint64_t searchGeneration_ = 0;
  std::uint64_t targetGeneration_ = 0;
  std::uint64_t sourceGeneration_ = 0;
  HeapMode heapMode_ = HeapMode::ReverseDistance;
  bool prepared_ = false;
};

} // namespace loom::pnr

#endif // LOOM_PNR_ENDPOINTROUTER_H
