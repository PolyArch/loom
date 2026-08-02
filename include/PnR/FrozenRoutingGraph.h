#ifndef LOOM_PNR_FROZENROUTINGGRAPH_H
#define LOOM_PNR_FROZENROUTINGGRAPH_H

#include "Mapping/Artifact.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::pnr {

struct PnrProblemInputs;

namespace detail {
class FrozenModelBuilder;
} // namespace detail

enum class FrozenRoutingEndpointOwnerKind {
  ComputeOccurrence,
  MemoryOccurrence,
  TransportResource,
};

enum class FrozenRoutingArcKind {
  PointToPoint,
  Traversal,
};

struct FrozenTransportResource {
  mapping::TransportResourceId id;
  mapping::TransportResourceKind kind;
  std::optional<::fabric::BoundaryDirection> boundaryDirection;
  PnrIndex endpointOffset;
  PnrIndex endpointCount;

  friend bool operator==(const FrozenTransportResource &lhs,
                         const FrozenTransportResource &rhs) {
    return lhs.id == rhs.id && lhs.kind == rhs.kind &&
           lhs.boundaryDirection == rhs.boundaryDirection &&
           lhs.endpointOffset == rhs.endpointOffset &&
           lhs.endpointCount == rhs.endpointCount;
  }
};

struct FrozenRoutingEndpoint {
  mapping::TransportEndpointId id;
  FrozenRoutingEndpointOwnerKind ownerKind;
  PnrIndex owner;
  mapping::PortDirection direction;
  mapping::PortKind portKind;
  ::fabric::DataPathKind transportKind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;

  friend bool operator==(const FrozenRoutingEndpoint &lhs,
                         const FrozenRoutingEndpoint &rhs) {
    return lhs.id == rhs.id && lhs.ownerKind == rhs.ownerKind &&
           lhs.owner == rhs.owner && lhs.direction == rhs.direction &&
           lhs.portKind == rhs.portKind &&
           lhs.transportKind == rhs.transportKind &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits;
  }
};

struct FrozenRoutingArc {
  PnrIndex target;
  FrozenRoutingArcKind kind;
  std::optional<PnrIndex> resource;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;

  friend bool operator==(const FrozenRoutingArc &lhs,
                         const FrozenRoutingArc &rhs) {
    return lhs.target == rhs.target && lhs.kind == rhs.kind &&
           lhs.resource == rhs.resource &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits;
  }
};

class FrozenRoutingReachabilityScratch {
public:
  bool contains(PnrIndex endpoint) const {
    return generation_ != 0 && endpoint < reachedGeneration_.size() &&
           reachedGeneration_[endpoint] == generation_;
  }

private:
  friend class FrozenRoutingGraph;

  std::vector<std::uint64_t> reachedGeneration_;
  std::vector<PnrIndex> worklist_;
  std::uint64_t generation_ = 0;
};

class FrozenRoutingGraph {
public:
  FrozenRoutingGraph(const FrozenRoutingGraph &) = delete;
  FrozenRoutingGraph(FrozenRoutingGraph &&) = default;
  FrozenRoutingGraph &operator=(const FrozenRoutingGraph &) = delete;
  FrozenRoutingGraph &operator=(FrozenRoutingGraph &&) = delete;

  llvm::ArrayRef<FrozenTransportResource> transportResources() const {
    return transportResources_;
  }
  llvm::ArrayRef<PnrIndex> resourceEndpointVertices() const {
    return resourceEndpointVertices_;
  }
  llvm::ArrayRef<FrozenRoutingEndpoint> routingEndpoints() const {
    return routingEndpoints_;
  }
  llvm::ArrayRef<PnrIndex> computeEndpointVertices() const {
    return computeEndpointVertices_;
  }
  llvm::ArrayRef<PnrIndex> memoryEndpointVertices() const {
    return memoryEndpointVertices_;
  }
  llvm::ArrayRef<PnrIndex> adjacencyOffsets() const {
    return adjacencyOffsets_;
  }
  llvm::ArrayRef<FrozenRoutingArc> routingArcs() const { return routingArcs_; }
  llvm::ArrayRef<PnrIndex> incomingAdjacencyOffsets() const {
    return incomingAdjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> incomingSourceVertices() const {
    return incomingSourceVertices_;
  }
  llvm::ArrayRef<PnrIndex> incomingForwardArcIndices() const {
    return incomingForwardArcIndices_;
  }
  void computeCompatibleReachability(
      PnrIndex source, mapping::PortKind portKind,
      std::uint32_t payloadWidthBits, std::uint32_t tagWidthBits,
      FrozenRoutingReachabilityScratch &scratch) const;

  friend bool operator==(const FrozenRoutingGraph &lhs,
                         const FrozenRoutingGraph &rhs) {
    return lhs.transportResources_ == rhs.transportResources_ &&
           lhs.resourceEndpointVertices_ == rhs.resourceEndpointVertices_ &&
           lhs.routingEndpoints_ == rhs.routingEndpoints_ &&
           lhs.computeEndpointVertices_ == rhs.computeEndpointVertices_ &&
           lhs.memoryEndpointVertices_ == rhs.memoryEndpointVertices_ &&
           lhs.adjacencyOffsets_ == rhs.adjacencyOffsets_ &&
           lhs.routingArcs_ == rhs.routingArcs_ &&
           lhs.incomingAdjacencyOffsets_ == rhs.incomingAdjacencyOffsets_ &&
           lhs.incomingSourceVertices_ == rhs.incomingSourceVertices_ &&
           lhs.incomingForwardArcIndices_ == rhs.incomingForwardArcIndices_;
  }
  friend bool operator!=(const FrozenRoutingGraph &lhs,
                         const FrozenRoutingGraph &rhs) {
    return !(lhs == rhs);
  }

private:
  FrozenRoutingGraph(std::vector<FrozenTransportResource> transportResources,
                     std::vector<PnrIndex> resourceEndpointVertices,
                     std::vector<FrozenRoutingEndpoint> routingEndpoints,
                     std::vector<PnrIndex> computeEndpointVertices,
                     std::vector<PnrIndex> memoryEndpointVertices,
                     std::vector<PnrIndex> adjacencyOffsets,
                     std::vector<FrozenRoutingArc> routingArcs,
                     std::vector<PnrIndex> incomingAdjacencyOffsets,
                     std::vector<PnrIndex> incomingSourceVertices,
                     std::vector<PnrIndex> incomingForwardArcIndices)
      : transportResources_(std::move(transportResources)),
        resourceEndpointVertices_(std::move(resourceEndpointVertices)),
        routingEndpoints_(std::move(routingEndpoints)),
        computeEndpointVertices_(std::move(computeEndpointVertices)),
        memoryEndpointVertices_(std::move(memoryEndpointVertices)),
        adjacencyOffsets_(std::move(adjacencyOffsets)),
        routingArcs_(std::move(routingArcs)),
        incomingAdjacencyOffsets_(std::move(incomingAdjacencyOffsets)),
        incomingSourceVertices_(std::move(incomingSourceVertices)),
        incomingForwardArcIndices_(std::move(incomingForwardArcIndices)) {}

  std::vector<FrozenTransportResource> transportResources_;
  std::vector<PnrIndex> resourceEndpointVertices_;
  std::vector<FrozenRoutingEndpoint> routingEndpoints_;
  std::vector<PnrIndex> computeEndpointVertices_;
  std::vector<PnrIndex> memoryEndpointVertices_;
  std::vector<PnrIndex> adjacencyOffsets_;
  std::vector<FrozenRoutingArc> routingArcs_;
  std::vector<PnrIndex> incomingAdjacencyOffsets_;
  std::vector<PnrIndex> incomingSourceVertices_;
  std::vector<PnrIndex> incomingForwardArcIndices_;

  friend class detail::FrozenModelBuilder;
};

namespace detail {

llvm::Error preflightFrozenRoutingGraphCapacity(
    std::uint64_t endpointCount, std::uint64_t resourceCount,
    std::uint64_t computeEndpointCount, std::uint64_t arcCount,
    std::uint64_t memoryEndpointCount = 0);

} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENROUTINGGRAPH_H
