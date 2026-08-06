#ifndef LOOM_PNR_ENDPOINTROUTINGTOPOLOGY_H
#define LOOM_PNR_ENDPOINTROUTINGTOPOLOGY_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr {

struct EndpointRoutingEndpoint final {
  ::loom::fabric::FabricTransportEndpointRef reference;
  ::loom::fabric::FabricPortDirection direction;
  ::fabric::DataPathType dataPath;
};

struct EndpointRoutingTraversal final {
  ::loom::fabric::FabricPhysicalTraversalRef reference;
  PnrIndex sourceOffset = 0;
  PnrIndex sourceCount = 0;
  PnrIndex destinationOffset = 0;
  PnrIndex destinationCount = 0;
};

struct EndpointRoutingArc final {
  PnrIndex target = 0;
  PnrIndex traversal = 0;
  std::uint32_t payloadCapacityBits = 0;
  std::uint32_t tagCapacityBits = 0;
};

/// Rebuildable dense routing projection of one exact Fabric root. Fabric
/// remains the owner of endpoints, traversals, widths, and replication.
class FrozenEndpointRoutingTopology final {
public:
  llvm::ArrayRef<EndpointRoutingEndpoint> endpoints() const {
    return endpoints_;
  }
  llvm::ArrayRef<EndpointRoutingTraversal> traversals() const {
    return traversals_;
  }
  llvm::ArrayRef<PnrIndex> traversalEndpoints() const {
    return traversalEndpoints_;
  }
  llvm::ArrayRef<PnrIndex> traversalReplicationGroups() const {
    return traversalReplicationGroups_;
  }
  llvm::ArrayRef<EndpointRoutingArc> arcs() const { return arcs_; }
  llvm::ArrayRef<PnrIndex> arcSources() const { return arcSources_; }
  llvm::ArrayRef<PnrIndex> adjacencyOffsets() const {
    return adjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets() const {
    return reverseAdjacencyOffsets_;
  }
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals() const {
    return reverseArcOrdinals_;
  }

private:
  std::vector<EndpointRoutingEndpoint> endpoints_;
  std::vector<EndpointRoutingTraversal> traversals_;
  std::vector<PnrIndex> traversalEndpoints_;
  std::vector<PnrIndex> traversalReplicationGroups_;
  std::vector<EndpointRoutingArc> arcs_;
  std::vector<PnrIndex> arcSources_;
  std::vector<PnrIndex> adjacencyOffsets_;
  std::vector<PnrIndex> reverseAdjacencyOffsets_;
  std::vector<PnrIndex> reverseArcOrdinals_;

  friend llvm::Expected<FrozenEndpointRoutingTopology>
  freezeEndpointRoutingTopology(
      const ::loom::fabric::FabricArtifactView &fabric);
};

llvm::Expected<FrozenEndpointRoutingTopology>
freezeEndpointRoutingTopology(const ::loom::fabric::FabricArtifactView &fabric);

} // namespace loom::pnr

#endif // LOOM_PNR_ENDPOINTROUTINGTOPOLOGY_H
