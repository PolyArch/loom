#ifndef LOOM_MAPPER_TOPOLOGYMODEL_H
#define LOOM_MAPPER_TOPOLOGYMODEL_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"

#include <cstdint>
#include <vector>

namespace loom {

class TopologyModel {
public:
  static constexpr unsigned kUnreachableDistance =
      static_cast<unsigned>(-1) / 4U;

  TopologyModel() = default;
  TopologyModel(const Graph &adg, const ADGFlattener &flattener);

  unsigned directedNodeDistance(IdIndex srcNode, IdIndex dstNode) const;
  unsigned undirectedNodeDistance(IdIndex lhsNode, IdIndex rhsNode) const;
  unsigned placementDistance(IdIndex srcNode, IdIndex dstNode) const;
  bool isWithinMoveRadius(IdIndex lhsNode, IdIndex rhsNode,
                          unsigned radius) const;
  std::vector<IdIndex> placeableNodesWithinRadius(IdIndex centerNode,
                                                  unsigned radius) const;

  double averagePlacementDistance(IdIndex hwNode) const;
  bool supportsGridCutLoad() const { return supportsGridCutLoad_; }

private:
  unsigned nodeCount_ = 0;
  unsigned fallbackDistance_ = 1;
  bool supportsGridCutLoad_ = false;

  std::vector<std::vector<IdIndex>> outgoingNeighbors_;
  std::vector<std::vector<IdIndex>> undirectedNeighbors_;
  std::vector<unsigned> directedDistances_;
  std::vector<unsigned> undirectedDistances_;
  std::vector<double> averagePlacementDistances_;
  std::vector<IdIndex> placeableNodes_;

  size_t flatIndex(IdIndex srcNode, IdIndex dstNode) const;
  void buildAdjacency(const Graph &adg);
  void buildDistanceTables(const Graph &adg);
  void inferGridTraits(const Graph &adg, const ADGFlattener &flattener);
};

} // namespace loom

#endif // LOOM_MAPPER_TOPOLOGYMODEL_H
