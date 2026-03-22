#include "loom/Mapper/TopologyModel.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>

namespace loom {

namespace {

bool isPlaceableHardwareNode(const Node *hwNode) {
  if (!hwNode)
    return false;
  llvm::StringRef resourceClass = getNodeAttrStr(hwNode, "resource_class");
  return resourceClass == "functional" || resourceClass == "memory";
}

void deduplicateNeighbors(std::vector<std::vector<IdIndex>> &neighbors) {
  for (auto &adj : neighbors) {
    std::sort(adj.begin(), adj.end());
    adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
  }
}

void runBreadthFirstSearch(
    IdIndex source, const std::vector<std::vector<IdIndex>> &neighbors,
    std::vector<unsigned> &distances, unsigned unreachableDistance) {
  if (source >= static_cast<IdIndex>(neighbors.size()))
    return;

  std::queue<IdIndex> worklist;
  distances[source] = 0;
  worklist.push(source);

  while (!worklist.empty()) {
    IdIndex current = worklist.front();
    worklist.pop();
    unsigned baseDistance = distances[current];
    for (IdIndex next : neighbors[current]) {
      if (next >= static_cast<IdIndex>(neighbors.size()))
        continue;
      if (distances[next] != unreachableDistance)
        continue;
      distances[next] = baseDistance + 1U;
      worklist.push(next);
    }
  }
}

} // namespace

TopologyModel::TopologyModel(const Graph &adg, const ADGFlattener &flattener)
    : nodeCount_(static_cast<unsigned>(adg.nodes.size())),
      fallbackDistance_(std::max<unsigned>(1U, nodeCount_ * 4U)) {
  outgoingNeighbors_.assign(nodeCount_, {});
  undirectedNeighbors_.assign(nodeCount_, {});
  directedDistances_.assign(static_cast<size_t>(nodeCount_) * nodeCount_,
                            kUnreachableDistance);
  undirectedDistances_.assign(static_cast<size_t>(nodeCount_) * nodeCount_,
                              kUnreachableDistance);
  averagePlacementDistances_.assign(nodeCount_,
                                    static_cast<double>(fallbackDistance_));

  buildAdjacency(adg);
  buildDistanceTables(adg);
  inferGridTraits(adg, flattener);
}

size_t TopologyModel::flatIndex(IdIndex srcNode, IdIndex dstNode) const {
  return static_cast<size_t>(srcNode) * nodeCount_ + static_cast<size_t>(dstNode);
}

void TopologyModel::buildAdjacency(const Graph &adg) {
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(adg.edges.size());
       ++edgeId) {
    const Edge *edge = adg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID) {
      continue;
    }

    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    if (srcNode == dstNode || srcNode >= static_cast<IdIndex>(nodeCount_) ||
        dstNode >= static_cast<IdIndex>(nodeCount_)) {
      continue;
    }

    outgoingNeighbors_[srcNode].push_back(dstNode);
    undirectedNeighbors_[srcNode].push_back(dstNode);
    undirectedNeighbors_[dstNode].push_back(srcNode);
  }

  deduplicateNeighbors(outgoingNeighbors_);
  deduplicateNeighbors(undirectedNeighbors_);
}

void TopologyModel::buildDistanceTables(const Graph &adg) {
  placeableNodes_.clear();
  placeableNodes_.reserve(adg.nodes.size());
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    if (isPlaceableHardwareNode(adg.getNode(nodeId)))
      placeableNodes_.push_back(nodeId);
  }

  for (IdIndex source = 0; source < static_cast<IdIndex>(nodeCount_); ++source) {
    std::vector<unsigned> directed(nodeCount_, kUnreachableDistance);
    std::vector<unsigned> undirected(nodeCount_, kUnreachableDistance);
    runBreadthFirstSearch(source, outgoingNeighbors_, directed,
                          kUnreachableDistance);
    runBreadthFirstSearch(source, undirectedNeighbors_, undirected,
                          kUnreachableDistance);

    for (IdIndex dst = 0; dst < static_cast<IdIndex>(nodeCount_); ++dst) {
      directedDistances_[flatIndex(source, dst)] = directed[dst];
      undirectedDistances_[flatIndex(source, dst)] = undirected[dst];
    }

    double weightedSum = 0.0;
    unsigned weightedCount = 0;
    for (IdIndex dst : placeableNodes_) {
      if (dst == source)
        continue;
      unsigned distance = undirected[dst];
      if (distance == kUnreachableDistance)
        distance = fallbackDistance_;
      weightedSum += static_cast<double>(distance);
      ++weightedCount;
    }
    if (weightedCount != 0)
      averagePlacementDistances_[source] = weightedSum / weightedCount;
  }
}

void TopologyModel::inferGridTraits(const Graph &adg,
                                    const ADGFlattener &flattener) {
  std::set<int> rows;
  std::set<int> cols;
  unsigned placeableNodes = 0;
  unsigned placeableWithGrid = 0;

  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *hwNode = adg.getNode(nodeId);
    if (!isPlaceableHardwareNode(hwNode))
      continue;
    ++placeableNodes;
    auto [row, col] = flattener.getNodeGridPos(nodeId);
    if (row < 0 || col < 0)
      continue;
    ++placeableWithGrid;
    rows.insert(row);
    cols.insert(col);
  }

  supportsGridCutLoad_ = placeableNodes != 0 &&
                         placeableWithGrid == placeableNodes &&
                         rows.size() > 1 && cols.size() > 1;
}

unsigned TopologyModel::directedNodeDistance(IdIndex srcNode,
                                             IdIndex dstNode) const {
  if (srcNode == INVALID_ID || dstNode == INVALID_ID ||
      srcNode >= static_cast<IdIndex>(nodeCount_) ||
      dstNode >= static_cast<IdIndex>(nodeCount_)) {
    return fallbackDistance_;
  }
  unsigned distance = directedDistances_[flatIndex(srcNode, dstNode)];
  return distance == kUnreachableDistance ? fallbackDistance_ : distance;
}

unsigned TopologyModel::undirectedNodeDistance(IdIndex lhsNode,
                                               IdIndex rhsNode) const {
  if (lhsNode == INVALID_ID || rhsNode == INVALID_ID ||
      lhsNode >= static_cast<IdIndex>(nodeCount_) ||
      rhsNode >= static_cast<IdIndex>(nodeCount_)) {
    return fallbackDistance_;
  }
  unsigned distance = undirectedDistances_[flatIndex(lhsNode, rhsNode)];
  return distance == kUnreachableDistance ? fallbackDistance_ : distance;
}

unsigned TopologyModel::placementDistance(IdIndex srcNode, IdIndex dstNode) const {
  if (srcNode == INVALID_ID || dstNode == INVALID_ID ||
      srcNode >= static_cast<IdIndex>(nodeCount_) ||
      dstNode >= static_cast<IdIndex>(nodeCount_)) {
    return fallbackDistance_;
  }
  unsigned directed = directedDistances_[flatIndex(srcNode, dstNode)];
  if (directed != kUnreachableDistance)
    return directed;
  unsigned undirected = undirectedDistances_[flatIndex(srcNode, dstNode)];
  return undirected == kUnreachableDistance ? fallbackDistance_ : undirected;
}

bool TopologyModel::isWithinMoveRadius(IdIndex lhsNode, IdIndex rhsNode,
                                       unsigned radius) const {
  if (radius == 0)
    return true;
  return undirectedNodeDistance(lhsNode, rhsNode) <= radius;
}

std::vector<IdIndex>
TopologyModel::placeableNodesWithinRadius(IdIndex centerNode,
                                          unsigned radius) const {
  std::vector<IdIndex> nearbyNodes;
  if (centerNode == INVALID_ID || centerNode >= static_cast<IdIndex>(nodeCount_))
    return nearbyNodes;
  nearbyNodes.reserve(placeableNodes_.size());
  for (IdIndex hwNode : placeableNodes_) {
    if (hwNode == centerNode)
      continue;
    if (!isWithinMoveRadius(centerNode, hwNode, radius))
      continue;
    nearbyNodes.push_back(hwNode);
  }
  return nearbyNodes;
}

double TopologyModel::averagePlacementDistance(IdIndex hwNode) const {
  if (hwNode == INVALID_ID || hwNode >= static_cast<IdIndex>(nodeCount_))
    return static_cast<double>(fallbackDistance_);
  return averagePlacementDistances_[hwNode];
}

} // namespace loom
