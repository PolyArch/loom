//===-- mapping_state_map_edge.cpp - Edge map/unmap + reverse mapping --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify mapEdge/unmapEdge maintain hwEdgeToSwEdges reverse mapping.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  // DFG: 2 nodes, 2 ports, 1 edge.
  auto sw0 = std::make_unique<Node>();
  sw0->kind = Node::OperationNode;
  dfg.addNode(std::move(sw0));

  auto sw1 = std::make_unique<Node>();
  sw1->kind = Node::OperationNode;
  dfg.addNode(std::move(sw1));

  auto swOut = std::make_unique<Port>();
  swOut->parentNode = 0;
  swOut->direction = Port::Output;
  dfg.addPort(std::move(swOut));

  auto swIn = std::make_unique<Port>();
  swIn->parentNode = 1;
  swIn->direction = Port::Input;
  dfg.addPort(std::move(swIn));

  auto swEdge = std::make_unique<Edge>();
  swEdge->srcPort = 0;
  swEdge->dstPort = 1;
  IdIndex swEdgeId = dfg.addEdge(std::move(swEdge));
  dfg.getPort(0)->connectedEdges.push_back(swEdgeId);
  dfg.getPort(1)->connectedEdges.push_back(swEdgeId);

  // ADG: 2 nodes, 2 ports, 1 physical edge (port 0 -> port 1).
  auto hw0 = std::make_unique<Node>();
  adg.addNode(std::move(hw0));

  auto hw1 = std::make_unique<Node>();
  adg.addNode(std::move(hw1));

  auto hwOut = std::make_unique<Port>();
  hwOut->parentNode = 0;
  hwOut->direction = Port::Output;
  adg.addPort(std::move(hwOut));

  auto hwIn = std::make_unique<Port>();
  hwIn->parentNode = 1;
  hwIn->direction = Port::Input;
  adg.addPort(std::move(hwIn));

  auto hwEdge = std::make_unique<Edge>();
  hwEdge->srcPort = 0;
  hwEdge->dstPort = 1;
  IdIndex hwEdgeId = adg.addEdge(std::move(hwEdge));
  adg.getPort(0)->connectedEdges.push_back(hwEdgeId);
  adg.getPort(1)->connectedEdges.push_back(hwEdgeId);

  MappingState state;
  state.init(dfg, adg);

  // Map the SW edge along the path [hwOut=0, hwIn=1].
  llvm::SmallVector<IdIndex, 8> path = {0, 1};
  auto r = state.mapEdge(swEdgeId, path, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);

  // Verify forward mapping.
  TEST_ASSERT(state.swEdgeToHwPaths[swEdgeId].size() == 2);
  TEST_ASSERT(state.swEdgeToHwPaths[swEdgeId][0] == 0);
  TEST_ASSERT(state.swEdgeToHwPaths[swEdgeId][1] == 1);

  // Verify reverse mapping: hwEdge 0 should contain swEdge 0.
  TEST_ASSERT(state.hwEdgeToSwEdges[hwEdgeId].size() == 1);
  TEST_ASSERT(state.hwEdgeToSwEdges[hwEdgeId][0] == swEdgeId);

  // Double-map should fail.
  r = state.mapEdge(swEdgeId, path, dfg, adg);
  TEST_ASSERT(r == ActionResult::FailedHardConstraint);

  // Unmap the edge.
  r = state.unmapEdge(swEdgeId, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);

  // Forward and reverse mappings should be cleared.
  TEST_ASSERT(state.swEdgeToHwPaths[swEdgeId].empty());
  TEST_ASSERT(state.hwEdgeToSwEdges[hwEdgeId].empty());

  return 0;
}
