//===-- mapping_state_init.cpp - MappingState initialization test --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify MappingState::init() correctly sizes all vectors.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  // Create DFG: 3 nodes, 4 ports, 2 edges.
  for (int i = 0; i < 3; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    dfg.addNode(std::move(n));
  }
  for (int i = 0; i < 4; ++i) {
    auto p = std::make_unique<Port>();
    dfg.addPort(std::move(p));
  }
  for (int i = 0; i < 2; ++i) {
    auto e = std::make_unique<Edge>();
    dfg.addEdge(std::move(e));
  }

  // Create ADG: 5 nodes, 8 ports, 3 edges.
  for (int i = 0; i < 5; ++i) {
    auto n = std::make_unique<Node>();
    adg.addNode(std::move(n));
  }
  for (int i = 0; i < 8; ++i) {
    auto p = std::make_unique<Port>();
    adg.addPort(std::move(p));
  }
  for (int i = 0; i < 3; ++i) {
    auto e = std::make_unique<Edge>();
    adg.addEdge(std::move(e));
  }

  MappingState state;
  state.init(dfg, adg);

  // Forward mappings sized to DFG.
  TEST_ASSERT(state.swNodeToHwNode.size() == 3);
  TEST_ASSERT(state.swPortToHwPort.size() == 4);
  TEST_ASSERT(state.swEdgeToHwPaths.size() == 2);

  // Reverse mappings sized to ADG.
  TEST_ASSERT(state.hwNodeToSwNodes.size() == 5);
  TEST_ASSERT(state.hwPortToSwPorts.size() == 8);
  TEST_ASSERT(state.hwEdgeToSwEdges.size() == 3);

  // Temporal assignments sized correctly.
  TEST_ASSERT(state.temporalPEAssignments.size() == 3);
  TEST_ASSERT(state.temporalSWAssignments.size() == 5);
  TEST_ASSERT(state.registerAssignments.size() == 2);

  // All forward mappings start as INVALID_ID.
  for (size_t i = 0; i < state.swNodeToHwNode.size(); ++i)
    TEST_ASSERT(state.swNodeToHwNode[i] == INVALID_ID);
  for (size_t i = 0; i < state.swPortToHwPort.size(); ++i)
    TEST_ASSERT(state.swPortToHwPort[i] == INVALID_ID);

  // Cost metrics start at 0.
  TEST_ASSERT(state.totalCost == 0.0);
  TEST_ASSERT(state.placementPressure == 0.0);

  return 0;
}
