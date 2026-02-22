//===-- mapping_state_map_node.cpp - Node map/unmap test -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify mapNode and unmapNode maintain forward/reverse mapping consistency.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <algorithm>

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  // DFG: 2 operation nodes.
  for (int i = 0; i < 2; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    dfg.addNode(std::move(n));
  }

  // ADG: 3 hardware nodes.
  for (int i = 0; i < 3; ++i) {
    auto n = std::make_unique<Node>();
    adg.addNode(std::move(n));
  }

  MappingState state;
  state.init(dfg, adg);

  // Map sw0 -> hw1.
  auto r = state.mapNode(0, 1, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);
  TEST_ASSERT(state.swNodeToHwNode[0] == 1);
  TEST_ASSERT(state.hwNodeToSwNodes[1].size() == 1);
  TEST_ASSERT(state.hwNodeToSwNodes[1][0] == 0);

  // Map sw1 -> hw2.
  r = state.mapNode(1, 2, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);
  TEST_ASSERT(state.swNodeToHwNode[1] == 2);

  // Double-map should fail.
  r = state.mapNode(0, 2, dfg, adg);
  TEST_ASSERT(r == ActionResult::FailedHardConstraint);

  // Unmap sw0.
  r = state.unmapNode(0, dfg, adg);
  TEST_ASSERT(r == ActionResult::Success);
  TEST_ASSERT(state.swNodeToHwNode[0] == INVALID_ID);
  TEST_ASSERT(state.hwNodeToSwNodes[1].empty());

  // Unmap already unmapped should fail.
  r = state.unmapNode(0, dfg, adg);
  TEST_ASSERT(r == ActionResult::FailedHardConstraint);

  // Verify action log recorded only successful actions.
  // mapNode(0,1), mapNode(1,2), unmapNode(0) = 3 entries.
  // Failed attempts (double-map, double-unmap) are not logged.
  TEST_ASSERT(state.actionLog.size() == 3);
  TEST_ASSERT(state.actionLog[0].type == ActionRecord::MAP_NODE);
  TEST_ASSERT(state.actionLog[2].type == ActionRecord::UNMAP_NODE);

  // State validity check.
  TEST_ASSERT(state.isValid());

  return 0;
}
