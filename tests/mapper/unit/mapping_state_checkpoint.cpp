//===-- mapping_state_checkpoint.cpp - Checkpoint/restore test -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify MappingState::save() and restore() preserve full state.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  Graph dfg;
  Graph adg;

  for (int i = 0; i < 2; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    dfg.addNode(std::move(n));
  }
  for (int i = 0; i < 3; ++i) {
    auto n = std::make_unique<Node>();
    adg.addNode(std::move(n));
  }

  MappingState state;
  state.init(dfg, adg);

  // Map sw0 -> hw1 and save checkpoint.
  state.mapNode(0, 1, dfg, adg);
  TEST_ASSERT(state.swNodeToHwNode[0] == 1);

  auto cp = state.save();

  // Map sw1 -> hw2 after checkpoint.
  state.mapNode(1, 2, dfg, adg);
  TEST_ASSERT(state.swNodeToHwNode[1] == 2);
  TEST_ASSERT(state.actionLog.size() == 2);

  // Restore checkpoint: sw1 should be unmapped, action log trimmed.
  state.restore(cp);
  TEST_ASSERT(state.swNodeToHwNode[0] == 1);
  TEST_ASSERT(state.swNodeToHwNode[1] == INVALID_ID);
  TEST_ASSERT(state.actionLog.size() == 1);

  // Reverse mapping should also be restored.
  TEST_ASSERT(state.hwNodeToSwNodes[1].size() == 1);
  TEST_ASSERT(state.hwNodeToSwNodes[2].empty());

  return 0;
}
