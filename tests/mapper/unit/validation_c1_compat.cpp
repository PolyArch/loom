//===-- validation_c1_compat.cpp - C1 mapping consistency test ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that mapping consistency is maintained: forward and reverse mappings
// agree, which is the prerequisite for C1 validation.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

using namespace loom;

int main() {
  // DFG: 3 operation nodes.
  Graph dfg;
  for (int i = 0; i < 3; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    dfg.addNode(std::move(n));
  }

  // ADG: 3 hardware nodes.
  Graph adg;
  for (int i = 0; i < 3; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    adg.addNode(std::move(n));
  }

  MappingState state;
  state.init(dfg, adg);

  // Map each sw node to corresponding hw node.
  for (int i = 0; i < 3; ++i) {
    auto r = state.mapNode(i, i, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
  }

  // Verify forward -> reverse consistency.
  for (int i = 0; i < 3; ++i) {
    IdIndex hw = state.swNodeToHwNode[i];
    TEST_ASSERT(hw == static_cast<IdIndex>(i));

    // Reverse mapping should contain this sw node.
    bool found = false;
    for (IdIndex sw : state.hwNodeToSwNodes[hw]) {
      if (sw == static_cast<IdIndex>(i))
        found = true;
    }
    TEST_ASSERT(found);
  }

  // isValid() should pass.
  TEST_ASSERT(state.isValid());

  // Unmap sw1, verify consistency still holds.
  state.unmapNode(1, dfg, adg);
  TEST_ASSERT(state.swNodeToHwNode[1] == INVALID_ID);
  TEST_ASSERT(state.hwNodeToSwNodes[1].empty());
  TEST_ASSERT(state.isValid());

  // Map sw1 to a different hw node (hw0 - which already has sw0).
  // Both sw0 and sw1 on hw0 - valid for temporal PEs.
  state.mapNode(1, 0, dfg, adg);
  TEST_ASSERT(state.hwNodeToSwNodes[0].size() == 2);
  TEST_ASSERT(state.isValid());

  return 0;
}
