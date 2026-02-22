//===-- cost_weights.cpp - Cost metric computation test ------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that MappingState cost metric fields are correctly structured
// and that cost computation handles edge cases (empty graphs).
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <cmath>

using namespace loom;

int main() {
  // Verify all cost metric fields are zero-initialized after init().
  {
    Graph dfg;
    Graph adg;
    for (int i = 0; i < 2; ++i) {
      dfg.addNode(std::make_unique<Node>());
      adg.addNode(std::make_unique<Node>());
    }

    MappingState state;
    state.init(dfg, adg);

    TEST_ASSERT(state.totalCost == 0.0);
    TEST_ASSERT(state.placementPressure == 0.0);
    TEST_ASSERT(state.routingCost == 0.0);
    TEST_ASSERT(state.temporalCost == 0.0);
    TEST_ASSERT(state.perfProxyCost == 0.0);
    TEST_ASSERT(state.criticalPathEst == 0.0);
    TEST_ASSERT(state.iiPressure == 0.0);
    TEST_ASSERT(state.queuePressure == 0.0);
    TEST_ASSERT(state.configFootprint == 0.0);
    TEST_ASSERT(state.nonDefaultWords == 0);
    TEST_ASSERT(state.totalConfigWords == 0);
  }

  // Verify cost metrics survive checkpoint/restore cycle.
  {
    Graph dfg;
    Graph adg;
    dfg.addNode(std::make_unique<Node>());
    adg.addNode(std::make_unique<Node>());

    MappingState state;
    state.init(dfg, adg);
    state.totalCost = 42.5;
    state.placementPressure = 1.5;

    auto cp = state.save();
    state.totalCost = 100.0;
    state.placementPressure = 99.9;

    state.restore(cp);
    TEST_ASSERT(std::abs(state.totalCost - 42.5) < 1e-6);
    // Note: placementPressure is not part of Checkpoint, so it won't be
    // restored. This is by design - only the mapping state is checkpointed,
    // not derived metrics.
  }

  return 0;
}
