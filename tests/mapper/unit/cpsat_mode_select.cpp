//===-- cpsat_mode_select.cpp - CP-SAT mode selection test ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify CPSATSolver::selectMode() correctly chooses mode based on DFG size
// and profile string.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  // Small DFG (<= 50 nodes) with default profile -> FULL_PROBLEM.
  {
    Graph dfg;
    for (int i = 0; i < 10; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::FULL_PROBLEM);
  }

  // Large DFG (> 50 nodes) with default profile -> SUB_PROBLEM.
  {
    Graph dfg;
    for (int i = 0; i < 100; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::SUB_PROBLEM);
  }

  // Profile "cpsat_full" forces FULL_PROBLEM regardless of size.
  {
    Graph dfg;
    for (int i = 0; i < 200; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "cpsat_full", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::FULL_PROBLEM);
  }

  // Profile "heuristic_only" -> DISABLED.
  {
    Graph dfg;
    for (int i = 0; i < 5; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "heuristic_only", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::DISABLED);
  }

  // Exactly at threshold (50) -> FULL_PROBLEM.
  {
    Graph dfg;
    for (int i = 0; i < 50; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::FULL_PROBLEM);
  }

  // One above threshold (51) -> SUB_PROBLEM.
  {
    Graph dfg;
    for (int i = 0; i < 51; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::SUB_PROBLEM);
  }

  return 0;
}
