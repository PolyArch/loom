//===-- cpsat_available.cpp - CP-SAT availability test -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify CPSATSolver::isAvailable() reports correct status and that stub
// solvers return appropriate diagnostics when OR-Tools is not linked.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  // Check availability flag matches compile-time config.
#ifdef LOOM_HAS_ORTOOLS
  TEST_ASSERT(CPSATSolver::isAvailable());
#else
  TEST_ASSERT(!CPSATSolver::isAvailable());
#endif

  // When not available, solveFullProblem should return failure with diagnostic.
  if (!CPSATSolver::isAvailable()) {
    Graph dfg;
    Graph adg;
    CandidateSet candidates;
    ConnectivityMatrix connectivity;
    CPSATSolver solver;
    auto result = solver.solveFullProblem(dfg, adg, candidates, connectivity);
    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
  }

  return 0;
}
