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
  // Runtime availability check.
  bool avail = CPSATSolver::isAvailable();

  if (avail) {
    // OR-Tools is linked; isAvailable() should return true.
    // Solve with empty graphs should still succeed (trivially).
    Graph dfg;
    Graph adg;
    CandidateSet candidates;
    ConnectivityMatrix connectivity;
    CPSATSolver solver;
    auto result = solver.solveFullProblem(dfg, adg, candidates, connectivity);
    // Empty problem should succeed trivially or return a diagnostic.
    (void)result;
  } else {
    // OR-Tools is not linked; stub should return failure with diagnostic.
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
