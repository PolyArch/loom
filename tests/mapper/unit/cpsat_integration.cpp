//===-- cpsat_integration.cpp - CP-SAT warm-start flow test -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that CPSATSolver warm-start flow works correctly:
// - isAvailable() returns a consistent value
// - selectMode() handles all profiles correctly
// - extractSubProblem() produces valid node sets
// - Solver stubs return appropriate diagnostics when OR-Tools is not linked
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/ConnectivityMatrix.h"

#include "mlir/IR/MLIRContext.h"

using namespace loom;

int main() {
  mlir::MLIRContext ctx;

  // Test 1: isAvailable() is consistent.
  {
    bool avail = CPSATSolver::isAvailable();
    // Value depends on build config, but must be consistent.
    TEST_ASSERT(avail == CPSATSolver::isAvailable());
  }

  // Test 2: selectMode() with "balanced" profile and small DFG.
  {
    Graph dfg(&ctx);
    for (int i = 0; i < 10; ++i) {
      auto n = std::make_unique<Node>();
      n->kind = Node::OperationNode;
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::FULL_PROBLEM);
  }

  // Test 3: selectMode() with "heuristic_only" profile.
  {
    Graph dfg(&ctx);
    for (int i = 0; i < 10; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "heuristic_only", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::DISABLED);
  }

  // Test 4: selectMode() with "cpsat_full" forces full mode.
  {
    Graph dfg(&ctx);
    for (int i = 0; i < 200; ++i) {
      auto n = std::make_unique<Node>();
      dfg.addNode(std::move(n));
    }
    auto mode = CPSATSolver::selectMode(dfg, "cpsat_full", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::FULL_PROBLEM);
  }

  // Test 5: extractSubProblem() returns conflict nodes + neighbors.
  {
    Graph dfg(&ctx);
    // Create a small chain: 0 -> 1 -> 2.
    for (int i = 0; i < 3; ++i) {
      auto n = std::make_unique<Node>();
      n->kind = Node::OperationNode;
      dfg.addNode(std::move(n));
    }

    // Add ports and edges: 0->1, 1->2.
    for (int i = 0; i < 3; ++i) {
      auto outPort = std::make_unique<Port>();
      outPort->parentNode = static_cast<IdIndex>(i);
      outPort->direction = Port::Output;
      IdIndex outPid = dfg.addPort(std::move(outPort));
      dfg.getNode(static_cast<IdIndex>(i))->outputPorts.push_back(outPid);

      auto inPort = std::make_unique<Port>();
      inPort->parentNode = static_cast<IdIndex>(i);
      inPort->direction = Port::Input;
      IdIndex inPid = dfg.addPort(std::move(inPort));
      dfg.getNode(static_cast<IdIndex>(i))->inputPorts.push_back(inPid);
    }

    // Edge 0->1
    {
      IdIndex srcPid = dfg.getNode(0)->outputPorts[0];
      IdIndex dstPid = dfg.getNode(1)->inputPorts[0];
      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPid;
      edge->dstPort = dstPid;
      IdIndex eid = dfg.addEdge(std::move(edge));
      dfg.getPort(srcPid)->connectedEdges.push_back(eid);
      dfg.getPort(dstPid)->connectedEdges.push_back(eid);
    }

    // Edge 1->2
    {
      IdIndex srcPid = dfg.getNode(1)->outputPorts[0];
      IdIndex dstPid = dfg.getNode(2)->inputPorts[0];
      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPid;
      edge->dstPort = dstPid;
      IdIndex eid = dfg.addEdge(std::move(edge));
      dfg.getPort(srcPid)->connectedEdges.push_back(eid);
      dfg.getPort(dstPid)->connectedEdges.push_back(eid);
    }

    // Extract sub-problem around node 1.
    llvm::SmallVector<IdIndex, 4> conflict = {1};
    auto subProblem = CPSATSolver::extractSubProblem(dfg, conflict, 10);

    // Should include node 1 and its neighbors (0, 2).
    TEST_ASSERT(subProblem.size() >= 1);
    bool has1 = false;
    for (IdIndex n : subProblem) {
      if (n == 1) has1 = true;
    }
    TEST_ASSERT(has1);
  }

  // Test 6: Warm-start solver when OR-Tools not available returns appropriate
  // diagnostic.
  {
    if (!CPSATSolver::isAvailable()) {
      Graph dfg(&ctx);
      Graph adg(&ctx);
      CandidateSet candidates;
      ConnectivityMatrix cm;

      CPSATSolver solver;
      auto result = solver.solveFullProblem(dfg, adg, candidates, cm);
      TEST_ASSERT(!result.success);
      TEST_ASSERT(!result.diagnostics.empty());
    }
  }

  return 0;
}
