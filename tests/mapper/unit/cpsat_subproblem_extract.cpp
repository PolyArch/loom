//===-- cpsat_subproblem_extract.cpp - Sub-problem extraction test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify CPSATSolver::extractSubProblem() expands conflict set correctly.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  // Build a small DFG: 5 nodes in a chain (0->1->2->3->4).
  Graph dfg;
  for (int i = 0; i < 5; ++i) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OperationNode;
    IdIndex nid = dfg.addNode(std::move(n));

    auto inPort = std::make_unique<Port>();
    inPort->parentNode = nid;
    inPort->direction = Port::Input;
    IdIndex inPid = dfg.addPort(std::move(inPort));
    dfg.getNode(nid)->inputPorts.push_back(inPid);

    auto outPort = std::make_unique<Port>();
    outPort->parentNode = nid;
    outPort->direction = Port::Output;
    IdIndex outPid = dfg.addPort(std::move(outPort));
    dfg.getNode(nid)->outputPorts.push_back(outPid);
  }

  // Create edges: 0->1, 1->2, 2->3, 3->4.
  // Port layout: node i has input=2*i, output=2*i+1.
  for (int i = 0; i < 4; ++i) {
    auto e = std::make_unique<Edge>();
    IdIndex srcOut = 2 * i + 1; // output port of node i
    IdIndex dstIn = 2 * (i + 1); // input port of node i+1
    e->srcPort = srcOut;
    e->dstPort = dstIn;
    IdIndex eid = dfg.addEdge(std::move(e));
    dfg.getPort(srcOut)->connectedEdges.push_back(eid);
    dfg.getPort(dstIn)->connectedEdges.push_back(eid);
  }

  // Extract sub-problem around node 2, max 10 nodes.
  llvm::SmallVector<IdIndex, 4> conflict = {2};
  auto subproblem = CPSATSolver::extractSubProblem(dfg, conflict, 10);

  // Should include node 2 plus neighbors (1 and 3).
  TEST_ASSERT(subproblem.size() >= 3);

  // Node 2 must be in the result.
  bool has2 = false;
  for (IdIndex n : subproblem) {
    if (n == 2)
      has2 = true;
  }
  TEST_ASSERT(has2);

  // With max 2 nodes, should be limited.
  auto limited = CPSATSolver::extractSubProblem(dfg, conflict, 2);
  TEST_ASSERT(limited.size() <= 2);

  // Extract with multiple conflicts.
  llvm::SmallVector<IdIndex, 4> multiConflict = {0, 4};
  auto multi = CPSATSolver::extractSubProblem(dfg, multiConflict, 10);
  // Should include 0, 4, and their neighbors (1, 3).
  TEST_ASSERT(multi.size() >= 4);

  return 0;
}
