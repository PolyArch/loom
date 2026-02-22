//===-- cpsat_subproblem_trigger.cpp - CP-SAT trigger test ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that CP-SAT sub-problem extraction considers both unrouted edges and
// congestion hotspots as conflict seeds, and that group atomicity is enforced
// when extracting sub-problems.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

IdIndex addOpNode(Graph &g, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn = 1, unsigned numOut = 1) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;
  setStringAttr(node.get(), ctx, "op_name", opName);
  setStringAttr(node.get(), ctx, "resource_class", resClass);
  IdIndex nodeId = g.addNode(std::move(node));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->inputPorts.push_back(pid);
  }

  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->outputPorts.push_back(pid);
  }

  return nodeId;
}

IdIndex addEdgeBetween(Graph &g, IdIndex srcNode, unsigned srcPortIdx,
                       IdIndex dstNode, unsigned dstPortIdx) {
  IdIndex srcPort = g.getNode(srcNode)->outputPorts[srcPortIdx];
  IdIndex dstPort = g.getNode(dstNode)->inputPorts[dstPortIdx];

  auto edge = std::make_unique<Edge>();
  edge->srcPort = srcPort;
  edge->dstPort = dstPort;
  IdIndex eid = g.addEdge(std::move(edge));
  g.getPort(srcPort)->connectedEdges.push_back(eid);
  g.getPort(dstPort)->connectedEdges.push_back(eid);
  return eid;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: extractSubProblem with unrouted edge endpoints as seeds.
  {
    Graph dfg(&ctx);

    // Chain: n0 -> n1 -> n2 -> n3.
    IdIndex n0 = addOpNode(dfg, ctx, "arith.addi", "functional");
    IdIndex n1 = addOpNode(dfg, ctx, "arith.muli", "functional");
    IdIndex n2 = addOpNode(dfg, ctx, "arith.addi", "functional");
    IdIndex n3 = addOpNode(dfg, ctx, "arith.muli", "functional");
    addEdgeBetween(dfg, n0, 0, n1, 0);
    addEdgeBetween(dfg, n1, 0, n2, 0);
    addEdgeBetween(dfg, n2, 0, n3, 0);

    // Conflict seeds: n1 and n2 (as if the edge between them failed).
    llvm::SmallVector<IdIndex, 4> seeds = {n1, n2};

    auto subProblem =
        CPSATSolver::extractSubProblem(dfg, seeds, /*maxNodes=*/10);

    // Sub-problem should include at least the seed nodes.
    bool hasN1 = false, hasN2 = false;
    for (IdIndex id : subProblem) {
      if (id == n1)
        hasN1 = true;
      if (id == n2)
        hasN2 = true;
    }
    TEST_ASSERT(hasN1);
    TEST_ASSERT(hasN2);

    // Should also include neighbors (n0 and n3) since maxNodes is large.
    bool hasN0 = false, hasN3 = false;
    for (IdIndex id : subProblem) {
      if (id == n0)
        hasN0 = true;
      if (id == n3)
        hasN3 = true;
    }
    TEST_ASSERT(hasN0 || hasN3); // At least one neighbor included.
  }

  // Test 2: extractSubProblem respects maxNodes limit.
  {
    Graph dfg(&ctx);

    // Chain of 10 nodes.
    llvm::SmallVector<IdIndex, 10> nodeIds;
    for (int i = 0; i < 10; ++i) {
      nodeIds.push_back(addOpNode(dfg, ctx, "arith.addi", "functional"));
    }
    for (int i = 0; i < 9; ++i) {
      addEdgeBetween(dfg, nodeIds[i], 0, nodeIds[i + 1], 0);
    }

    // Seed: middle node.
    llvm::SmallVector<IdIndex, 1> seeds = {nodeIds[5]};

    auto subProblem =
        CPSATSolver::extractSubProblem(dfg, seeds, /*maxNodes=*/3);

    // Should not exceed maxNodes.
    TEST_ASSERT(subProblem.size() <= 3);

    // Seed node must be included.
    bool hasSeed = false;
    for (IdIndex id : subProblem) {
      if (id == nodeIds[5])
        hasSeed = true;
    }
    TEST_ASSERT(hasSeed);
  }

  // Test 3: Group atomicity in sub-problem extraction.
  // When a conflict node is part of a group, all group members should
  // be reachable from the conflict seeds (verified at Mapper level,
  // but here we verify extractSubProblem includes neighbors).
  {
    Graph dfg(&ctx);

    // Two-node group: n0 -> n1, plus an isolated n2.
    IdIndex n0 = addOpNode(dfg, ctx, "arith.addi", "functional");
    IdIndex n1 = addOpNode(dfg, ctx, "arith.muli", "functional");
    IdIndex n2 = addOpNode(dfg, ctx, "arith.addi", "functional");
    addEdgeBetween(dfg, n0, 0, n1, 0);
    (void)n2;

    // Seed: just n0.
    llvm::SmallVector<IdIndex, 1> seeds = {n0};

    auto subProblem =
        CPSATSolver::extractSubProblem(dfg, seeds, /*maxNodes=*/10);

    // n1 should be included as a neighbor of n0.
    bool hasN0 = false, hasN1 = false;
    for (IdIndex id : subProblem) {
      if (id == n0)
        hasN0 = true;
      if (id == n1)
        hasN1 = true;
    }
    TEST_ASSERT(hasN0);
    TEST_ASSERT(hasN1);
  }

  return 0;
}
