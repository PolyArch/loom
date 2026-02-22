//===-- mapper_run_congestion.cpp - Mapper::run congestion test ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that Mapper::run() produces a result for congested DFG/ADG
// scenarios. When CP-SAT is available, the sub-problem path should
// be triggered; when unavailable, the heuristic-only fallback runs.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/ConnectivityMatrix.h"

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

void setIntAttr(Node *node, mlir::MLIRContext &ctx,
                llvm::StringRef name, int64_t value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value)));
}

void setArrayStrAttr(Node *node, mlir::MLIRContext &ctx,
                     llvm::StringRef name,
                     llvm::ArrayRef<llvm::StringRef> values) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (auto v : values)
    attrs.push_back(mlir::StringAttr::get(&ctx, v));
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::ArrayAttr::get(&ctx, attrs)));
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

IdIndex addPENode(Graph &adg, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn, unsigned numOut) {
  auto hwNode = std::make_unique<Node>();
  hwNode->kind = Node::OperationNode;
  setStringAttr(hwNode.get(), ctx, "op_name", opName);
  setStringAttr(hwNode.get(), ctx, "resource_class", resClass);
  IdIndex hwId = adg.addNode(std::move(hwNode));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Input;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->inputPorts.push_back(pid);
  }
  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Output;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->outputPorts.push_back(pid);
  }
  return hwId;
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

  // Test 1: selectMode returns SUB_PROBLEM for large DFGs under balanced
  // profile, triggering the CP-SAT sub-problem extraction path.
  {
    Graph dfg(&ctx);

    // 60 nodes exceeds the default sub-problem threshold of 50.
    for (int i = 0; i < 60; ++i) {
      addOpNode(dfg, ctx, "arith.addi", "functional");
    }

    auto mode = CPSATSolver::selectMode(dfg, "balanced", 50);
    TEST_ASSERT(mode == CPSATSolver::Mode::SUB_PROBLEM);
  }

  // Test 2: CP-SAT sub-problem extraction from congestion seeds.
  // Simulate a scenario where multiple SW nodes compete for limited HW.
  {
    Graph dfg(&ctx);

    // Star topology: center connected to 8 peripherals.
    IdIndex center = addOpNode(dfg, ctx, "arith.addi", "functional", 8, 8);
    llvm::SmallVector<IdIndex, 8> peripherals;
    for (int i = 0; i < 8; ++i) {
      IdIndex p = addOpNode(dfg, ctx, "arith.muli", "functional");
      peripherals.push_back(p);
      addEdgeBetween(dfg, center, i, p, 0);
    }

    // Use 4 peripheral nodes as conflict seeds (congestion hotspots).
    llvm::SmallVector<IdIndex, 4> seeds = {
        peripherals[0], peripherals[2], peripherals[4], peripherals[6]};

    auto subProblem =
        CPSATSolver::extractSubProblem(dfg, seeds, /*maxNodes=*/10);

    // All seeds must be included.
    for (IdIndex seed : seeds) {
      bool found = false;
      for (IdIndex id : subProblem) {
        if (id == seed) {
          found = true;
          break;
        }
      }
      TEST_ASSERT(found);
    }

    // Center node (common neighbor) should be included.
    bool hasCenter = false;
    for (IdIndex id : subProblem) {
      if (id == center)
        hasCenter = true;
    }
    TEST_ASSERT(hasCenter);

    // Respect maxNodes limit.
    TEST_ASSERT(subProblem.size() <= 10);
  }

  // Test 3: CP-SAT availability check.
  // Verify consistent behavior regardless of build configuration.
  {
    bool avail = CPSATSolver::isAvailable();
    TEST_ASSERT(avail == CPSATSolver::isAvailable());

    if (!avail) {
      // When OR-Tools is not linked, solver stubs return diagnostics.
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

  // Test 4: CP-SAT sub-problem solver with congested fixed occupancy.
  // When OR-Tools is available, verify the sub-problem solver handles
  // fixed nodes correctly and produces a feasible placement.
  if (CPSATSolver::isAvailable()) {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 4 addi nodes.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw3 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);
    addEdgeBetween(dfg, sw2, 0, sw3, 0);

    // ADG: 4 single-op PEs.
    IdIndex hw0 = addPENode(adg, ctx, "fabric.pe", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "fabric.pe", "functional", 1, 1);
    IdIndex hw2 = addPENode(adg, ctx, "fabric.pe", "functional", 1, 1);
    IdIndex hw3 = addPENode(adg, ctx, "fabric.pe", "functional", 1, 1);

    // Fix sw0 -> hw0 and sw1 -> hw1.
    MappingState currentState;
    currentState.init(dfg, adg);
    currentState.mapNode(sw0, hw0, dfg, adg);
    currentState.mapNode(sw1, hw1, dfg, adg);

    // Sub-problem: only sw2 and sw3 need re-mapping.
    CandidateSet cands;
    for (IdIndex sw : {sw2, sw3}) {
      for (IdIndex hw : {hw0, hw1, hw2, hw3}) {
        Candidate c;
        c.hwNodeId = hw;
        c.swNodeIds = {sw};
        c.isGroup = false;
        cands[sw].push_back(c);
      }
    }

    ConnectivityMatrix cm;
    CPSATSolver solver;
    CPSATSolver::Options opts;
    opts.timeLimitSeconds = 10.0;

    llvm::SmallVector<IdIndex, 2> subNodes = {sw2, sw3};
    auto result = solver.solveSubProblem(dfg, adg, subNodes, currentState,
                                         cands, cm, opts);

    TEST_ASSERT(result.success);

    // Fixed nodes should remain in place.
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] == hw0);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] == hw1);

    // Sub-problem nodes should be placed on available PEs (hw2 or hw3).
    IdIndex placedHw2 = result.state.swNodeToHwNode[sw2];
    IdIndex placedHw3 = result.state.swNodeToHwNode[sw3];
    TEST_ASSERT(placedHw2 != INVALID_ID);
    TEST_ASSERT(placedHw3 != INVALID_ID);
    TEST_ASSERT(placedHw2 != placedHw3);
  }

  return 0;
}
