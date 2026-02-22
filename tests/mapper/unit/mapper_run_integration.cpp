//===-- mapper_run_integration.cpp - Mapper::run integration test --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Integration test that calls Mapper::run() on minimal DFG/ADG pairs to
// exercise the full placement/routing/validation pipeline, including
// memory capacity, routing, and congestion scenarios.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
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

void setIntAttr(Node *node, mlir::MLIRContext &ctx,
                llvm::StringRef name, int64_t value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value)));
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

void addADGEdge(Graph &adg, IdIndex srcPort, IdIndex dstPort) {
  auto e = std::make_unique<Edge>();
  e->srcPort = srcPort;
  e->dstPort = dstPort;
  IdIndex eid = adg.addEdge(std::move(e));
  adg.getPort(srcPort)->connectedEdges.push_back(eid);
  adg.getPort(dstPort)->connectedEdges.push_back(eid);
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Minimal successful mapping (2-node chain, 2-PE ADG).
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                result.state.swNodeToHwNode[sw1]);
  }

  // Test 2: Incompatible hardware produces diagnostics-bearing failure.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
  }

  // Test 3: 3-node chain with central routing switch (deterministic success).
  // A central switch provides any-to-any routing between PEs regardless
  // of placement order. All 3 nodes must be placed on distinct PEs.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);
    addEdgeBetween(dfg, sw1, 0, sw2, 0);

    // ADG: 3 functional PEs (1 in, 1 out each) + central switch (3 in, 3 out).
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw2 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 3, 3);

    // PE outputs -> switch inputs.
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(csw)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(hw1)->outputPorts[0],
               adg.getNode(csw)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(hw2)->outputPorts[0],
               adg.getNode(csw)->inputPorts[2]);

    // Switch outputs -> PE inputs.
    addADGEdge(adg, adg.getNode(csw)->outputPorts[0],
               adg.getNode(hw0)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[1],
               adg.getNode(hw1)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[2],
               adg.getNode(hw2)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw2] != INVALID_ID);

    // All on distinct functional PEs (not on the switch).
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                result.state.swNodeToHwNode[sw1]);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] !=
                result.state.swNodeToHwNode[sw2]);
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                result.state.swNodeToHwNode[sw2]);
  }

  // Test 4: Memory numRegion capacity through Mapper::run.
  // DFG: 2 memory ops + 2 functional ops in a chain.
  // ADG: 1 memory node (numRegion=2) + 2 functional PEs.
  // Both memory ops should be placed on the same memory node.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: mem0 -> add0, mem1 -> add1
    IdIndex mem0 =
        addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    IdIndex mem1 =
        addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    IdIndex add0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex add1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, mem0, 0, add0, 0);
    addEdgeBetween(dfg, mem1, 0, add1, 0);

    // ADG: 1 memory node with numRegion=2, 2 functional PEs, 1 switch.
    IdIndex hwMem = addPENode(adg, ctx, "fabric.extmemory", "memory", 2, 2);
    setIntAttr(adg.getNode(hwMem), ctx, "numRegion", 2);

    IdIndex hwPE0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hwPE1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    // Central switch for routing (4 in, 4 out).
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 4, 4);

    // All outputs -> switch inputs.
    addADGEdge(adg, adg.getNode(hwMem)->outputPorts[0],
               adg.getNode(csw)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(hwMem)->outputPorts[1],
               adg.getNode(csw)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(hwPE0)->outputPorts[0],
               adg.getNode(csw)->inputPorts[2]);
    addADGEdge(adg, adg.getNode(hwPE1)->outputPorts[0],
               adg.getNode(csw)->inputPorts[3]);

    // Switch outputs -> all inputs.
    addADGEdge(adg, adg.getNode(csw)->outputPorts[0],
               adg.getNode(hwMem)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[1],
               adg.getNode(hwMem)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[2],
               adg.getNode(hwPE0)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[3],
               adg.getNode(hwPE1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);

    // Both memory ops must be placed on the single memory node.
    TEST_ASSERT(result.state.swNodeToHwNode[mem0] == hwMem);
    TEST_ASSERT(result.state.swNodeToHwNode[mem1] == hwMem);

    // The functional ops must be on distinct PEs.
    TEST_ASSERT(result.state.swNodeToHwNode[add0] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[add1] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[add0] !=
                result.state.swNodeToHwNode[add1]);
  }

  // Test 5: Memory over-capacity (3 memory ops, numRegion=2) produces
  // failure because the ADG cannot accommodate all 3 memory ops.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);

    // Only 1 memory node with numRegion=2: cannot host 3 memory ops.
    IdIndex hwMem = addPENode(adg, ctx, "fabric.extmemory", "memory", 2, 2);
    setIntAttr(adg.getNode(hwMem), ctx, "numRegion", 2);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;

    Mapper::Result result = mapper.run(dfg, adg, opts);

    // Should fail: 3 memory ops cannot fit in numRegion=2.
    TEST_ASSERT(!result.success);
  }

  // Test 6: Congested scenario exercising the full pipeline.
  // Create a DFG with enough nodes to trigger congestion detection
  // in Mapper::run. When OR-Tools is available, this may enter the
  // CP-SAT sub-problem path; otherwise, the heuristic-only path runs.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 8 functional nodes in a chain (enough to stress placement
    // on a limited 4-PE ADG, creating resource contention).
    const int numDfgNodes = 8;
    llvm::SmallVector<IdIndex, 8> swNodes;
    for (int i = 0; i < numDfgNodes; ++i) {
      swNodes.push_back(
          addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1));
    }
    for (int i = 0; i < numDfgNodes - 1; ++i) {
      addEdgeBetween(dfg, swNodes[i], 0, swNodes[i + 1], 0);
    }

    // ADG: only 4 functional PEs (half the DFG), creating congestion.
    // Each PE has 2 inputs + 2 outputs for full mesh connectivity.
    const int numHwPEs = 4;
    llvm::SmallVector<IdIndex, 4> hwPEs;
    for (int i = 0; i < numHwPEs; ++i) {
      hwPEs.push_back(
          addPENode(adg, ctx, "arith.addi", "functional", 2, 2));
    }
    // Full mesh connectivity between all PEs.
    for (int i = 0; i < numHwPEs; ++i) {
      for (int j = 0; j < numHwPEs; ++j) {
        if (i == j)
          continue;
        unsigned outIdx = (j < i) ? static_cast<unsigned>(j)
                                  : static_cast<unsigned>(j - 1);
        if (outIdx < 2) {
          unsigned inIdx = (i < j) ? static_cast<unsigned>(i)
                                   : static_cast<unsigned>(i - 1);
          if (inIdx < 2) {
            addADGEdge(adg,
                       adg.getNode(hwPEs[i])->outputPorts[outIdx],
                       adg.getNode(hwPEs[j])->inputPorts[inIdx]);
          }
        }
      }
    }

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 15.0;
    opts.seed = 42;
    opts.profile = "balanced";
    opts.maxGlobalRestarts = 5;

    Mapper::Result result = mapper.run(dfg, adg, opts);

    // 8 DFG nodes on 4 PEs: the mapper must fail because non-temporal
    // PEs enforce single-occupancy. This exercises the full refinement
    // pipeline (placement, routing, repair, restart) under congestion.
    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());

    // Verify the pipeline did attempt placement (not an early-exit).
    // At least some nodes should have been placed before failure.
    int placedCount = 0;
    for (int i = 0; i < numDfgNodes; ++i) {
      if (result.state.swNodeToHwNode[swNodes[i]] != INVALID_ID)
        ++placedCount;
    }
    TEST_ASSERT(placedCount > 0);
  }

  return 0;
}
