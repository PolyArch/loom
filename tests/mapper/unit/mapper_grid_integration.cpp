//===-- mapper_grid_integration.cpp - Grid topology integration test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Integration test that calls Mapper::run() on non-trivial ADG topologies
// with routing switches to exercise diamond fan-out/fan-in, multi-cluster
// routing, and diagnostics specificity for incompatible operations.
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

  // Test 1: 2x2 diamond DFG on a 4-PE grid with routing switch.
  // DFG: fan-out from A to B and C, then fan-in from B and C to D.
  // ADG: 4 functional PEs (arith.addi, 2 in, 2 out each) + 1 central
  // routing switch (4 in, 4 out). PE outputs connect to switch inputs,
  // switch outputs connect to PE inputs.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG diamond: A -> B, A -> C, B -> D, C -> D.
    IdIndex swA = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 2);
    IdIndex swB = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex swC = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex swD = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addEdgeBetween(dfg, swA, 0, swB, 0); // A out0 -> B in0
    addEdgeBetween(dfg, swA, 1, swC, 0); // A out1 -> C in0
    addEdgeBetween(dfg, swB, 0, swD, 0); // B out0 -> D in0
    addEdgeBetween(dfg, swC, 0, swD, 1); // C out0 -> D in1

    // ADG: 4 functional PEs (2 in, 2 out each) + central switch (8 in, 8 out).
    // Each PE gets 2 unique switch input/output port pairs for fan-out routing.
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 2, 2);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 2, 2);
    IdIndex hw2 = addPENode(adg, ctx, "arith.addi", "functional", 2, 2);
    IdIndex hw3 = addPENode(adg, ctx, "arith.addi", "functional", 2, 2);
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 8, 8);

    // PE outputs -> unique switch inputs (2 per PE).
    for (int p = 0; p < 4; ++p) {
      IdIndex pe = (p == 0) ? hw0 : (p == 1) ? hw1 : (p == 2) ? hw2 : hw3;
      addADGEdge(adg, adg.getNode(pe)->outputPorts[0],
                 adg.getNode(csw)->inputPorts[p * 2]);
      addADGEdge(adg, adg.getNode(pe)->outputPorts[1],
                 adg.getNode(csw)->inputPorts[p * 2 + 1]);
    }

    // Switch outputs -> PE inputs (2 per PE).
    for (int p = 0; p < 4; ++p) {
      IdIndex pe = (p == 0) ? hw0 : (p == 1) ? hw1 : (p == 2) ? hw2 : hw3;
      addADGEdge(adg, adg.getNode(csw)->outputPorts[p * 2],
                 adg.getNode(pe)->inputPorts[0]);
      addADGEdge(adg, adg.getNode(csw)->outputPorts[p * 2 + 1],
                 adg.getNode(pe)->inputPorts[1]);
    }

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);

    // All 4 DFG nodes must be placed on distinct PEs.
    TEST_ASSERT(result.state.swNodeToHwNode[swA] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[swB] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[swC] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[swD] != INVALID_ID);

    TEST_ASSERT(result.state.swNodeToHwNode[swA] !=
                result.state.swNodeToHwNode[swB]);
    TEST_ASSERT(result.state.swNodeToHwNode[swA] !=
                result.state.swNodeToHwNode[swC]);
    TEST_ASSERT(result.state.swNodeToHwNode[swA] !=
                result.state.swNodeToHwNode[swD]);
    TEST_ASSERT(result.state.swNodeToHwNode[swB] !=
                result.state.swNodeToHwNode[swC]);
    TEST_ASSERT(result.state.swNodeToHwNode[swB] !=
                result.state.swNodeToHwNode[swD]);
    TEST_ASSERT(result.state.swNodeToHwNode[swC] !=
                result.state.swNodeToHwNode[swD]);

    // No node should be placed on the routing switch.
    TEST_ASSERT(result.state.swNodeToHwNode[swA] != csw);
    TEST_ASSERT(result.state.swNodeToHwNode[swB] != csw);
    TEST_ASSERT(result.state.swNodeToHwNode[swC] != csw);
    TEST_ASSERT(result.state.swNodeToHwNode[swD] != csw);
  }

  // Test 2: 4x4 style grid with 6 nodes and 2 routing switches.
  // DFG: 6-node chain (arith.addi).
  // ADG: 6 functional PEs (1 in, 1 out) + 2 switches (3 in, 3 out each),
  // partitioned into 2 groups of 3 PEs. Each group's PEs connect to their
  // local switch. Cross-group routing through switch-to-switch link.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 6-node chain.
    llvm::SmallVector<IdIndex, 6> swNodes;
    for (int i = 0; i < 6; ++i) {
      swNodes.push_back(
          addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1));
    }
    for (int i = 0; i < 5; ++i) {
      addEdgeBetween(dfg, swNodes[i], 0, swNodes[i + 1], 0);
    }

    // ADG: Group A: PEs 0-2 + switch A; Group B: PEs 3-5 + switch B.
    llvm::SmallVector<IdIndex, 6> hwPEs;
    for (int i = 0; i < 6; ++i) {
      hwPEs.push_back(
          addPENode(adg, ctx, "arith.addi", "functional", 1, 1));
    }

    IdIndex swA = addPENode(adg, ctx, "fabric.switch", "routing", 3, 3);
    IdIndex swB = addPENode(adg, ctx, "fabric.switch", "routing", 3, 3);

    // Group A: PEs 0-2 <-> switch A.
    for (int i = 0; i < 3; ++i) {
      addADGEdge(adg, adg.getNode(hwPEs[i])->outputPorts[0],
                 adg.getNode(swA)->inputPorts[i]);
      addADGEdge(adg, adg.getNode(swA)->outputPorts[i],
                 adg.getNode(hwPEs[i])->inputPorts[0]);
    }

    // Group B: PEs 3-5 <-> switch B.
    for (int i = 0; i < 3; ++i) {
      addADGEdge(adg, adg.getNode(hwPEs[3 + i])->outputPorts[0],
                 adg.getNode(swB)->inputPorts[i]);
      addADGEdge(adg, adg.getNode(swB)->outputPorts[i],
                 adg.getNode(hwPEs[3 + i])->inputPorts[0]);
    }

    // Cross-group link: switch A output[2] -> switch B input[2]
    // and switch B output[2] -> switch A input[2] (bidirectional).
    // Use dedicated extra ports for cross-switch link.
    // Add extra ports to each switch for the cross-link.
    {
      auto portOut = std::make_unique<Port>();
      portOut->parentNode = swA;
      portOut->direction = Port::Output;
      IdIndex pidOut = adg.addPort(std::move(portOut));
      adg.getNode(swA)->outputPorts.push_back(pidOut);

      auto portIn = std::make_unique<Port>();
      portIn->parentNode = swB;
      portIn->direction = Port::Input;
      IdIndex pidIn = adg.addPort(std::move(portIn));
      adg.getNode(swB)->inputPorts.push_back(pidIn);

      addADGEdge(adg, pidOut, pidIn); // swA -> swB
    }
    {
      auto portOut = std::make_unique<Port>();
      portOut->parentNode = swB;
      portOut->direction = Port::Output;
      IdIndex pidOut = adg.addPort(std::move(portOut));
      adg.getNode(swB)->outputPorts.push_back(pidOut);

      auto portIn = std::make_unique<Port>();
      portIn->parentNode = swA;
      portIn->direction = Port::Input;
      IdIndex pidIn = adg.addPort(std::move(portIn));
      adg.getNode(swA)->inputPorts.push_back(pidIn);

      addADGEdge(adg, pidOut, pidIn); // swB -> swA
    }

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 15.0;
    opts.seed = 42;
    opts.profile = "balanced";
    opts.maxGlobalRestarts = 5;

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);

    // All 6 nodes must be placed.
    for (int i = 0; i < 6; ++i) {
      TEST_ASSERT(result.state.swNodeToHwNode[swNodes[i]] != INVALID_ID);
    }

    // All 6 nodes must be on distinct PEs.
    for (int i = 0; i < 6; ++i) {
      for (int j = i + 1; j < 6; ++j) {
        TEST_ASSERT(result.state.swNodeToHwNode[swNodes[i]] !=
                    result.state.swNodeToHwNode[swNodes[j]]);
      }
    }
  }

  // Test 3: Diagnostics specificity test.
  // DFG: 2 nodes with incompatible types (arith.muli and arith.shli).
  // ADG: PEs only support arith.addi.
  // Expected: mapping fails and diagnostics contain useful info.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.shli", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: 2 PEs that only support arith.addi.
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    // Diagnostics should contain something meaningful (not just whitespace).
    bool hasContent = false;
    for (char c : result.diagnostics) {
      if (c != ' ' && c != '\n' && c != '\t' && c != '\r') {
        hasContent = true;
        break;
      }
    }
    TEST_ASSERT(hasContent);
  }

  return 0;
}
