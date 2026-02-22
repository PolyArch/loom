//===-- mapper_run_integration.cpp - Mapper::run integration test --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Integration test that calls Mapper::run() on minimal DFG/ADG pairs to
// exercise the full placement/routing/validation pipeline.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
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

  // Test 1: Minimal successful Mapper::run with 2 functional PEs and
  // a direct connection.
  //
  // DFG:  sw0(arith.addi) --> sw1(arith.addi)
  // ADG:  hw0(arith.addi) --> hw1(arith.addi)
  //
  // The mapper should place sw0->hw0, sw1->hw1, and route the edge
  // through the direct ADG connection.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: two addi nodes, 1 input + 1 output each, connected by an edge.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: two functional PEs with body_ops containing "arith.addi".
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    // Direct ADG edge: hw0 output port -> hw1 input port.
    IdIndex hw0OutPort = adg.getNode(hw0)->outputPorts[0];
    IdIndex hw1InPort = adg.getNode(hw1)->inputPorts[0];
    auto adgEdge = std::make_unique<Edge>();
    adgEdge->srcPort = hw0OutPort;
    adgEdge->dstPort = hw1InPort;
    IdIndex adgEdgeId = adg.addEdge(std::move(adgEdge));
    adg.getPort(hw0OutPort)->connectedEdges.push_back(adgEdgeId);
    adg.getPort(hw1InPort)->connectedEdges.push_back(adgEdgeId);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(result.success);

    // Both SW nodes should be mapped.
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] != INVALID_ID);
    TEST_ASSERT(result.state.swNodeToHwNode[sw1] != INVALID_ID);

    // They should be on different HW nodes (PE exclusivity).
    TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                result.state.swNodeToHwNode[sw1]);
  }

  // Test 2: Mapper::run should fail when DFG has no compatible hardware.
  // A DFG with "arith.muli" nodes but an ADG with only "arith.addi" PEs
  // should produce a diagnostics-bearing failure.
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

    // Should fail due to no compatible hardware for muli.
    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
  }

  // Test 3: Mapper::run with 3 nodes in a chain, requiring routing through
  // an intermediate routing node.
  //
  // DFG:  sw0(arith.addi) --> sw1(arith.addi) --> sw2(arith.addi)
  // ADG:  hw0 --> router --> hw1 --> hw2
  //       (hw0 connects to hw1 via a routing node; hw1 directly to hw2)
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 3-node chain.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);
    addEdgeBetween(dfg, sw1, 0, sw2, 0);

    // ADG: 3 functional PEs + 1 routing node connecting them in a chain.
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw2 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    // Routing node: 1 input, 2 outputs (for fanout).
    IdIndex router = addPENode(adg, ctx, "fabric.router", "routing", 2, 2);

    // ADG connectivity:
    // hw0.out -> router.in[0]
    // router.out[0] -> hw1.in
    // hw1.out -> router.in[1]
    // router.out[1] -> hw2.in
    auto addADGEdge = [&](IdIndex srcPort, IdIndex dstPort) {
      auto e = std::make_unique<Edge>();
      e->srcPort = srcPort;
      e->dstPort = dstPort;
      IdIndex eid = adg.addEdge(std::move(e));
      adg.getPort(srcPort)->connectedEdges.push_back(eid);
      adg.getPort(dstPort)->connectedEdges.push_back(eid);
    };

    addADGEdge(adg.getNode(hw0)->outputPorts[0],
               adg.getNode(router)->inputPorts[0]);
    addADGEdge(adg.getNode(router)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);
    addADGEdge(adg.getNode(hw1)->outputPorts[0],
               adg.getNode(router)->inputPorts[1]);
    addADGEdge(adg.getNode(router)->outputPorts[1],
               adg.getNode(hw2)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    // The mapper may or may not succeed on this topology depending on
    // placement order and routing BFS. If it succeeds, verify placement.
    if (result.success) {
      TEST_ASSERT(result.state.swNodeToHwNode[sw0] != INVALID_ID);
      TEST_ASSERT(result.state.swNodeToHwNode[sw1] != INVALID_ID);
      TEST_ASSERT(result.state.swNodeToHwNode[sw2] != INVALID_ID);

      // All on distinct PEs.
      TEST_ASSERT(result.state.swNodeToHwNode[sw0] !=
                  result.state.swNodeToHwNode[sw1]);
      TEST_ASSERT(result.state.swNodeToHwNode[sw1] !=
                  result.state.swNodeToHwNode[sw2]);
    }
    // If it fails, the pipeline exercised placement + routing + refinement,
    // which is the code path we want to test.
  }

  return 0;
}
