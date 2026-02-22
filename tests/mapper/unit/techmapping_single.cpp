//===-- techmapping_single.cpp - Single-op tech-mapping test ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that TechMapper correctly matches single DFG operations to
// compatible ADG PE nodes.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/Graph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

/// Helper: add a named attribute to a node.
void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

/// Helper: create a simple operation node with op_name and resource_class.
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

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Direct name match - arith.addi maps to arith.addi PE.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addOpNode(adg, ctx, "arith.addi", "functional", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(!candidates[0].empty());
    TEST_ASSERT(candidates[0][0].hwNodeId == 0);
    TEST_ASSERT(candidates[0][0].swNodeIds.size() == 1);
    TEST_ASSERT(candidates[0][0].swNodeIds[0] == 0);
    TEST_ASSERT(!candidates[0][0].isGroup);
  }

  // Test 2: arith.cmpi matches arith.cmpi regardless of predicate variant.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.cmpi.eq", "functional", 2, 1);
    addOpNode(adg, ctx, "arith.cmpi.slt", "functional", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(!candidates[0].empty());
  }

  // Test 3: Incompatible operations - arith.addi does not match arith.muli.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addOpNode(adg, ctx, "arith.muli", "functional", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(candidates[0].empty());
  }

  // Test 4: Routing nodes are not placement targets.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addOpNode(adg, ctx, "fabric.switch", "routing", 2, 2);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(candidates[0].empty());
  }

  // Test 5: Port count compatibility - SW has more ports than HW.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 3, 1);
    addOpNode(adg, ctx, "arith.addi", "functional", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(candidates[0].empty());
  }

  // Test 6: Memory operation matching.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "handshake.load", "functional", 2, 1);
    addOpNode(adg, ctx, "fabric.memory", "memory", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(!candidates[0].empty());
  }

  // Test 7: Multiple candidates - DFG op matches multiple ADG nodes.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addOpNode(adg, ctx, "arith.addi", "functional", 2, 1);
    addOpNode(adg, ctx, "arith.addi", "functional", 2, 1);

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    TEST_ASSERT(candidates[0].size() == 2);
  }

  return 0;
}
