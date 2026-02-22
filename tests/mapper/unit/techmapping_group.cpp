//===-- techmapping_group.cpp - Multi-op group tech-mapping test --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that TechMapper finds multi-op group candidates when a PE body
// pattern matches a DFG subgraph.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/Graph.h"

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

void setArrayIntAttr(Node *node, mlir::MLIRContext &ctx,
                     llvm::StringRef name,
                     llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (auto v : values)
    attrs.push_back(mlir::IntegerAttr::get(
        mlir::IntegerType::get(&ctx, 64), v));
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

  // Test 1: Multi-op PE with body_ops={arith.addi, arith.muli}
  // DFG has addi -> muli chain, should create group candidate.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: addi -> muli
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: multi-op PE with body_ops = [arith.addi, arith.muli]
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNode.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(hwNode.get(), ctx, "body_edges", {0, 1});

    IdIndex hwId = adg.addNode(std::move(hwNode));

    // Add ports to the PE.
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    // Both DFG nodes should have candidates.
    TEST_ASSERT(candidates.count(sw0) > 0);
    TEST_ASSERT(candidates.count(sw1) > 0);

    // At least one candidate should be a group candidate.
    bool foundGroup = false;
    for (const auto &c : candidates[sw0]) {
      if (c.isGroup && c.swNodeIds.size() == 2) {
        foundGroup = true;
        break;
      }
    }
    TEST_ASSERT(foundGroup);
  }

  // Test 2: Multi-op PE where DFG doesn't match pattern (wrong connectivity).
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: muli -> addi (reverse order from PE body).
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: PE with body_ops = [arith.addi, arith.muli] (addi first).
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNode.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(hwNode.get(), ctx, "body_edges", {0, 1});

    IdIndex hwId = adg.addNode(std::move(hwNode));
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    // sw0 (muli) should NOT have a group candidate starting from it
    // since the PE body starts with addi.
    bool foundGroupForSw0 = false;
    if (candidates.count(sw0)) {
      for (const auto &c : candidates[sw0]) {
        if (c.isGroup && c.swNodeIds.size() == 2)
          foundGroupForSw0 = true;
      }
    }
    // The group matching starts from addi-compatible nodes,
    // so sw0 (muli) won't be the root of a group match.
    // sw1 (addi) could start a group if it finds a muli neighbor.
    // The edge goes muli->addi, so sw1's input neighbor is sw0 (muli).
    // That should match body_ops[1] = arith.muli.

    // Verify: sw0 (muli) should NOT have a group candidate rooted on it,
    // since the PE body starts with addi. Group candidates are anchored
    // to the first body op.
    TEST_ASSERT(!foundGroupForSw0);

    // With reversed connectivity (muli->addi vs PE body addi->muli),
    // the group match also fails for sw1 as root because the required
    // DFG edge direction doesn't match the PE body edge direction.
    // Multi-op PEs return false for isSingleOpCompatible, so neither
    // node gets any candidates from this PE.
    bool foundGroupForSw1 = false;
    if (candidates.count(sw1)) {
      for (const auto &c : candidates[sw1]) {
        if (c.isGroup && c.swNodeIds.size() == 2)
          foundGroupForSw1 = true;
      }
    }
    TEST_ASSERT(!foundGroupForSw1);
  }

  // Test 3: Single-op PE body should not produce group candidates.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);

    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNode.get(), ctx, "body_ops", {"arith.addi"});

    IdIndex hwId = adg.addNode(std::move(hwNode));
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    TechMapper mapper;
    auto candidates = mapper.map(dfg, adg);

    TEST_ASSERT(candidates.count(0) > 0);
    // Should have a single-op candidate, not a group.
    bool anyGroup = false;
    for (const auto &c : candidates[0]) {
      if (c.isGroup)
        anyGroup = true;
    }
    TEST_ASSERT(!anyGroup);
    TEST_ASSERT(!candidates[0].empty());
  }

  return 0;
}
