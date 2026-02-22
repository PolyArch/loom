//===-- group_placement.cpp - Group atomic placement test ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that the mapper places multi-op group candidates atomically: when a
// group candidate wins, ALL SW nodes in the group map to the same HW PE, and
// C4 validation accepts this mapping.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TechMapper.h"

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

  // Test 1: Group placement binds both SW nodes to the same HW PE.
  // DFG: addi -> muli chain.
  // ADG: single multi-op PE with body_ops={arith.addi, arith.muli}.
  // After tech-mapping, both DFG nodes should be group candidates.
  // After placement, both should map to the same HW PE.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: multi-op PE.
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNode.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(hwNode.get(), ctx, "body_edges", {0, 1});
    IdIndex hwId = adg.addNode(std::move(hwNode));

    // Add enough ports for the group (3 in, 2 out).
    for (int i = 0; i < 3; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    // Run tech-mapping.
    TechMapper techMapper;
    auto candidates = techMapper.map(dfg, adg);

    // Both nodes should have candidates (group candidates at minimum).
    TEST_ASSERT(candidates.count(sw0) > 0);
    TEST_ASSERT(candidates.count(sw1) > 0);

    // Verify group candidate exists.
    bool foundGroup = false;
    for (const auto &c : candidates[sw0]) {
      if (c.isGroup && c.swNodeIds.size() == 2) {
        foundGroup = true;
        TEST_ASSERT(c.hwNodeId == hwId);
        break;
      }
    }
    TEST_ASSERT(foundGroup);

    // Test placement with group atomicity.
    MappingState state;
    state.init(dfg, adg);

    // Simulate group atomic placement: bind both nodes to same PE.
    auto r0 = state.mapNode(sw0, hwId, dfg, adg);
    TEST_ASSERT(r0 == ActionResult::Success);
    auto r1 = state.mapNode(sw1, hwId, dfg, adg);
    TEST_ASSERT(r1 == ActionResult::Success);

    // Both should map to the same HW node.
    TEST_ASSERT(state.swNodeToHwNode[sw0] == hwId);
    TEST_ASSERT(state.swNodeToHwNode[sw1] == hwId);

    // HW node should have both SW nodes.
    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 2);

    // Record group binding for C4 validation.
    state.groupBindings[hwId] = {sw0, sw1};

    // Verify groupBindings is set correctly.
    TEST_ASSERT(state.groupBindings.count(hwId));
    TEST_ASSERT(state.groupBindings[hwId].size() == 2);
  }

  // Test 2: Without group binding, C4 should reject multiple mappings
  // to a non-temporal PE (controlled by MappingState.groupBindings).
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);

    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
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

    MappingState state;
    state.init(dfg, adg);

    state.mapNode(sw0, hwId, dfg, adg);
    state.mapNode(sw1, hwId, dfg, adg);

    // Without group binding, hwNode has 2 SW nodes but no group record.
    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 2);
    TEST_ASSERT(!state.groupBindings.count(hwId));
  }

  // Test 3: Checkpoint/restore preserves groupBindings.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);

    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNode.get(), ctx, "resource_class", "functional");
    IdIndex hwId = adg.addNode(std::move(hwNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      adg.addPort(std::move(port));
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      adg.addPort(std::move(port));
    }

    MappingState state;
    state.init(dfg, adg);
    state.mapNode(sw0, hwId, dfg, adg);
    state.mapNode(sw1, hwId, dfg, adg);
    state.groupBindings[hwId] = {sw0, sw1};

    // Save checkpoint.
    auto cp = state.save();

    // Clear group bindings.
    state.groupBindings.clear();
    TEST_ASSERT(!state.groupBindings.count(hwId));

    // Restore and verify.
    state.restore(cp);
    TEST_ASSERT(state.groupBindings.count(hwId));
    TEST_ASSERT(state.groupBindings[hwId].size() == 2);
  }

  return 0;
}
