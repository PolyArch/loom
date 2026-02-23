//===-- techmapping_exhaustiveness.cpp - Tech-mapping completeness -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify tech-mapping exhaustiveness and overlap: CandidateBuilder failure
// diagnostics for incompatible DFG nodes, success when all nodes have
// candidates, multiple PE types producing multiple candidates, and overlap
// scenarios where two multi-op PE groups claim the same DFG node.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/CandidateBuilder.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/TechMapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

/// Helper: add a named string attribute to a node.
void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

/// Helper: add an array-of-strings attribute to a node.
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

/// Helper: add an array-of-integers attribute to a node.
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

/// Helper: create an operation node for DFG.
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

/// Helper: create a PE node for ADG.
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

/// Helper: add an edge between operation nodes by port index.
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

  // Test 1: CandidateBuilder returns failure with diagnostics when a DFG
  // node has no compatible PE.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG has an arith.muli, but ADG only provides arith.addi PEs.
    addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 2, 1);

    CandidateBuilder builder;
    CandidateBuilder::Result result = builder.build(dfg, adg);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(result.failedNode == 0);
    TEST_ASSERT(!result.diagnostics.empty());
    // Verify diagnostics contain the expected error code.
    TEST_ASSERT(result.diagnostics.find("CPL_MAPPER_NO_COMPATIBLE_HW") !=
                std::string::npos);
    // Verify diagnostics mention the operation name.
    TEST_ASSERT(result.diagnostics.find("arith.muli") != std::string::npos);
  }

  // Test 2: CandidateBuilder returns success when all DFG nodes have candidates.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: addi -> muli chain.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: matching PEs for both operations.
    addPENode(adg, ctx, "arith.addi", "functional", 2, 1);
    addPENode(adg, ctx, "arith.muli", "functional", 2, 1);

    CandidateBuilder builder;
    CandidateBuilder::Result result = builder.build(dfg, adg);

    TEST_ASSERT(result.success);
    TEST_ASSERT(result.failedNode == INVALID_ID);
    TEST_ASSERT(result.diagnostics.empty());

    // Both DFG nodes must have non-empty candidate sets.
    TEST_ASSERT(result.candidates.count(sw0) > 0);
    TEST_ASSERT(!result.candidates[sw0].empty());
    TEST_ASSERT(result.candidates.count(sw1) > 0);
    TEST_ASSERT(!result.candidates[sw1].empty());
  }

  // Test 3: Multiple PE types for one operation - CandidateBuilder includes
  // all compatible options.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: single arith.addi operation.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);

    // ADG: 3 compatible arith.addi PEs.
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex hw2 = addPENode(adg, ctx, "arith.addi", "functional", 2, 1);

    CandidateBuilder builder;
    CandidateBuilder::Result result = builder.build(dfg, adg);

    TEST_ASSERT(result.success);
    TEST_ASSERT(result.candidates.count(sw0) > 0);

    // All 3 PEs should appear as candidates.
    auto &cands = result.candidates[sw0];
    TEST_ASSERT(cands.size() == 3);

    // Verify all 3 HW node IDs are present.
    bool foundHw0 = false, foundHw1 = false, foundHw2 = false;
    for (const auto &c : cands) {
      if (c.hwNodeId == hw0) foundHw0 = true;
      if (c.hwNodeId == hw1) foundHw1 = true;
      if (c.hwNodeId == hw2) foundHw2 = true;
    }
    TEST_ASSERT(foundHw0);
    TEST_ASSERT(foundHw1);
    TEST_ASSERT(foundHw2);
  }

  // Test 4: Overlap scenario - when two multi-op PE groups could claim the
  // same DFG node, both appear in candidates.
  // PE-A body: {arith.addi, arith.muli} (addi -> muli)
  // PE-B body: {arith.addi, arith.subi} (addi -> subi)
  // DFG: addi -> muli, addi -> subi
  // The addi node is covered by both PE-A and PE-B group patterns.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: addi fans out to muli and subi.
    IdIndex swAdd = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 2);
    IdIndex swMul = addOpNode(dfg, ctx, "arith.muli", "functional", 2, 1);
    IdIndex swSub = addOpNode(dfg, ctx, "arith.subi", "functional", 2, 1);
    addEdgeBetween(dfg, swAdd, 0, swMul, 0);
    addEdgeBetween(dfg, swAdd, 1, swSub, 0);

    // PE-A: multi-op PE with body_ops = [arith.addi, arith.muli].
    auto hwNodeA = std::make_unique<Node>();
    hwNodeA->kind = Node::OperationNode;
    setStringAttr(hwNodeA.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNodeA.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNodeA.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    setArrayIntAttr(hwNodeA.get(), ctx, "body_edges", {0, 1});
    IdIndex hwA = adg.addNode(std::move(hwNodeA));
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwA;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwA)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwA;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwA)->outputPorts.push_back(pid);
    }

    // PE-B: multi-op PE with body_ops = [arith.addi, arith.subi].
    auto hwNodeB = std::make_unique<Node>();
    hwNodeB->kind = Node::OperationNode;
    setStringAttr(hwNodeB.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(hwNodeB.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(hwNodeB.get(), ctx, "body_ops",
                    {"arith.addi", "arith.subi"});
    setArrayIntAttr(hwNodeB.get(), ctx, "body_edges", {0, 1});
    IdIndex hwB = adg.addNode(std::move(hwNodeB));
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwB;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwB)->inputPorts.push_back(pid);
    }
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwB;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwB)->outputPorts.push_back(pid);
    }

    CandidateBuilder builder;
    CandidateBuilder::Result result = builder.build(dfg, adg);

    // The addi node should have candidates from both PE-A and PE-B groups.
    TEST_ASSERT(result.candidates.count(swAdd) > 0);

    bool foundGroupA = false, foundGroupB = false;
    for (const auto &c : result.candidates[swAdd]) {
      if (c.isGroup && c.hwNodeId == hwA)
        foundGroupA = true;
      if (c.isGroup && c.hwNodeId == hwB)
        foundGroupB = true;
    }
    // Both multi-op PE groups should produce candidates for the addi node.
    TEST_ASSERT(foundGroupA);
    TEST_ASSERT(foundGroupB);

    // The muli node should have a group candidate from PE-A.
    TEST_ASSERT(result.candidates.count(swMul) > 0);
    bool mulHasGroupA = false;
    for (const auto &c : result.candidates[swMul]) {
      if (c.isGroup && c.hwNodeId == hwA)
        mulHasGroupA = true;
    }
    TEST_ASSERT(mulHasGroupA);

    // The subi node should have a group candidate from PE-B.
    TEST_ASSERT(result.candidates.count(swSub) > 0);
    bool subHasGroupB = false;
    for (const auto &c : result.candidates[swSub]) {
      if (c.isGroup && c.hwNodeId == hwB)
        subHasGroupB = true;
    }
    TEST_ASSERT(subHasGroupB);
  }

  return 0;
}
