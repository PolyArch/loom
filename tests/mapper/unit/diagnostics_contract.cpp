//===-- diagnostics_contract.cpp - Diagnostics contract test ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies the diagnostics contract: that failed mappings produce specific
// constraint-class prefixed messages with meaningful content.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <vector>

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

/// Check that a diagnostics string contains non-whitespace content.
bool hasContent(const std::string &diag) {
  for (char c : diag) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
      return true;
  }
  return false;
}

/// Check that a diagnostics string contains a given substring.
bool containsSubstring(const std::string &diag, const std::string &sub) {
  return diag.find(sub) != std::string::npos;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Incompatible ops -> CandidateBuilder failure.
  // DFG has arith.muli but ADG only supports arith.addi.
  // Diagnostics must contain "CPL_MAPPER_NO_COMPATIBLE_HW" and the op name.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    TEST_ASSERT(hasContent(result.diagnostics));
    TEST_ASSERT(containsSubstring(result.diagnostics,
                                  "CPL_MAPPER_NO_COMPATIBLE_HW"));
    TEST_ASSERT(containsSubstring(result.diagnostics, "arith.muli"));
  }

  // Test 2: Routing failure -> "Routing failed" diagnostic.
  // DFG has 2 connected nodes but ADG has 2 disconnected PEs (no routing
  // path between them). Diagnostics must contain "Routing failed".
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // Two disconnected PEs: no edges between them.
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    TEST_ASSERT(containsSubstring(result.diagnostics, "Routing failed"));
  }

  // Test 3: Capacity violation -> "Placement failed" or "Validation failed: C4:".
  // DFG has 3 nodes (arith.addi) but ADG only has 2 PEs. The mapper cannot
  // place all 3 DFG nodes on 2 non-temporal PEs. Diagnostics must indicate
  // either a placement failure or a C4 validation failure.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);
    addEdgeBetween(dfg, sw1, 0, sw2, 0);

    // Only 2 PEs with direct connection.
    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);

    // Not enough PEs for all DFG nodes.
    TEST_ASSERT(!result.success);
    TEST_ASSERT(!result.diagnostics.empty());
    // The mapper must report either a placement-level failure or a C4
    // validation constraint (capacity exceeded).
    bool hasCapacityDiag =
        containsSubstring(result.diagnostics, "Placement failed") ||
        containsSubstring(result.diagnostics, "C4:") ||
        containsSubstring(result.diagnostics, "Validation failed");
    TEST_ASSERT(hasCapacityDiag);
  }

  return 0;
}
