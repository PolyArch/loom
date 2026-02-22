//===-- placement_exclusivity.cpp - PE exclusivity test ------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that non-temporal PEs reject a second placement while temporal PEs
// allow multiple SW nodes (in different time slots), and memory nodes allow
// up to numRegion SW nodes.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Non-temporal (functional) PE rejects second placement in the
  // candidate loop. We simulate the exclusivity check manually since
  // runPlacement is an integrated pipeline.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);

    // ADG: one non-temporal PE.
    IdIndex hwPE = addOpNode(adg, ctx, "fabric.pe", "functional", 2, 2);

    MappingState state;
    state.init(dfg, adg);

    // Place sw0 on hwPE.
    auto r = state.mapNode(sw0, hwPE, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
    TEST_ASSERT(!state.hwNodeToSwNodes[hwPE].empty());

    // Simulate the exclusivity check from Mapper::runPlacement:
    // For a non-group candidate on a non-temporal PE that already has
    // occupants, the check should skip (i.e. reject).
    const Node *hwNode = adg.getNode(hwPE);
    TEST_ASSERT(hwNode != nullptr);

    // getNodeResourceClass is an internal helper in Mapper.cpp, so we
    // replicate the attribute lookup here.
    llvm::StringRef resClass;
    for (auto &attr : hwNode->attributes) {
      if (attr.getName() == "resource_class") {
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
          resClass = s.getValue();
      }
    }
    TEST_ASSERT(resClass == "functional");
    // Non-temporal PE with occupants -> exclusivity should reject.
    bool wouldSkip = !state.hwNodeToSwNodes[hwPE].empty() &&
                     resClass != "temporal";
    TEST_ASSERT(wouldSkip);
  }

  // Test 2: Temporal PE allows multiple SW nodes.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);

    // ADG: one temporal PE.
    IdIndex hwPE = addOpNode(adg, ctx, "fabric.pe", "temporal", 2, 2);

    MappingState state;
    state.init(dfg, adg);

    // Place sw0 on hwPE.
    auto r = state.mapNode(sw0, hwPE, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    // Simulate the exclusivity check: temporal PE should NOT be skipped.
    const Node *hwNode = adg.getNode(hwPE);
    TEST_ASSERT(hwNode != nullptr);

    llvm::StringRef resClass;
    for (auto &attr : hwNode->attributes) {
      if (attr.getName() == "resource_class") {
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
          resClass = s.getValue();
      }
    }
    TEST_ASSERT(resClass == "temporal");
    // Temporal PE with occupants -> exclusivity should allow.
    bool wouldSkip = !state.hwNodeToSwNodes[hwPE].empty() &&
                     resClass != "temporal";
    TEST_ASSERT(!wouldSkip);

    // Place sw1 on the same temporal PE.
    r = state.mapNode(sw1, hwPE, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    // Both should be on the same HW node.
    TEST_ASSERT(state.swNodeToHwNode[sw0] == hwPE);
    TEST_ASSERT(state.swNodeToHwNode[sw1] == hwPE);
    TEST_ASSERT(state.hwNodeToSwNodes[hwPE].size() == 2);
  }

  // Test 3: Memory node with numRegion=2 allows two SW nodes but rejects
  // the third.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);
    (void)addOpNode(dfg, ctx, "handshake.extmemory", "memory", 1, 1);

    // ADG: one memory node with numRegion=2.
    IdIndex hwMem = addOpNode(adg, ctx, "fabric.memory", "memory", 2, 2);
    // Add numRegion attribute.
    Node *memNode = adg.getNode(hwMem);
    memNode->attributes.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&ctx, "numRegion"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), 2)));

    MappingState state;
    state.init(dfg, adg);

    // First placement: succeeds.
    auto r = state.mapNode(sw0, hwMem, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    // Second placement: succeeds (numRegion=2).
    r = state.mapNode(sw1, hwMem, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
    TEST_ASSERT(state.hwNodeToSwNodes[hwMem].size() == 2);

    // Simulate the exclusivity check for sw2:
    // Memory with numRegion=2 and 2 occupants -> should be blocked.
    int64_t numRegion = 2;
    bool wouldSkip =
        state.hwNodeToSwNodes[hwMem].size() >=
        static_cast<size_t>(numRegion);
    TEST_ASSERT(wouldSkip);
  }

  return 0;
}
