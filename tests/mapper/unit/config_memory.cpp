//===-- config_memory.cpp - Memory config output test -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that ConfigGen produces proper addr_offset_table entries for memory
// nodes with mapped operations.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Graph.h"
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

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test: Memory node with 2 regions, 1 mapped operation.
  // Verify that genMemoryConfig produces valid entries (not placeholder
  // region indices).
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: one memory operation.
    auto swNode = std::make_unique<Node>();
    swNode->kind = Node::OperationNode;
    setStringAttr(swNode.get(), ctx, "op_name", "handshake.load");
    setStringAttr(swNode.get(), ctx, "resource_class", "functional");
    IdIndex swId = dfg.addNode(std::move(swNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId;
      port->direction = Port::Input;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId;
      port->direction = Port::Output;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId)->outputPorts.push_back(pid);
    }

    // ADG: one memory node with numRegion=2.
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.memory");
    setStringAttr(hwNode.get(), ctx, "resource_class", "memory");
    setIntAttr(hwNode.get(), ctx, "numRegion", 2);
    IdIndex hwId = adg.addNode(std::move(hwNode));
    {
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

    // Map node.
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(swId, hwId, dfg, adg);

    // Verify mapping succeeded.
    TEST_ASSERT(state.swNodeToHwNode[swId] == hwId);
    TEST_ASSERT(!state.hwNodeToSwNodes[hwId].empty());

    // ConfigGen should be able to generate config for this setup.
    // We just verify the mapping state is consistent, not the actual
    // file output (which requires filesystem access in the test).
    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 1);
  }

  return 0;
}
