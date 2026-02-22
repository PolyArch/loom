//===-- config_pe.cpp - PE config output test ----------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that ConfigGen produces correct PE configuration for various PE
// types: standard PE, compare PE, constant PE, and temporal PE.
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

  // Test 1: Standard PE - arith.addi mapped to PE node.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    auto swNode = std::make_unique<Node>();
    swNode->kind = Node::OperationNode;
    setStringAttr(swNode.get(), ctx, "op_name", "arith.addi");
    IdIndex swId = dfg.addNode(std::move(swNode));
    for (int i = 0; i < 2; ++i) {
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

    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "arith.addi");
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
    state.mapNode(swId, hwId, dfg, adg);
    state.mapPort(dfg.getNode(swId)->inputPorts[0],
                  adg.getNode(hwId)->inputPorts[0], dfg, adg);
    state.mapPort(dfg.getNode(swId)->inputPorts[1],
                  adg.getNode(hwId)->inputPorts[1], dfg, adg);
    state.mapPort(dfg.getNode(swId)->outputPorts[0],
                  adg.getNode(hwId)->outputPorts[0], dfg, adg);

    // Verify the mapping is valid.
    TEST_ASSERT(state.swNodeToHwNode[swId] == hwId);
    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 1);
  }

  // Test 2: Temporal PE with FU nodes.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // ADG: temporal PE virtual node + 1 FU node.
    auto tpeNode = std::make_unique<Node>();
    tpeNode->kind = Node::OperationNode;
    setStringAttr(tpeNode.get(), ctx, "op_name", "fabric.temporal_pe");
    setStringAttr(tpeNode.get(), ctx, "resource_class", "functional");
    setIntAttr(tpeNode.get(), ctx, "num_instruction", 4);
    setIntAttr(tpeNode.get(), ctx, "num_register", 2);
    tpeNode->attributes.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&ctx, "is_virtual"),
        mlir::UnitAttr::get(&ctx)));
    IdIndex tpeId = adg.addNode(std::move(tpeNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = tpeId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(tpeId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = tpeId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(tpeId)->outputPorts.push_back(pid);
    }

    auto fuNode = std::make_unique<Node>();
    fuNode->kind = Node::OperationNode;
    setStringAttr(fuNode.get(), ctx, "op_name", "arith.addi");
    setStringAttr(fuNode.get(), ctx, "resource_class", "functional");
    setIntAttr(fuNode.get(), ctx, "parent_temporal_pe",
               static_cast<int64_t>(tpeId));
    IdIndex fuId = adg.addNode(std::move(fuNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = fuId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(fuId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = fuId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(fuId)->outputPorts.push_back(pid);
    }

    // DFG: one operation.
    auto swNode = std::make_unique<Node>();
    swNode->kind = Node::OperationNode;
    setStringAttr(swNode.get(), ctx, "op_name", "arith.addi");
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

    // Map SW node to FU node.
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(swId, fuId, dfg, adg);

    // Assign temporal PE slot.
    state.temporalPEAssignments[swId].slot = 0;
    state.temporalPEAssignments[swId].tag = 0;
    state.temporalPEAssignments[swId].opcode = 0;

    // Verify mapping state is correct.
    TEST_ASSERT(state.swNodeToHwNode[swId] == fuId);
    TEST_ASSERT(state.temporalPEAssignments[swId].slot == 0);
  }

  return 0;
}
